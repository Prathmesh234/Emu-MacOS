"""Download a Tinker LoRA checkpoint and publish it to the HuggingFace Hub.

By default this downloads the sampler_weights/000050 checkpoint from the
Qwen3.5-397B-A17B VLM SFT run (same checkpoint sample_vlm.py points at),
converts it to a standalone PEFT adapter, and pushes it to HuggingFace.

The published artifact references the base model
(``Qwen/Qwen3.5-397B-A17B``) in ``adapter_config.json``, so consumers can
load both pieces with:

    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-397B-A17B")
    model = PeftModel.from_pretrained(base, "ppbhatt500/<repo>")

Prerequisites:

    # one-time, in the training/ venv
    uv pip install tinker tinker-cookbook

    # credentials (also picked up from training/.env)
    export TINKER_API_KEY=...
    export HF_TOKEN=...

Usage:

    uv run python publish_checkpoint_to_hf.py
    uv run python publish_checkpoint_to_hf.py --repo-id ppbhatt500/my-repo --public
    uv run python publish_checkpoint_to_hf.py --merge   # also push merged model

Notes:
    * ``weights.download`` and ``weights.build_lora_adapter`` do not spin up
      a training or sampling client, so they don't consume Tinker compute
      credits. Only the HF Hub upload uses bandwidth.
    * ``--merge`` materialises the full ~800 GB bf16 base model on disk
      and uploads it. Don't enable that unless you actually have the disk,
      RAM, and HF Hub quota for it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---- defaults (match sample_vlm.py / train_vlm_sft.py) ----------------------

DEFAULT_CHECKPOINT = (
    "tinker://2c65dd6d-78ae-53e5-a7c1-00b03ea21097:train:0/"
    "sampler_weights/000050"
)
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-397B-A17B"
DEFAULT_REPO_ID = "ppbhatt500/Emu-Qwen3.5-397B-A17B-LoRA-step50"
DEFAULT_WORKDIR = Path(__file__).parent / "data" / "checkpoints" / "step50"

# ---- env loading (mirrors sample_vlm.py) ------------------------------------


def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ---- main pipeline ----------------------------------------------------------


def _download_checkpoint(tinker_path: str, output_dir: Path) -> Path:
    from tinker_cookbook import weights

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] Downloading {tinker_path}", flush=True)
    print(f"      → {output_dir}", flush=True)
    adapter_dir = weights.download(
        tinker_path=tinker_path,
        output_dir=str(output_dir),
    )
    print(f"      ✓ extracted to {adapter_dir}", flush=True)
    return Path(adapter_dir)


def _build_peft_adapter(
    base_model: str, adapter_dir: Path, output_dir: Path
) -> Path:
    import shutil

    # ── Workaround for upstream issue ────────────────────────────────────
    # weights.build_lora_adapter internally calls
    # tinker_cookbook.weights._artifacts.resolve_model_dir(base_model)
    # which does snapshot_download(repo_id) WITH NO allow_patterns,
    # pulling the entire base model. For Qwen3.5-397B-A17B that's
    # ~800 GB, which is absurd because the only thing the adapter
    # conversion needs from the base model is the set of weight key
    # names (used for QKV-fusion planning, MoE detection, and tied-
    # embedding remaps).
    #
    # The full key list is already available in
    # `model.safetensors.index.json` — a few-MB manifest. So we:
    #   1. Pre-download only json/text/tokenizer/model-code files.
    #   2. Monkey-patch resolve_model_dir() to return that local dir.
    #   3. Monkey-patch get_model_state_keys() to read keys from the
    #      safetensors index json instead of opening every shard.
    # ──────────────────────────────────────────────────────────────────────
    metadata_dir = _prefetch_base_model_metadata(base_model)
    _install_metadata_only_patches(base_model, metadata_dir)

    from tinker_cookbook import weights

    # build_lora_adapter raises FileExistsError if output_path already
    # exists, so wipe any stale output from a previous run.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[2/4] Converting Tinker adapter → PEFT format", flush=True)
    print(f"      base_model={base_model}", flush=True)
    print(f"      → {output_dir}", flush=True)
    weights.build_lora_adapter(
        base_model=base_model,
        adapter_path=str(adapter_dir),
        output_path=str(output_dir),
        trust_remote_code=True,
    )
    files = sorted(output_dir.iterdir())
    for f in files:
        size_mb = f.stat().st_size / 1e6
        print(f"      {f.name:40s} {size_mb:>10.2f} MB", flush=True)
    return output_dir


def _prefetch_base_model_metadata(base_model: str) -> Path:
    """Download only metadata/config files for the base model.

    Skips all ``*.safetensors`` weight shards — we only need the index
    json (which lists every key) plus config/tokenizer/jinja files.
    """
    from huggingface_hub import snapshot_download

    print(f"[2a/4] Pre-fetching base model metadata "
          f"(config + tokenizer + safetensors.index, no weights)",
          flush=True)
    local_dir = snapshot_download(
        repo_id=base_model,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.jinja",
            "*.py",
            "tokenizer*",
            "merges*",
            "vocab*",
            "README.md",
            "LICENSE",
            "LICENSE.txt",
        ],
    )
    index = Path(local_dir) / "model.safetensors.index.json"
    if not index.exists():
        raise SystemExit(
            f"{base_model} has no model.safetensors.index.json — this "
            "metadata-only fast path can't reconstruct the weight key "
            "list. Re-run with the full snapshot_download (will pull "
            "the entire base model)."
        )
    print(f"       → {local_dir} (index lists "
          f"{len(_read_index_keys(index))} weight keys)", flush=True)
    return Path(local_dir)


def _read_index_keys(index_path: Path) -> set[str]:
    import json
    with open(index_path) as f:
        return set(json.load(f)["weight_map"].keys())


def _install_metadata_only_patches(base_model: str,
                                   metadata_dir: Path) -> None:
    """Monkey-patch the cookbook to avoid downloading full base weights."""
    from tinker_cookbook.weights import _artifacts

    index_path = metadata_dir / "model.safetensors.index.json"
    cached_keys = _read_index_keys(index_path)

    # Some Qwen3.5 logic also needs shapes — but the adapter path only
    # uses keys (set membership). We expose shapes too in case future
    # cookbook versions start checking them; values are placeholder ().
    cached_shapes = {k: () for k in cached_keys}

    _orig_resolve = _artifacts.resolve_model_dir
    _orig_keys = _artifacts.get_model_state_keys
    _orig_shapes = _artifacts.get_model_state_shapes

    def _patched_resolve(model_id: str):
        if model_id == base_model:
            return metadata_dir
        return _orig_resolve(model_id)

    def _patched_keys(model_dir):
        if Path(model_dir).resolve() == metadata_dir.resolve():
            return cached_keys
        return _orig_keys(model_dir)

    def _patched_shapes(model_dir):
        if Path(model_dir).resolve() == metadata_dir.resolve():
            return cached_shapes
        return _orig_shapes(model_dir)

    _artifacts.resolve_model_dir = _patched_resolve
    _artifacts.get_model_state_keys = _patched_keys
    _artifacts.get_model_state_shapes = _patched_shapes

    # _adapter and _export import these symbols by name at module load,
    # so patch the rebound references too.
    import tinker_cookbook.weights._adapter as _adapter_mod
    import tinker_cookbook.weights._export as _export_mod
    for mod in (_adapter_mod, _export_mod):
        if hasattr(mod, "resolve_model_dir"):
            mod.resolve_model_dir = _patched_resolve
        if hasattr(mod, "get_model_state_keys"):
            mod.get_model_state_keys = _patched_keys
        if hasattr(mod, "get_model_state_shapes"):
            mod.get_model_state_shapes = _patched_shapes


def _build_merged_model(
    base_model: str, adapter_dir: Path, output_dir: Path, dtype: str
) -> Path:
    from tinker_cookbook import weights

    if output_dir.exists():
        raise FileExistsError(
            f"--merge output dir already exists: {output_dir}. "
            "Move or delete it before re-running with --merge."
        )
    print(f"[2b/4] Merging adapter into base model "
          f"(dtype={dtype}, this needs ~800 GB free)", flush=True)
    print(f"       → {output_dir}", flush=True)
    weights.build_hf_model(
        base_model=base_model,
        adapter_path=str(adapter_dir),
        output_path=str(output_dir),
        dtype=dtype,
        trust_remote_code=True,
    )
    return output_dir


def _download_raw_from_hf(repo_id: str, output_dir: Path,
                          hf_token: str | None) -> Path:
    """Rehydrate a previously-published raw Tinker checkpoint from HF.

    Used when we want to convert an already-uploaded raw adapter into PEFT
    format without paying for another `weights.download()` round-trip from
    Tinker (which also burns a time-limited signed URL on the cookbook side).
    """
    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] Rehydrating raw Tinker checkpoint from HF: {repo_id}",
          flush=True)
    print(f"      → {output_dir}", flush=True)
    local = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(output_dir),
        token=hf_token,
        # The raw upload contains only adapter_model.safetensors,
        # adapter_config.json, and README.md. Pull just those — no .git
        # history, no orphaned files.
        allow_patterns=["adapter_model*", "adapter_config*", "*.json",
                        "*.safetensors"],
    )
    print(f"      ✓ {local}", flush=True)
    return Path(local)


def _write_peft_readme(peft_dir: Path, base_model: str, repo_id: str,
                       checkpoint_uri: str,
                       republished_from_raw: str | None) -> None:
    """Overwrite the model card so the repo clearly advertises PEFT/vLLM
    compatibility (overrides whatever tinker_cookbook generates).

    Required when republishing a previously-raw repo: the repo name still
    has 'tinker-raw' in it, and stale READMEs from the earlier raw upload
    would otherwise confuse consumers about what they're loading.
    """
    rehydrate_note = ""
    if republished_from_raw:
        rehydrate_note = (
            f"\n> **Republished:** this repo previously hosted a raw "
            f"Tinker checkpoint at the same id "
            f"(`{republished_from_raw}`). It has been replaced in-place "
            f"with the PEFT/vLLM-compatible adapter below. The repo name "
            f"still contains `tinker-raw` for backward compatibility with "
            f"existing references, but the contents are now standard "
            f"PEFT.\n"
        )

    readme = peft_dir / "README.md"
    content = f"""---
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
tags:
- peft
- lora
- vllm
- vision-language-model
- computer-use
- desktop-agent
- emu
- osworld
- sft
datasets:
- xlangai/ubuntu_osworld_verified_trajs
- xlangai/computer-agent-arena
license: apache-2.0
language:
- en
---

# {repo_id}

**PEFT/vLLM-compatible LoRA adapter** for
[{base_model}](https://huggingface.co/{base_model}),
trained with [Tinker](https://thinkingmachines.ai/tinker) +
[tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
and converted to PEFT format with `weights.build_lora_adapter()`.
{rehydrate_note}
Original Tinker checkpoint: `{checkpoint_uri}`

## Load with PEFT

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "{base_model}", trust_remote_code=True
)
model = PeftModel.from_pretrained(base, "{repo_id}")
```

## Serve with vLLM

```bash
vllm serve {base_model} \\
    --enable-lora \\
    --lora-modules emu-step50={repo_id} \\
    --trust-remote-code
```

Then point requests at `model="emu-step50"`.

## Training

- **Recipe:** vision SFT on real OSWorld + Computer-Agent-Arena Gemini
  trajectories rewritten into Emu's remote-mode harness format. See
  `training/train_vlm_sft.py` in the
  [emu-macos repo](https://github.com/Prathmesh234/Emu-MacOS).
- **Base:** `{base_model}` (Hybrid + Vision MoE, qwen3_5_moe family).
- **LoRA rank:** 32
- **Renderer:** `qwen3_5_disable_thinking`
- **Checkpoint:** step 50
"""
    readme.write_text(content)


def _publish(
    model_path: Path,
    repo_id: str,
    *,
    base_model: str,
    private: bool,
    hf_token: str | None,
    label: str,
) -> str:
    from tinker_cookbook import weights
    from tinker_cookbook.weights import ModelCardConfig

    print(f"[{label}] Publishing {model_path} → {repo_id} "
          f"(private={private})", flush=True)
    card = ModelCardConfig(
        base_model=base_model,
        datasets=["xlangai/ubuntu_osworld_verified_trajs",
                  "xlangai/computer-agent-arena"],
        tags=["vision-language-model", "computer-use", "desktop-agent",
              "emu", "osworld", "sft"],
        license="apache-2.0",
        language=["en"],
    )
    url = weights.publish_to_hf_hub(
        model_path=str(model_path),
        repo_id=repo_id,
        private=private,
        token=hf_token,
        model_card=card,
    )
    print(f"      ✓ {url}", flush=True)
    return url


def _write_raw_readme(adapter_dir: Path, base_model: str, repo_id: str,
                      checkpoint_uri: str) -> None:
    """Write a README explaining this is a raw Tinker checkpoint."""
    readme = adapter_dir / "README.md"
    if readme.exists():
        return
    content = f"""---
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
tags:
- tinker
- tinker-cookbook
- raw-tinker-checkpoint
- lora
- vision-language-model
- computer-use
- desktop-agent
- emu
- osworld
- sft
datasets:
- xlangai/ubuntu_osworld_verified_trajs
- xlangai/computer-agent-arena
license: apache-2.0
language:
- en
---

# {repo_id}

**Raw Tinker LoRA checkpoint** for [{base_model}](https://huggingface.co/{base_model}),
trained with [Tinker](https://thinkingmachines.ai/tinker) + [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).

> This repo contains the **unconverted** Tinker adapter
> (`adapter_model.safetensors` + `adapter_config.json` as emitted by
> `weights.download()`). It is **not** in PEFT format yet — the conversion
> from Tinker's internal LoRA key layout to PEFT's expected names is a
> RAM-heavy operation for MoE models (30-60 GB for Qwen3.5-397B-A17B)
> and was deferred to the consumer side.

Original Tinker checkpoint: `{checkpoint_uri}`

## Producing a PEFT adapter from this repo

On a machine with 64+ GB RAM:

```bash
pip install tinker tinker-cookbook
huggingface-cli download {repo_id} --local-dir ./raw_adapter

python -c "
from tinker_cookbook import weights
weights.build_lora_adapter(
    base_model='{base_model}',
    adapter_path='./raw_adapter',
    output_path='./peft_adapter',
    trust_remote_code=True,
)
"
```

`./peft_adapter` is then a standard PEFT directory you can load with:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "{base_model}", trust_remote_code=True
)
model = PeftModel.from_pretrained(base, "./peft_adapter")
```

## Training

- **Recipe:** vision SFT on real OSWorld + Computer-Agent-Arena Gemini
  trajectories rewritten into Emu's remote-mode harness format. See
  `training/train_vlm_sft.py` in the
  [emu-macos repo](https://github.com/Prathmesh234/Emu-MacOS).
- **Base:** `{base_model}` (Hybrid + Vision MoE, qwen3_5_moe family).
- **LoRA rank:** 32
- **Renderer:** `qwen3_5_disable_thinking`
- **Checkpoint:** step 50
"""
    readme.write_text(content)


def _publish_raw(adapter_dir: Path, repo_id: str, *, base_model: str,
                 private: bool, hf_token: str | None) -> str:
    from huggingface_hub import HfApi

    print(f"[3/4] Publishing RAW Tinker checkpoint {adapter_dir} → "
          f"{repo_id} (private={private})", flush=True)
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model",
                    private=private, exist_ok=True)
    # upload_large_folder is designed for multi-GB uploads: it splits
    # the work into background workers, persists progress to
    # .huggingface/ inside adapter_dir, and resumes on retry. The
    # single-shot upload_folder() can deadlock on the final commit for
    # files in the 10+ GB range (observed at ~98% with hf_xet).
    try:
        api.upload_large_folder(
            folder_path=str(adapter_dir),
            repo_id=repo_id,
            repo_type="model",
            print_report=True,
        )
    except AttributeError:
        # Older huggingface_hub: fall back to upload_folder.
        api.upload_folder(folder_path=str(adapter_dir), repo_id=repo_id,
                          repo_type="model")
    url = f"https://huggingface.co/{repo_id}"
    print(f"      ✓ {url}", flush=True)
    return url


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help=f"Tinker sampler checkpoint URI. "
                         f"Default: {DEFAULT_CHECKPOINT}")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                    help=f"HF base model id. Default: {DEFAULT_BASE_MODEL}")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                    help=f"Destination HF Hub repo id. "
                         f"Default: {DEFAULT_REPO_ID}")
    ap.add_argument("--workdir", default=str(DEFAULT_WORKDIR), type=Path,
                    help=f"Local scratch dir for downloaded + converted "
                         f"artifacts. Default: {DEFAULT_WORKDIR}")
    ap.add_argument("--public", action="store_true",
                    help="Create the HF repo as public. Default is private.")
    ap.add_argument("--skip-download", action="store_true",
                    help="Reuse an existing download at "
                         "<workdir>/raw_adapter (skips step 1).")
    ap.add_argument("--skip-publish", action="store_true",
                    help="Only download + convert; do not upload to HF.")
    ap.add_argument("--upload-raw", action="store_true",
                    help="Upload the raw Tinker checkpoint as-is, "
                         "without running the PEFT conversion. Use this "
                         "when the local machine doesn't have enough RAM "
                         "to expand expert LoRA tensors (the 397B MoE "
                         "conversion needs 30-60 GB RAM). The consumer can "
                         "run build_lora_adapter() locally to produce the "
                         "PEFT format from this raw upload.")
    ap.add_argument("--from-hf-raw", nargs="?", const="__same__",
                    default=None, metavar="REPO_ID",
                    help="Rehydrate the raw Tinker checkpoint from an "
                         "existing HF repo instead of re-downloading from "
                         "Tinker. Pass with no value to default to "
                         "--repo-id (typical when republishing the same "
                         "repo as PEFT/vLLM-compatible). Implies "
                         "--skip-download and does NOT require "
                         "TINKER_API_KEY.")
    ap.add_argument("--merge", action="store_true",
                    help="Also build a fully merged HF model (~800 GB bf16) "
                         "and publish it to <repo-id>-merged. Requires huge "
                         "disk + RAM.")
    ap.add_argument("--merge-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="dtype for the merged model. Default: bfloat16.")
    ap.add_argument("--merged-repo-id", default=None,
                    help="HF repo id for the merged model. Default: "
                         "<repo-id>-merged.")
    args = ap.parse_args()

    _load_dotenv()

    workdir: Path = args.workdir
    raw_dir = workdir / "raw_adapter"
    peft_dir = workdir / "peft_adapter"
    merged_dir = workdir / "merged_model"

    hf_token = os.environ.get("HF_TOKEN")
    if not args.skip_publish and not hf_token:
        # huggingface_hub can also use the cached login; just warn.
        print("[warn] HF_TOKEN not set (env or .env). publish_to_hf_hub "
              "will fall back to `hf auth login` cache.", file=sys.stderr,
              flush=True)

    # Resolve --from-hf-raw sentinel ("" → use --repo-id).
    from_hf_raw = args.from_hf_raw
    if from_hf_raw == "__same__":
        from_hf_raw = args.repo_id
    if from_hf_raw and args.upload_raw:
        raise SystemExit("--from-hf-raw and --upload-raw are mutually "
                         "exclusive (the whole point of --from-hf-raw is "
                         "to convert an already-uploaded raw checkpoint "
                         "into PEFT format).")

    if (not args.skip_download and not from_hf_raw
            and not os.environ.get("TINKER_API_KEY")):
        raise SystemExit("TINKER_API_KEY not set (env or .env) — "
                         "needed by weights.download()")

    # 1. Download from Tinker (or rehydrate from a previously-uploaded
    #    raw HF repo when --from-hf-raw is set).
    if args.skip_download:
        if not raw_dir.exists():
            raise SystemExit(f"--skip-download set but {raw_dir} is empty. "
                             "Run once without --skip-download first.")
        adapter_dir = raw_dir
        print(f"[1/4] Skipping download, reusing {adapter_dir}", flush=True)
    elif from_hf_raw:
        adapter_dir = _download_raw_from_hf(from_hf_raw, raw_dir, hf_token)
    else:
        adapter_dir = _download_checkpoint(args.checkpoint, raw_dir)

    # Raw-upload short-circuit. We push the unmodified Tinker checkpoint
    # to HF so a machine with more RAM can run build_lora_adapter() later
    # without re-downloading from Tinker (which costs an extra round-trip
    # and burns the time-limited signed URL).
    if args.upload_raw:
        if args.merge:
            raise SystemExit("--upload-raw and --merge are mutually "
                             "exclusive.")
        _write_raw_readme(adapter_dir, args.base_model, args.repo_id,
                          args.checkpoint)
        if args.skip_publish:
            print(f"[4/4] --skip-publish set; raw checkpoint ready at "
                  f"{adapter_dir}", flush=True)
            return
        private = not args.public
        _publish_raw(adapter_dir, args.repo_id,
                     base_model=args.base_model,
                     private=private, hf_token=hf_token)
        print("\nDone. On a box with 64+ GB RAM, finish the conversion "
              "with:")
        print(f"    huggingface-cli download {args.repo_id} "
              f"--local-dir ./raw_adapter")
        print(f"    python -c \"from tinker_cookbook import weights; "
              f"weights.build_lora_adapter("
              f"base_model='{args.base_model}', "
              f"adapter_path='./raw_adapter', "
              f"output_path='./peft_adapter', "
              f"trust_remote_code=True)\"")
        return

    # 2. Convert to PEFT format (small, no base weights downloaded).
    _build_peft_adapter(args.base_model, adapter_dir, peft_dir)

    # 2c. Write our own README so the repo advertises PEFT/vLLM use
    #     (overrides whatever tinker_cookbook would otherwise generate
    #     during publish_to_hf_hub).
    _write_peft_readme(peft_dir, args.base_model, args.repo_id,
                       args.checkpoint,
                       republished_from_raw=from_hf_raw)

    # 3. (optional) merge into a full HF model.
    if args.merge:
        _build_merged_model(args.base_model, adapter_dir, merged_dir,
                            args.merge_dtype)

    # 4. Publish.
    if args.skip_publish:
        print(f"[4/4] --skip-publish set; PEFT adapter ready at {peft_dir}",
              flush=True)
        if args.merge:
            print(f"       merged model ready at {merged_dir}", flush=True)
        return

    private = not args.public
    _publish(peft_dir, args.repo_id, base_model=args.base_model,
             private=private, hf_token=hf_token, label="3/4")

    if args.merge:
        merged_repo = (args.merged_repo_id
                       or f"{args.repo_id}-merged")
        _publish(merged_dir, merged_repo, base_model=args.base_model,
                 private=private, hf_token=hf_token, label="4/4")

    print("\nDone. Load the adapter with:")
    print(f"    from peft import PeftModel")
    print(f"    from transformers import AutoModelForCausalLM")
    print(f'    base = AutoModelForCausalLM.from_pretrained('
          f'"{args.base_model}", trust_remote_code=True)')
    print(f'    model = PeftModel.from_pretrained(base, "{args.repo_id}")')


if __name__ == "__main__":
    main()
