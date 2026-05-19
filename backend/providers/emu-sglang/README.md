# emu-sglang

Self-contained launcher for serving the Emu fine-tuned
`Qwen/Qwen3.5-397B-A17B` with **SGLang**, using the LoRA adapter at
`ppbhatt500/Emu-Qwen3.5-397B-A17B-LoRA-step50-tinker-raw` on HuggingFace.

## ⚠️ Compatibility caveat

The tinker-cookbook documents that **SGLang does not yet support
MoE expert-LoRA** serving for the `qwen3_5_moe` model family — and
`Qwen/Qwen3.5-397B-A17B` is exactly that family. Two outcomes:

- **Best case:** startup succeeds and the adapter is silently
  ignored on expert layers (you get the base model with a few
  non-expert LoRA deltas applied).
- **Likely case:** SGLang errors out at load time with a shape or
  "target module not supported" message.

The recommended way to serve this checkpoint under SGLang is to
**merge the LoRA into a full HF model first**, then point SGLang at
that merged directory — see "Merged-model fallback" below.

vLLM treats this case as experimental (see `../emu-vllm/`) and is more
likely to work as-is.

## What the LoRA path does

1. `prepare_adapter.sh` downloads the **raw Tinker checkpoint** from HF
   and runs `tinker_cookbook.weights.build_lora_adapter(...)` to remap
   it into the standard PEFT format that SGLang expects.
2. `start.sh` exec's `python -m sglang.launch_server --enable-lora
   --lora-paths emu=<peft_adapter> ...` and prints the OpenAI-compatible
   endpoint URL.

Both steps are idempotent.

## Requirements

- **Hardware:** ≥ 8× H100 80GB (or A100 80GB). Defaults to `--tp-size 8`.
- **Host RAM:** ≥ 60 GB free for the one-time raw → PEFT conversion.
- **Disk:** ≥ 30 GB under `$WORKDIR` for adapters, **plus ~800 GB**
  if you use the merged-model fallback.
- **Python deps:** `sglang[all]>=0.4`, `tinker-cookbook`, `huggingface_hub`.
  ```bash
  uv pip install 'sglang[all]>=0.4' tinker-cookbook
  ```
- **Env:** `HF_TOKEN` (picked up from `backend/providers/emu-sglang/.env` or
  `training/.env`).

## Quickstart — LoRA path

```bash
bash backend/providers/emu-sglang/start.sh
```

The script prints the endpoint banner once SGLang is launching, e.g.

```
OpenAI-compatible : http://0.0.0.0:30000/v1
Native generate   : http://0.0.0.0:30000/generate
```

## Calling the adapter

OpenAI-style chat completion (use `<base>:<adapter>` syntax for the
LoRA, or just `<base>` to hit the base weights):

```bash
curl http://localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.5-397B-A17B:emu",
    "messages": [
      {"role": "user", "content": "Plan how to open Safari and search the web."}
    ],
    "max_tokens": 256
  }'
```

Native SGLang `/generate` API (specify `lora_path` per request):

```bash
curl http://localhost:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Plan how to open Safari and search the web.",
    "sampling_params": {"max_new_tokens": 256, "temperature": 0.2},
    "lora_path": "emu"
  }'
```

## Merged-model fallback

If the LoRA path fails (most likely outcome for this MoE), merge the
adapter into a full HF model on a high-RAM, high-disk host and serve
that. The merge **needs ≥ ~800 GB free disk** and significant RAM —
plan accordingly.

```bash
# 1. On a high-RAM host, build the merged model once:
python - <<'PY'
from tinker_cookbook import weights
weights.build_hf_model(
    base_model="Qwen/Qwen3.5-397B-A17B",
    adapter_path="/path/to/raw_adapter",  # output of `huggingface-cli download ...`
    output_path="/path/to/merged_emu_397b",
    trust_remote_code=True,
)
PY

# 2. Point SGLang at the merged directory — no LoRA flags needed.
MERGED_MODEL_DIR=/path/to/merged_emu_397b bash backend/providers/emu-sglang/start.sh
```

In merged mode the start banner reflects the merged-model path and the
adapter prep step is skipped entirely.

## Configuration

| Var | Default | Notes |
|---|---|---|
| `HF_TOKEN` | _required (LoRA mode)_ | read access to the adapter repo |
| `MERGED_MODEL_DIR` | _empty_ | if set, skip LoRA and serve this dir |
| `HOST` | `0.0.0.0` | bind address |
| `PORT` | `30000` | SGLang's default |
| `BASE_MODEL` | `Qwen/Qwen3.5-397B-A17B` | HF model id |
| `ADAPTER_REPO` | `ppbhatt500/Emu-Qwen3.5-397B-A17B-LoRA-step50-tinker-raw` | raw checkpoint repo |
| `ADAPTER_NAME` | `emu` | name used in `model=` / `lora_path=` |
| `TP_SIZE` | `8` | `--tp-size` |
| `DP_SIZE` | `1` | `--dp-size` |
| `CONTEXT_LENGTH` | `32768` | matches training `MAX_LENGTH` |
| `MAX_LORAS_PER_BATCH` | `2` | base slot + one adapter |
| `MAX_LORA_RANK` | `32` | matches training `LORA_RANK` |
| `LORA_BACKEND` | `triton` | or `csgmv` |
| `LORA_TARGET_MODULES` | `all` | or a comma-list e.g. `q_proj,k_proj,...` |
| `MEM_FRACTION_STATIC` | `0.90` | per-GPU memory headroom |
| `DTYPE` | `bfloat16` | |
| `EXTRA_SGLANG_ARGS` | _empty_ | appended verbatim |
| `WORKDIR` | `~/.cache/emu-serve` | scratch for adapters |
