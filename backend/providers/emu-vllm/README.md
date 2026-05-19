# emu-vllm

Self-contained launcher for serving the Emu fine-tuned
`Qwen/Qwen3.5-397B-A17B` with vLLM, using the LoRA adapter at
`ppbhatt500/Emu-Qwen3.5-397B-A17B-LoRA-step50-tinker-raw` on HuggingFace.

## What it does

1. `prepare_adapter.sh` downloads the **raw Tinker checkpoint** from HF and
   runs `tinker_cookbook.weights.build_lora_adapter(...)` to remap it into
   the standard PEFT format that vLLM expects.
2. `start.sh` exec's `vllm serve` with `--enable-lora --lora-modules
   emu=<peft_adapter>` and prints the OpenAI-compatible endpoint URL.

Both steps are idempotent — re-running skips work that's already on disk.

## Requirements

- **Hardware:** ≥ 8× H100 80GB (or A100 80GB), single node. The base
  model is ~800 GB in bf16; defaults to `--tensor-parallel-size 8`.
- **Host RAM:** ≥ 60 GB free for the one-time raw → PEFT conversion.
  (Steady-state serving is GPU-bound; RAM only matters during prep.)
- **Disk:** ≥ 30 GB free under `$WORKDIR` (raw + PEFT adapter copies).
- **Python deps:** `vllm>=0.7`, `tinker-cookbook`, `huggingface_hub`.
  ```bash
  uv pip install 'vllm>=0.7' tinker-cookbook
  ```
- **Env:** `HF_TOKEN` with read access to the private adapter repo. Picked
  up from `backend/providers/emu-vllm/.env` (preferred) or `training/.env`.

## Quickstart

```bash
bash backend/providers/emu-vllm/start.sh
```

The script prints the endpoint banner once vLLM is launching, e.g.

```
http://0.0.0.0:8000/v1
```

## Calling the LoRA

OpenAI-style chat completion against the adapter:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "emu",
    "messages": [
      {"role": "user", "content": "Plan how to open Safari and search the web."}
    ],
    "max_tokens": 256
  }'
```

Substitute `"model": "Qwen/Qwen3.5-397B-A17B"` to hit the base model
without the adapter (useful for A/B comparisons).

## Configuration

All knobs are env vars (override in `backend/providers/emu-vllm/.env` or shell):

| Var | Default | Notes |
|---|---|---|
| `HF_TOKEN` | _required_ | read access to the adapter repo |
| `HOST` | `0.0.0.0` | bind address |
| `PORT` | `8000` | HTTP port |
| `BASE_MODEL` | `Qwen/Qwen3.5-397B-A17B` | HF model id |
| `ADAPTER_REPO` | `ppbhatt500/Emu-Qwen3.5-397B-A17B-LoRA-step50-tinker-raw` | raw checkpoint repo |
| `ADAPTER_NAME` | `emu` | name used in `model=` |
| `TENSOR_PARALLEL_SIZE` | `8` | must divide attention heads |
| `MAX_MODEL_LEN` | `32768` | matches training `MAX_LENGTH` |
| `MAX_LORA_RANK` | `32` | matches training `LORA_RANK` |
| `DTYPE` | `bfloat16` | |
| `GPU_MEMORY_UTILIZATION` | `0.92` | per-GPU memory headroom |
| `EXTRA_VLLM_ARGS` | _empty_ | appended verbatim to `vllm serve` |
| `WORKDIR` | `~/.cache/emu-serve` | scratch for adapters |

## Known issues

- vLLM treats MoE expert LoRA serving for `qwen3_5` as **experimental** —
  the tinker-cookbook prints a warning during conversion. If you hit
  shape/dtype errors at serve time, the workaround is to use
  `tinker_cookbook.weights.build_hf_model(...)` instead to materialise a
  fully-merged ~800 GB model and serve that without `--enable-lora`.
- The first PEFT conversion run consumes 30–60 GB host RAM. On boxes
  with less, run the conversion separately on a high-RAM host and rsync
  the resulting `peft_adapter/` onto the GPU box, then start with
  `ADAPTER_DIR=/path/to/peft_adapter bash start.sh`.
