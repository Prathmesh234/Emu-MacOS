#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server hosting
# Qwen/Qwen3.5-397B-A17B with the Emu LoRA adapter from HuggingFace.
#
# Adapter prep (download + raw→PEFT conversion) is done by
# prepare_adapter.sh; this script invokes it then execs vllm.
#
# Quickstart:
#   bash emu-vllm/start.sh
#
# Optional env overrides (read from emu-vllm/.env, <repo>/training/.env, env):
#   HF_TOKEN                  read access to the adapter repo (REQUIRED)
#   BASE_MODEL                default: Qwen/Qwen3.5-397B-A17B
#   ADAPTER_REPO              default: ppbhatt500/Emu-Qwen3.5-397B-A17B-LoRA-step50-tinker-raw
#   ADAPTER_NAME              served-model-name of the LoRA  (default: emu)
#   HOST                      default: 0.0.0.0
#   PORT                      default: 8000
#   TENSOR_PARALLEL_SIZE      default: 8        (8x H100 80GB minimum for bf16)
#   PIPELINE_PARALLEL_SIZE    default: 1
#   GPU_MEMORY_UTILIZATION    default: 0.92
#   MAX_MODEL_LEN             default: 32768    (matches train_vlm_sft MAX_LENGTH)
#   MAX_LORAS                 default: 1
#   MAX_LORA_RANK             default: 32       (matches LORA_RANK in training)
#   DTYPE                     default: bfloat16
#   EXTRA_VLLM_ARGS           appended verbatim to the vllm serve cmdline
#   WORKDIR / ADAPTER_DIR     see prepare_adapter.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

load_env() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  set -a
  # shellcheck disable=SC1090
  source <(tr -d '\r' < "$f")
  set +a
}
load_env "$REPO_ROOT/training/.env"
load_env "$SCRIPT_DIR/.env"

: "${BASE_MODEL:=Qwen/Qwen3.5-397B-A17B}"
: "${ADAPTER_NAME:=emu}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${TENSOR_PARALLEL_SIZE:=8}"
: "${PIPELINE_PARALLEL_SIZE:=1}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
: "${MAX_MODEL_LEN:=32768}"
: "${MAX_LORAS:=1}"
: "${MAX_LORA_RANK:=32}"
: "${DTYPE:=bfloat16}"
: "${EXTRA_VLLM_ARGS:=}"
: "${WORKDIR:=$HOME/.cache/emu-serve}"
: "${ADAPTER_DIR:=$WORKDIR/peft_adapter}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "[start_vllm] ERROR: 'vllm' not found in PATH." >&2
  echo "             Install with:  uv pip install 'vllm>=0.7' tinker-cookbook" >&2
  exit 1
fi

# 1. Make sure the PEFT adapter exists.
bash "$SCRIPT_DIR/prepare_adapter.sh"

# 2. Launch vLLM (OpenAI-compatible server).
echo "[start_vllm] Launching vLLM ..."
echo "             base model       : $BASE_MODEL"
echo "             adapter          : $ADAPTER_NAME=$ADAPTER_DIR"
echo "             tensor parallel  : $TENSOR_PARALLEL_SIZE"
echo "             dtype            : $DTYPE"
echo "             max model len    : $MAX_MODEL_LEN"
echo
echo "  ╔══════════════════════════════════════════════════════════════════╗"
echo "  ║ Once ready, the OpenAI-compatible API will be at:                ║"
echo "  ║   http://$HOST:$PORT/v1                                          ║"
echo "  ║   - models endpoint : http://$HOST:$PORT/v1/models               ║"
echo "  ║   - chat endpoint   : http://$HOST:$PORT/v1/chat/completions     ║"
echo "  ║   - health check    : http://$HOST:$PORT/health                  ║"
echo "  ║                                                                  ║"
echo "  ║ Specify model='$ADAPTER_NAME' in requests to use the LoRA;       ║"
echo "  ║ model='$BASE_MODEL' will hit the base model.                     ║"
echo "  ╚══════════════════════════════════════════════════════════════════╝"
echo

# shellcheck disable=SC2086
exec vllm serve "$BASE_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype "$DTYPE" \
  --trust-remote-code \
  --enable-lora \
  --max-loras "$MAX_LORAS" \
  --max-lora-rank "$MAX_LORA_RANK" \
  --lora-modules "${ADAPTER_NAME}=${ADAPTER_DIR}" \
  $EXTRA_VLLM_ARGS
