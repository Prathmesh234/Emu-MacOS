#!/usr/bin/env bash
# Launch an SGLang OpenAI-compatible server hosting
# Qwen/Qwen3.5-397B-A17B with the Emu LoRA adapter from HuggingFace.
#
# IMPORTANT: SGLang's LoRA support does NOT yet cover the
# qwen3_5_moe expert-LoRA case (tinker-cookbook prints this warning
# during the raw→PEFT conversion). If you hit a "LoRA target module
# not supported" or shape mismatch error at startup, set
# MERGED_MODEL_DIR=/path/to/merged_model in the env to fall back to
# serving a fully-merged HF model with no adapter — produced via
# `python -c "from tinker_cookbook import weights;
# weights.build_hf_model(base_model='Qwen/Qwen3.5-397B-A17B',
# adapter_path=$RAW_DIR, output_path='/some/dir',
# trust_remote_code=True)"` on a high-RAM, high-disk host.
#
# Quickstart:
#   bash emu-sglang/start.sh
#
# Optional env overrides (read from emu-sglang/.env, <repo>/training/.env, env):
#   HF_TOKEN                  read access to the adapter repo (REQUIRED when
#                             MERGED_MODEL_DIR is unset)
#   BASE_MODEL                default: Qwen/Qwen3.5-397B-A17B
#   ADAPTER_REPO              default: ppbhatt500/Emu-Qwen3.5-397B-A17B-LoRA-step50-tinker-raw
#   ADAPTER_NAME              default: emu
#   MERGED_MODEL_DIR          if set, skip the LoRA path and serve the
#                             fully-merged HF model from this directory.
#   HOST                      default: 0.0.0.0
#   PORT                      default: 30000   (SGLang's default)
#   TP_SIZE                   default: 8       (tensor parallel)
#   DP_SIZE                   default: 1       (data parallel)
#   CONTEXT_LENGTH            default: 32768
#   MAX_LORAS_PER_BATCH       default: 2       (one slot for base, one for adapter)
#   MAX_LORA_RANK             default: 32      (matches LORA_RANK in training)
#   LORA_BACKEND              default: triton  (or 'csgmv')
#   LORA_TARGET_MODULES       default: all
#   MEM_FRACTION_STATIC       default: 0.90
#   DTYPE                     default: bfloat16
#   EXTRA_SGLANG_ARGS         appended verbatim
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
: "${PORT:=30000}"
: "${TP_SIZE:=8}"
: "${DP_SIZE:=1}"
: "${CONTEXT_LENGTH:=32768}"
: "${MAX_LORAS_PER_BATCH:=2}"
: "${MAX_LORA_RANK:=32}"
: "${LORA_BACKEND:=triton}"
: "${LORA_TARGET_MODULES:=all}"
: "${MEM_FRACTION_STATIC:=0.90}"
: "${DTYPE:=bfloat16}"
: "${EXTRA_SGLANG_ARGS:=}"
: "${WORKDIR:=$HOME/.cache/emu-serve}"
: "${ADAPTER_DIR:=$WORKDIR/peft_adapter}"
: "${MERGED_MODEL_DIR:=}"

if ! python -c "import sglang" >/dev/null 2>&1; then
  echo "[start_sglang] ERROR: 'sglang' python package not found." >&2
  echo "               Install with:  uv pip install 'sglang[all]>=0.4' tinker-cookbook" >&2
  exit 1
fi

print_banner() {
  local model_label="$1"
  echo
  echo "  ╔══════════════════════════════════════════════════════════════════╗"
  echo "  ║ Once ready, the SGLang server will expose:                       ║"
  echo "  ║   OpenAI-compatible : http://$HOST:$PORT/v1                      ║"
  echo "  ║   Native generate   : http://$HOST:$PORT/generate                ║"
  echo "  ║   Models list       : http://$HOST:$PORT/v1/models               ║"
  echo "  ║   Health            : http://$HOST:$PORT/health                  ║"
  echo "  ║                                                                  ║"
  echo "  ║ Serving: $model_label"
  echo "  ╚══════════════════════════════════════════════════════════════════╝"
  echo
}

# ── Mode A: fully-merged model (no LoRA). ─────────────────────────────
if [[ -n "$MERGED_MODEL_DIR" ]]; then
  if [[ ! -f "$MERGED_MODEL_DIR/config.json" ]]; then
    echo "[start_sglang] ERROR: MERGED_MODEL_DIR=$MERGED_MODEL_DIR has no config.json" >&2
    exit 1
  fi
  echo "[start_sglang] Serving merged model from $MERGED_MODEL_DIR (no LoRA)."
  print_banner "$MERGED_MODEL_DIR (merged, no LoRA)"
  # shellcheck disable=SC2086
  exec python -m sglang.launch_server \
    --model-path "$MERGED_MODEL_DIR" \
    --host "$HOST" \
    --port "$PORT" \
    --tp-size "$TP_SIZE" \
    --dp-size "$DP_SIZE" \
    --context-length "$CONTEXT_LENGTH" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    $EXTRA_SGLANG_ARGS
fi

# ── Mode B: base model + LoRA adapter. ────────────────────────────────
bash "$SCRIPT_DIR/prepare_adapter.sh"

echo "[start_sglang] Launching SGLang ..."
echo "             base model       : $BASE_MODEL"
echo "             adapter          : $ADAPTER_NAME=$ADAPTER_DIR"
echo "             tensor parallel  : $TP_SIZE"
echo "             dtype            : $DTYPE"
echo "             context length   : $CONTEXT_LENGTH"
echo "             lora backend     : $LORA_BACKEND"
echo "             lora target mods : $LORA_TARGET_MODULES"
echo
echo "[start_sglang] NOTE: SGLang's MoE expert-LoRA support for qwen3_5"
echo "             is experimental/unsupported. If startup fails, see"
echo "             MERGED_MODEL_DIR in the header comment of this script."
print_banner "$BASE_MODEL + LoRA '$ADAPTER_NAME' @ $ADAPTER_DIR"

# shellcheck disable=SC2086
exec python -m sglang.launch_server \
  --model-path "$BASE_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tp-size "$TP_SIZE" \
  --dp-size "$DP_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --dtype "$DTYPE" \
  --trust-remote-code \
  --enable-lora \
  --lora-paths "${ADAPTER_NAME}=${ADAPTER_DIR}" \
  --max-loras-per-batch "$MAX_LORAS_PER_BATCH" \
  --max-lora-rank "$MAX_LORA_RANK" \
  --lora-target-modules "$LORA_TARGET_MODULES" \
  --lora-backend "$LORA_BACKEND" \
  $EXTRA_SGLANG_ARGS
