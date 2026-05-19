#!/usr/bin/env bash
# Prepare the PEFT adapter from the raw Tinker checkpoint on HuggingFace.
#
# Downloads ppbhatt500/Emu-Qwen3.5-397B-A17B-LoRA-step50-tinker-raw (12.7 GB),
# then runs tinker_cookbook.weights.build_lora_adapter() to remap key
# names into PEFT format. The conversion step needs ~30-60 GB host RAM
# for this 397B MoE model.
#
# Idempotent: if $ADAPTER_DIR/adapter_config.json + adapter_model.safetensors
# already exist, prep is skipped.
#
# Required env (read from emu-vllm/.env, then <repo>/training/.env, then real env):
#   HF_TOKEN                HuggingFace token with read access to the
#                           private adapter repo.
#
# Optional env:
#   BASE_MODEL              default: Qwen/Qwen3.5-397B-A17B
#   ADAPTER_REPO            default: ppbhatt500/Emu-Qwen3.5-397B-A17B-LoRA-step50-tinker-raw
#   WORKDIR                 default: $HOME/.cache/emu-serve
#   ADAPTER_DIR             default: $WORKDIR/peft_adapter
#   RAW_DIR                 default: $WORKDIR/raw_adapter
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load .env files (CRLF-tolerant). Later files override earlier ones.
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
: "${ADAPTER_REPO:=ppbhatt500/Emu-Qwen3.5-397B-A17B-LoRA-step50-tinker-raw}"
: "${WORKDIR:=$HOME/.cache/emu-serve}"
: "${RAW_DIR:=$WORKDIR/raw_adapter}"
: "${ADAPTER_DIR:=$WORKDIR/peft_adapter}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[prepare_adapter] ERROR: HF_TOKEN not set. Put it in $REPO_ROOT/training/.env or $SCRIPT_DIR/.env" >&2
  exit 1
fi

mkdir -p "$WORKDIR"

if [[ -f "$ADAPTER_DIR/adapter_model.safetensors" && -f "$ADAPTER_DIR/adapter_config.json" ]]; then
  echo "[prepare_adapter] PEFT adapter already at $ADAPTER_DIR — skipping prep."
  exit 0
fi

# 1. Download raw Tinker checkpoint (skip if present).
if [[ -f "$RAW_DIR/adapter_model.safetensors" && -f "$RAW_DIR/adapter_config.json" ]]; then
  echo "[prepare_adapter] Raw adapter present at $RAW_DIR — skipping download."
else
  echo "[prepare_adapter] Downloading $ADAPTER_REPO → $RAW_DIR ..."
  HF_TOKEN="$HF_TOKEN" python -m huggingface_hub.commands.huggingface_cli \
    download "$ADAPTER_REPO" \
    --local-dir "$RAW_DIR" \
    --token "$HF_TOKEN"
fi

# 2. Convert raw → PEFT. Needs tinker-cookbook + 30-60 GB RAM.
echo "[prepare_adapter] Converting raw → PEFT at $ADAPTER_DIR ..."
echo "                  (this needs ~30-60 GB host RAM for the MoE expert expand)"
python - <<PY
import os, shutil
from tinker_cookbook import weights

base_model = "$BASE_MODEL"
raw_dir    = "$RAW_DIR"
out_dir    = "$ADAPTER_DIR"

# build_lora_adapter raises FileExistsError if out_dir exists.
if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)

weights.build_lora_adapter(
    base_model=base_model,
    adapter_path=raw_dir,
    output_path=out_dir,
    trust_remote_code=True,
)
print(f"[prepare_adapter] wrote {out_dir}")
PY

echo "[prepare_adapter] done. PEFT adapter at $ADAPTER_DIR"
