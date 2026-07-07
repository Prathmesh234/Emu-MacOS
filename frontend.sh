#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# frontend.sh — One-command Emu frontend launcher
#
# What it does:
#   1. Installs Node.js dependencies (if needed)
#   2. Starts the Electron desktop app
#
# Usage:
#   chmod +x frontend.sh
#   ./frontend.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[emu]${NC} $*"; }
ok()    { echo -e "${GREEN}[emu]${NC} $*"; }
err()   { echo -e "${RED}[emu]${NC} $*" >&2; }

# ── Pre-flight checks ───────────────────────────────────────────────────────

if ! command -v node &>/dev/null; then
    err "Node.js is not installed. Install v18+ from https://nodejs.org/"
    exit 1
fi

if ! command -v npm &>/dev/null; then
    err "npm is not found. It should come with Node.js."
    exit 1
fi

# ── Install dependencies ────────────────────────────────────────────────────

cd "$SCRIPT_DIR"

if [ ! -d "node_modules" ]; then
    info "Installing Node.js dependencies..."
    npm install
    ok "Dependencies installed"
else
    info "node_modules exists — skipping install (run 'npm install' manually to update)"
fi

# ── Load .env if present ────────────────────────────────────────────────────
#
# Electron spawns both the backend AND the messaging runner as child
# processes. Both inherit our env. Sourcing backend/.env here means
# operator-tunable knobs (EMU_MESSAGING_*_MODE, reply-prefix overrides,
# provider keys for the backend) all flow from one file.
#
# The messaging runner additionally goes through _scrubbedEnv in
# frontend/process/messagingProcess.js, which allowlists only the
# EMU_MESSAGING_* family — provider API keys are NOT forwarded to it.

if [ -f "$SCRIPT_DIR/backend/.env" ]; then
    info "Loading environment from backend/.env"
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/backend/.env"
    set +a
fi

# ── Start Electron ──────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  Emu Frontend Starting${NC}"
echo -e "  Make sure the backend is running: ${CYAN}./backend.sh${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

exec npm start
