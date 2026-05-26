# Emu

Emu is a desktop automation agent for macOS that combines screen understanding,
LLM planning, and OS-level action execution. It runs locally on your machine,
binds only to loopback, and stores all session artifacts under `~/.emu/`.

---

## Architecture

| Component | Path | Role |
|---|---|---|
| Electron shell | `main.js`, `preload.js`, `frontend/` | UI, action dispatch, child-process supervision |
| Backend agent | `backend/` | FastAPI server on `127.0.0.1:8000`, tool-calling loop, provider routing |
| Memory daemon | `daemon/` | macOS LaunchAgent that curates long-term memory under `.emu/` on a 2-minute tick |
| Coworker driver | `frontend/coworker-mode/emu-driver/` | Swift accessibility/screen helper bundled as `emu-cua-driver` |

All cross-process communication is gated by a per-launch random token written
to `.emu/.auth_token` (chmod 0600) and required as `X-Emu-Token` on every
backend request and WebSocket connection.

---

## Quick start

Two terminals:

```bash
# Terminal 1 — backend
./backend.sh

# Terminal 2 — frontend
./frontend.sh
```

Manual startup:

```bash
# Backend
cd backend
uv venv && uv sync
uv run uvicorn main:app --host 127.0.0.1 --port 8000

# Frontend (separate terminal)
npm install
npm start
```

For setup automation, see `install_emu_skill/INSTALL_EMU.MD`.

---

## Configuration

Provider credentials are read from environment variables. Pick one:

```bash
ANTHROPIC_API_KEY=
OPENROUTER_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
```

Optional overrides:

```bash
EMU_PROVIDER=          # force a specific provider
OPENAI_BASE_URL=       # for openai-compatible endpoints (allowlisted hosts only)
USE_OMNI_PARSER=1      # enable omni-parser screen understanding
EMU_DEV=1              # development mode
```

See `backend/.env.example` for the full list. **Do not commit `.env` files** —
`.env`, `.env.*`, `*.pem`, `*.key`, and SSH material are gitignored and
excluded from packaged builds.

For production use, prefer the macOS Keychain for provider keys (the daemon
already supports this — see `daemon/llm_client.py`).

---

## Features

### Coworker mode (default)

Emu drives target windows through the bundled `emu-cua-driver` Swift helper
without stealing the user's foreground app. The backend calls `cua_*` tools
that operate on a specific `pid` + `window_id`.

`cua_page execute_javascript` requires per-call user confirmation
(`user_has_confirmed_javascript=true`) before executing any JavaScript in a
browser tab. This is enforced server-side regardless of the LLM's intent.

### Memory daemon

Background memory curation runs out-of-process via launchd:

```bash
python3 -m daemon.install_macos install     # install
python3 -m daemon.install_macos status      # verify
python3 -m daemon.install_macos run-now     # one-shot tick
python3 -m daemon.install_macos uninstall   # remove
```

The daemon runs as a per-user LaunchAgent (not a system LaunchDaemon) and
operates only on files under `~/.emu/`. It exposes no IPC surface.

### Hermes Agent delegation

Long-running terminal/code tasks can be delegated to a locally-installed
[Hermes Agent](https://github.com/NousResearch/hermes-agent):

| Tool | Purpose |
|---|---|
| `invoke_hermes` | Spawn a background Hermes job, return job id |
| `check_hermes` | Poll status / retrieve final output |
| `cancel_hermes` | Terminate a running job |
| `list_hermes_jobs` | Enumerate jobs in current session |

The Hermes child process inherits a minimal env (PATH/HOME/LANG/TERM only) —
provider API keys are never forwarded.

### Inbound messaging bridge (WhatsApp + iMessage)

Message Emu from your phone — every allowlisted inbound message from
WhatsApp or iMessage funnels into a **single, persistent "Remote control"
session** that shows up in the History sidebar. See `messaging/README.md`
for full setup. Quick install:

```bash
cd messaging && npm install
```

Then create `~/.emu/messaging/allowlist.json` (mode 0600):

```json
{ "whatsapp": ["+15551234567"], "imessage": ["+15551234567"] }
```

The bridge is gated by the allowlist — without it, no inbound messages are
accepted. iMessage requires Full Disk Access and Automation → Messages
grants on Emu (see `messaging/imessage/README.md`).

### Provider support

Auto-detected providers, in order:

Anthropic, OpenRouter, Azure OpenAI, OpenAI-compatible, OpenAI, Gemini,
Bedrock, Fireworks, Together AI, Baseten, H Company, Modal fallback.

See `backend/providers/registry.py`.

---

## Build & package

```bash
npm run build:driver   # build the Swift emu-cua-driver
npm run pack           # build driver + create dist/mac-*/Emu.app (unsigned)
npm run dist           # build driver + run electron-builder release flow
```

The packaged app excludes `.env*`, `*.pem`, `*.key`, `.venv`, `__pycache__`,
`training/`, `Tests/`, and editor/agent dotfiles.

---

## Security model

- **Loopback only.** Backend binds `127.0.0.1:8000`. The TCP socket is
  guarded by a per-launch random token (`.emu/.auth_token`, chmod 0600).
- **Sandboxed shell.** `shell_exec` pins `cwd` and `HOME` to `~/.emu`,
  rejects network tools, privilege escalation, destructive operations, and
  interpreter `-c`/`-e`/`-f` invocations that would hide work from the
  path-scope check.
- **Browser JS gated.** `cua_page execute_javascript` requires explicit
  per-call user confirmation; `enable_javascript_apple_events` requires
  `user_has_confirmed_enabling`.
- **Strict app-name allowlist.** `raise_app` validates against
  `^[A-Za-z0-9 .\-+_/&()'!]+$` and rejects control characters and
  Unicode quote variants.
- **Navigation lockdown.** The Electron renderer denies any navigation
  outside the bundled `file://` pages; external URLs are routed through
  `shell.openExternal` after host allowlisting.
- **Memory daemon is jailed** to `~/.emu/`. All writes go through
  `daemon/policy.py` which uses path containment + filename pattern checks.
- **Hermes env is scrubbed.** The Hermes child process never sees provider
  API keys.

See `MACOS_PERMISSIONS.md` for the macOS Accessibility / Screen Recording
grant flow.

---

## Troubleshooting

### Backend 401

`X-Emu-Token` mismatch. Restart backend and frontend so both reload the
same token from `.emu/.auth_token`.

### Frontend can't connect

Confirm the backend is running on `http://127.0.0.1:8000` and no local
proxy is intercepting loopback.

### macOS actions fail silently

Grant Accessibility and Screen Recording to Emu in System Settings → Privacy
& Security, then restart the app. In coworker mode, use the in-app
permission card. Packaged `.dmg` flow: see `MACOS_PERMISSIONS.md`.

### Modal deployment fails

```bash
cd backend
uv run modal setup
```

Then retry `./backend.sh`.

### Hermes jobs stuck

Poll with `check_hermes` and `cancel_hermes` if there has been no output for
an extended period. Inspect `.emu/sessions/<id>/hermes/` for stderr output.

### Daemon not running

```bash
python3 -m daemon.install_macos status
```

Reinstall if needed. Confirm the app is not running from a translocation or
mounted-image path.

---

## Repository map

| Path | Contents |
|---|---|
| `main.js`, `preload.js` | Electron entrypoint and preload bridge |
| `frontend/` | Renderer UI, actions, services, store, styles |
| `frontend/coworker-mode/emu-driver/` | Swift `emu-cua-driver` source |
| `frontend/coworker-driver/` | Coworker mode operator docs |
| `backend/` | FastAPI agent harness, providers, tools, prompts |
| `daemon/` | Memory daemon runtime, policy, state, launchd installer |
| `messaging/` | Inbound messaging bridge runner (WhatsApp + iMessage) |
| `backend.sh`, `frontend.sh` | One-command startup scripts |

## Additional docs

- `install_emu_skill/INSTALL_EMU.MD` — automated install runbook
- `DOCUMENTATION.md` — extended documentation
- `MACOS_PERMISSIONS.md` — Accessibility & Screen Recording grant flow
- `backend/BACKEND.MD` — backend architecture
- `frontend/FRONTEND.md` — frontend architecture
- `daemon/DESIGN.md` — daemon design notes
- `frontend/coworker-mode/PLAN.md` — coworker mode plan
- `frontend/coworker-driver/README.md` — driver operator guide
