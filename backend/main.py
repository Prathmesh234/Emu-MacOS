import asyncio
import importlib
import os
import secrets
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Ensure the repo root is importable so `from daemon import run` resolves.
# main.py lives at <repo>/backend/main.py; the daemon package is at <repo>/daemon.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

import json

from models import (
    Action,
    ActionType,
    AgentRequest,
    AgentResponse,
    ActionCompleteRequest,
    CompactRequest,
    ContinueSessionRequest,
    ProviderSettingsRequest,
    StopRequest,
)
from context_manager import ContextManager
from workspace import ensure_session_dir
from utilities import ConnectionManager, log_entry, log_and_send, ipc_to_action_label, interpret_action_error
from tools import execute_agent_tool, auto_compact
from tools.coworker_tools import cancel_driver_calls

# ── Inference backend ────────────────────────────────────────────────────────
from providers.registry import load_provider, load_compact_provider
from context_manager.context import USE_OMNI_PARSER
from utilities.paths import get_emu_path

call_model, is_ready, ensure_ready, _provider_name = load_provider()
compact_model = load_compact_provider()

print(f"[config] OmniParser: {'ENABLED' if USE_OMNI_PARSER else 'DISABLED (direct screenshots)'}")

# ── Per-launch auth token ────────────────────────────────────────────────────
# Shared with the Electron frontend via .emu/.auth_token file.
AUTH_TOKEN = os.environ.get("EMU_AUTH_TOKEN") or secrets.token_hex(32)
_token_path = get_emu_path() / ".auth_token"
_token_path.parent.mkdir(parents=True, exist_ok=True)
_token_path.write_text(AUTH_TOKEN)
try:
    os.chmod(_token_path, 0o600)  # owner-only read/write
except OSError:
    pass  # Fallback if chmod not supported on this platform
print(f"[security] Auth token written to {_token_path}")


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Reject HTTP requests that don't carry a valid X-Emu-Token header."""

    async def dispatch(self, request, call_next):
        # CORS preflight and health checks are exempt
        if request.method == "OPTIONS" or request.url.path == "/health":
            return await call_next(request)
        # NOTE: Do NOT exempt requests based on an `Upgrade: websocket` header —
        # FastAPI routes by path, not by the upgrade header, so a plain HTTP
        # POST that sets `Upgrade: websocket` would otherwise bypass auth on
        # normal HTTP routes (e.g. /agent/step, which can run shell_exec).
        # Real WebSocket upgrades are dispatched on the separate ASGI `websocket`
        # scope that never passes through HTTP middleware, and the endpoint
        # validates the token from the query param itself.
        token = request.headers.get("x-emu-token", "")
        if not token or not secrets.compare_digest(token, AUTH_TOKEN):
            print(f"[security] 401 on {request.method} {request.url.path} — token present={bool(token)}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing auth token"},
            )
        return await call_next(request)


# ── Memory daemon ───────────────────────────────────────────────────────────
# The memory daemon is driven by macOS launchd (see daemon/launchd/ and
# daemon/install_macos.py), NOT by this process. That way it keeps ticking
# on its launchd interval even when uvicorn is shut down. The backend shares the
# same .emu/ directory with the daemon — no IPC between them.


app = FastAPI(title="Emulation Agent API")

# Middleware execution order in Starlette: LAST added = OUTERMOST.
# We need CORS to be outermost so its headers appear on ALL responses
# (including 401s from TokenAuth). So add CORS LAST.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(null|http://(127\.0\.0\.1|localhost)(:\d+)?)$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Emu-Token"],
)
app.add_middleware(TokenAuthMiddleware)

manager = ConnectionManager()
context_manager = ContextManager()

# Per-session stop tokens. A user can stop an in-flight step and immediately
# send a new message in the same session; tracking the specific step id keeps
# the old loop stopped without poisoning the new one.
_active_step_ids: dict[str, int] = {}
_stopped_step_ids: dict[str, int] = {}



async def _inject_coworker_perception(session_id: str) -> None:
    """
    Coworker-mode lightweight target reminder.

    Do NOT automatically call ``get_window_state`` here. AX snapshots can be
    slow or app-specific, and doing them before every model turn makes Stop
    feel broken because the request cannot return until the driver call
    unwinds. The model already has explicit ``cua_screenshot`` and
    ``cua_get_window_state`` tools; it should request those only when needed.
    """
    target = context_manager.get_coworker_target(session_id)
    if not target:
        return

    pid, window_id = target
    context_manager.add_user_message(
        session_id,
        "[coworker_target]\n"
        f"Current target: pid={pid} window_id={window_id}.\n"
        "No automatic screenshot or AX tree was captured for this turn. "
        "Call cua_screenshot or cua_get_window_state explicitly only if the next action needs fresh UI state."
    )


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer, got {raw!r}") from exc

    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer, got {raw!r}")

    return value


def _nonnegative_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a non-negative integer, got {raw!r}") from exc

    if value < 0:
        raise RuntimeError(f"{name} must be a non-negative integer, got {raw!r}")

    return value


MODEL_TIMEOUT_SECS = _positive_int_env("EMU_MODEL_TIMEOUT_SECS", 60)
MAX_TIMEOUT_RETRIES = _positive_int_env("EMU_MODEL_TIMEOUT_RETRIES", 1)
MAX_RATE_LIMIT_RETRIES = _nonnegative_int_env("EMU_MODEL_RATE_LIMIT_RETRIES", 2)
RATE_LIMIT_BASE_DELAY_SECS = _positive_int_env("EMU_MODEL_RATE_LIMIT_BASE_DELAY_SECS", 2)
RATE_LIMIT_MAX_DELAY_SECS = _positive_int_env("EMU_MODEL_RATE_LIMIT_MAX_DELAY_SECS", 30)
print(
    f"[config] Model timeout: {MODEL_TIMEOUT_SECS}s, retries: {MAX_TIMEOUT_RETRIES}, "
    f"rate-limit retries: {MAX_RATE_LIMIT_RETRIES}"
)


def _is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True
    text = f"{status or ''} {exc}".lower()
    return any(
        marker in text
        for marker in (
            "429",
            "rate limit",
            "rate_limit",
            "rate-limit",
            "too many requests",
            "too_many_requests",
            "resource exhausted",
            "quota exceeded",
            "throttl",
        )
    )


def _rate_limit_retry_delay(exc: Exception, attempt_index: int) -> int:
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return max(1, min(RATE_LIMIT_MAX_DELAY_SECS, retry_after))
    delay = RATE_LIMIT_BASE_DELAY_SECS * (2 ** max(0, attempt_index))
    return min(RATE_LIMIT_MAX_DELAY_SECS, delay)


def _retry_after_seconds(exc: Exception) -> int | None:
    headers = getattr(exc, "headers", None)
    response = getattr(exc, "response", None)
    if headers is None and response is not None:
        headers = getattr(response, "headers", None)

    value = None
    if headers is not None:
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
        except AttributeError:
            if isinstance(headers, dict):
                value = headers.get("retry-after") or headers.get("Retry-After")

    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


async def _sleep_with_stop(delay_s: int, is_stopped) -> bool:
    deadline = asyncio.get_running_loop().time() + max(0, delay_s)
    while True:
        if is_stopped():
            return False
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return True
        await asyncio.sleep(min(0.5, remaining))


def _call_sync_provider_with_rate_limit(func, *args, label: str):
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return func(*args)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt >= MAX_RATE_LIMIT_RETRIES:
                raise
            delay = _rate_limit_retry_delay(exc, attempt)
            print(
                f"[provider] {label} rate-limited — retrying in {delay}s "
                f"({attempt + 1}/{MAX_RATE_LIMIT_RETRIES})"
            )
            time.sleep(delay)


def _wrap_compact_model(func):
    def _wrapped(messages):
        return _call_sync_provider_with_rate_limit(func, messages, label="compact_model")

    return _wrapped


compact_model = _wrap_compact_model(compact_model)


# ── Provider settings config ─────────────────────────────────────────────────
# Maps each short provider name → the env vars used for its API key and model.

_PROVIDER_SETTINGS: dict[str, dict] = {
    "claude":       {"key_env": "ANTHROPIC_API_KEY",  "model_env": "ANTHROPIC_MODEL",    "default_model": "claude-sonnet-4-5"},
    "openai":       {"key_env": "OPENAI_API_KEY",      "model_env": "OPENAI_MODEL",        "default_model": "gpt-5.4"},
    "openrouter":   {"key_env": "OPENROUTER_API_KEY",  "model_env": "OPENROUTER_MODEL",    "default_model": "anthropic/claude-sonnet-4"},
    "gemini":       {"key_env": "GOOGLE_API_KEY",      "model_env": "GEMINI_MODEL",        "default_model": "gemini-3-flash-preview"},
    "fireworks":    {"key_env": "FIREWORKS_API_KEY",   "model_env": "FIREWORKS_MODEL",     "default_model": "accounts/fireworks/models/llama4-maverick-instruct-basic"},
    "together_ai":  {"key_env": "TOGETHER_API_KEY",    "model_env": "TOGETHER_MODEL",      "default_model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"},
    "baseten":      {"key_env": "BASETEN_API_KEY",     "model_env": "BASETEN_MODEL",       "default_model": "moonshotai/Kimi-K2.5"},
    "h_company":    {"key_env": "H_COMPANY_API_KEY",   "model_env": "H_COMPANY_MODEL",     "default_model": "holo3-35b-a3b"},
    "modal":        {"key_env": None,                  "model_env": None,                  "default_model": ""},
}


def _update_env_file(updates: dict) -> None:
    """Write key=value pairs into backend/.env, adding missing keys at the end."""
    env_path = Path(__file__).resolve().parent / ".env"
    lines = env_path.read_text().splitlines() if env_path.exists() else []

    written: set[str] = set()
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            written.add(key)
        else:
            new_lines.append(line)

    for key, value in updates.items():
        if key not in written:
            new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n")


def _normalize_model_option(option: dict) -> dict:
    model_id = str(option.get("id", "")).strip()
    if not model_id:
        return {}
    label = str(option.get("label") or model_id).strip()
    description = str(option.get("description") or "").strip()
    parameters = str(option.get("parameters") or "").strip()
    source = str(option.get("source") or "").strip()
    normalized = {
        "id": model_id,
        "label": label,
        "description": description,
        "parameters": parameters,
        "vision": bool(option.get("vision", True)),
        "source": source,
    }
    for optional_key in ("context", "pricing", "modalities", "released"):
        if optional_key in option:
            normalized[optional_key] = option[optional_key]
    return normalized


def _get_provider_model_options(provider: str) -> list[dict]:
    """Load the provider-owned curated model catalog, if it exists."""
    try:
        from providers.registry import _PROVIDER_MAP
        module_path = _PROVIDER_MAP.get(provider)
        if not module_path:
            return []
        catalog = importlib.import_module(f"{module_path}.models")
        raw_options = (
            catalog.get_model_options()
            if hasattr(catalog, "get_model_options")
            else getattr(catalog, "MODEL_OPTIONS", [])
        )
    except ModuleNotFoundError:
        return []
    except Exception as exc:
        print(f"[settings] Failed to load model catalog for {provider}: {exc}")
        return []

    normalized = [_normalize_model_option(opt) for opt in raw_options if isinstance(opt, dict)]
    return [opt for opt in normalized if opt]


def _provider_current_model(provider: str) -> str:
    cfg = _PROVIDER_SETTINGS.get(provider, {})
    model_env = cfg.get("model_env")
    if model_env:
        return os.environ.get(model_env, cfg.get("default_model", ""))
    return cfg.get("default_model", "")


def _current_provider_model() -> tuple[str, str]:
    provider = os.environ.get("EMU_PROVIDER", _provider_name).strip().lower()
    return provider, _provider_current_model(provider)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/settings/provider")
async def get_provider_settings():
    """Return current provider/model configuration (API key is masked)."""
    provider = os.environ.get("EMU_PROVIDER", _provider_name)
    cfg = _PROVIDER_SETTINGS.get(provider, {})
    key_env = cfg.get("key_env")
    model_env = cfg.get("model_env")

    api_key = os.environ.get(key_env, "") if key_env else ""
    model = (
        os.environ.get(model_env, cfg.get("default_model", ""))
        if model_env
        else cfg.get("default_model", "")
    )

    masked = (
        ("*" * max(0, len(api_key) - 4) + api_key[-4:]) if len(api_key) > 4 else "*" * len(api_key)
    )

    return {
        "provider": provider,
        "model": model,
        "api_key_set": bool(api_key),
        "api_key_preview": masked,
        "providers": list(_PROVIDER_SETTINGS.keys()),
        "default_models": {k: v["default_model"] for k, v in _PROVIDER_SETTINGS.items()},
        "model_options": _get_provider_model_options(provider),
    }


@app.get("/settings/provider/models")
async def get_provider_models(provider: str | None = None):
    """Return curated model options for a provider."""
    selected_provider = (provider or os.environ.get("EMU_PROVIDER", _provider_name)).strip().lower()
    if selected_provider not in _PROVIDER_SETTINGS:
        return JSONResponse(status_code=400, content={"detail": f"Unknown provider: {selected_provider}"})

    return {
        "provider": selected_provider,
        "model": _provider_current_model(selected_provider),
        "default_model": _PROVIDER_SETTINGS[selected_provider].get("default_model", ""),
        "models": _get_provider_model_options(selected_provider),
        "policy": "provider-curated vision-capable model catalog",
    }


@app.post("/settings/provider")
async def save_provider_settings(req: ProviderSettingsRequest):
    """Persist new provider/model/API key to .env and hot-reload the provider."""
    global call_model, is_ready, ensure_ready, _provider_name, compact_model

    provider = req.provider.strip().lower()
    if provider not in _PROVIDER_SETTINGS:
        return JSONResponse(status_code=400, content={"detail": f"Unknown provider: {provider}"})

    current_provider = os.environ.get("EMU_PROVIDER", _provider_name).strip().lower()
    cfg = _PROVIDER_SETTINGS[provider]
    requested_model = req.model.strip()
    if requested_model:
        model = requested_model
    elif provider == current_provider:
        model = _provider_current_model(provider) or cfg["default_model"]
    else:
        model = cfg["default_model"]
    api_key = req.api_key.strip()

    # Update environment in-process
    env_updates: dict[str, str] = {"EMU_PROVIDER": provider}
    os.environ["EMU_PROVIDER"] = provider
    if cfg["key_env"] and api_key:
        os.environ[cfg["key_env"]] = api_key
        env_updates[cfg["key_env"]] = api_key
    if cfg["model_env"] and model:
        os.environ[cfg["model_env"]] = model
        env_updates[cfg["model_env"]] = model

    # Persist to .env
    _update_env_file(env_updates)

    # Hot-reload provider module so the new env vars take effect.
    # We must reload the underlying client module(s) too — most providers
    # capture env vars (model name, base URL, etc.) into module-level
    # constants at import time. Reloading only the package __init__ keeps
    # the cached client module with stale values.
    try:
        from providers.registry import _PROVIDER_MAP
        module_path = _PROVIDER_MAP[provider]

        # Reload any already-loaded submodules of this provider package so
        # module-level constants like MODEL_NAME re-read os.environ.
        import sys
        for sub_name in [m for m in list(sys.modules) if m == module_path or m.startswith(module_path + ".")]:
            try:
                importlib.reload(sys.modules[sub_name])
            except Exception:
                pass

        mod = importlib.import_module(module_path)
        importlib.reload(mod)

        try:
            compact_mod = importlib.import_module(f"{module_path}.client_compact")
            importlib.reload(compact_mod)
            compact_model = _wrap_compact_model(compact_mod.compact)
        except Exception:
            pass

        call_model = mod.call_model
        is_ready = mod.is_ready
        ensure_ready = mod.ensure_ready
        _provider_name = provider
        print(f"[settings] Provider reloaded: {provider}  model={model}")
    except Exception as e:
        print(f"[settings] Reload failed: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Provider reload failed: {e}"})

    return {"status": "ok", "provider": provider, "model": model}


@app.post("/agent/session")
async def create_session():
    """Create a new agent session."""
    session_id = str(uuid.uuid4())
    session_dir = ensure_session_dir(session_id)
    print(f"[session] created {session_id}  dir={session_dir}")
    return {"session_id": session_id}


@app.get("/agent/session/{session_id}")
async def get_session(session_id: str):
    """Return current session state."""
    return {"session_id": session_id, "status": "active"}


# ── Session metadata ────────────────────────────────────────────────────────
# A side-channel for tagging sessions with a stable `kind` (e.g.
# "remote_control") and a human label. The messaging/ bridge calls this once
# when it provisions its singleton remote-control session so the History
# sidebar can pin and badge it.

_ALLOWED_SESSION_KINDS = {"remote_control"}


@app.post("/agent/session/{session_id}/metadata")
async def set_session_metadata(session_id: str, payload: dict):
    import re
    if not re.match(r"^[a-zA-Z0-9_-]+$", session_id):
        return JSONResponse(status_code=400, content={"detail": "Invalid session_id"})

    kind = str(payload.get("kind", "")).strip()
    label = str(payload.get("label", "")).strip()
    if kind and kind not in _ALLOWED_SESSION_KINDS:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Unsupported kind: {kind!r}"},
        )
    if len(label) > 200:
        label = label[:200]

    session_dir = ensure_session_dir(session_id)
    metadata_path = session_dir / "metadata.json"
    metadata: dict = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metadata = {}
    if kind:
        metadata["kind"] = kind
    elif "kind" in payload and payload["kind"] in ("", None):
        # Explicit clear: caller sent kind="" or kind=null to demote a
        # previously-tagged session (used when rotating the remote-control
        # session so old ones drop out of the pinned sidebar group).
        metadata.pop("kind", None)
    if label:
        metadata["label"] = label
    metadata["updated_at"] = time.time()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[session-metadata] {session_id}  kind={metadata.get('kind')}  label={metadata.get('label')}")
    return {"session_id": session_id, "metadata": metadata}


@app.get("/agent/session/{session_id}/metadata")
async def get_session_metadata(session_id: str):
    import re
    if not re.match(r"^[a-zA-Z0-9_-]+$", session_id):
        return JSONResponse(status_code=400, content={"detail": "Invalid session_id"})
    from workspace import get_sessions_dir
    metadata_path = get_sessions_dir() / session_id / "metadata.json"
    if not metadata_path.exists():
        return {"session_id": session_id, "metadata": {}}
    try:
        return {
            "session_id": session_id,
            "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
        }
    except (json.JSONDecodeError, OSError):
        return {"session_id": session_id, "metadata": {}}


@app.get("/messaging/remote_session_id")
async def get_remote_session_id():
    """
    Return the singleton remote-control session id that the messaging
    bridge has provisioned, or ``null`` when none exists yet.

    Used by the Electron renderer to open a passive observer WebSocket
    against that session so WhatsApp / iMessage activity shows up live in
    the desktop UI's History sidebar with the same fidelity as a session
    started locally.
    """
    remote_file = get_emu_path() / "messaging" / "remote_session.json"
    if not remote_file.exists():
        return {"session_id": None}
    try:
        data = json.loads(remote_file.read_text(encoding="utf-8"))
        sid = data.get("session_id")
        if not isinstance(sid, str) or not sid:
            return {"session_id": None}
        # Defense in depth: re-validate the id shape even though the
        # bridge writes it.
        import re
        if not re.match(r"^[a-zA-Z0-9_-]+$", sid):
            return {"session_id": None}
        return {
            "session_id": sid,
            "label": data.get("label") or "",
            "kind": data.get("kind") or "",
        }
    except (json.JSONDecodeError, OSError):
        return {"session_id": None}


@app.post("/agent/step")
async def agent_step(req: AgentRequest):
    """
    Main agent loop entry point.

    Simple loop:
      1. Add user input to context
      2. Call model
      3. If tool_calls → execute tools, add results to context, re-call model (repeat)
      4. If desktop action → validate, dispatch to frontend
      5. If done → send final message
    """
    session_id = req.session_id or str(uuid.uuid4())
    has_screenshot = bool(req.base64_screenshot)
    has_text = bool(req.user_message.strip())
    context_manager.set_agent_mode(session_id, req.agent_mode)
    active_provider, active_model = _current_provider_model()
    context_manager.set_active_model(session_id, active_provider, active_model)

    # Register this concrete /agent/step invocation before any await points.
    # Stop requests target the active step id, so a later user message can
    # start a fresh step without clearing the stop signal for this one.
    step_id = _active_step_ids.get(session_id, 0) + 1
    _active_step_ids[session_id] = step_id

    def _is_stopped() -> bool:
        return _stopped_step_ids.get(session_id) == step_id
    cancel_key = f"{session_id}:{step_id}"

    # ── 1. Add input to context ──────────────────────────────────────────────
    if has_screenshot:
        context_manager.add_screenshot_turn(session_id, req.base64_screenshot)
        log_entry(session_id, "[user] <screenshot>")
    if has_text:
        context_manager.add_user_message(session_id, req.user_message)
        await log_and_send(session_id, f"[user] {req.user_message}", manager)

    # ── 1b. Coworker-mode per-turn AX perception (PLAN §4.6) ─────────────────
    # If we know the active (pid, window_id), refresh the AX tree + window
    # screenshot via the local driver and inject as a user-side perception
    # block before the model is called. Best-effort — silent skip on failure.
    if req.agent_mode == "coworker":
        await _inject_coworker_perception(session_id)

    if _is_stopped():
        print(f"[agent/step] Stop signal received before model loop — aborting (step_id={step_id})")
        return {"session_id": session_id, "status": "stopped", "done": False}

    # ── Log ──────────────────────────────────────────────────────────────────
    history = context_manager._history.get(session_id, [])
    print(f"\n{'=' * 60}")
    print(
        f"[agent/step] session={session_id}  mode={'screenshot' if has_screenshot else 'text'}"
        f"  agent_mode={req.agent_mode}  source={req.source}  provider={active_provider}  model={active_model}"
        f"  chain={len(history)}"
    )
    if has_text:
        print(f"  message: {req.user_message[:120]}")
    print(f"{'=' * 60}\n")

    await manager.send(session_id, {"type": "status", "message": "Processing..."})

    # ── Ensure backend ready ─────────────────────────────────────────────────
    if not is_ready():
        await manager.send(session_id, {"type": "status", "message": "Warming up..."})
    try:
        await asyncio.to_thread(ensure_ready, timeout=300, poll_interval=5)
    except TimeoutError as e:
        await manager.send(session_id, {"type": "error", "message": str(e)})
        return {"session_id": session_id, "status": "error", "error": str(e)}

    if _is_stopped():
        print(f"[agent/step] Stop signal received after warmup — aborting (step_id={step_id})")
        return {"session_id": session_id, "status": "stopped", "done": False}

    # ── 2. Model loop — tool calls resolved server-side, actions go to frontend
    # No fixed tool-call ceiling: a coworker task may legitimately chain
    # many `cua_*` calls (raise → snapshot → click → snapshot → click …)
    # before yielding a `done`. The previous `MAX_TOOL_LOOPS` cap forced
    # synthetic screenshots / done actions in the middle of useful work.
    # The user stop button is the user-facing escape, and `_is_stopped()`
    # checks below break the loop within one tool call of `/agent/stop`.
    response: AgentResponse | None = None
    plan_pending_review: str | None = None  # set when update_plan is called
    hermes_started = False
    plan_review_enabled = has_text and not req.user_message.startswith((
        "[PLAN APPROVED]",
        "User wants to refine the plan:",
    ))

    loop_i = 0
    while True:
        # ── Stop checkpoint (user clicked stop) ──────────────────────────
        # Bail before the next model call so we don't burn another
        # inference + tool round-trip after the user said stop.
        if _is_stopped():
            print(f"[agent/step] Stop signal received — aborting loop (loop_i={loop_i})")
            return {"session_id": session_id, "status": "stopped", "done": False}
        loop_i += 1

        # Call model (with timeout + provider-agnostic rate-limit retry)
        try:
            for rate_attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
                try:
                    for timeout_attempt in range(MAX_TIMEOUT_RETRIES + 1):
                        try:
                            agent_req = context_manager.build_request(session_id)
                            response = await asyncio.wait_for(
                                asyncio.to_thread(call_model, agent_req),
                                timeout=MODEL_TIMEOUT_SECS,
                            )
                            break  # success
                        except asyncio.TimeoutError:
                            if timeout_attempt < MAX_TIMEOUT_RETRIES:
                                print(f"[agent/step] Model call timed out after {MODEL_TIMEOUT_SECS}s — retrying ({timeout_attempt + 1}/{MAX_TIMEOUT_RETRIES})")
                                await manager.send(session_id, {
                                    "type": "status",
                                    "message": f"Model response timed out, retrying... ({timeout_attempt + 1}/{MAX_TIMEOUT_RETRIES})",
                                })
                            else:
                                raise TimeoutError(
                                    f"Model did not respond after {MAX_TIMEOUT_RETRIES} retries "
                                    f"(timeout={MODEL_TIMEOUT_SECS}s each)"
                                )
                    break  # success
                except Exception as rate_exc:
                    if not _is_rate_limit_error(rate_exc) or rate_attempt >= MAX_RATE_LIMIT_RETRIES:
                        raise
                    delay = _rate_limit_retry_delay(rate_exc, rate_attempt)
                    print(
                        f"[agent/step] Provider rate-limited — retrying in {delay}s "
                        f"({rate_attempt + 1}/{MAX_RATE_LIMIT_RETRIES})"
                    )
                    await manager.send(session_id, {
                        "type": "status",
                        "message": (
                            f"Provider rate limit hit; retrying in {delay}s... "
                            f"({rate_attempt + 1}/{MAX_RATE_LIMIT_RETRIES})"
                        ),
                    })
                    if not await _sleep_with_stop(delay, _is_stopped):
                        print(f"[agent/step] Stop signal received during rate-limit backoff — aborting (loop_i={loop_i})")
                        return {"session_id": session_id, "status": "stopped", "done": False}
        except Exception as e:
            from pydantic import ValidationError
            if isinstance(e, ValidationError):
                print(f"[agent/step] Pydantic Validation failed: {e}")
                # Inject the exact error into the context so the model can fix its own JSON schema violation
                context_manager.add_user_message(
                    session_id,
                    f"[SCHEMA VALIDATION FAILED] The JSON you returned was invalid or violated the schema limits:\n{e}\n\nPlease fix your formatting and try again."
                )
                await manager.send(session_id, {"type": "status", "message": "Pydantic schema validation error, returning to model to fix..."})
                continue  # Re-call the model internally

            print(f"[agent/step] Inference failed: {e}")
            await manager.send(session_id, {"type": "error", "message": f"Inference failed: {e}"})
            return {"session_id": session_id, "status": "error", "error": str(e)}

        if _is_stopped():
            print(f"[agent/step] Stop signal received after model response — aborting (loop_i={loop_i})")
            return {"session_id": session_id, "status": "stopped", "done": False}

        print(f"[agent/step] Response in {response.inference_time_ms}ms"
              f"  tool_calls={bool(response.tool_calls)}"
              f"  action={response.action.type.value if response.action else 'none'}"
              f"  done={response.done}")

        # ── 3. Tool calls? Execute and loop ──────────────────────────────────
        if response.tool_calls:
            # Store assistant turn with tool calls (OpenAI format for context)
            raw_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in response.tool_calls
            ]
            context_manager.add_tool_call_turn(session_id, raw_tool_calls)
            screenshots_for_context = []

            # Execute each tool and add results. Once the assistant tool-call
            # turn is recorded, every tool_call_id must get exactly one tool
            # result or OpenAI/Azure-compatible providers reject future turns.
            for tool_index, tc in enumerate(response.tool_calls):
                # ── Stop checkpoint inside the tool batch ──────────────
                # Multi-tool batches can take a few seconds (clicks +
                # snapshots). Check on every iteration so the user
                # doesn't have to wait through the whole batch after
                # clicking stop.
                if _is_stopped():
                    print(f"[agent/step] Stop signal received mid-batch — aborting (tool={tc.name})")
                    for pending_tc in response.tool_calls[tool_index:]:
                        context_manager.add_tool_result_turn(
                            session_id,
                            pending_tc.id,
                            pending_tc.name,
                            "[TOOL CANCELLED] User stopped execution before this tool ran.",
                        )
                    return {"session_id": session_id, "status": "stopped", "done": False}

                try:
                    args = json.loads(tc.arguments) if tc.arguments else {}
                except json.JSONDecodeError:
                    args = {}

                tool_ok, tool_err = context_manager.coworker_validator.validate_tool_call(
                    session_id,
                    tc.name,
                    args,
                    agent_mode=req.agent_mode,
                )
                if not tool_ok:
                    result = f"[TOOL REJECTED] {tool_err}"
                    context_manager.add_tool_result_turn(session_id, tc.id, tc.name, result)
                    print(f"[tool:reject] {tc.name}({json.dumps(args, ensure_ascii=False)[:120]}) → {tool_err}")
                    await log_and_send(
                        session_id,
                        f"[tool] {tc.name}({json.dumps(args, ensure_ascii=False)[:200]}) → {result[:300]}",
                        manager,
                        metadata={
                            "tool_name": tc.name,
                            "args": json.dumps(args, ensure_ascii=False),
                            "result": result,
                            "status": "rejected",
                        },
                    )
                    continue

                await manager.send(session_id, {
                    "type": "tool_event",
                    "event": "tool_call_started",
                    "tool": tc.name,
                    "args": json.dumps(args, ensure_ascii=False)[:300],
                })
                log_entry(
                    session_id,
                    f"[tool:start] {tc.name}({json.dumps(args, ensure_ascii=False)[:500]})",
                    metadata={
                        "tool_name": tc.name,
                        "args": json.dumps(args, ensure_ascii=False),
                        "status": "started",
                    },
                )
                print(f"[tool:start] {tc.name}({json.dumps(args, ensure_ascii=False)[:120]})")

                try:
                    raw_result = await execute_agent_tool(
                        session_id, tc.name, args, manager, context_manager, compact_model,
                        agent_mode=req.agent_mode,
                        cancel_key=cancel_key,
                    )
                except Exception as tool_exc:
                    raw_result = f"[TOOL ERROR] {type(tool_exc).__name__}: {tool_exc}"
                screenshot_for_context = None
                if isinstance(raw_result, dict):
                    result = str(raw_result.get("text") or "")
                    screenshot_for_context = raw_result.get("screenshot")
                else:
                    result = str(raw_result)

                context_manager.add_tool_result_turn(session_id, tc.id, tc.name, result)
                if screenshot_for_context:
                    screenshots_for_context.append(screenshot_for_context)
                context_manager.coworker_validator.record_tool_result(
                    session_id,
                    tc.name,
                    args,
                    result,
                    agent_mode=req.agent_mode,
                )
                print(f"[tool] {tc.name}({json.dumps(args)[:80]}) → {result[:150]}")
                await log_and_send(
                    session_id,
                    f"[tool] {tc.name}({json.dumps(args, ensure_ascii=False)[:200]}) → {result[:300]}",
                    manager,
                    metadata={"tool_name": tc.name, "args": json.dumps(args, ensure_ascii=False), "result": result},
                )

                if _is_stopped():
                    print(f"[agent/step] Stop signal received after tool returned — aborting (tool={tc.name})")
                    for pending_tc in response.tool_calls[tool_index + 1:]:
                        context_manager.add_tool_result_turn(
                            session_id,
                            pending_tc.id,
                            pending_tc.name,
                            "[TOOL CANCELLED] User stopped execution before this tool ran.",
                        )
                    return {"session_id": session_id, "status": "stopped", "done": False}

                # If update_plan was called, capture the plan content for review
                if plan_review_enabled and tc.name == "update_plan":
                    plan_content = args.get("content", "")
                    if plan_content:
                        plan_pending_review = plan_content

                # Hermes runs as a background job. Once it is successfully
                # assigned, yield control back to the user instead of asking
                # the model to immediately continue/poll in the same turn.
                if tc.name == "invoke_hermes" and result.startswith("Hermes job started:"):
                    hermes_started = True
                    break

            if _is_stopped():
                print(f"[agent/step] Stop signal received after tool batch — aborting (loop_i={loop_i})")
                return {"session_id": session_id, "status": "stopped", "done": False}

            for screenshot in screenshots_for_context:
                context_manager.add_screenshot_turn(session_id, screenshot)

            if hermes_started:
                response = AgentResponse(
                    action=Action(type=ActionType.DONE),
                    done=True,
                    confidence=1.0,
                    final_message=(
                        "Hermes agent has been assigned and is running in the background. "
                        "Feel free to check in after some time."
                    ),
                    reasoning_content=(
                        "Hermes job started asynchronously; yielding control back to the user."
                    ),
                )
                break

            # ── Plan review gate: pause loop and wait for user approval ──────
            if plan_pending_review:
                print(f"[plan-review] Plan created — pausing for user approval")
                await manager.send(session_id, {
                    "type": "plan_review",
                    "content": plan_pending_review,
                })
                return {
                    "session_id": session_id,
                    "status": "plan_pending",
                    "done": False,
                }

            await manager.send(session_id, {
                "type": "status",
                "message": f"Tools: {', '.join(tc.name for tc in response.tool_calls)}. Continuing...",
            })
            continue  # re-call model with tool results

        # No tool calls — we have a desktop action or done.

        # ── Safety: ensure we have an action ──────────────────────────────────────
        if not response.action:
            response.action = Action(type=ActionType.SCREENSHOT)
            response.done = False

        # ── Recovery: model returned an unparseable / unknown action.
        # The provider wraps plain-text or malformed-JSON responses as
        # action=unknown. Never promote these to "done" — a garbled action
        # payload is not evidence the task is complete. Instead, nudge the
        # model to re-orient with a fresh screenshot and try again.
        if response.action.type == ActionType.UNKNOWN:
            preview = (response.final_message or "").strip()[:200]
            print(f"[agent/step] unknown action received, requesting retry: {preview!r}")
            context_manager.add_assistant_turn(
                session_id,
                response.model_dump_json(exclude_none=True),
            )
            context_manager.add_user_message(
                session_id,
                "[MALFORMED RESPONSE] Your previous response could not be parsed as a "
                "valid desktop action. Do NOT treat this as task completion. Take a "
                "screenshot to re-orient and then emit a well-formed action JSON.",
            )
            response.action = Action(type=ActionType.SCREENSHOT)
            response.done = False
            response.final_message = None
            await manager.send(session_id, {"type": "status", "message": "Unparseable action, retaking screenshot..."})
            continue

        action_type = response.action.type.value
        action_payload = response.action.model_dump(exclude_none=True)

        # ── 4. Action validation ─────────────────────────────────────────────────
        if req.agent_mode == "coworker" and response.action.type != ActionType.DONE:
            bad_action = response.action.type.value
            print(f"[coworker-validator] REJECTED remote action in coworker mode: {bad_action}")
            context_manager.add_assistant_turn(
                session_id,
                response.model_dump_json(exclude_none=True),
            )
            context_manager.add_user_message(
                session_id,
                "[COWORKER ACTION REJECTED] You returned a remote-mode desktop action "
                f"`{bad_action}`. In coworker mode, remote actions like navigate_and_click, "
                "key_press, scroll, type_text, screenshot, mouse_move, drag, and click actions "
                "are invalid because they bypass the backend emu-cua-driver tool workflow and "
                "do not carry the required pid/window_id/element_index context.\n\n"
                "Use function tools instead: raise_app/list_running_apps/cua_list_windows for "
                "discovery, cua_get_window_state for AX state, cua_click/cua_press_key/"
                "cua_scroll/cua_type_text for interaction, and only emit the raw JSON `done` "
                "action when the task is complete or blocked."
            )
            await manager.send(session_id, {
                "type": "status",
                "message": f"Rejected remote action {bad_action} in coworker mode; asking model to use cua_* tools.",
            })
            continue

        is_valid, error_msg = context_manager.remote_validator.validate(
            session_id, action_payload
        )

        if not is_valid:
            print(f"[validator] REJECTED: {error_msg}")
            context_manager.add_assistant_turn(
                session_id,
                response.model_dump_json(exclude_none=True),
            )
            context_manager.add_user_message(
                session_id,
                f"[ACTION REJECTED] {error_msg}\nChoose a different action.",
            )
            await manager.send(session_id, {"type": "status", "message": f"Rejected: {error_msg}. Retrying..."})
            continue  # Re-call the model internally

        # ── 4b. Catch bogus "done" from truncated action parse ───────────────────
        if response.done and response.action and response.action.type == ActionType.DONE:
            done_ok, done_err = context_manager.remote_validator.validate_done_response(
                response.final_message
            )
            if not done_ok:
                print(f"[validator] REJECTED done: {done_err}")
                context_manager.add_user_message(
                    session_id,
                    "[MALFORMED RESPONSE] Your previous response was garbled — it looked like "
                    "a truncated desktop action, not a real completion. Take a screenshot to "
                    "re-orient and then continue with your task."
                )
                response.action = Action(type=ActionType.SCREENSHOT)
                response.done = False
                response.final_message = None
                await manager.send(session_id, {"type": "status", "message": "Truncated response detected, retaking screenshot..."})
                continue

            if req.agent_mode == "coworker":
                verify_ok, verify_err = context_manager.coworker_validator.validate_done_response(
                    session_id,
                    response.final_message,
                )
                if not verify_ok:
                    print(f"[coworker-validator] REJECTED unverified done: {verify_err}")
                    context_manager.add_assistant_turn(
                        session_id,
                        response.model_dump_json(exclude_none=True),
                    )
                    context_manager.add_user_message(
                        session_id,
                        f"[COWORKER VERIFY REQUIRED] {verify_err}",
                    )
                    await manager.send(session_id, {
                        "type": "status",
                        "message": "Coworker action needs verification before reporting success.",
                    })
                    continue

        break  # Valid desktop action or done, dispatch to frontend

    # ── 5. Dispatch action to frontend ────────────────────────────────────────
    if not response or not response.action:
        response = response or AgentResponse(action=Action(type=ActionType.SCREENSHOT), done=False)
        if not response.action:
            response.action = Action(type=ActionType.SCREENSHOT)
            response.done = False

    action_type = response.action.type.value
    action_payload = response.action.model_dump(exclude_none=True)

    response_json = response.model_dump_json(exclude_none=True)
    context_manager.add_assistant_turn(session_id, response_json)

    # Log the assistant's action/response
    reasoning_preview = (response.reasoning_content or "")[:200]
    if response.done:
        await log_and_send(
            session_id,
            f"[assistant] DONE — {response.final_message or 'Task complete.'}",
            manager,
            metadata={"done": True, "final_message": response.final_message or "Task complete."},
        )
    else:
        await log_and_send(
            session_id,
            f"[action] {action_type}  confidence={response.confidence}  reasoning={reasoning_preview}",
            manager,
            metadata={
                "action_type": action_type,
                "confidence": response.confidence,
                "reasoning": response.reasoning_content or "",
                "action": action_payload,
            },
        )

    # Safety-net auto-compaction
    needs_compact = context_manager.needs_compaction(session_id)
    if needs_compact:
        print(f"[auto-compact] Safety net triggered at {context_manager.chain_length(session_id)} messages")

    needs_confirm = False

    step_payload = {
        "type":                 "step",
        "reasoning":            response_json,
        "reasoning_content":    response.reasoning_content or "",
        "action":               action_payload,
        "done":                 response.done,
        "confidence":           response.confidence,
        "final_message":        response.final_message or ("Task complete." if response.done else None),
        "requires_confirmation": needs_confirm,
        "chain_length":         context_manager.chain_length(session_id),
        "needs_compaction":     needs_compact,
    }

    # Coworker capture path (PLAN §6.5): expose the active (pid, window_id)
    # so the renderer's captureForStep can pull a target-window screenshot
    # via emu-cua-driver instead of desktopCapturer on the next loop turn.
    # Always include the key in coworker mode (None clears a stale target).
    if req.agent_mode == "coworker":
        target = context_manager.get_coworker_target(session_id)
        if target:
            pid, window_id = target
            step_payload["coworker_target"] = {"pid": pid, "window_id": window_id}
        else:
            step_payload["coworker_target"] = None

    await manager.send(session_id, step_payload)

    # Run auto-compact if needed
    if needs_compact and not response.done:
        await auto_compact(session_id, context_manager, compact_model, manager)

    if response.done:
        await manager.send(session_id, {
            "type": "done",
            "message": response.final_message or "Task complete.",
        })

    return {
        "session_id": session_id,
        "status": "done" if response.done else "action_dispatched",
        "action": action_payload,
        "done": response.done,
        "confidence": response.confidence,
        "final_message": response.final_message,
    }


@app.post("/action/complete")
async def action_complete(req: ActionCompleteRequest):
    """Electron notifies backend that a dispatched action has finished."""
    print(f"[action/complete] session={req.session_id} channel={req.ipc_channel} ok={req.success}")

    text_parts = []

    is_shell = req.ipc_channel in ("shell:exec", "shell_exec")

    if is_shell:
        # Shell exec: always inject stdout; inject stderr on failure
        if req.output:
            text_parts.append(f"[shell_exec output]\n{req.output}")
        if req.error and not req.success:
            text_parts.append(f"[shell_exec error]\n{req.error}")
    elif not req.success and req.error:
        # Non-shell actions: only inject on failure with interpreted guidance
        action_label = ipc_to_action_label(req.ipc_channel)
        text_parts.append(interpret_action_error(req.error, action_label))

    if text_parts:
        context_manager.add_user_message(req.session_id, "\n".join(text_parts))
        print(f"[action/complete] injected feedback into context ({len(' '.join(text_parts))} chars)")

    return {"acknowledged": True}





@app.post("/agent/stop")
async def agent_stop(req: StopRequest):
    """User interrupted the agent flow."""
    session_id = req.session_id
    # Mark the currently active step as stopped FIRST so its loop bails out
    # at the next checkpoint. The token is step-specific: if the user sends a
    # new message immediately after stopping, the old loop still stops and the
    # new loop gets a fresh token.
    active_step_id = _active_step_ids.get(session_id)
    if active_step_id is not None:
        _stopped_step_ids[session_id] = active_step_id
        killed = cancel_driver_calls(f"{session_id}:{active_step_id}")
    else:
        killed = 0
    context_manager.add_user_message(session_id, "STOP")
    print(f"[agent/stop] session={session_id} step_id={active_step_id} killed_driver_calls={killed}")

    await manager.send(session_id, {
        "type": "stopped",
        "message": "Agent stopped by user.",
    })

    return {"session_id": session_id, "status": "stopped"}


@app.post("/agent/compact")
async def compact_context(req: CompactRequest):
    """Manual compact endpoint (user-triggered via UI)."""
    session_id = req.session_id
    chain_len = context_manager.chain_length(session_id)
    print(f"[compact] session={session_id}  chain_length={chain_len}")

    if chain_len <= 4:
        return {
            "session_id": session_id,
            "status": "skipped",
            "message": "Context chain is too short to compact.",
            "chain_length": chain_len,
        }

    compact_messages = context_manager.get_compact_messages(session_id)

    await manager.send(session_id, {
        "type": "status",
        "message": "Compacting context...",
    })

    try:
        summary = compact_model(compact_messages)
    except Exception as e:
        print(f"[compact] Failed: {e}")
        await manager.send(session_id, {
            "type": "error",
            "message": f"Context compaction failed: {e}",
        })
        return {"session_id": session_id, "status": "error", "error": str(e)}

    context_manager.reset_with_summary(session_id, summary)
    new_len = context_manager.chain_length(session_id)

    print(f"[compact] Done. {chain_len} → {new_len} messages")

    await manager.send(session_id, {
        "type": "status",
        "message": f"Context compacted: {chain_len} → {new_len} messages.",
    })

    return {
        "session_id": session_id,
        "status": "compacted",
        "previous_length": chain_len,
        "new_length": new_len,
    }


@app.get("/sessions/history")
async def sessions_history():
    """Return all sessions with their first user message for the history sidebar."""
    from workspace import get_sessions_dir
    sessions_dir = get_sessions_dir()
    results = []

    if not sessions_dir.is_dir():
        return {"sessions": []}

    import re as _re
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        # Only process directories with valid session-id names (skip symlinks, junctions)
        if not _re.match(r'^[a-zA-Z0-9_-]+$', session_dir.name) or session_dir.is_symlink():
            continue
        conv_path = session_dir / "logs" / "conversation.json"
        # Load optional metadata (kind/label) for sidebar pinning + badging.
        metadata: dict = {}
        metadata_path = session_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                metadata = {}
        if not conv_path.exists():
            # Sessions with metadata but no conversation yet (e.g. a freshly
            # provisioned remote-control session) should still appear so the
            # user can see the channel is wired up.
            if metadata.get("kind"):
                results.append({
                    "session_id": session_dir.name,
                    "preview": metadata.get("label", "Remote control"),
                    "message_count": 0,
                    "last_active": metadata.get("updated_at", session_dir.stat().st_mtime),
                    "kind": metadata.get("kind"),
                    "label": metadata.get("label"),
                })
            continue
        try:
            with open(conv_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages = data.get("messages", [])
            if not messages and not metadata.get("kind"):
                continue
            # Find the first user message for the preview. Strip the
            # [whatsapp:…] / [imessage:…] origin prefix the bridge prepends
            # so the sidebar shows the actual question, not the routing tag.
            first_user_content = ""
            for m in messages:
                if m.get("role") != "user":
                    continue
                raw = (m.get("content") or "").strip()
                if not raw or raw == "<screenshot>":
                    continue
                first_user_content = _re.sub(
                    r'^\[(?:whatsapp|imessage|discord):[^\]]+\]\s*',
                    '',
                    raw,
                )
                break
            preview = first_user_content or metadata.get("label") or ""
            if not preview:
                continue
            mtime = conv_path.stat().st_mtime
            entry = {
                "session_id": session_dir.name,
                "preview": preview[:80],
                "message_count": len(messages),
                "last_active": mtime,
            }
            if metadata.get("kind"):
                entry["kind"] = metadata["kind"]
            if metadata.get("label"):
                entry["label"] = metadata["label"]
            results.append(entry)
        except (json.JSONDecodeError, KeyError):
            continue

    # Sort by last_active descending (most recent first)
    results.sort(key=lambda s: s["last_active"], reverse=True)
    return {"sessions": results}


@app.get("/sessions/{session_id}/messages")
async def session_messages(session_id: str):
    """Return the full conversation.json for a given session."""
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        return JSONResponse(status_code=400, content={"detail": "Invalid session_id"})
    from workspace import get_sessions_dir
    conv_path = get_sessions_dir() / session_id / "logs" / "conversation.json"
    if not conv_path.exists():
        return {"messages": []}
    try:
        with open(conv_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, KeyError):
        return {"messages": []}


@app.get("/sessions/{session_id}/latest_screenshot")
async def session_latest_screenshot(session_id: str):
    """
    Return raw bytes of the most recent screenshot captured for a session.

    Used by the messaging bridge to attach a current frame to outbound
    replies (e.g. a WhatsApp "done" message gets the final desktop snapshot
    so the remote sender can confirm the agent's outcome visually).
    """
    import re
    from starlette.responses import Response

    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        return JSONResponse(status_code=400, content={"detail": "Invalid session_id"})

    cached = context_manager.get_latest_screenshot(session_id)
    if not cached:
        return JSONResponse(status_code=404, content={"detail": "No screenshot available"})

    mime, body = cached
    return Response(content=body, media_type=mime or "image/png")


@app.post("/agent/session/continue")
async def continue_session(req: ContinueSessionRequest):
    """
    Rehydrate backend context for an existing session so the next user message
    continues it under the same visible session id.
    """
    import re

    old_id = req.previous_session_id
    if not re.match(r'^[a-zA-Z0-9_-]+$', old_id):
        return JSONResponse(status_code=400, content={"detail": "Invalid session_id"})

    from workspace import get_sessions_dir
    old_dir = get_sessions_dir() / old_id
    conv_path = old_dir / "logs" / "conversation.json"

    if not conv_path.exists():
        return JSONResponse(status_code=404, content={"detail": "Session not found"})

    try:
        with open(conv_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        old_messages = data.get("messages", [])
    except (json.JSONDecodeError, KeyError):
        return JSONResponse(status_code=400, content={"detail": "Could not read session"})

    context_manager.clear_session(old_id)
    context_manager.set_agent_mode(old_id, req.agent_mode)
    context_manager.preload_from_conversation(old_id, old_messages)

    print(f"[session] continued {old_id}")
    return {"session_id": old_id}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    # Validate auth token from query parameter before accepting
    token = ws.query_params.get("token", "")
    if not secrets.compare_digest(token, AUTH_TOKEN):
        await ws.close(code=4001, reason="Invalid auth token")
        return
    await manager.connect(session_id, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id)
