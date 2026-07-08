"""
client.py — OpenRouter API client

Uses the OpenAI Chat Completions API format against OpenRouter's endpoint.
Supports native function calling for agent tools (plan, memory, skills, etc.).
Desktop actions are returned as JSON text responses.

Docs: https://openrouter.ai/docs/quickstart

Environment:
    OPENROUTER_API_KEY  — required
    OPENROUTER_MODEL    — optional (default: anthropic/claude-sonnet-4)
    OPENROUTER_REQUEST_TIMEOUT_SECS — optional (default: 60)
"""

import json
import os
import re
import time

from openai import OpenAI

from models import Action, AgentRequest, AgentResponse, MessageRole, ToolCallInfo, safe_build_action
from providers.agent_tools import get_agent_tools_openai

# -- Configuration -----------------------------------------------------------

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL_NAME = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")
PROVIDER_NAME = os.environ.get("OPENROUTER_PROVIDER_NAME", "")
BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 8000
TEMPERATURE = 0.6


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


REQUEST_TIMEOUT_SECS = _positive_int_env(
    "OPENROUTER_REQUEST_TIMEOUT_SECS",
    _positive_int_env("EMU_MODEL_TIMEOUT_SECS", 60),
)

SCREENSHOT_PREFIX = "data:image/"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=REQUEST_TIMEOUT_SECS)


# -- Public API (matches provider interface) ---------------------------------

def call_model(agent_req: AgentRequest) -> AgentResponse:
    system_prompt, messages = _build_messages(agent_req)

    kwargs = {
        "model": MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "tools": get_agent_tools_openai(agent_req.agent_mode),
        "messages": [{"role": "system", "content": system_prompt}] + messages,
    }

    if PROVIDER_NAME:
        kwargs["extra_body"] = {
            "provider": {
                "order": [PROVIDER_NAME],
                "allow_fallbacks": False
            }
        }

    start = time.time()
    resp = client.chat.completions.create(**kwargs)
    elapsed_ms = int((time.time() - start) * 1000)

    return _parse_response(resp, elapsed_ms)


def is_ready() -> bool:
    return bool(API_KEY)


def ensure_ready(**kwargs) -> None:
    if not API_KEY:
        raise RuntimeError(
            "[openrouter] OPENROUTER_API_KEY is not set. "
            "Get one at https://openrouter.ai/keys"
        )


# -- Message builder ---------------------------------------------------------

def _build_messages(req: AgentRequest) -> tuple[str, list[dict]]:
    """Build (system_prompt, messages) in OpenAI Chat Completions format."""
    system_prompt = ""
    raw: list[dict] = []

    for pm in req.previous_messages:
        if pm.role == MessageRole.system:
            system_prompt = pm.content

        elif pm.role == MessageRole.assistant:
            msg = {"role": "assistant"}
            if pm.tool_calls:
                msg["tool_calls"] = pm.tool_calls
                msg["content"] = pm.content if pm.content else None
            else:
                msg["content"] = pm.content
            raw.append(msg)

        elif pm.role == MessageRole.tool:
            msg = {
                "role": "tool",
                "tool_call_id": pm.tool_call_id or "",
                "content": pm.content,
            }
            if pm.tool_name:
                msg["name"] = pm.tool_name
            raw.append(msg)

        elif pm.role == MessageRole.user:
            if pm.content.startswith(SCREENSHOT_PREFIX):
                raw.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": pm.content}},
                        {
                            "type": "text",
                            "text": (
                                "[SCREENSHOT] Current screen state above. Reply with "
                                "exactly ONE of: (a) one function tool call, or (b) one "
                                "desktop action as plain JSON text in your message, e.g. "
                                '{"action": {"type": "navigate_and_click", "coordinates": '
                                '{"x": 0.5, "y": 0.5}}, "done": false}. Desktop actions '
                                "(screenshot, clicks, type_text, key_press, scroll, drag, "
                                "wait, done) are NEVER function tools."
                            ),
                        },
                    ],
                })
            else:
                raw.append({"role": "user", "content": pm.content})

    return system_prompt or "", raw


# -- Response parser ---------------------------------------------------------

def _parse_response(resp, elapsed_ms: int) -> AgentResponse:
    choice = resp.choices[0] if resp.choices else None
    message = choice.message if choice else None
    content = message.content or "" if message else ""

    reasoning = getattr(message, "reasoning_content", None) if message else None

    # ── Tool calls? Return them directly — main.py will execute and re-call ──
    if message and message.tool_calls:
        if len(message.tool_calls) == 1 and message.tool_calls[0].function.name == "done":
            try:
                args = json.loads(message.tool_calls[0].function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            return AgentResponse(
                action=Action(type="done"),
                done=True,
                final_message=args.get("final_message") or "Task complete.",
                reasoning_content=reasoning,
                inference_time_ms=elapsed_ms,
                model_name=MODEL_NAME,
            )

        tool_calls = [
            ToolCallInfo(
                id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments,
            )
            for tc in message.tool_calls
        ]
        print(f"[openrouter] tool_calls: {[tc.name for tc in tool_calls]}")
        return AgentResponse(
            tool_calls=tool_calls,
            done=False,
            reasoning_content=reasoning,
            inference_time_ms=elapsed_ms,
            model_name=MODEL_NAME,
        )

    # ── No tool calls — parse JSON desktop action from text ──────────────────
    data = _extract_json(content)
    data = _sanitize_single_action(data)

    raw_action = data.get("action", {"type": "done"})
    if isinstance(raw_action, str):
        raw_action = {"type": raw_action}

    action = safe_build_action(raw_action, "openrouter")
    is_done_action = action.type.value == "done"

    return AgentResponse(
        action=action,
        done=bool(data.get("done", is_done_action)) or is_done_action,
        final_message=data.get("final_message"),
        confidence=data.get("confidence", 1.0),
        reasoning_content=reasoning,
        inference_time_ms=elapsed_ms,
        model_name=MODEL_NAME,
    )


def _sanitize_single_action(data: dict) -> dict:
    """Enforce one-action-per-response. Strip extra actions the model may batch."""
    for key in ("next", "next_action", "actions", "step2", "then"):
        if key in data:
            print(f"[openrouter] WARN: stripped batched key '{key}' from response")
            del data[key]

    fm = data.get("final_message")
    if isinstance(fm, str) and fm.strip().startswith("{"):
        try:
            inner = json.loads(fm)
            if isinstance(inner, dict) and "action" in inner:
                print(f"[openrouter] WARN: final_message contained nested action — extracting")
                data["action"] = inner["action"]
                data["done"] = inner.get("done", False)
                data["confidence"] = inner.get("confidence", data.get("confidence", 1.0))
                data["final_message"] = inner.get("final_message")
        except (json.JSONDecodeError, ValueError):
            pass

    return data


def _extract_json(content: str) -> dict:
    """Pull the JSON object out of the model's text response."""
    text = content.strip()

    # Strip markdown fences
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    # Extract first JSON object
    brace_start = text.find("{")
    if brace_start != -1:
        try:
            decoder = json.JSONDecoder()
            obj, end_idx = decoder.raw_decode(text, brace_start)
            remainder = text[end_idx:].strip()
            if remainder:
                print(f"[openrouter] WARN: multiple objects, using first. Discarded: {remainder[:120]}")
            return obj
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Common JSON repairs
    repaired = text
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    repaired = re.sub(r'"x"\s*:\s*(\d+)\s*,\s*(\d+)\s*}', r'"x":\1,"y":\2}', repaired)
    repaired = re.sub(
        r'"coordinates"\s*:\s*\{\s*"?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*"?\s*\}',
        r'"coordinates":{"x":\1,"y":\2}',
        repaired,
    )
    repaired = re.sub(
        r'"coordinates"\s*:\s*\{\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\}',
        r'"coordinates":{"x":\1,"y":\2}',
        repaired,
    )
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Unparseable — wrap as unknown so the validator sends a format-error back to the model
    print(f"[openrouter] INFO: plain-text response, wrapping as unknown:\n  {content[:200]}")
    return {"action": {"type": "unknown"}, "done": False, "final_message": content.strip(), "confidence": 0.0}
