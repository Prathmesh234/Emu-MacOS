"""
context_manager/context.py

Chains PreviousMessage turns per session so each AgentRequest to the model
carries the right conversation history.

Key features:
  - Screenshot management: only latest screenshot kept as image data
  - Action validation: blocks invalid actions at runtime (replaces prompt rules)
  - Plan auto-injection: injects plan.md every N turns to keep model anchored
  - Model-driven compaction: model calls compact_context tool, harness executes
"""

import json
import os
import sys

from models import AgentRequest, PreviousMessage, MessageRole, ScreenAnnotation, ScreenElement
from prompts import build_system_prompt, build_coworker_system_prompt
from prompts.plan_prompt import PLAN_DIRECTIVE, PLAN_REMINDER
from workspace import build_workspace_context, get_device_details, is_bootstrap_needed, read_bootstrap
from context_manager.coworker_validator import CoworkerActionValidator
from context_manager.remote_validator import RemoteActionValidator
from utilities.tool_history import repair_orphan_tool_calls

# OmniParser toggle
USE_OMNI_PARSER = (
    "--use-omni-parser" in sys.argv
    or os.environ.get("USE_OMNI_PARSER", "").lower() in ("1", "true", "yes")
)

# Prefix used to identify screenshot messages in history
SCREENSHOT_PREFIX = "data:image/"

# Maximum messages before middle-trimming (safety net)
MAX_CHAIN_LENGTH = 300

# Auto-compaction safety net — only fires if model doesn't self-compact.
# Set high: the model should call compact_context before this triggers.
AUTO_COMPACT_THRESHOLD = 300

# Number of recent messages to preserve verbatim during compaction.
# Only the older messages before this tail window get summarised.
KEEP_RECENT_MESSAGES = 30

# Plan auto-injection interval (every N assistant turns)
PLAN_INJECT_INTERVAL = 40

# Injected at the end of a preloaded context to orient the model for continuation
CONTEXT_CONTINUATION_MSG = """\
[CONTEXT CONTINUATION]

You are resuming a previous session. The conversation history above is everything that happened in that session.

When the user sends their next message:
- If they say "continue" or give no new objective, resume the original goal with a fresh plan that picks up from where you left off.
- If they provide new or additional context, incorporate it into your approach.

Either way, you MUST call update_plan before taking any desktop actions. The user will review and approve the plan before you proceed.\
"""


def _active_model_block(provider: str, model: str) -> str:
    provider = (provider or "unknown").strip() or "unknown"
    model = (model or "unknown").strip() or "unknown"
    return f"""\

<active_model>
Provider: {provider}
Model: {model}

If the user asks what model you are, which model is running, or whether the
model changed, answer from this active_model block. It supersedes any earlier
conversation text about the active model.
</active_model>"""


def _mode_transition_block(previous_mode: str, current_mode: str) -> str:
    if previous_mode == current_mode:
        return ""

    if current_mode == "remote":
        return """\

<mode_transition>
This session has just switched from coworker mode to remote mode.

Historical coworker context may mention `cua_*` function tools, `(pid, window_id)`,
and `[element_index N]` values from emu-cua-driver snapshots. Treat those as
history only: they describe what was attempted and observed, but they are NOT
valid remote-mode commands and old element_index values are not usable here.

Current mode: REMOTE. You do not have `cua_*` desktop-driver tools in this
request. Use only the remote action JSON protocol from this system prompt —
emitted as plain JSON text in your message content, NEVER as function/tool
calls (only the function tools listed in <channels> may be tool calls):
  - `screenshot`: ask the harness for a fresh screen image.
  - `navigate_and_click` / `navigate_and_right_click` / `navigate_and_triple_click`:
    move to normalized screen coordinates and click.
  - `left_click` / `right_click` / `double_click` / `triple_click`: click at the
    current cursor position.
  - `mouse_move`: move the cursor to normalized coordinates without clicking.
  - `drag`: drag from `coordinates` to `end_coordinates`.
  - `scroll`: scroll up/down by amount.
  - `type_text`: type the provided text into the current focus.
  - `key_press`: press one key, optionally with modifiers.
  - `wait`: pause briefly.
  - `done`: finish, ask one question, or report a blocker.

If the prior coworker task was stopped, resume from the user's goal and the
useful observations in history, but take a fresh `screenshot` before any
coordinate-dependent action unless the current screen state is already obvious.
</mode_transition>"""

    if current_mode == "coworker":
        return """\

<mode_transition>
This session has just switched from remote mode to coworker mode.

Historical remote context may mention desktop action JSON such as
`screenshot`, `navigate_and_click`, `type_text`, `key_press`, `scroll`, or
normalized screen coordinates. Treat those as history only: they describe what
was attempted and observed, but they are NOT valid coworker interaction
commands except for `done`.

Current mode: COWORKER. The only action JSON you may emit is `done`. All real
desktop interaction must use function tools available in this request:
  - Discovery: prefer non-disruptive `list_running_apps`, `cua_list_apps`,
    and `cua_list_windows` for existing targets. Use `cua_launch_app` or
    `raise_app` only when no usable running target exists or opening/launching
    is explicitly required. If the user explicitly approved a disruptive
    foreground fallback, `bring_app_frontmost`.
  - Perception: `cua_get_window_state` for AX tree + element_index cache,
    `cua_screenshot` for pixels, `cua_zoom` for tiny visual details.
  - Interaction: `cua_click`, `cua_right_click`, `cua_double_click`,
    `cua_type_text`, `cua_set_value`, `cua_press_key`, `cua_hotkey`,
    `cua_scroll`, `cua_drag`.
  - Browser DOM: `cua_page` when AX is sparse and page/DOM data is needed.
  - Finish: raw `done` JSON only when complete, blocked, or asking one question.

Avoid disrupting the user's active app/Space. Do not use `raise_app` as a
routine targeting primitive, and do not use `bring_app_frontmost` unless the
user explicitly approves foregrounding after background `cua_*` paths are
insufficient.

For browser URL/navigation requests, first discover existing browser windows.
Use `cua_launch_app(..., urls=[...])` only when opening/navigating is actually
required and no lower-disruption route is available. Normalize bare domains
with `https://`, choose user-named/existing/Chrome browser, and do not
click/type the address bar plus Return.

If the prior remote task was stopped, resume from the user's goal and the useful
observations in history, but first rebuild coworker targeting with discovery and
fresh `cua_get_window_state` before using any `element_index`. Remote
coordinates do not translate into coworker element indices.
</mode_transition>"""

    return ""


class ContextManager:
    """
    Maintains conversation history per session.

    Typical call order:
    1. add_user_message(session_id, text)           → user types a task
    2. build_request(session_id)                     → send to model
    3. add_assistant_turn(session_id, response_json) → model response
    4. add_screenshot_turn(session_id, base64)       → screenshot from frontend
    5. build_request(session_id)                     → send again
    6. Repeat 3-5
    """

    # Plan reminder injection interval (every N assistant turns, offset from plan inject)
    PLAN_REMINDER_INTERVAL = 40

    def __init__(self):
        self._history: dict[str, list[PreviousMessage]] = {}
        self._step_offset: dict[str, int] = {}
        self._plan_injected: dict[str, bool] = {}
        self._agent_mode: dict[str, str] = {}
        self._active_model: dict[str, tuple[str, str]] = {}
        # Per-session active (pid, window_id) tracked for coworker-mode
        # per-turn AX perception injection (see PLAN §4.6).
        self._coworker_target: dict[str, tuple[int, int]] = {}
        # Cache of the most recent screenshot per session, kept as
        # (mime_type, raw_bytes). Used by GET /sessions/{id}/latest_screenshot
        # so the messaging bridge can attach a current frame to outbound
        # replies without scanning the full history each time.
        self._latest_screenshot: dict[str, tuple[str, bytes]] = {}
        self.remote_validator = RemoteActionValidator()
        self.coworker_validator = CoworkerActionValidator()

    def _build_system_prompt(self, session_id: str, mode: str) -> str:
        from workspace import is_hermes_setup_needed
        workspace_ctx = build_workspace_context()
        bootstrap_mode = is_bootstrap_needed()
        bootstrap_content = read_bootstrap() if bootstrap_mode else ""
        hermes_setup_mode = (not bootstrap_mode) and is_hermes_setup_needed()
        device_info = get_device_details()
        if mode == "coworker":
            return build_coworker_system_prompt(
                workspace_context=workspace_ctx,
                session_id=session_id,
                device_details=device_info,
            )
        return build_system_prompt(
            workspace_ctx,
            session_id=session_id,
            bootstrap_mode=bootstrap_mode,
            bootstrap_content=bootstrap_content,
            device_details=device_info,
            use_omni_parser=USE_OMNI_PARSER,
            hermes_setup_mode=hermes_setup_mode,
        )

    def _replace_system_prompt(
        self,
        session_id: str,
        mode: str,
        previous_mode: str | None = None,
    ) -> None:
        history = self._history.get(session_id)
        if not history:
            return

        prompt = self._build_system_prompt(session_id, mode).strip()
        if previous_mode and previous_mode != mode:
            prompt += _mode_transition_block(previous_mode, mode)

        for idx, message in enumerate(history):
            if message.role == MessageRole.system:
                history[idx] = PreviousMessage(role=MessageRole.system, content=prompt)
                return

        history.insert(0, PreviousMessage(role=MessageRole.system, content=prompt))

    def _get(self, session_id: str) -> list[PreviousMessage]:
        """Return existing history or bootstrap a new session with the system prompt."""
        if session_id not in self._history:
            system_prompt = self._build_system_prompt(
                session_id,
                self._agent_mode.get(session_id, "coworker"),
            )
            self._history[session_id] = [
                PreviousMessage(role=MessageRole.system, content=system_prompt.strip())
            ]
        return self._history[session_id]

    def add_user_message(self, session_id: str, text: str) -> None:
        """Append a text-only user message.

        On the first real user task (not system feedback like tool results),
        auto-injects the planning directive so the model plans before acting.
        """
        if not text or not text.strip():
            return

        history = self._get(session_id)
        history.append(
            PreviousMessage(role=MessageRole.user, content=text)
        )

        # Inject planning directive on first real user message
        # Skip if: already injected, or message is system feedback (tool results, action results, etc.)
        is_system_feedback = text.startswith("[") and any(
            text.startswith(prefix) for prefix in [
                "[update_plan result]", "[read_plan result]",
                "[compact_context result]", "[shell_exec", "[ACTION REJECTED]",
                "[PLAN CHECKPOINT", "[CONTEXT CONTINUATION]", "[PLANNING REQUIRED]",
                "[PLAN CHECK]", "[TOOL LOOP", "[SCHEMA VALIDATION",
                "[MALFORMED RESPONSE]", "[ACTION FAILED",
            ]
        )

        if not self._plan_injected.get(session_id) and not is_system_feedback:
            # Check if there are any assistant turns yet (no = first task)
            has_assistant_turns = any(m.role == MessageRole.assistant for m in history)
            if not has_assistant_turns:
                history.append(
                    PreviousMessage(role=MessageRole.user, content=PLAN_DIRECTIVE)
                )
                self._plan_injected[session_id] = True
                print(f"[plan] Injected planning directive for session {session_id}")

    def set_agent_mode(self, session_id: str, mode: str) -> None:
        """Record the current frontend mode for the next model request."""
        if mode not in ("coworker", "remote"):
            raise ValueError(f"invalid agent mode: {mode}")
        previous_mode = self._agent_mode.get(session_id)
        self._agent_mode[session_id] = mode
        if previous_mode and previous_mode != mode and session_id in self._history:
            if mode == "remote":
                self.clear_coworker_target(session_id)
            self._replace_system_prompt(session_id, mode, previous_mode)
            print(f"[agent_mode] session={session_id} {previous_mode} → {mode}; refreshed system prompt")

    def set_active_model(self, session_id: str, provider: str, model: str) -> None:
        """Record the selected inference provider/model for prompt metadata.

        When the model changes mid-conversation, also inject a user-role notice
        into history so the new model isn't primed by the previous model's
        self-identification turns (LLMs frequently parrot prior answers about
        their identity when the conversation history shows them).
        """
        new_pair = (provider or "", model or "")
        previous_pair = self._active_model.get(session_id)
        self._active_model[session_id] = new_pair

        if (
            previous_pair
            and previous_pair != new_pair
            and session_id in self._history
            and any(m.role == MessageRole.assistant for m in self._history[session_id])
        ):
            prev_label = previous_pair[1] or previous_pair[0] or "previous model"
            new_label = model or provider or "new model"
            notice = (
                f"[model switched mid-session: {prev_label} → {new_label}. "
                f"Earlier assistant turns were generated by the previous model. "
                f"Do not repeat or rely on any prior self-identification — "
                f"you are now {new_label}.]"
            )
            self._history[session_id].append(
                PreviousMessage(role=MessageRole.user, content=notice)
            )
            print(f"[active_model] session={session_id} {previous_pair} → {new_pair}; injected switch notice")

    def set_coworker_target(self, session_id: str, pid: int, window_id: int) -> None:
        """Record the active (pid, window_id) for coworker-mode per-turn perception."""
        try:
            self._coworker_target[session_id] = (int(pid), int(window_id))
        except (TypeError, ValueError):
            return

    def get_coworker_target(self, session_id: str) -> tuple[int, int] | None:
        """Return the active (pid, window_id) tuple, or None if unknown."""
        return self._coworker_target.get(session_id)

    def clear_coworker_target(self, session_id: str) -> None:
        self._coworker_target.pop(session_id, None)

    def add_screenshot_turn(self, session_id: str, base64_screenshot: str) -> None:
        """Append a screenshot as a user message, optionally via OmniParser."""
        if not base64_screenshot.startswith("data:"):
            if base64_screenshot.startswith("/9j"):
                mime = "image/jpeg"
            elif base64_screenshot.startswith("UklGR"):
                mime = "image/webp"
            else:
                mime = "image/png"
            base64_screenshot = f"data:{mime};base64,{base64_screenshot}"

        # Cache the raw bytes for the GET /sessions/{id}/latest_screenshot
        # endpoint. We decode once here so the HTTP path is a constant-time
        # lookup + write instead of base64-decoding on every request.
        try:
            import base64 as _b64
            header, b64body = base64_screenshot.split(",", 1)
            mime_part = header.split(";", 1)[0].split(":", 1)[1] if ":" in header else "image/png"
            self._latest_screenshot[session_id] = (mime_part, _b64.b64decode(b64body))
        except Exception:
            # Don't let cache failures break the agent loop.
            pass

        annotations = None

        if USE_OMNI_PARSER:
            raw_b64 = base64_screenshot.split(",", 1)[1] if "," in base64_screenshot else base64_screenshot
            try:
                from providers.modal.omni_parser.client import parse_screenshot_b64
                result = parse_screenshot_b64(raw_b64, include_annotated=True)

                if result.annotated_image_base64:
                    base64_screenshot = f"data:image/png;base64,{result.annotated_image_base64}"

                elements = [
                    ScreenElement(
                        id=e.id, type=e.type, content=e.content,
                        bbox_pixel=e.bbox_pixel, center_pixel=e.center_pixel,
                        interactable=e.interactable,
                    )
                    for e in result.elements
                ]
                annotations = ScreenAnnotation(
                    elements=elements,
                    image_width=result.image_width,
                    image_height=result.image_height,
                    latency_ms=result.latency_ms,
                )
                img_w = result.image_width or 1
                img_h = result.image_height or 1
                print(f"[omniparser] {len(elements)} elements detected ({result.latency_ms}ms)")
                for e in elements:
                    tag = e.type.upper()
                    ncx = round(e.center_pixel[0] / img_w, 4)
                    ncy = round(e.center_pixel[1] / img_h, 4)
                    coords = f"center_norm=({ncx},{ncy})"
                    lbl = e.content if e.content else ""
                    click = "  [clickable]" if e.interactable else ""
                    print(f"  [{e.id}] {tag}  label=\"{lbl}\"  {coords}{click}")
            except Exception as e:
                print(f"[omniparser] Failed, using raw screenshot: {e}")
        else:
            print("[screenshot] OmniParser skipped — sending direct screenshot to model")

        self._get(session_id).append(
            PreviousMessage(
                role=MessageRole.user,
                content=base64_screenshot,
                annotations=annotations,
            )
        )

    def add_assistant_turn(self, session_id: str, response_json: str) -> None:
        """Append the model's full JSON response to the chain."""
        self._get(session_id).append(
            PreviousMessage(role=MessageRole.assistant, content=response_json)
        )

    def add_tool_call_turn(self, session_id: str, tool_calls: list[dict], content: str = "") -> None:
        """Record an assistant message that contains tool calls (no desktop action)."""
        self._get(session_id).append(
            PreviousMessage(
                role=MessageRole.assistant,
                content=content or "",
                tool_calls=tool_calls,
            )
        )

    def add_tool_result_turn(self, session_id: str, tool_call_id: str, tool_name: str, result: str) -> None:
        """Record the result of a tool call."""
        self._get(session_id).append(
            PreviousMessage(
                role=MessageRole.tool,
                content=result,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
        )

    def _repair_orphan_tool_calls(self, session_id: str) -> None:
        repaired, inserted, converted = repair_orphan_tool_calls(self._get(session_id))
        if inserted or converted:
            self._history[session_id] = repaired
            print(
                f"[tool-history] Repaired {inserted} orphan tool call(s), "
                f"converted {converted} orphan tool result(s) for session {session_id}"
            )

    # Placeholder that replaces old screenshots
    SCREENSHOT_PLACEHOLDER = "[A screenshot was taken here and reviewed by you]"

    def _count_assistant_turns(self, session_id: str) -> int:
        """Count assistant turns in current history."""
        return sum(1 for m in self._get(session_id) if m.role == MessageRole.assistant)

    def _should_inject_plan(self, session_id: str) -> bool:
        """Check if it's time to auto-inject plan.md content."""
        turns = self._count_assistant_turns(session_id) + self._step_offset.get(session_id, 0)
        return turns > 0 and turns % PLAN_INJECT_INTERVAL == 0

    def _should_inject_plan_reminder(self, session_id: str) -> bool:
        """Check if it's time for a lightweight plan reminder."""
        turns = self._count_assistant_turns(session_id) + self._step_offset.get(session_id, 0)
        return turns > 0 and turns % self.PLAN_REMINDER_INTERVAL == 0

    def build_request(self, session_id: str) -> AgentRequest:
        """
        Build an AgentRequest from the current history.

        - Only latest screenshot kept as image data
        - Plan auto-injected every PLAN_INJECT_INTERVAL turns
        - Middle-trimmed if chain exceeds MAX_CHAIN_LENGTH
        """
        self._repair_orphan_tool_calls(session_id)
        history = self._get(session_id)

        step_index = (
            self._step_offset.get(session_id, 0)
            + sum(1 for m in history if m.role == MessageRole.assistant)
        )

        # Get original user message
        user_message = ""
        for m in history:
            if m.role == MessageRole.user and not m.content.startswith(SCREENSHOT_PREFIX):
                user_message = m.content
                break

        # Auto-inject plan every N turns — but skip if model recently read plan via tools
        _recently_read_plan = any(
            m.role == MessageRole.tool
            and m.tool_name in ("read_plan", "read_session_file")
            and "plan" in (m.content or "").lower()
            for m in history[-6:]
        )
        _recent_tool_rejection = any(
            m.role == MessageRole.tool
            and (m.content or "").startswith("[TOOL REJECTED]")
            for m in history[-6:]
        )

        if self._should_inject_plan(session_id) and not _recently_read_plan and not _recent_tool_rejection:
            from workspace import read_session_plan
            plan = read_session_plan(session_id)
            if plan:
                plan_msg = (
                    f"[PLAN CHECKPOINT — Step {step_index}]\n"
                    f"Here is your current plan for reference:\n\n{plan}\n\n"
                    f"Continue from the next incomplete step."
                )
                history.append(
                    PreviousMessage(role=MessageRole.user, content=plan_msg)
                )
                print(f"[plan-inject] Auto-injected plan.md at step {step_index}")

        # Lightweight plan reminder (offset from full plan injection)
        elif self._should_inject_plan_reminder(session_id) and not _recently_read_plan and not _recent_tool_rejection:
            history.append(
                PreviousMessage(role=MessageRole.user, content=PLAN_REMINDER)
            )
            print(f"[plan-remind] Injected plan reminder at step {step_index}")

        # Build trimmed copy: replace all but latest screenshot
        trimmed: list[PreviousMessage] = []
        last_screenshot_idx = -1
        for i, m in enumerate(history):
            if m.role == MessageRole.user and m.content.startswith(SCREENSHOT_PREFIX):
                last_screenshot_idx = i

        for i, m in enumerate(history):
            if (m.role == MessageRole.user
                    and m.content.startswith(SCREENSHOT_PREFIX)
                    and i != last_screenshot_idx):
                trimmed.append(
                    PreviousMessage(role=m.role, content=self.SCREENSHOT_PLACEHOLDER, timestamp=m.timestamp)
                )
            else:
                trimmed.append(m)
                if i == last_screenshot_idx and m.annotations and m.annotations.elements:
                    element_text = self._format_annotations(m.annotations)
                    trimmed.append(
                        PreviousMessage(role=MessageRole.user, content=element_text, timestamp=m.timestamp)
                    )

        # Middle-trim safety net
        if len(trimmed) > MAX_CHAIN_LENGTH:
            keep_head = 2
            keep_tail = MAX_CHAIN_LENGTH - keep_head
            trimmed = trimmed[:keep_head] + [
                PreviousMessage(
                    role=MessageRole.user,
                    content=(
                        "[Earlier turns were trimmed. Use read_plan to see your full task plan.]"
                    ),
                )
            ] + trimmed[-keep_tail:]

        trimmed, inserted, converted = repair_orphan_tool_calls(trimmed)
        if inserted or converted:
            print(
                f"[tool-history] Sanitized provider request for {session_id}: "
                f"inserted={inserted}, converted={converted}"
            )

        active_model = self._active_model.get(session_id)
        if active_model:
            provider, model = active_model
            model_block = _active_model_block(provider, model)
            for idx, message in enumerate(trimmed):
                if message.role == MessageRole.system:
                    trimmed[idx] = PreviousMessage(
                        role=MessageRole.system,
                        content=message.content.rstrip() + model_block,
                        timestamp=message.timestamp,
                    )
                    break
            else:
                trimmed.insert(0, PreviousMessage(role=MessageRole.system, content=model_block.strip()))

        return AgentRequest(
            session_id=session_id,
            user_message=user_message,
            base64_screenshot="",
            previous_messages=trimmed,
            step_index=step_index,
            agent_mode=self._agent_mode.get(session_id, "coworker"),
        )

    def clear_session(self, session_id: str) -> None:
        """Remove all history for a session."""
        self._history.pop(session_id, None)
        self._step_offset.pop(session_id, None)
        self._plan_injected.pop(session_id, None)
        self._agent_mode.pop(session_id, None)
        self._active_model.pop(session_id, None)
        self._coworker_target.pop(session_id, None)
        self._latest_screenshot.pop(session_id, None)
        self.remote_validator.clear(session_id)
        self.coworker_validator.clear(session_id)

    def get_latest_screenshot(self, session_id: str) -> tuple[str, bytes] | None:
        """Return (mime_type, raw_bytes) of the most recent screenshot, or None."""
        return self._latest_screenshot.get(session_id)

    def preload_from_conversation(self, session_id: str, old_messages: list[dict]) -> None:
        """
        Seed a session's in-memory context with user/assistant turns from its
        conversation.json, then append a continuation directive so the model
        knows to re-plan before acting.
        """
        history = self._get(session_id)  # bootstraps system prompt

        for msg in old_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "user" and content != "<screenshot>":
                history.append(PreviousMessage(role=MessageRole.user, content=content))
            elif role == "assistant":
                history.append(PreviousMessage(role=MessageRole.assistant, content=content))
            # tool / action / system entries are intentionally skipped

        history.append(PreviousMessage(role=MessageRole.user, content=CONTEXT_CONTINUATION_MSG))
        # Mark plan as injected — the continuation message above handles replanning
        self._plan_injected[session_id] = True
        print(f"[context] preloaded {len(old_messages)} prior messages into {session_id}")

    @staticmethod
    def _format_annotations(ann: ScreenAnnotation) -> str:
        """Format screen annotations as compact text for the model."""
        w = ann.image_width or 1
        h = ann.image_height or 1
        lines = [f"[SCREEN ELEMENTS] {ann.image_width}x{ann.image_height} — {len(ann.elements)} elements (coords normalized 0-1)"]
        for e in ann.elements:
            x1, y1, x2, y2 = e.bbox_pixel
            cx, cy = e.center_pixel
            nx1, ny1 = round(x1 / w, 4), round(y1 / h, 4)
            nx2, ny2 = round(x2 / w, 4), round(y2 / h, 4)
            ncx, ncy = round(cx / w, 4), round(cy / h, 4)
            lbl = e.content if e.content else ""
            click = "  [clickable]" if e.interactable else ""
            lines.append(f'  [{e.id}] {e.type.upper()}  label="{lbl}"  bbox=({nx1},{ny1},{nx2},{ny2})  center=({ncx},{ncy}){click}')
        return "\n".join(lines)

    SCREEN_ELEMENTS_PREFIX = "[SCREEN ELEMENTS]"

    def get_compact_messages(self, session_id: str) -> list[PreviousMessage]:
        """
        Return pre-filtered messages for the compaction model.
        Only the OLDER portion of history (excluding the recent tail that will
        be kept verbatim) is returned.  Strips system prompt, screenshots,
        element data, and verbose reasoning.
        """
        full_history = self._get(session_id)
        # Only compact the older portion; the recent tail is kept as-is
        cutoff = max(0, len(full_history) - KEEP_RECENT_MESSAGES)
        history = full_history[:cutoff]
        messages: list[PreviousMessage] = []
        last_was_screenshot_placeholder = False

        for m in history:
            if m.role == MessageRole.system:
                continue
            elif m.role == MessageRole.user and m.content.startswith(SCREENSHOT_PREFIX):
                if not last_was_screenshot_placeholder:
                    messages.append(
                        PreviousMessage(role=m.role, content=self.SCREENSHOT_PLACEHOLDER, timestamp=m.timestamp)
                    )
                    last_was_screenshot_placeholder = True
                continue
            elif m.role == MessageRole.user and m.content.startswith(self.SCREEN_ELEMENTS_PREFIX):
                continue
            elif m.role == MessageRole.assistant:
                last_was_screenshot_placeholder = False
                if m.tool_calls:
                    # Summarize tool calls with arguments for richer compaction
                    tc_parts = []
                    for tc in m.tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "?")
                        args = fn.get("arguments", "")
                        if isinstance(args, str) and len(args) > 200:
                            args = args[:200] + "..."
                        tc_parts.append(f"{name}({args})" if args else name)
                    messages.append(
                        PreviousMessage(role=m.role, content=f"TOOL_CALLS: {'; '.join(tc_parts)}", timestamp=m.timestamp)
                    )
                else:
                    trimmed = self._trim_assistant_for_compact(m.content)
                    messages.append(
                        PreviousMessage(role=m.role, content=trimmed, timestamp=m.timestamp)
                    )
            elif m.role == MessageRole.tool:
                # Compact tool results — keep enough detail for meaningful summaries
                result_preview = m.content[:1000] + "..." if len(m.content) > 1000 else m.content
                messages.append(
                    PreviousMessage(role=MessageRole.user, content=f"TOOL_RESULT: {result_preview}", timestamp=m.timestamp)
                )
            else:
                last_was_screenshot_placeholder = False
                messages.append(m)

        return messages

    @staticmethod
    def _trim_assistant_for_compact(content: str) -> str:
        """Pre-compress an assistant message for compaction."""
        try:
            data = json.loads(content)
            parts = []
            action = data.get("action", {})
            action_type = action.get("type", "unknown")
            parts.append(f"ACTION: {action_type}")

            if action_type == "shell_exec" and "command" in action:
                parts.append(f"CMD: {action['command']}")
            elif action_type == "type_text" and "text" in action:
                parts.append(f"TEXT: {action['text']}")
            elif action_type == "key_press":
                key = action.get("key", "")
                mods = action.get("modifiers", [])
                combo = "+".join(mods + [key]) if mods else key
                parts.append(f"KEY: {combo}")
            elif action_type in (
                "mouse_move",
                "navigate_and_click",
                "navigate_and_right_click",
                "navigate_and_triple_click",
            ) and "coordinates" in action:
                c = action["coordinates"]
                parts.append(f"TO: ({c.get('x', 0):.3f}, {c.get('y', 0):.3f})")

            if data.get("done"):
                parts.append("DONE=true")
            if data.get("final_message"):
                parts.append(f"MSG: {data['final_message']}")

            reasoning = data.get("reasoning", "")
            if reasoning and len(reasoning) > 300:
                reasoning = reasoning[:300] + "..."
            if reasoning:
                parts.append(f"WHY: {reasoning}")

            return " | ".join(parts)
        except (json.JSONDecodeError, TypeError, AttributeError):
            if len(content) > 500:
                return content[:500] + "..."
            return content

    def reset_with_summary(self, session_id: str, summary: str) -> None:
        """
        Replace the older portion of the context chain with a compacted summary
        while preserving the most recent KEEP_RECENT_MESSAGES verbatim.
        Result: system prompt + summary directive + recent tail.
        """
        from prompts.compact_prompt import CONTINUATION_DIRECTIVE

        system_prompt = self._build_system_prompt(
            session_id,
            self._agent_mode.get(session_id, "coworker"),
        )

        old_history = self._history.get(session_id, [])

        # Preserve step count across compactions (only count compacted portion)
        cutoff = max(0, len(old_history) - KEEP_RECENT_MESSAGES)
        compacted_portion = old_history[:cutoff]
        recent_tail = old_history[cutoff:]

        # Only count assistant steps in the compacted portion
        compacted_steps = sum(1 for m in compacted_portion if m.role == MessageRole.assistant)
        self._step_offset[session_id] = self._step_offset.get(session_id, 0) + compacted_steps

        total_steps = self._step_offset[session_id]
        user_content = (
            CONTINUATION_DIRECTIVE
            .replace("{session_id}", session_id)
            .replace("{summary}", summary)
            .replace("{step_count}", str(total_steps))
        )

        # Strip any system messages from the recent tail (we re-add a fresh one)
        recent_tail = [m for m in recent_tail if m.role != MessageRole.system]

        self._history[session_id] = [
            PreviousMessage(role=MessageRole.system, content=system_prompt.strip()),
            PreviousMessage(role=MessageRole.user, content=user_content),
        ] + recent_tail

    def needs_compaction(self, session_id: str) -> bool:
        """
        Safety-net compaction check. Only fires at a high threshold —
        the model should call compact_context before this triggers.
        """
        return len(self._get(session_id)) > AUTO_COMPACT_THRESHOLD

    def estimate_token_count(self, session_id: str) -> int:
        """Rough token estimate for the current context chain."""
        total = 0
        for m in self._get(session_id):
            if m.content.startswith(SCREENSHOT_PREFIX):
                total += 1000
            else:
                total += len(m.content) // 4
        return total

    def chain_length(self, session_id: str) -> int:
        """Return the number of messages in a session's context chain."""
        return len(self._get(session_id))

    def get_raw_history(self, session_id: str) -> list[PreviousMessage]:
        """Return the full, untruncated conversation history for logging."""
        return list(self._get(session_id))
