"""
context_manager/coworker_validator.py

Runtime validation for coworker-mode function-tool calls and completion
verification. Remote desktop action validation lives in remote_validator.py.
"""

import hashlib
import json


_DROP = object()


class CoworkerActionValidator:
    """Stateful per-session validator for coworker tool calls/results."""

    _COWORKER_INTERACTIVE_TOOLS = {
        "cua_click",
        "cua_right_click",
        "cua_double_click",
        "cua_press_key",
        "cua_hotkey",
        "cua_type_text",
        "cua_type_text_chars",
        "cua_set_value",
        "cua_scroll",
        "cua_drag",
    }

    _COWORKER_PERCEPTION_TOOLS = {
        "cua_get_window_state",
        "cua_screenshot",
        "cua_zoom",
        "cua_list_windows",
        "cua_list_apps",
        "list_running_apps",
        "raise_app",
        "cua_launch_app",
    }

    _COWORKER_STATE_REFRESH_TOOLS = _COWORKER_PERCEPTION_TOOLS | {
        "cua_page",
        "cua_get_config",
        "cua_set_config",
    }

    _FAILURE_BUDGET_TOOLS = _COWORKER_INTERACTIVE_TOOLS | {
        "cua_page",
        "cua_get_window_state",
        "cua_screenshot",
        "cua_zoom",
    }

    _BLOCKED_DONE_WORDS = (
        "can't",
        "cannot",
        "couldn't",
        "unable",
        "failed",
        "blocked",
        "limitation",
        "not possible",
        "not safely",
        "need you",
        "please",
    )

    _CLICK_TOOLS = {"cua_click", "cua_right_click", "cua_double_click"}
    MAX_CONSECUTIVE_TOOL_FAILURES = 2
    DEFAULT_UNVERIFIED_REPEAT_LIMIT = 2
    _UNVERIFIED_REPEAT_LIMITS = {
        "cua_click": 1,
        "cua_right_click": 1,
        "cua_double_click": 1,
        "cua_scroll": 3,
    }

    def __init__(self):
        self._coworker_pending_verification: dict[str, str] = {}
        # (tool_key, fail_count, summary) tracks consecutive identical failures.
        self._tool_failures: dict[str, tuple[str, int, str]] = {}
        # (tool_key, count) tracks repeated successful interactions that have
        # not been followed by a state refresh/perception tool yet.
        self._unverified_interactions: dict[str, tuple[str, int]] = {}
        # (rejection_key, count) tracks repeated pre-dispatch validation loops.
        self._tool_rejections: dict[str, tuple[str, int]] = {}

    def clear(self, session_id: str) -> None:
        """Reset all state for a session."""
        self._coworker_pending_verification.pop(session_id, None)
        self._tool_failures.pop(session_id, None)
        self._unverified_interactions.pop(session_id, None)
        self._tool_rejections.pop(session_id, None)

    def _remember_rejection(self, session_id: str, key: str, message: str) -> tuple[bool, str]:
        current_key, count = self._tool_rejections.get(session_id, ("", 0))
        next_count = count + 1 if current_key == key else 1
        self._tool_rejections[session_id] = (key, next_count)
        if next_count >= 3:
            return False, (
                f"{message} This exact invalid tool call has been rejected "
                f"{next_count} times in a row. Stop retrying it: take a fresh "
                "`cua_get_window_state`/`cua_screenshot`, choose one addressing "
                "mode, or report the blocker."
            )
        return False, message

    def _click_key(self, name: str, args: dict | None) -> str:
        """Return the executable click target, ignoring ignored provider defaults."""
        args = args or {}
        pid = args.get("pid")
        window_id = args.get("window_id")
        element_index = args.get("element_index")
        if element_index is not None and window_id is not None:
            return f"{name}:element:{pid}:{window_id}:{element_index}"

        x = args.get("x")
        y = args.get("y")
        if x is not None and y is not None:
            try:
                x = round(float(x), 1)
                y = round(float(y), 1)
            except (TypeError, ValueError):
                pass
            return f"{name}:pixel:{pid}:{window_id}:{x}:{y}"

        return f"{name}:invalid:{pid}:{window_id}"

    def _tool_key(self, name: str, args: dict | None) -> str:
        """Return a stable, compact key for repeated-call detection."""
        if name in self._CLICK_TOOLS:
            return self._click_key(name, args)

        normalized = self._normalize_value(args or {})
        if normalized is _DROP:
            normalized = {}
        try:
            raw = json.dumps(
                normalized,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            raw = repr(normalized)
        if len(raw) > 500:
            digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]
            raw = raw[:240] + f"...#{digest}"
        return f"{name}:{raw}"

    def _normalize_value(self, value):
        """Normalize provider-filled optionals so equivalent calls share keys."""
        if value is None or value == "" or value == []:
            return _DROP
        if isinstance(value, bool) or isinstance(value, int):
            return value
        if isinstance(value, float):
            return round(value, 2)
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            items = []
            for item in value:
                normalized = self._normalize_value(item)
                if normalized is not _DROP:
                    items.append(normalized)
            return items if items else _DROP
        if isinstance(value, dict):
            items = {}
            for key in sorted(value):
                normalized = self._normalize_value(value[key])
                if normalized is not _DROP:
                    items[str(key)] = normalized
            return items if items else _DROP
        return repr(value)

    def _failure_summary(self, result: str) -> str:
        """Bucket common failures without overfitting to one app or website."""
        text = " ".join(str(result or "").split())
        lowered = text.lower()
        if "allow javascript from apple events" in lowered:
            return "JavaScript from Apple Events is disabled"
        if "timed out" in lowered or "timeout" in lowered:
            return "tool timed out"
        if "permission" in lowered or "not authorized" in lowered or "not authorised" in lowered:
            return "permission or authorization failure"
        if "daemon unavailable" in lowered or "connection refused" in lowered or "socket" in lowered:
            return "driver daemon unavailable"
        if "ax action" in lowered and "failed" in lowered:
            return "AX action failed"
        if "missing required" in lowered:
            return "missing required argument"
        return text[:180] or "tool returned an error"

    def _unverified_repeat_limit(self, name: str) -> int:
        return self._UNVERIFIED_REPEAT_LIMITS.get(
            name,
            self.DEFAULT_UNVERIFIED_REPEAT_LIMIT,
        )

    def _validate_click_target(self, name: str, args: dict | None) -> tuple[bool, str]:
        """Reject ambiguous click targeting before it can execute the wrong path."""
        args = args or {}
        has_element_target = (
            args.get("element_index") is not None
            and args.get("window_id") is not None
        )
        has_xy = args.get("x") is not None and args.get("y") is not None
        if not has_element_target or not has_xy:
            return True, ""

        try:
            is_zero_default = float(args.get("x")) == 0 and float(args.get("y")) == 0
        except (TypeError, ValueError):
            is_zero_default = False
        if is_zero_default:
            return True, ""

        try:
            is_element_zero = int(args.get("element_index")) == 0
        except (TypeError, ValueError):
            is_element_zero = False
        if is_element_zero:
            return True, ""

        return False, (
            f"`{name}` has an invalid mixed target: provide either "
            "`element_index` + `window_id` OR pixel `x` + `y`, not both. "
            "For a pixel click, omit `element_index` and `action`; for an AX "
            "click, omit `x`, `y`, `modifier`, `count`, and `from_zoom`."
        )

    def validate_tool_call(
        self,
        session_id: str,
        name: str,
        args: dict | None,
        agent_mode: str = "remote",
    ) -> tuple[bool, str]:
        """Validate backend function-tool calls before execution."""
        if agent_mode != "coworker" and (
            name == "list_running_apps" or name.startswith("cua_")
        ):
            return False, (
                f"`{name}` is only available in coworker mode. The current "
                "mode is remote, so do not call emu-cua-driver tools. "
                "Desktop actions in remote mode are NOT function tools: do "
                "not emit another tool call for them. Instead reply with "
                "plain JSON text in your message content, e.g. "
                '{"action": {"type": "screenshot"}, "done": false}. '
                "Valid remote action types: screenshot, navigate_and_click, "
                "navigate_and_right_click, navigate_and_triple_click, "
                "left_click, right_click, double_click, triple_click, "
                "mouse_move, drag, scroll, type_text, key_press, wait, done."
            )

        if agent_mode == "coworker" and name in self._CLICK_TOOLS:
            target_ok, target_err = self._validate_click_target(name, args)
            if not target_ok:
                key = self._click_key(name, args)
                return self._remember_rejection(session_id, f"reject:{key}", target_err)

        if agent_mode == "coworker" and name in self._FAILURE_BUDGET_TOOLS:
            key = self._tool_key(name, args)
            current_key, count, summary = self._tool_failures.get(session_id, ("", 0, ""))
            if current_key == key and count >= self.MAX_CONSECUTIVE_TOOL_FAILURES:
                return self._remember_rejection(
                    session_id,
                    f"failed:{key}",
                    (
                        f"`{name}` has already failed {count} times in a row "
                        f"with the same arguments ({summary}). Do not retry the "
                        "same call. Change one thing first: refresh state, choose "
                        "a different target, use a different tool, or report the blocker."
                    ),
                )

        if agent_mode == "coworker" and name in self._COWORKER_INTERACTIVE_TOOLS:
            key = self._click_key(name, args)
            if name not in self._CLICK_TOOLS:
                key = self._tool_key(name, args)
            current_key, count = self._unverified_interactions.get(session_id, ("", 0))
            limit = self._unverified_repeat_limit(name)
            if current_key == key and count >= limit:
                return self._remember_rejection(
                    session_id,
                    f"unverified:{key}",
                    (
                        f"`{name}` already ran {count} time(s) with the same "
                        "arguments without a successful state refresh afterward. "
                        "Do not keep posting the same input. Verify with "
                        "`cua_get_window_state`, `cua_screenshot`, `cua_page`, "
                        "or `cua_list_windows`, then either choose a new strategy "
                        "or report the blocker."
                    ),
                )

        return True, ""

    def record_tool_result(
        self,
        session_id: str,
        name: str,
        args: dict | None,
        result: str,
        agent_mode: str = "remote",
    ) -> None:
        """Update coworker tool history after a function tool returns."""
        if agent_mode != "coworker":
            return

        ok = result.startswith(f"[{name}]")

        if name in self._FAILURE_BUDGET_TOOLS:
            key = self._tool_key(name, args)
            if ok:
                self._tool_failures[session_id] = ("", 0, "")
            else:
                current_key, count, _ = self._tool_failures.get(session_id, ("", 0, ""))
                self._tool_failures[session_id] = (
                    key,
                    count + 1 if current_key == key else 1,
                    self._failure_summary(result),
                )

        if name in self._COWORKER_STATE_REFRESH_TOOLS:
            if ok:
                self._coworker_pending_verification.pop(session_id, None)
                self._unverified_interactions.pop(session_id, None)
                self._tool_rejections.pop(session_id, None)
            return

        if name not in self._COWORKER_INTERACTIVE_TOOLS:
            return

        if ok:
            key = self._tool_key(name, args)
            current_key, count = self._unverified_interactions.get(session_id, ("", 0))
            self._unverified_interactions[session_id] = (
                key,
                count + 1 if current_key == key else 1,
            )
            self._coworker_pending_verification[session_id] = name
            self._tool_rejections.pop(session_id, None)

    def validate_done_response(self, session_id: str, final_message: str | None) -> tuple[bool, str]:
        """
        Prevent success-shaped final answers immediately after an unverified
        coworker interaction. Honest blocked/limitation messages are allowed.
        """
        pending = self._coworker_pending_verification.get(session_id)
        if not pending:
            return True, ""

        text = (final_message or "").lower()
        if any(word in text for word in self._BLOCKED_DONE_WORDS):
            return True, ""

        return False, (
            f"The last coworker interaction (`{pending}`) has not been verified "
            "by a successful cua_get_window_state/cua_screenshot/list_windows "
            "call. Do not claim success from a posted click/key alone. Verify "
            "the UI state first; if it did not change, switch strategy or report "
            "the limitation."
        )
