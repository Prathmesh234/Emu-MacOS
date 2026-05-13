# Cursor-Context ("Emu Pointer") — Functional Spec

**Status:** Draft v1 — research + design, no implementation yet
**Owner:** Emu agent platform
**Date:** 2026-05-13
**Inspired by:** Google DeepMind "Magic Pointer" (announced 2026-05-12 alongside Googlebook)

---

## 1. Background

On 2026-05-12, Google DeepMind announced **Magic Pointer**: a reimagining of the desktop cursor as an AI-grounded interaction surface. The core insight is that the cursor position is *already* the user's implicit "this" — by treating it as context, the AI no longer needs the user to translate what they're looking at into a written prompt. Users hover, issue a 2–3 word command (or wiggle/gesture), and the AI acts on whatever is under the pointer.

Magic Pointer's full implementation depends on:
- An on-device NPU (Intel Wildcat Lake-class) for sub-100ms cursor-following inference
- OS-level integration (Googlebook = Android + ChromeOS hybrid)
- Hybrid on-device + cloud architecture for ambient always-on responsiveness
- A separate web-only preview in Chrome (cloud-only)

Emu's environment is materially different:
- macOS-only, sandboxed Electron app
- No on-device AI inference (no CoreML/Metal/NPU pipeline)
- All LLM calls are cloud round-trips (200–500ms floor)
- AX-mediated control through the bundled `emu-cua-driver`

This spec defines **Emu Pointer** — a feasible adaptation that captures Magic Pointer's *design philosophy* (cursor-as-implicit-context, short commands, no prompt tax) within Emu's existing architecture, without requiring NPU access or OS-level hooks.

---

## 2. Goals & Non-Goals

### Goals
1. Let a user point at *anything* on screen and invoke an AI action with a short command (no need to describe the target).
2. Surface zero-latency local actions for common entity types (dates, URLs, phone numbers, addresses) via macOS-native detectors.
3. Render results in a lightweight, cursor-anchored overlay that does not steal focus from the user's active window.
4. Reuse Emu's existing tool-dispatch / coworker-driver / provider stack — no new LLM provider integration.
5. Keep latency budget under **500ms** for the local fast-path and **1.5s** for the cloud vision-grounded path.

### Non-Goals
- Continuous ambient cursor-following inference. Without an NPU, every hover would be a cloud call — cost-prohibitive and laggy.
- Generative image compositing (Magic Pointer's "couch + living room" demo). Out of scope for v1.
- Voice or gesture input. Trigger is hotkey + typed short command only.
- Cross-platform support. macOS-only, matching Emu's current scope.
- Replacing the existing autonomous coworker agent. Emu Pointer is *additive* and *user-initiated*, not autonomous.

---

## 3. User Experience

### 3.1 Primary flow ("Point + Press + Type")

1. User hovers over content in **any application** (browser, PDF, mail, IDE).
2. User presses a configurable global hotkey (default proposal: ⌃⌘Space — chosen to avoid conflict with Spotlight / Raycast).
3. A small **cursor-anchored panel** appears 12px below-right of the cursor, with:
   - An identified entity chip if a local detector matched ("📅 May 22, 2026")
   - 3–4 suggested action chips ("Summarize", "Translate", "Schedule", "Ask…")
   - A compact text input focused for free-form short commands
4. User either clicks a suggested chip or types a short command and hits Enter.
5. Panel inlines the result, or hands off to the main Emu coworker loop for multi-step actions.
6. ESC or click-away dismisses; ⌥-modifier on the hotkey pins the panel for follow-ups.

### 3.2 Entity fast-path

When the AX text under the cursor contains a recognized entity (via `NSDataDetector`), the panel surfaces an instant local action with **zero network cost**:

| Entity        | Suggested instant action                          |
| ------------- | ------------------------------------------------- |
| Date / time   | "Schedule in Calendar"                            |
| Phone number  | "Call via FaceTime" / "Copy"                      |
| URL           | "Open" / "Summarize page"                         |
| Email address | "Compose new mail"                                |
| Address       | "Open in Maps"                                    |
| Flight number | "Track flight" (external API)                     |
| Tracking #    | "Track shipment" (external API)                   |

These are dispatched to macOS via existing `open` URL schemes (`tel:`, `mailto:`, `maps://`, `x-apple-calevent://`); no LLM needed.

### 3.3 Vision-grounded path

When no local detector matches, or the user requests an analytical action (Summarize / Compare / Explain), Emu Pointer:
1. Grabs a focused screenshot crop (~512×384 centered on cursor) via the existing screenshot pipeline.
2. Pulls the AX element-at-point (new driver call — see §5.2) for structured metadata.
3. Sends `{crop, ax_element, user_command}` to the agent loop as a one-turn tool call (not a multi-step plan).
4. Streams the response back into the panel.

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  macOS                                                      │
│                                                             │
│  ⌃⌘Space ──────┐                                            │
│                ▼                                            │
│   ┌──────────────────────┐    ┌────────────────────────┐    │
│   │ main.js              │    │ emu-cua-driver         │    │
│   │  globalShortcut      │───▶│  cua_get_element_at_   │    │
│   │  (NEW)               │    │  point (NEW)           │    │
│   └──────┬───────────────┘    └────────────────────────┘    │
│          │ ipc                                              │
│          ▼                                                  │
│   ┌──────────────────────┐                                  │
│   │ pointer-panel        │                                  │
│   │  BrowserWindow (NEW) │                                  │
│   │  - cursor-anchored   │                                  │
│   │  - click-through edge│                                  │
│   │  - focused composer  │                                  │
│   └──────┬───────────────┘                                  │
│          │ WebSocket                                        │
│          ▼                                                  │
│   ┌──────────────────────┐    ┌────────────────────────┐    │
│   │ backend/main.py      │───▶│ NSDataDetector         │    │
│   │  POST /pointer       │    │  (local, no LLM)       │    │
│   │  (NEW endpoint)      │    └────────────────────────┘    │
│   └──────┬───────────────┘                                  │
│          │ fallback                                         │
│          ▼                                                  │
│   ┌──────────────────────┐                                  │
│   │ agent loop (existing)│                                  │
│   │  + new "pointer_act" │                                  │
│   │   tool (NEW)         │                                  │
│   └──────────────────────┘                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Components to Build

### 5.1 Global hotkey registration — `main.js`

**Current state:** No `globalShortcut` usage exists anywhere in `main.js`. The feature relies on click-driven IPC.

**To add:**
- Register `globalShortcut.register('Control+Command+Space', onPointerInvoke)` in `app.whenReady()`.
- Make the accelerator configurable via a new key in user settings (`~/.emu/settings.json` → `pointer.hotkey`).
- Unregister on `will-quit`.
- On invoke: capture cursor position synchronously via the same mechanism `actions/getMousePosition.js:13-51` uses (`cliclick p`), then post an `ipc` event `pointer:invoke` carrying `{x, y, timestamp}`.

**Effort:** ~50 LOC, single file.

### 5.2 New driver tool: `cua_get_element_at_point`

**Current state:** `cua_get_cursor_position` (`backend/tools/coworker_tools.py:682-685`) returns coordinates only. `cua_get_window_state` (`coworker_tools.py:217-263`) walks the entire AX tree — too heavyweight for a hover query. **No element-at-point query exists.**

**To add (Swift side, `emu-cua-driver`):**
- New endpoint that calls `AXUIElementCopyElementAtPosition(systemWideElement, x, y, &element)`.
- Returns `{role, sub_role, title, value, frame, actions[], pid, window_id, element_index?}`.
  - If the element matches one already in a cached tree walk, return its `element_index` so existing click/type tools can target it. Otherwise return `null` and the caller decides whether to refresh the tree.
- Latency budget: <50ms (single AX SPI call, no full tree walk).

**To add (Python side, `backend/tools/coworker_tools.py`):**
- New tool spec following the `_fn()` pattern at `coworker_tools.py:97-765`. Suggested name `cua_get_element_at_point`, params `{x: int, y: int}`.
- Wire through `dispatcher.py:388-427` automatically once added to `COWORKER_DRIVER_TOOLS_OPENAI`.

**Effort:** ~80 LOC Swift + ~40 LOC Python.

### 5.3 New backend endpoint: `POST /pointer`

**Current state:** Backend is a long-running agent loop over WebSocket. There is no fast one-shot path optimized for cursor-context queries.

**To add (`backend/main.py`):**
- New REST endpoint `POST /pointer` with body `{cursor: {x, y}, screenshot_b64, user_command?: str, intent_hint?: enum}`.
- Internally:
  1. Call `cua_get_element_at_point` to get AX context.
  2. Extract `value`/`title`/`description` text from the element.
  3. Run `NSDataDetector` over that text (via a small helper invoked through the driver — see §5.4) to extract entities.
  4. If entity match AND no `user_command`: return `{mode: "local", entity, suggested_actions[]}` (zero LLM cost, <100ms).
  5. Else if `user_command` is short and analytical: route to the agent loop with a new `pointer_act` tool (§5.5).
  6. Return streaming response.

**Effort:** ~150 LOC, single new endpoint.

### 5.4 Local entity detection helper

**Current state:** No `NSDataDetector` usage anywhere in the project.

**To add (Swift in `emu-cua-driver`):**
- Driver endpoint `detect_entities(text: String) -> [{type, range, value, normalized}]` wrapping `NSDataDetector(types: NSTextCheckingResult.CheckingType.allTypes.rawValue)`.
- Types: `.date`, `.phoneNumber`, `.link`, `.address`, plus heuristic regex for flight/tracking numbers.
- Pure Swift, zero network. <5ms per call.

**Effort:** ~60 LOC Swift.

### 5.5 New agent tool: `pointer_act`

**Current state:** Existing tools (`cua_click`, `cua_type`, etc.) are full coworker actions — they assume the model is *acting on* the screen, not *answering about* it.

**To add (`backend/tools/coworker_tools.py` and `backend/providers/agent_tools.py:575-579`):**
- New tool `pointer_act` with schema `{action: enum["summarize", "explain", "translate", "compare", "answer"], context_image_b64, ax_element_json, user_command?}`.
- Distinct from the action-driver tools — this one is purely *observational + generative*. The model receives the crop + AX context as input and is instructed to return a short answer (one paragraph max) rather than plan a click sequence.
- New system-prompt fragment in `backend/prompts/` instructing the model that when invoked through `pointer_act`, it should be terse, not initiate multi-turn plans, and never call `cua_click` or other action tools.

**Effort:** ~80 LOC + a ~30-line prompt fragment.

### 5.6 Pointer panel UI — new `BrowserWindow`

**Current state:** Emu has exactly one overlay — `frontend/border.html` (47 lines, static animated border). No floating-panel infrastructure exists.

**To add (`main.js` + new `frontend/pages/PointerPanel.js` + `frontend/pointer-panel.html`):**
- New `BrowserWindow` created on demand with:
  - `frame: false, transparent: true, alwaysOnTop: true, skipTaskbar: true, focusable: true`
  - `level: 'screen-saver'` so it floats above all apps
  - `hasShadow: true` for visual polish
  - Position anchored to cursor on invoke; clamp to active display bounds
  - Auto-dismiss on blur unless ⌥-pinned
- Inside: small React panel (matches frontend's existing stack — see `frontend/pages/Chat.js`) with:
  - Entity chip row (top)
  - Action chip row (middle)
  - Compact composer with focus-on-mount (bottom)
  - Result region that grows on response
- IPC: receives `pointer:invoke` from main.js with cursor coords; calls backend `POST /pointer`.

**Effort:** ~300 LOC including styles. Largest single piece of work.

### 5.7 Settings & permissions

**Current state:** Permissions doc at `MACOS_PERMISSIONS.md`. Emu already requests Screen Recording + Accessibility.

**To add:**
- No new permissions. Both required entitlements (Screen Recording for the crop, Accessibility for element-at-point) are already requested.
- New settings keys in `~/.emu/settings.json`:
  ```jsonc
  {
    "pointer": {
      "enabled": true,
      "hotkey": "Control+Command+Space",
      "panel_offset": [12, 12],
      "auto_local_entities": true,
      "vision_path_enabled": true,
      "max_panel_width_px": 420
    }
  }
  ```
- Settings UI in the existing preferences page to toggle and rebind.

---

## 6. Latency Budget

| Path                            | Target | Notes                                                       |
| ------------------------------- | ------ | ----------------------------------------------------------- |
| Hotkey → panel visible          | <60ms  | Pre-warmed BrowserWindow, no LLM call                       |
| AX element-at-point fetch       | <50ms  | Single AX SPI call                                          |
| Local entity match → action     | <100ms | Pure Swift detection, no network                            |
| Vision-grounded one-shot answer | <1.5s  | Crop + AX context + Claude Haiku 4.5 (small model on purpose) |
| Hand-off to full coworker loop  | n/a    | Reuses existing path                                        |

**Model choice for vision-grounded path:** Use `claude-haiku-4-5` (already wired in via `backend/providers/registry.py`). The interaction is single-turn and bounded; the larger `claude-sonnet-4-5` is unnecessary and would push past the 1.5s budget.

---

## 7. Risks & Open Questions

1. **AX accuracy for non-native apps.** `AXUIElementCopyElementAtPosition` returns sensible elements for native AppKit and Catalyst apps. Electron-based apps (VS Code, Slack, Discord) expose limited AX trees, and web content in browsers depends on the browser exposing AXWebArea. Mitigation: when AX returns empty, fall back to the vision-grounded path using the cursor-centered crop alone.

2. **Focus-stealing.** The panel must be `focusable: true` to receive typed commands, but must not yank focus from the user's active window before the hotkey fires. Solution: capture the previously-focused window via `cua_get_focused_window` before showing the panel, and restore on dismiss (already a pattern in coworker mode).

3. **Multi-display / Retina coordinate normalization.** `cua_get_cursor_position` returns screen points; `AXUIElementCopyElementAtPosition` expects screen points; screenshots are in pixels. We already track `_scaleX`/`_scaleY` in `frontend/actions/screenshot.js:64-71`. Reuse that, don't reinvent.

4. **Hotkey conflicts.** ⌃⌘Space is unused by macOS defaults but conflicts with Alfred's secondary hotkey for some users. Make rebindable from settings on day one (not v2).

5. **Cost.** Even with the local fast-path, a heavy user could trigger 50–100 vision calls/day. At Haiku 4.5 pricing this is acceptable for individual users; flag for telemetry from launch.

6. **Privacy.** The panel sees whatever the cursor is over, including potentially sensitive content (passwords, private messages). The panel must:
   - Never auto-send to the cloud without a user trigger (hotkey + command).
   - Honor a deny-list of sensitive AX roles (`AXSecureTextField`).
   - Respect macOS's "Enhanced Privacy" mode if set.

7. **Should pinned mode allow multi-turn follow-ups?** Open question — adds complexity but unlocks the "Compare these two things I'm pointing at sequentially" flow. Recommend: v2.

---

## 8. Phased Rollout

### Phase 1 — Local fast-path only (1 week, 1 engineer)
- §5.1 global hotkey
- §5.2 `cua_get_element_at_point` (Swift + Python)
- §5.4 `NSDataDetector` helper
- §5.6 panel UI (entity chips + action chips, no composer yet)
- §5.7 settings

**Deliverable:** Point at a date/URL/phone, press hotkey, get a one-click action. Zero LLM cost.

### Phase 2 — Vision-grounded short commands (1 week)
- §5.3 `POST /pointer` endpoint
- §5.5 `pointer_act` tool + prompt fragment
- Composer in panel UI
- Streaming response rendering

**Deliverable:** "Summarize this," "Translate this," "Explain this code" — single-turn cursor-grounded answers.

### Phase 3 — Coworker hand-off (3 days)
- "Do this" command triggers the full existing coworker loop, pre-seeded with cursor context as the first observation.
- Panel collapses into a chip showing live agent progress.

**Deliverable:** Cursor-grounded entry point for multi-step automation.

### Phase 4 — Polish (ongoing)
- Pinned multi-turn mode
- Per-app entity heuristics (e.g., GitHub issue numbers, Jira tickets)
- Custom user-defined entity-action mappings
- Telemetry + cost dashboard

---

## 9. What We Are *Not* Trying to Match

For honesty about the gap with Magic Pointer:

| Magic Pointer feature                          | Emu Pointer equivalent                          |
| ---------------------------------------------- | ----------------------------------------------- |
| Sub-100ms ambient cursor-following inference   | **Not attempted.** Hotkey-triggered only.       |
| Voice and gesture activation                   | **Not attempted.** Hotkey + typed command only. |
| On-device NPU semantic entity extraction       | Cloud Haiku 4.5 for generic, local NSDataDetector for common types. |
| OS-level always-on presence in every app       | Electron global hotkey + AX. Same reach in practice, but explicit trigger. |
| Generative image compositing                   | **Not in v1.**                                  |
| Glowbar hardware integration                   | n/a.                                            |

What Emu *can* match: the **core design philosophy** — cursor position as implicit context, short commands instead of long prompts, no app-switching to talk to the AI. Within that design space, the 500ms / 1.5s latency targets are sufficient for the feature to feel responsive even without an NPU.

---

## 10. Summary of Concrete Work Items

| # | Item                                            | Files touched                                                                   | Est. LOC |
| - | ----------------------------------------------- | ------------------------------------------------------------------------------- | -------- |
| 1 | Global hotkey registration                      | `main.js`                                                                       | ~50      |
| 2 | `cua_get_element_at_point` driver tool          | `emu-cua-driver/*` (Swift), `backend/tools/coworker_tools.py`                   | ~120     |
| 3 | `NSDataDetector` helper                         | `emu-cua-driver/*` (Swift)                                                      | ~60      |
| 4 | `POST /pointer` backend endpoint                | `backend/main.py`                                                               | ~150     |
| 5 | `pointer_act` agent tool + prompt               | `backend/tools/coworker_tools.py`, `backend/providers/agent_tools.py`, `backend/prompts/` | ~110     |
| 6 | Pointer panel BrowserWindow + React UI          | `main.js`, `frontend/pointer-panel.html`, `frontend/pages/PointerPanel.js`      | ~300     |
| 7 | Settings keys + preferences UI                  | `~/.emu/settings.json` schema, `frontend/pages/Settings*`                       | ~80      |
| 8 | Telemetry hooks (cost + latency)                | `backend/utilities/`                                                            | ~40      |

**Total estimated effort:** ~910 LOC across ~10 files, plus Swift driver work. Two-engineer-weeks for Phases 1–2, an additional week for Phase 3.

---

## 11. References

- DeepMind — *Shaping the future of AI interaction by reimagining the mouse pointer* (2026-05-12)
- *Introducing Googlebook, designed for Gemini Intelligence* — Google blog, 2026-05-12
- Gemini 2.5 Computer Use model — `ai.google.dev/gemini-api/docs/computer-use`
- Internal: `backend/tools/coworker_tools.py:97-765` — tool spec pattern
- Internal: `backend/tools/dispatcher.py:214-458` — dispatch routing
- Internal: `frontend/actions/getMousePosition.js:13-51` — cursor position via `cliclick`
- Internal: `frontend/actions/screenshot.js:62-100` — capture pipeline + scale factor caching
- Internal: `frontend/border.html` — current overlay reference implementation
- Internal: `MACOS_PERMISSIONS.md` — existing entitlement set
