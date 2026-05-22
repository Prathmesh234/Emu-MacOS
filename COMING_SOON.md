# Coming Soon

Planned features for Emu. Design notes only — nothing here is implemented yet.
Each section is meant to be small enough that an engineer can pick it up
without reading the rest of the file.

---

## 1. Inbound messaging bridge (Discord / WhatsApp / iMessage)

### Goal

Let users send a prompt to Emu from a chat app they already use, have the
agent run it on their Mac, and receive the result in the same conversation.
Same model loop as the in-app chat — different transport.

### High-level flow

```
chat app  →  bridge process  →  POST /agent/step  →  agent loop  →  reply on same channel
```

The bridge processes are thin adapters. They do not run the agent themselves;
they translate inbound messages into `AgentRequest` objects and forward
outbound text/screenshots back to the originating conversation.

### Per-platform transport

Pick well-maintained open-source libraries that other projects (Beeper,
Matrix bridges, the various open-source assistant stacks) already rely on,
so we inherit their reliability work instead of reinventing it.

| Platform  | Library                          | Notes |
|-----------|----------------------------------|-------|
| Discord   | `discord.js` (Node)              | Runs inside Electron main. Bot token + intents only — no user-account login. |
| WhatsApp  | `@whiskeysockets/Baileys` (Node) | Same library mautrix-whatsapp, Beeper, and many open assistants use. QR-pair once; session stored under `.emu/bridges/whatsapp/`. End-to-end encryption preserved. |
| iMessage  | Read `~/Library/Messages/chat.db` (SQLite, read-only) + send via AppleScript to `Messages.app` | macOS-native, no third-party server. Requires Full Disk Access for the chat.db read. |

If users later want a single inbox across all three with one auth surface,
the next step is a small Matrix homeserver plus the `mautrix-imessage`,
`mautrix-whatsapp`, and `mautrix-discord` bridges. Defer until the direct
path is stable.

### Spec

**Inbound**

1. Bridge subscribes to messages on its platform.
2. Filter: only messages from an allowlisted handle (per-platform contact
   ID stored in `.emu/bridges/allowlist.json`). Unmatched messages are
   dropped and counted in `.emu/bridges/dropped.log`.
3. Build an `AgentRequest` with the message text, the user's default
   `agent_mode`, and a new `source` field (`"discord" | "whatsapp" | "imessage"`).
4. POST to `http://127.0.0.1:8000/agent/step` with the same `X-Emu-Token`
   the local UI uses.

**Outbound**

- Each assistant text turn and the final `done.message` are posted back to
  the same conversation.
- Screenshots go as platform-native attachments where supported (Discord:
  file upload; WhatsApp: image message; iMessage: AppleScript image send).
- A single sanitizer strips `.emu/` paths and TCC-sensitive paths from
  every outbound message before it leaves the host.

**Security**

- Allowlist is mandatory. No allowlist entry → no agent invocation, full stop.
- Bridge ↔ backend uses the existing `X-Emu-Token` — we do not open a new
  auth surface on `127.0.0.1:8000`.
- Per-handle rate limit: queue, don't drop. Losing user intent is worse
  than a few seconds of delay.
- Bridge processes run with the same scrubbed env we already use for
  Hermes — no provider API keys, no `.env`.

**Out of scope for v1**

- Group chats — DMs only.
- Reactions, voice notes, video, file uploads from the user.
- Cross-platform conversation threading.

### Files (planned)

```
bridges/
  discord/        index.js, README.md
  whatsapp/       index.js, README.md
  imessage/       index.js, send.applescript, README.md
  runner.js       supervises enabled bridges; lifecycle owned by Electron main
backend/main.py   accept optional `source` field on AgentRequest
.emu/bridges/
  allowlist.json
  dropped.log
  <platform>/.token, <platform>/session/
```

---

## 2. Locked-screen computer use

### Goal

Keep `remote`-mode Emu turns running after the user locks their Mac —
clicks, key events, and screenshots all keep working — without dismissing
the lock screen, leaking visuals to bystanders, or fighting the user if
they come back to the keyboard mid-turn.

### Why screenshots don't go black

When the screen locks, `loginwindow` takes the foreground console session
and draws the lock UI on top. The user session keeps compositing
underneath; apps don't pause. A normal per-user process SCStream'ing at
that point captures `loginwindow` (black/lock frames) — that's today's
failure mode.

The fix is to capture from a **system-context LaunchDaemon** with its own
Screen Recording TCC grant. From that tier, `SCStream` can be filtered to
the user-session display below the lock layer, and `CGEventPost` to
`cghidEventTap` reaches the user session whether locked or not. The lock
screen is never dismissed.

### Approach

Reuse the existing Swift sources under `frontend/coworker-mode/emu-driver/`.
Build a sibling executable target `emu-locked-helper` that:

- Installs as a **LaunchDaemon** at `/Library/LaunchDaemons/com.emu.locked-helper.plist`
  via `SMAppService.daemon(plistName:)`. This is separate from the existing
  per-user LaunchAgent (`com.emu.emu-cua-driver`), which keeps handling
  unlocked sessions.
- Exposes XPC: `beginLockedTurn(grant)`, `endLockedTurn(grant)`,
  `captureFrame()`, `postEvent(...)`, `getDisplays()`. Only callable from
  the signed Electron parent.
- Routes capture and input to the helper as soon as
  `CGSessionCopyCurrentDictionary[kCGSSessionScreenIsLocked] == 1`.

### Safeguards (mirror Codex "locked use")

1. **Short-lived authorization** — backend mints a per-turn HMAC grant in
   `backend/auth/locked_grant.py` (expiry ≤ 90s, single-use nonce). The
   helper validates every XPC call against it.
2. **Covered display fallback** — only used if SCStream-under-lock returns
   black on the user's macOS version. Helper unlocks via a Keychain-stored
   credential, drops a `CGShieldingWindow` above `CGShieldingWindowLevel()`
   for the turn, re-locks at the end. SCStream filter excludes the shield
   so the model still sees the real UI.
3. **Relock on local input** — helper installs a listen-only `CGEventTap`
   at `cghidEventTap` for the duration of the turn. Any event whose source
   is not the helper's synthetic post ends the turn and re-locks.
4. **Manual-unlock fallback** — helper watches for `com.apple.SecurityAgent`
   becoming frontmost (sudo, Touch ID, FileVault prompts). On detection,
   end the turn cleanly and surface `manual_unlock_required`. No retries.

### Files (planned)

```
frontend/coworker-mode/emu-driver/Sources/EmuLockedHelper/
  XPC.swift, Capture.swift, Input.swift, Shield.swift,
  AuthWatcher.swift, EventTap.swift
daemon/launchd/com.emu.locked-helper.plist.template   LaunchDaemon, not Agent
daemon/install_macos.py                               install/uninstall via SMAppService
backend/auth/locked_grant.py                          HMAC grant minting + verification
backend/main.py                                       attach grant to remote actions when locked
frontend/process/EmuCuaDriverProcess.js               detect lock state, route to helper
```

### Prerequisites

- Developer ID Application certificate. LaunchDaemons cannot be ad-hoc
  signed; current signing per `frontend/coworker-mode/PLAN.md` would block
  install.
- macOS 13+ for `SMAppService`.
- One-time user grant: System Settings → Privacy & Security → Screen
  Recording → `emu-locked-helper`. This is a separate TCC entry from the
  Electron app.

### Composition with feature 1

When a `remote` turn is triggered from Discord/WhatsApp/iMessage and the
Mac is already locked, the messaging bridge → backend → locked-helper path
means a user can message Emu from anywhere and have it work on their
machine even after they've walked away and locked it. The bridge supplies
the `source`, the backend mints the grant, the helper runs the turn under
the four safeguards above.
