# Emu inbound messaging bridge

Lets you message Emu from WhatsApp or iMessage and have the agent run the
request on your Mac, with the reply landing back in the same chat.

All inbound messages from **all** allowlisted senders (across both
platforms) funnel into a **single, persistent Emu session** — the
"Remote control" session — which shows up in the in-app History sidebar
and just keeps growing over time. Local desktop chats are unaffected.

## Architecture at a glance

```
WhatsApp / iMessage chat
         │
         ▼
  bridge (Baileys / chat.db poll)        ← messaging/whatsapp, messaging/imessage
         │
         ▼
  common/dispatcher.js                   ← allowlist, queue, route memory
         │
         ▼  POST /agent/step  (X-Emu-Token)
  backend (FastAPI 127.0.0.1:8000)
         │
         ▼  /ws/<remote_session_id>
  dispatcher → outboundRouter
         │
         ▼  bridge.sendMessage(handle, sanitized_text)
  back to the originating WhatsApp / iMessage chat
```

The dispatcher pins the same `session_id` for every inbound message, so
`.emu/sessions/<remote_session_id>/logs/conversation.json` is one
continuous log the in-app History sidebar can render exactly like any
other past session.

## Files

```
messaging/
  package.json
  runner.js                supervisor; one Node process owns all bridges
  common/
    paths.js               .emu/messaging/* path helpers
    log.js                 JSONL + stderr logger with secret redaction
    allowlist.js           reads .emu/messaging/allowlist.json
    sanitizer.js           strips .emu/, /Users/<name>/, TCC paths from outbound
    agentClient.js         backend HTTP + /ws/<id> client (X-Emu-Token)
    sessionBinder.js       singleton "remote_control" session id
    outboundRouter.js      per-turn (platform, handle) reply route
    rateLimit.js           per-handle FIFO queue
    dispatcher.js          inbound → /agent/step → outbound reply
  whatsapp/
    index.js               Baileys WhatsApp bridge
    README.md
  imessage/
    index.js               chat.db poll + osascript send
    send.applescript       the only AppleScript ever invoked
    README.md
```

Runtime state:

```
.emu/messaging/            chmod 0700, gitignored, excluded from packaging
  allowlist.json           operator-edited list of allowed handles
  remote_session.json      pinned remote-control session id
  dropped.log              one line per dropped (non-allowlisted) message
  whatsapp/
    auth/                  Baileys multi-file auth state (chmod 0700)
    bridge.log             JSONL bridge events
  imessage/
    last_rowid.txt         chat.db ROWID high-water mark
    bridge.log
```

## Allowlist — mandatory

The bridge accepts **zero** messages by default. Create
`.emu/messaging/allowlist.json` (mode 0600) with the handles you'll
message from:

```json
{
  "whatsapp": ["+15551234567"],
  "imessage": ["+15551234567", "you@icloud.com"]
}
```

Phone numbers are normalized (digits-only with leading `+`), so any
formatting variation in the file or in chat.db lines up. Non-matching
senders are silently dropped and counted in `dropped.log` — we never
reply to a non-allowlisted contact (that would confirm to spammers the
number is live).

Group chats are dropped (DMs only for v1).

## Operator commands

```bash
# Sanity-check the configuration without opening any sockets.
node messaging/runner.js --selfcheck

# Run the bridges with verbose logging.
EMU_MESSAGING_DEBUG=1 node messaging/runner.js

# Dry-run mode: accept inbound, log it, but don't POST to /agent/step
# (used by tests).
EMU_MESSAGING_DRY_RUN=1 node messaging/runner.js
```

Environment switches read by `runner.js`:

| Variable                     | Effect                                                       |
|------------------------------|--------------------------------------------------------------|
| `EMU_DISABLE_MESSAGING=1`    | Skip the runner entirely.                                    |
| `EMU_DISABLE_WHATSAPP=1`     | Run iMessage only.                                           |
| `EMU_DISABLE_IMESSAGE=1`     | Run WhatsApp only.                                           |
| `EMU_MESSAGING_AGENT_MODE`   | `coworker` (default) or `remote`.                            |
| `EMU_MESSAGING_DRY_RUN=1`    | Log inbound, skip backend post.                              |
| `EMU_MESSAGING_DEBUG=1`      | Emit debug lines from each bridge.                           |
| `IMESSAGE_DB`                | Override `~/Library/Messages/chat.db` (testing).             |

## Security

- **Allowlist mandatory.** No allowlist = no inbound accepted.
- **Existing X-Emu-Token reused.** Bridge ↔ backend uses the per-launch
  token in `.emu/.auth_token` — no new auth surface on `127.0.0.1:8000`.
- **Outbound sanitization.** Every outbound message goes through
  `common/sanitizer.js` which strips `.emu/`, the user's home directory,
  `~/Library/`, `/private/var/folders/`, and obvious auth tokens.
- **Env scrub.** The Electron supervisor spawns the runner with a minimal
  env (no provider API keys), same shape as the Hermes child.
- **State jailed.** Baileys auth and the iMessage cursor live only under
  `.emu/messaging/` with mode 0700. Gitignored and excluded from packaged
  builds.
- **Never `osascript -e`.** The iMessage bridge invokes
  `osascript send.applescript <handle> <text>` via `execFile`, never
  builds AppleScript source dynamically.

## Out of scope for v1

- Discord bridge (folder reserved; not yet implemented).
- Group chats, reactions, voice notes, video, attachments from the user.
- Sending screenshots **back** as attachments — v1 sends text only.
- Cross-platform conversation threading.

See `/COMING_SOON.md` § "Inbound messaging bridge" for the longer design
note this implementation is based on.
