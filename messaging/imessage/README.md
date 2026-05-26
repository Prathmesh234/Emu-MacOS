# messaging/imessage — iMessage bridge

Inbound: poll `~/Library/Messages/chat.db` (read-only) at 1.5 s for new
messages. Outbound: send via `osascript send.applescript <handle>
<text>`. macOS-native, no third-party server, no third-party send-path
npm dep.

This is the same approach mautrix-imessage uses for its read pipeline.
We deliberately do **not** depend on third-party iMessage npm packages
(e.g. `osa-imessage`) so the only AppleScript that ever runs in this
process is the file in this directory.

## Required permissions

iMessage on macOS is gated by two separate user grants. Both are
documented in `MACOS_PERMISSIONS.md`; the bridge logs a clear error the
first time either is missing.

| Grant                                      | Where to set                                       | Why                              |
|--------------------------------------------|----------------------------------------------------|----------------------------------|
| **Full Disk Access** for the Emu binary    | System Settings → Privacy & Security → Full Disk Access | Read `chat.db`                  |
| **Automation → Messages**                  | System Settings → Privacy & Security → Automation  | Send via Messages.app            |

First-launch error you'll see if Full Disk Access is missing:

```
[messaging/imessage] error chat-db-permission-denied {"path":"…","hint":"Grant Full Disk Access to Emu in System Settings → Privacy & Security"}
[messaging/imessage] warn imessage-bridge-disabled
```

## Allowlist

```json
{ "imessage": ["+15551234567", "you@icloud.com"] }
```

Handles match `handle.id` from `chat.db` after light normalization
(phone numbers → `+digits`, emails → lowercased). DMs only; group chats
are filtered out by checking `chat_handle_join` cardinality.

## Cursor

We track the highest `message.ROWID` we've delivered in
`.emu/messaging/imessage/last_rowid.txt` so we never re-deliver. On
**first start**, we seed the cursor to the current `MAX(ROWID)` — the
bridge starts from "right now" instead of replaying months of backlog.

## Send

`send.applescript` runs against the **iMessage** service only (we look
up `1st service whose service type = iMessage`). If you want SMS
fallback, edit the script — we deliberately ship iMessage-only for v1 so
sends are E2E-encrypted Apple → Apple.
