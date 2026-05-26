# messaging/whatsapp — WhatsApp bridge (Baileys)

Uses [`@whiskeysockets/Baileys`](https://github.com/WhiskeySockets/Baileys)
to speak the WhatsApp Web multi-device protocol. Same library
mautrix-whatsapp, Beeper, Evolution API, n8n's WhatsApp node, Chatwoot,
and the bulk of OSS WhatsApp assistants use. End-to-end encryption is
preserved by the WhatsApp protocol itself; messages are decrypted only
inside this local process.

## Pairing

First run prints a QR code to stderr (visible in the runner log and the
Electron main process stdout):

```
[messaging/whatsapp] info qr-ready
█▀▀▀▀▀█  ▀▄▀█▄▄ █▀▀▀▀▀█
…
```

On your phone:

1. WhatsApp → **Settings → Linked Devices → Link a Device**
2. Scan the QR.
3. The bridge logs `connected` and persists the auth state under
   `.emu/messaging/whatsapp/auth/` (chmod 0700).

After that, the bridge reconnects automatically on every Emu start.

If WhatsApp force-logs-out the linked device (you remove it from your
phone, the server rotates keys, …), the bridge writes
`.emu/messaging/whatsapp/REPAIR_NEEDED` so the operator knows to re-pair.

## Allowlist

`.emu/messaging/allowlist.json`:

```json
{ "whatsapp": ["+15551234567"] }
```

Numbers are normalized to E.164-ish (`+` + digits only). Inbound messages
from any other handle are dropped and counted in
`.emu/messaging/dropped.log` — we never auto-reply to a non-allowlisted
sender.

Group chats (`*@g.us`) are always dropped (DMs only for v1).

## Outbound

`sendMessage(jid, { text })` only. No image attachments, voice notes, or
reactions in v1. Outbound text is run through `common/sanitizer.js`
before send.
