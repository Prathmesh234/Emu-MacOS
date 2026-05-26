// messaging/common/allowlist.js — per-platform sender allowlist
//
// File format (.emu/messaging/allowlist.json):
//   {
//     "whatsapp": ["+15551234567", "+15557654321"],
//     "imessage": ["+15551234567", "alice@example.com"]
//   }
//
// Empty file or empty list per platform → that platform accepts zero
// messages. The bridge still starts (e.g. WhatsApp QR pair) so the
// operator can pair the account, but every inbound is dropped and
// logged in .emu/messaging/dropped.log until the operator edits the
// allowlist.

const fs = require('fs');
const { allowlistPath } = require('./paths');

function _normalize(handle) {
    // Phone numbers: strip everything but leading + and digits.
    // Email addresses: lowercase, trim.
    const trimmed = String(handle || '').trim();
    if (!trimmed) return '';
    if (trimmed.includes('@')) return trimmed.toLowerCase();
    const digits = trimmed.replace(/[^0-9+]/g, '');
    return digits.startsWith('+') ? digits : `+${digits}`;
}

function loadAllowlist() {
    const path = allowlistPath();
    if (!fs.existsSync(path)) {
        return { whatsapp: [], imessage: [], discord: [] };
    }
    try {
        const raw = JSON.parse(fs.readFileSync(path, 'utf8'));
        return {
            whatsapp: (raw.whatsapp || []).map(_normalize).filter(Boolean),
            imessage: (raw.imessage || []).map(_normalize).filter(Boolean),
            discord:  (raw.discord  || []).map((d) => String(d || '').trim()).filter(Boolean),
        };
    } catch (err) {
        // Bad JSON must NOT fail open — treat as "empty allowlist, drop everything".
        process.stderr.write(`[messaging/allowlist] failed to parse ${path}: ${err.message}\n`);
        return { whatsapp: [], imessage: [], discord: [] };
    }
}

function isAllowed(platform, handle) {
    const list = loadAllowlist()[platform] || [];
    if (list.length === 0) return false;
    if (platform === 'discord') return list.includes(String(handle || '').trim());
    const normalized = _normalize(handle);
    return Boolean(normalized) && list.includes(normalized);
}

function anyConfigured() {
    const a = loadAllowlist();
    return (a.whatsapp.length + a.imessage.length + a.discord.length) > 0;
}

module.exports = { loadAllowlist, isAllowed, anyConfigured, _normalize };
