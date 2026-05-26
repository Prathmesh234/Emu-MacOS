// messaging/common/sanitizer.js — strip host-sensitive paths from outbound text
//
// Outbound messages leave the host on user channels (WhatsApp/iMessage).
// Anything containing `.emu/`, the user's home dir, /private/var/folders,
// or other TCC-sensitive paths is rewritten before send. This is the only
// place where we get to scrub before the platform's transport takes over.

const os = require('os');
const path = require('path');

function _userPrefixes() {
    const home = os.homedir() || '';
    const prefixes = [];
    if (home) {
        prefixes.push(home);
        // /Users/<name>  — common alias seen in stack traces and logs
        const base = path.basename(home);
        if (base) prefixes.push(`/Users/${base}`);
    }
    return prefixes;
}

function sanitize(text) {
    if (typeof text !== 'string') return text;
    let out = text;

    // .emu/* (case-sensitive) — single combined replace so we don't
    // double-substitute when the second pattern matches the `<.emu>`
    // that the first one produces. Path body stops at closing-quote /
    // paren boundaries so "(.emu/foo)" doesn't swallow the ).
    out = out.replace(/(^|[\s'"`(])\.emu(\/[^\s'"`),>]*)?/g, (_m, lead, rest) =>
        lead + '<.emu>' + (rest ? '/…' : '')
    );

    // TCC-sensitive macOS paths
    out = out.replace(/\/private\/var\/folders\/[A-Za-z0-9_+/=\-]+/g, '<sandbox>');
    out = out.replace(/~\/Library\/[^\s'")]+/g, '~/<library>');
    out = out.replace(/\/Library\/Caches\/[^\s'")]+/g, '/Library/Caches/<…>');
    out = out.replace(/\/Library\/Application Support\/[^\s'")]+/g, '/Library/Application Support/<…>');

    // Per-user home and /Users/<name>
    for (const prefix of _userPrefixes()) {
        if (!prefix) continue;
        const escaped = prefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        out = out.replace(new RegExp(escaped + '(?=$|[\\s\'"`),>])', 'g'), '~');
        out = out.replace(new RegExp(escaped + '\\/', 'g'), '~/');
    }

    // Auth token (paranoia)
    out = out.replace(/[A-Fa-f0-9]{64}/g, '<token>');

    // Hard cap so a runaway model can't dump a megabyte to a phone.
    const MAX = 3500;
    if (out.length > MAX) {
        out = out.slice(0, MAX) + '\n…[truncated]';
    }
    return out;
}

module.exports = { sanitize };
