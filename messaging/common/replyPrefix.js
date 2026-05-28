// messaging/common/replyPrefix.js — outbound prefix + echo suppression
//
// When a bridge runs in self-chat mode (e.g. WhatsApp WHATSAPP_MODE=self-chat,
// iMessage IMESSAGE_MODE=self-chat) the agent reads + writes messages in a
// chat where it is BOTH sender and recipient. Without active suppression
// every reply the agent sends would arrive back as a new "user message",
// driving an infinite loop.
//
// Three layers of defense, used together:
//
//   1. Reply prefix   — every outbound message is prepended with a fixed,
//                       unusual string. Inbound messages whose body starts
//                       with that string are recognised as our own echoes
//                       and dropped at the bridge boundary.
//   2. Recently-sent  — a bounded Set of the last N outbound message IDs
//        ID cache       returned by the underlying transport (WhatsApp:
//                       msg.key.id). Catches the rare case where the prefix
//                       is stripped by transport re-encoding.
//   3. Scope          — self-chat mode itself only accepts `fromMe` traffic
//                       from the user's own chat thread; everything else
//                       is rejected at the bridge layer.
//
// This module owns layers 1 + 2 (transport-agnostic helpers). Layer 3 lives
// in the bridge that knows its own JID/handle semantics.

const DEFAULT_PREFIX = '🤖 *Emu*\n────────\n';

let _cachedPrefix = null;

function getPrefix() {
    if (_cachedPrefix !== null) return _cachedPrefix;
    const raw = process.env.EMU_MESSAGING_REPLY_PREFIX;
    if (typeof raw === 'string' && raw.length > 0) {
        // Allow operators to encode escapes (e.g. "\n") in the env value so
        // they don't have to wrestle with shell quoting for a literal newline.
        _cachedPrefix = raw.replace(/\\n/g, '\n').replace(/\\t/g, '\t');
    } else {
        _cachedPrefix = DEFAULT_PREFIX;
    }
    return _cachedPrefix;
}

// Test-only hook so unit tests can override the prefix without monkey-patching
// process.env mid-run. Production code should never call this.
function _resetForTests() { _cachedPrefix = null; }

function wrap(text) {
    if (typeof text !== 'string' || text.length === 0) return text;
    const prefix = getPrefix();
    // Defensive: don't double-prefix if upstream already wrapped (e.g. the
    // sanitizer ran twice, or a caller pre-wrapped). Idempotent on retries.
    if (text.startsWith(prefix)) return text;
    return prefix + text;
}

function isPrefixed(text) {
    if (typeof text !== 'string' || text.length === 0) return false;
    return text.startsWith(getPrefix());
}

function stripPrefix(text) {
    if (!isPrefixed(text)) return text;
    return text.slice(getPrefix().length);
}

// Bounded FIFO set of message IDs we've recently sent on this transport.
// Insertion-ordered Set + size cap keeps lookup O(1) and memory bounded.
function createRecentlySentCache(max = 200) {
    const ids = new Set();
    return {
        add(id) {
            if (!id) return;
            if (ids.has(id)) return;
            ids.add(id);
            if (ids.size > max) {
                // Sets preserve insertion order; pop the oldest.
                const oldest = ids.values().next().value;
                ids.delete(oldest);
            }
        },
        has(id) {
            return Boolean(id) && ids.has(id);
        },
        size() {
            return ids.size;
        },
    };
}

module.exports = {
    getPrefix,
    wrap,
    isPrefixed,
    stripPrefix,
    createRecentlySentCache,
    _resetForTests,
};
