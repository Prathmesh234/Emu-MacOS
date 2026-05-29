// messaging/common/silenceLibsignal.js — suppress noisy libsignal dumps.
//
// Baileys' bundled libsignal-protocol implementation emits multi-line
// `console.log('Closing session:', SessionEntry { ... })` dumps every time
// a Signal Double-Ratchet session rotates. These are not errors and bypass
// every pino/log level we configure because they're emitted via raw
// `console.log` from inside node_modules. They flood stderr (which the
// Electron supervisor mirrors into the main log).
//
// We intercept `console.log` / `console.warn` once at runner boot and drop
// the well-known libsignal patterns. Anything else passes through unchanged.

const _SUPPRESS_PATTERNS = [
    /^Closing session:/,           // libsignal session rotation dump
    /^Closing open session in favor of incoming prekey bundle/,
    /^Deleting session closed at/,
];

function _shouldSuppress(args) {
    if (!args || args.length === 0) return false;
    const first = args[0];
    if (typeof first !== 'string') return false;
    for (const re of _SUPPRESS_PATTERNS) {
        if (re.test(first)) return true;
    }
    return false;
}

let _installed = false;

function install() {
    if (_installed) return;
    _installed = true;
    const origLog = console.log.bind(console);
    const origWarn = console.warn.bind(console);
    console.log = function patchedLog(...args) {
        if (_shouldSuppress(args)) return;
        return origLog(...args);
    };
    console.warn = function patchedWarn(...args) {
        if (_shouldSuppress(args)) return;
        return origWarn(...args);
    };
}

module.exports = { install };
