// messaging/common/log.js — line-buffered append-only logger
//
// Writes JSONL to .emu/messaging/<platform>/bridge.log and mirrors a
// single human-readable line to stderr with a "[messaging/<platform>] "
// prefix. The Electron supervisor (messagingProcess.js) captures stderr
// and stamps it into the main log so operators see bridge activity
// without tailing files.

const fs = require('fs');
const { bridgeLogPath, droppedLogPath } = require('./paths');

function _redact(str) {
    if (typeof str !== 'string') return str;
    return str
        .replace(/sk-[A-Za-z0-9]{16,}/g, 'sk-[REDACTED]')
        .replace(/Bearer\s+[A-Za-z0-9._\-]{12,}/gi, 'Bearer [REDACTED]')
        .replace(/(X-Emu-Token['"]?\s*[:=]\s*['"]?)[A-Za-z0-9]{8,}/gi, '$1[REDACTED]')
        .replace(/(api[_-]?key['"]?\s*[:=]\s*['"]?)[^\s,'"}]+/gi, '$1[REDACTED]');
}

function _stamp() {
    return new Date().toISOString();
}

function makeLogger(platform) {
    const filePath = bridgeLogPath(platform);
    const prefix = `[messaging/${platform}]`;

    function writeLine(level, message, fields) {
        const entry = {
            ts: _stamp(),
            level,
            message: _redact(message),
        };
        if (fields && typeof fields === 'object') {
            for (const [k, v] of Object.entries(fields)) {
                entry[k] = typeof v === 'string' ? _redact(v) : v;
            }
        }
        try {
            fs.appendFileSync(filePath, JSON.stringify(entry) + '\n', { mode: 0o600 });
        } catch (err) {
            // Logging must never throw out of the bridge loop.
            process.stderr.write(`${prefix} log-append-failed ${err.message}\n`);
        }
        const human = fields && Object.keys(fields).length
            ? ` ${JSON.stringify(fields)}`
            : '';
        process.stderr.write(`${prefix} ${level} ${entry.message}${human}\n`);
    }

    return {
        info:  (msg, fields) => writeLine('info',  msg, fields),
        warn:  (msg, fields) => writeLine('warn',  msg, fields),
        error: (msg, fields) => writeLine('error', msg, fields),
        debug: (msg, fields) => {
            if (process.env.EMU_MESSAGING_DEBUG === '1') writeLine('debug', msg, fields);
        },
    };
}

function logDropped(platform, handle, reason) {
    const line = JSON.stringify({
        ts: _stamp(),
        platform,
        handle: _redact(String(handle || '')),
        reason: _redact(String(reason || '')),
    });
    try {
        fs.appendFileSync(droppedLogPath(), line + '\n', { mode: 0o600 });
    } catch (_) { /* best-effort */ }
}

module.exports = { makeLogger, logDropped };
