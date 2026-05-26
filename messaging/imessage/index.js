// messaging/imessage/index.js — iMessage bridge
//
// Inbound:  poll ~/Library/Messages/chat.db (read-only) at 1.5s. Track
//           high-water ROWID under .emu/messaging/imessage/last_rowid.txt
//           so we never re-deliver.
// Outbound: execFile('osascript', ['send.applescript', handle, text]).
//           No osascript -e ever; the script body is static and bundled.
//
// Required user grants (documented in README.md):
//   - Full Disk Access on Emu.app  → so chat.db is readable
//   - Automation → Messages         → so osascript can drive Messages.app

const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFile } = require('child_process');
const { platformDir } = require('../common/paths');
const { makeLogger } = require('../common/log');
const allowlist = require('../common/allowlist');

const PLATFORM = 'imessage';
const CHAT_DB = process.env.IMESSAGE_DB || path.join(os.homedir(), 'Library', 'Messages', 'chat.db');
const APPLESCRIPT_PATH = path.join(__dirname, 'send.applescript');
const POLL_INTERVAL_MS = 1500;

let Database;

function _lazyLoadSqlite() {
    if (!Database) {
        Database = require('better-sqlite3');
    }
    return Database;
}

function _loadCursor() {
    const file = path.join(platformDir(PLATFORM), 'last_rowid.txt');
    if (!fs.existsSync(file)) return 0;
    try {
        const raw = fs.readFileSync(file, 'utf8').trim();
        const n = parseInt(raw, 10);
        return Number.isFinite(n) ? n : 0;
    } catch {
        return 0;
    }
}

function _saveCursor(rowid) {
    const file = path.join(platformDir(PLATFORM), 'last_rowid.txt');
    try {
        fs.writeFileSync(file, String(rowid), { mode: 0o600 });
    } catch (_) { /* best-effort */ }
}

function _normalizeAppleHandle(id) {
    // chat.db's `handle.id` is either an E.164 phone ("+15551234567") or
    // an iCloud email. Pass through as-is after trimming.
    return String(id || '').trim();
}

function _openDb(logger) {
    let DB;
    try {
        DB = _lazyLoadSqlite();
    } catch (err) {
        logger.error('better-sqlite3-load-failed', { error: err.message });
        return null;
    }
    if (!fs.existsSync(CHAT_DB)) {
        logger.warn('chat-db-not-found', { path: CHAT_DB });
        return null;
    }
    try {
        // readonly + fileMustExist guards against accidental writes.
        return new DB(CHAT_DB, { readonly: true, fileMustExist: true });
    } catch (err) {
        if (/permission/i.test(err.message) || err.code === 'SQLITE_CANTOPEN') {
            logger.error('chat-db-permission-denied', {
                path: CHAT_DB,
                hint: 'Grant Full Disk Access to Emu in System Settings → Privacy & Security',
            });
        } else {
            logger.error('chat-db-open-failed', { error: err.message });
        }
        return null;
    }
}

// Returns rows for messages with ROWID > cursor, in ascending order.
//
// `is_from_me = 0` filters out our own outbound replies.
// `cache_has_attachments = 0` and `text IS NOT NULL` keep us to plain text.
// `handle.id` is the buddy. Group chats are excluded by joining only
// 1:1 chats from chat_message_join with chat_handle_join cardinality of 1.
const QUERY = `
    SELECT
        m.ROWID            AS rowid,
        m.text             AS text,
        m.attributedBody   AS attributed_body,
        h.id               AS handle_id,
        c.guid             AS chat_guid,
        m.date             AS date,
        (
            SELECT COUNT(*) FROM chat_handle_join chj
            WHERE chj.chat_id = c.ROWID
        )                  AS handle_count
    FROM message m
    JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
    JOIN chat c                ON c.ROWID = cmj.chat_id
    LEFT JOIN handle h         ON h.ROWID = m.handle_id
    WHERE m.ROWID > ?
      AND m.is_from_me = 0
      AND m.text IS NOT NULL
      AND m.text != ''
    ORDER BY m.ROWID ASC
    LIMIT 50
`;

async function startIMessage({ dispatcher }) {
    const logger = makeLogger(PLATFORM);

    if (process.platform !== 'darwin') {
        logger.info('skipping-non-darwin');
        return { stop() {} };
    }
    if (allowlist.loadAllowlist().imessage.length === 0) {
        logger.info('skipping-no-allowlist-entries');
        return { stop() {} };
    }

    let db = _openDb(logger);
    if (!db) {
        logger.warn('imessage-bridge-disabled');
        return { stop() {} };
    }

    let cursor = _loadCursor();
    if (cursor === 0) {
        // First start: skip the entire backlog so we don't replay months
        // of old messages on launch. Capture the current max ROWID and
        // start fresh from there.
        try {
            const row = db.prepare('SELECT MAX(ROWID) AS m FROM message').get();
            cursor = (row && row.m) || 0;
            _saveCursor(cursor);
            logger.info('initialized-cursor-skipping-backlog', { cursor });
        } catch (err) {
            logger.warn('initial-cursor-query-failed', { error: err.message });
        }
    }

    const select = db.prepare(QUERY);

    let stopped = false;
    let timer = null;

    function _reopen() {
        try { db && db.close(); } catch (_) { /* ignore */ }
        db = _openDb(logger);
    }

    async function _tick() {
        if (stopped || !db) return;
        let rows = [];
        try {
            rows = select.all(cursor);
        } catch (err) {
            logger.warn('query-failed-reopening', { error: err.message });
            _reopen();
            return;
        }
        for (const row of rows) {
            cursor = row.rowid;
            _saveCursor(cursor);
            if (row.handle_count !== 1) {
                logger.debug('skip-group-chat', { rowid: row.rowid });
                continue;
            }
            const handle = _normalizeAppleHandle(row.handle_id);
            if (!handle) continue;
            const text = String(row.text || '').trim();
            if (!text) continue;
            try {
                await dispatcher.handleInbound({
                    platform: PLATFORM,
                    handle,
                    text,
                    meta: { rowid: row.rowid, chat_guid: row.chat_guid },
                });
            } catch (err) {
                logger.error('inbound-dispatch-failed', { error: err.message });
            }
        }
    }

    function _send(handle, text) {
        return new Promise((resolve, reject) => {
            execFile(
                'osascript',
                [APPLESCRIPT_PATH, handle, text],
                { timeout: 20_000 },
                (err, stdout, stderr) => {
                    if (err) return reject(err);
                    const out = String(stdout || '').trim();
                    if (out.startsWith('ERR')) {
                        return reject(new Error(out));
                    }
                    if (stderr && stderr.trim()) {
                        // osascript writes UI authorization warnings to stderr;
                        // surface them at warn level but don't fail the send.
                        logger.warn('osascript-stderr', { stderr: stderr.trim() });
                    }
                    resolve(out);
                },
            );
        });
    }

    dispatcher.registerSender(PLATFORM, async (handle, text) => {
        try {
            await _send(handle, text);
            logger.info('sent', { handle, length: text.length });
        } catch (err) {
            logger.error('send-failed', { handle, error: err.message });
            throw err;
        }
    });

    logger.info('imessage-bridge-started', { cursor, dbPath: CHAT_DB });
    timer = setInterval(() => { _tick().catch(() => {}); }, POLL_INTERVAL_MS);
    timer.unref && timer.unref();

    return {
        stop() {
            stopped = true;
            if (timer) clearInterval(timer);
            try { db && db.close(); } catch (_) { /* ignore */ }
            logger.info('stopped');
        },
    };
}

module.exports = { startIMessage };
