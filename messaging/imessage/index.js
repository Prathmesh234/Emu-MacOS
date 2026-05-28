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
const replyPrefix = require('../common/replyPrefix');

const PLATFORM = 'imessage';
const CHAT_DB = process.env.IMESSAGE_DB || path.join(os.homedir(), 'Library', 'Messages', 'chat.db');
const APPLESCRIPT_PATH = path.join(__dirname, 'send.applescript');
const POLL_INTERVAL_MS = 1500;

// MODE: "bot" (default, legacy) or "self-chat".
//
//   bot       — chat.db only surfaces messages from other contacts. Our
//               own outbound sends (via osascript) are ignored. This is
//               the safe default for any deployment where the agent has
//               its own iMessage account / Apple ID.
//
//   self-chat — the bridge accepts messages with `is_from_me = 1` so the
//               operator can drive the agent from any iMessage chat that
//               syncs across their Apple ID (Note to Self, or a chat with
//               their own number from a second device). Anti-loop is
//               enforced by:
//                 (a) restricting accepted chats to a configured "self
//                     handles" set (defaults to the imessage allowlist —
//                     in self-chat mode the operator's own handle is the
//                     command surface), and
//                 (b) dropping any inbound message whose body starts with
//                     Emu's reply prefix (those are our own osascript
//                     sends, replayed to us via the iCloud message sync).
//
// Implementation parallels the WhatsApp self-chat path so operators get
// the same env shape on both platforms.
function _mode() {
    const raw = String(process.env.EMU_MESSAGING_IMESSAGE_MODE || 'bot').toLowerCase();
    return raw === 'self-chat' ? 'self-chat' : 'bot';
}

// Build the normalised set of chat identifiers that count as "the
// operator's own chat" for self-chat mode. Source precedence:
//   1. EMU_MESSAGING_IMESSAGE_SELF_HANDLES (comma-separated). Lets the
//      operator narrow to a specific chat if they have multiple handles.
//   2. Fallback: the imessage allowlist. In self-chat mode the operator's
//      OWN handle is what they put in the allowlist; using the allowlist
//      as the self-handle source is the zero-config experience.
function _buildSelfHandleSet(logger) {
    const set = new Set();
    const raw = String(process.env.EMU_MESSAGING_IMESSAGE_SELF_HANDLES || '').trim();
    if (raw) {
        for (const piece of raw.split(',')) {
            const normalized = allowlist._normalize(piece);
            if (normalized) set.add(normalized);
        }
    }
    if (set.size === 0) {
        for (const handle of allowlist.loadAllowlist().imessage) {
            set.add(handle);
        }
    }
    if (set.size === 0 && logger) {
        logger.warn('self-chat-no-self-handle-resolved', {
            hint: 'add your own handle to .emu/messaging/allowlist.json or set EMU_MESSAGING_IMESSAGE_SELF_HANDLES=+15551234567',
        });
    }
    return set;
}

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
// We deliberately do NOT filter on `is_from_me` in SQL anymore — the JS
// tick loop decides based on the configured mode. `bot` mode drops
// `is_from_me=1`, `self-chat` mode does the opposite. Pulling both keeps
// the bridge mode switchable without touching the prepared statement.
//
// `cache_has_attachments = 0` is not enforced; we only require non-empty
// text. `handle.id` is the buddy; `chat.chat_identifier` is the chat's
// canonical address (used in self-chat mode to confirm Note-to-Self).
// Group chats are excluded by joining only 1:1 chats from
// chat_handle_join with cardinality 1.
const QUERY = `
    SELECT
        m.ROWID            AS rowid,
        m.text             AS text,
        m.attributedBody   AS attributed_body,
        m.is_from_me       AS is_from_me,
        h.id               AS handle_id,
        c.guid             AS chat_guid,
        c.chat_identifier  AS chat_identifier,
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
    const mode = _mode();
    // Built once at startup; not hot-reloaded if the operator edits the
    // allowlist mid-run. Matches the WhatsApp bridge's selfJids semantics.
    const selfHandles = mode === 'self-chat' ? _buildSelfHandleSet(logger) : new Set();
    logger.info('mode-selected', {
        mode,
        selfHandles: mode === 'self-chat' ? Array.from(selfHandles) : undefined,
    });

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
            const fromMe = row.is_from_me === 1;
            const text = String(row.text || '').trim();
            if (!text) continue;

            let handle;
            if (mode === 'self-chat') {
                // Scope: self-chat mode ignores third-party inbound. The
                // operator's own personal Apple ID is the only command
                // surface; messages from contacts go nowhere even if the
                // contact happens to be allowlisted.
                if (!fromMe) {
                    logger.debug('self-chat-drop-not-from-me', { rowid: row.rowid });
                    continue;
                }
                // Echo suppression: anything starting with our reply
                // prefix is an osascript send we just made, replayed back
                // to us via iCloud message sync.
                if (replyPrefix.isPrefixed(text)) {
                    logger.debug('self-chat-drop-prefixed-echo', {
                        rowid: row.rowid,
                        preview: text.slice(0, 40),
                    });
                    continue;
                }
                // Scope check: the chat must be one of the operator's own
                // self-chat threads (typically Note-to-Self, where the
                // chat_identifier is the operator's own phone/email).
                const chatId = allowlist._normalize(row.chat_identifier);
                if (!chatId || !selfHandles.has(chatId)) {
                    logger.debug('self-chat-drop-not-self-chat', {
                        rowid: row.rowid,
                        chat_identifier: row.chat_identifier,
                        selfHandlesSize: selfHandles.size,
                    });
                    continue;
                }
                handle = chatId;
            } else {
                // bot mode (legacy): drop our own outbound, accept anything
                // else and let the dispatcher's allowlist gate decide.
                if (fromMe) {
                    logger.debug('bot-mode-drop-from-me', { rowid: row.rowid });
                    continue;
                }
                handle = _normalizeAppleHandle(row.handle_id);
                if (!handle) continue;
            }

            try {
                await dispatcher.handleInbound({
                    platform: PLATFORM,
                    handle,
                    text,
                    meta: { rowid: row.rowid, chat_guid: row.chat_guid, mode },
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

    // iMessage outbound is text-only. We deliberately omit sendImage so
    // the router falls back to text+caption when an attachment was
    // requested (see outboundRouter.sendImageToActive). Images over SMS
    // fallback are unreliable enough that we'd rather skip cleanly.
    dispatcher.registerSender(PLATFORM, {
        sendText: async (handle, text) => {
            // Always prefix outbound. In self-chat mode this is what the
            // tick loop uses to recognise + drop our own echoes via
            // chat.db sync. In bot mode the prefix is cosmetic but keeps
            // both modes byte-identical so operators don't see different
            // formatting when they flip modes.
            const body = replyPrefix.wrap(text);
            try {
                await _send(handle, body);
                logger.info('sent', { handle, length: body.length, mode });
            } catch (err) {
                logger.error('send-failed', { handle, error: err.message });
                throw err;
            }
        },
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
