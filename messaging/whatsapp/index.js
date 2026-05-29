// messaging/whatsapp/index.js — WhatsApp bridge via @whiskeysockets/Baileys
//
// Pairing: first run prints a QR to the bridge log (and stderr). The user
// scans it from WhatsApp → Settings → Linked Devices. After that, the
// auth state is persisted under .emu/messaging/whatsapp/auth/ and the
// bridge reconnects automatically.
//
// We never use a phone-number-based "pairing code" flow; QR pair only,
// because the pairing-code path requires registering the device against
// a phone number which is awkward to script and a worse security trade.

const fs = require('fs');
const path = require('path');
const { platformDir } = require('../common/paths');
const { makeLogger } = require('../common/log');
const allowlist = require('../common/allowlist');
const replyPrefix = require('../common/replyPrefix');

const PLATFORM = 'whatsapp';

// MODE: "bot" (default, legacy) or "self-chat".
//
//   bot       — the bridge runs as a dedicated WhatsApp account (the
//               recommended production deployment, per Hermes' own docs).
//               We drop every msg.key.fromMe message because anything we
//               sent is, by definition, our own reply.
//
//   self-chat — the bridge runs on the operator's PERSONAL WhatsApp
//               account, and the only valid command surface is the
//               operator messaging themselves in their own self-chat
//               thread. We accept msg.key.fromMe ONLY when the chat is
//               the operator's own JID, and we reject every non-fromMe
//               message so randos who happen to message the operator's
//               personal number can never trigger the agent.
//
// Pattern adopted from NousResearch/hermes-agent
// scripts/whatsapp-bridge/bridge.js (MIT) — the only OSS agent we found
// that solved single-user self-messaging without a second SIM.
function _mode() {
    const raw = String(process.env.EMU_MESSAGING_WHATSAPP_MODE || 'bot').toLowerCase();
    return raw === 'self-chat' ? 'self-chat' : 'bot';
}

let baileys; // lazy require so a missing dep on selfcheck doesn't crash
function _lazyLoadBaileys() {
    if (!baileys) {
        baileys = require('@whiskeysockets/baileys');
    }
    return baileys;
}

function _jidToHandle(jid) {
    // "15551234567@s.whatsapp.net" → "+15551234567"
    if (typeof jid !== 'string') return '';
    const at = jid.indexOf('@');
    const local = at === -1 ? jid : jid.slice(0, at);
    const digits = local.replace(/[^0-9]/g, '');
    return digits ? `+${digits}` : '';
}

function _isGroupJid(jid) {
    return typeof jid === 'string' && jid.endsWith('@g.us');
}

// Normalize a JID to its bare form: strip the ":N" device suffix Baileys
// appends to its own account JID, then lowercase the domain. Used to
// compare "is this chat the user's own self-chat?".
//
// Examples:
//   "14092392410:7@s.whatsapp.net" → "14092392410@s.whatsapp.net"
//   "67504034582761@lid"           → "67504034582761@lid"      (no change)
//   "14092392410@s.whatsapp.net"   → "14092392410@s.whatsapp.net"
function _bareJid(jid) {
    if (typeof jid !== 'string' || !jid) return '';
    const at = jid.indexOf('@');
    if (at === -1) return jid;
    const local = jid.slice(0, at).split(':')[0];
    const domain = jid.slice(at + 1).toLowerCase();
    return `${local}@${domain}`;
}

// Build the set of JIDs that count as "the operator's own chat" for self-
// chat mode. We always include the classic `<user.id>` and the newer LID
// form (`<user.lid>`) because WhatsApp routes some accounts through @lid
// after the privacy-preserving Login ID rollout. The user-supplied
// EMU_MESSAGING_WHATSAPP_SELF_JIDS lets operators add aliases (e.g. an
// alternate number that also hits their self-chat) without code changes.
function _buildSelfJidSet(sockUser, logger) {
    const set = new Set();
    function add(j) {
        const bare = _bareJid(j);
        if (bare) set.add(bare);
    }
    if (sockUser && typeof sockUser === 'object') {
        if (sockUser.id) add(sockUser.id);
        if (sockUser.lid) add(sockUser.lid);
    }
    const extra = String(process.env.EMU_MESSAGING_WHATSAPP_SELF_JIDS || '').trim();
    if (extra) {
        for (const piece of extra.split(',')) {
            const trimmed = piece.trim();
            if (trimmed) add(trimmed);
        }
    }
    if (set.size === 0 && logger) {
        logger.warn('self-chat-no-self-jid-resolved', {
            hint: 'set EMU_MESSAGING_WHATSAPP_SELF_JIDS=14092392410@s.whatsapp.net',
        });
    }
    return set;
}

// Resolve the canonical "+phone" handle for self-chat dispatch. Per-message
// remoteJid can arrive either as the classic phone JID (15551234567@s.whatsapp.net)
// OR as a meaningless LID (67504034582761@lid) — the LID is a privacy-
// preserving identifier with NO relationship to the phone number, so
// digit-extracting it gives garbage like "+67504034582761" that will never
// be in the operator's allowlist.
//
// In self-chat mode every accepted message routes back to the same operator,
// so we resolve their handle ONCE (preferring sockUser.id which is the
// classic phone JID, falling back to allowlist[0] if even that is LID-only).
function _resolveSelfHandle(sockUser, allowlistEntries, logger) {
    if (sockUser && typeof sockUser === 'object' && typeof sockUser.id === 'string') {
        const bare = _bareJid(sockUser.id);
        // Only trust the JID-derived handle if it's NOT a LID — LID digits
        // are not a phone number and we'd be lying to the dispatcher.
        if (bare && !bare.endsWith('@lid')) {
            const h = _jidToHandle(bare);
            if (h) return h;
        }
    }
    // Fallback: the operator opted into self-chat by allowlisting their own
    // number, so the first WhatsApp allowlist entry is by definition them.
    const fallback = (allowlistEntries || []).find((h) => typeof h === 'string' && h.startsWith('+'));
    if (fallback) return fallback;
    if (logger) {
        logger.warn('self-chat-no-self-handle-resolved', {
            hint: 'add your +E.164 number to the WhatsApp allowlist',
        });
    }
    return '';
}


function _extractText(msg) {
    const m = msg && msg.message;
    if (!m) return '';
    if (m.conversation) return m.conversation;
    if (m.extendedTextMessage && m.extendedTextMessage.text) return m.extendedTextMessage.text;
    if (m.imageMessage && m.imageMessage.caption) return m.imageMessage.caption;
    return '';
}

async function startWhatsApp({ dispatcher }) {
    const logger = makeLogger(PLATFORM);
    if (allowlist.loadAllowlist().whatsapp.length === 0) {
        logger.info('skipping-no-allowlist-entries');
        return { stop() {} };
    }

    let api;
    try {
        api = _lazyLoadBaileys();
    } catch (err) {
        logger.error('baileys-load-failed', { error: err.message });
        return { stop() {} };
    }

    const { default: makeWASocket, useMultiFileAuthState, DisconnectReason, fetchLatestBaileysVersion } = api;
    const authDir = path.join(platformDir(PLATFORM), 'auth');
    fs.mkdirSync(authDir, { recursive: true, mode: 0o700 });
    try { fs.chmodSync(authDir, 0o700); } catch (_) { /* ignore */ }

    const { state, saveCreds } = await useMultiFileAuthState(authDir);

    let version;
    try {
        ({ version } = await fetchLatestBaileysVersion());
    } catch (err) {
        logger.warn('version-fetch-failed-using-baked-in', { error: err.message });
    }

    // Baileys wants a pino logger with a `child` method. Wrap our logger.
    let pinoLogger;
    try {
        pinoLogger = require('pino')({ level: 'warn' });
    } catch (_) {
        // pino is a peer dep of Baileys; if missing, the require above
        // already threw before we get here.
        pinoLogger = { level: 'warn', child: () => pinoLogger, fatal() {}, error() {}, warn() {}, info() {}, debug() {}, trace() {} };
    }

    let sock = null;
    let stopped = false;
    let qrShown = false;
    const mode = _mode();
    // self-chat mode bookkeeping:
    //   selfJids   — set of bare JIDs that count as "my own self-chat"; we
    //                rebuild this every time the socket reconnects because
    //                `sock.user` only populates on `connection: open`.
    //   selfHandle — canonical "+phone" form for dispatcher routing,
    //                resolved once at socket open. We do NOT derive it
    //                per-message because msg.key.remoteJid can be a LID
    //                (privacy-preserving identifier with garbage digits).
    //   recentlySent — IDs of messages we just sent, used to fast-drop
    //                echo-backs on the rare path where prefix detection fails
    //                (e.g. transport-level re-encoding strips the leading glyph).
    let selfJids = new Set();
    let selfHandle = '';
    const recentlySent = replyPrefix.createRecentlySentCache(200);
    logger.info('mode-selected', { mode });

    function _attach(socket) {
        socket.ev.on('creds.update', saveCreds);

        socket.ev.on('connection.update', (update) => {
            const { connection, lastDisconnect, qr } = update;
            if (qr && !qrShown) {
                qrShown = true;
                try {
                    const qrt = require('qrcode-terminal');
                    qrt.generate(qr, { small: true });
                } catch (_) { /* fall back below */ }
                logger.info('qr-ready', { hint: 'WhatsApp → Settings → Linked Devices → Link a Device' });
            }
            if (connection === 'open') {
                qrShown = false;
                selfJids = _buildSelfJidSet(socket.user, logger);
                if (mode === 'self-chat') {
                    selfHandle = _resolveSelfHandle(
                        socket.user,
                        (allowlist.loadAllowlist().whatsapp || []),
                        logger,
                    );
                }
                logger.info('connected', {
                    user: socket.user?.id,
                    mode,
                    selfJids: mode === 'self-chat' ? Array.from(selfJids) : undefined,
                    selfHandle: mode === 'self-chat' ? selfHandle : undefined,
                });
            }
            if (connection === 'close') {
                const statusCode = lastDisconnect?.error?.output?.statusCode;
                const shouldReconnect = statusCode !== DisconnectReason?.loggedOut;
                logger.warn('disconnected', { statusCode, shouldReconnect });
                if (statusCode === DisconnectReason?.loggedOut) {
                    // Write a flag so the operator knows to re-pair.
                    const flag = path.join(platformDir(PLATFORM), 'REPAIR_NEEDED');
                    try { fs.writeFileSync(flag, new Date().toISOString(), { mode: 0o600 }); } catch (_) {}
                }
                if (!stopped && shouldReconnect) {
                    setTimeout(() => { if (!stopped) sock = _open(); }, 3_000);
                }
            }
        });

        socket.ev.on('messages.upsert', async ({ messages, type }) => {
            // In `bot` mode we only care about live inbound (`notify`).
            // In `self-chat` mode the operator's own sends arrive as
            // `append` events (the protocol's "your other linked device
            // just sent something" delivery path), so we MUST accept both
            // event types. This is exactly the Hermes bridge's fix.
            if (type !== 'notify' && type !== 'append') return;
            for (const msg of messages || []) {
                try {
                    if (!msg.message) continue;
                    const jid = msg.key?.remoteJid || '';
                    if (_isGroupJid(jid)) {
                        logger.debug('skip-group', { jid });
                        continue;
                    }
                    const fromMe = Boolean(msg.key?.fromMe);
                    const bareChat = _bareJid(jid);
                    let handle;

                    if (mode === 'self-chat') {
                        // Scope: in self-chat mode, ONLY the user's own
                        // self-chat thread is a valid command surface.
                        if (!fromMe) {
                            logger.debug('self-chat-drop-not-from-me', { jid });
                            continue;
                        }
                        if (!selfJids.has(bareChat)) {
                            logger.debug('self-chat-drop-not-self-chat', {
                                jid, bareChat, selfJidsSize: selfJids.size,
                            });
                            continue;
                        }
                        // Echo suppression layer 1 (ID cache).
                        const msgId = msg.key?.id;
                        if (msgId && recentlySent.has(msgId)) {
                            logger.debug('self-chat-drop-our-own-id', { msgId });
                            continue;
                        }
                        // Echo suppression layer 2 (reply prefix).
                        const text = _extractText(msg);
                        if (!text) continue;
                        if (replyPrefix.isPrefixed(text)) {
                            logger.debug('self-chat-drop-prefixed-echo', {
                                preview: text.slice(0, 40),
                            });
                            continue;
                        }
                        // In self-chat mode the route target IS the
                        // operator themselves. We use the canonical
                        // selfHandle resolved at socket-open rather than
                        // _jidToHandle(bareChat), because bareChat can be
                        // a LID like "67504034582761@lid" whose digits are
                        // NOT a phone number and won't match the allowlist.
                        handle = selfHandle;
                        if (!handle) {
                            logger.warn('self-chat-no-handle-derived', { jid, bareChat });
                            continue;
                        }
                        await dispatcher.handleInbound({
                            platform: PLATFORM,
                            handle,
                            text,
                            meta: { jid, mode: 'self-chat' },
                        });
                    } else {
                        // bot mode (legacy): drop our own outbound, accept
                        // everything else. Allowlist gate happens downstream.
                        if (fromMe) continue;
                        handle = _jidToHandle(jid);
                        const text = _extractText(msg);
                        if (!text) continue;
                        await dispatcher.handleInbound({
                            platform: PLATFORM,
                            handle,
                            text,
                            meta: { jid },
                        });
                    }
                } catch (err) {
                    logger.error('inbound-failed', { error: err.message });
                }
            }
        });
    }

    function _open() {
        const socket = makeWASocket({
            version,
            auth: state,
            logger: pinoLogger,
            printQRInTerminal: false, // we render via qrcode-terminal ourselves
            browser: ['Emu', 'Desktop', '1.0'],
            // Conservatively keep history sync off so we don't pull years of
            // chat into the local cache; we only need new inbound messages.
            shouldSyncHistoryMessage: () => false,
            markOnlineOnConnect: false,
        });
        _attach(socket);
        return socket;
    }

    sock = _open();

    dispatcher.registerSender(PLATFORM, {
        sendText: async (handle, text) => {
            if (!sock || !sock.user) {
                logger.warn('send-skipped-no-socket', { handle });
                return;
            }
            const digits = String(handle || '').replace(/[^0-9]/g, '');
            if (!digits) {
                logger.warn('send-skipped-bad-handle', { handle });
                return;
            }
            const jid = `${digits}@s.whatsapp.net`;
            // In self-chat mode, prefix every outbound so we can recognise
            // and drop our own echoes when they bounce back as `append`
            // events. In bot mode the prefix is harmless cosmetic; we add
            // it unconditionally so the agent's voice is identifiable.
            const body = replyPrefix.wrap(text);
            try {
                const result = await sock.sendMessage(jid, { text: body });
                // Belt + suspenders: cache the WA-assigned msg ID so we
                // can drop the echo even if a future server re-encoding
                // strips the prefix.
                const sentId = result?.key?.id;
                if (sentId) recentlySent.add(sentId);
                logger.info('sent', { handle, length: body.length, mode });
            } catch (err) {
                logger.error('send-failed', { handle, error: err.message });
                throw err;
            }
        },
        sendImage: async (handle, image, caption) => {
            if (!sock || !sock.user) {
                logger.warn('send-image-skipped-no-socket', { handle });
                return;
            }
            const digits = String(handle || '').replace(/[^0-9]/g, '');
            if (!digits) {
                logger.warn('send-image-skipped-bad-handle', { handle });
                return;
            }
            if (!image || !Buffer.isBuffer(image.buffer)) {
                logger.warn('send-image-skipped-bad-payload', { handle });
                return;
            }
            const jid = `${digits}@s.whatsapp.net`;
            // Baileys accepts a Buffer for the `image` field. WhatsApp
            // re-encodes server-side, so we don't need to convert mime
            // types; PNG and JPEG both work.
            const msg = { image: image.buffer };
            // Always wrap captions with the reply prefix in self-chat
            // mode, otherwise the bridge would echo an attachment whose
            // caption looks like a user command. Empty caption stays empty
            // (no prefix-only "ghost" message).
            if (caption) {
                msg.caption = replyPrefix.wrap(caption);
            }
            try {
                const result = await sock.sendMessage(jid, msg);
                const sentId = result?.key?.id;
                if (sentId) recentlySent.add(sentId);
                logger.info('sent-image', {
                    handle,
                    bytes: image.buffer.length,
                    captionLength: msg.caption ? msg.caption.length : 0,
                    mode,
                });
            } catch (err) {
                logger.error('send-image-failed', { handle, error: err.message });
                throw err;
            }
        },
    });

    return {
        stop() {
            stopped = true;
            try { sock && sock.end && sock.end(); } catch (_) { /* ignore */ }
            logger.info('stopped');
        },
    };
}

module.exports = { startWhatsApp };
