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

const PLATFORM = 'whatsapp';

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
                logger.info('connected', { user: socket.user?.id });
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
            if (type !== 'notify') return;
            for (const msg of messages || []) {
                try {
                    if (!msg.message) continue;
                    if (msg.key?.fromMe) continue;
                    const jid = msg.key?.remoteJid || '';
                    if (_isGroupJid(jid)) {
                        logger.debug('skip-group', { jid });
                        continue;
                    }
                    const handle = _jidToHandle(jid);
                    const text = _extractText(msg);
                    if (!text) continue;
                    await dispatcher.handleInbound({
                        platform: PLATFORM,
                        handle,
                        text,
                        meta: { jid },
                    });
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

    dispatcher.registerSender(PLATFORM, async (handle, text) => {
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
        try {
            await sock.sendMessage(jid, { text });
            logger.info('sent', { handle, length: text.length });
        } catch (err) {
            logger.error('send-failed', { handle, error: err.message });
            throw err;
        }
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
