// messaging/common/agentClient.js — HTTP + WS client for the local agent
//
// Wraps the loopback FastAPI backend. The only auth surface is the
// per-launch X-Emu-Token, same as the Electron renderer uses. We open a
// long-lived WebSocket per session id and dispatch streamed events to
// the caller-supplied handler.

const fs = require('fs');
const WebSocket = require('ws');
const { authTokenPath } = require('./paths');

const BACKEND_URL = 'http://127.0.0.1:8000';
const WS_URL = 'ws://127.0.0.1:8000';

function readToken() {
    try {
        return fs.readFileSync(authTokenPath(), 'utf8').trim();
    } catch {
        return '';
    }
}

function _headers() {
    return {
        'Content-Type': 'application/json',
        'X-Emu-Token': readToken(),
    };
}

async function _fetch(method, path, body, { timeoutMs = 15_000 } = {}) {
    const controller = new AbortController();
    const timer = timeoutMs > 0 ? setTimeout(() => controller.abort(), timeoutMs) : null;
    try {
        const res = await fetch(`${BACKEND_URL}${path}`, {
            method,
            headers: _headers(),
            body: body ? JSON.stringify(body) : undefined,
            signal: controller.signal,
        });
        if (!res.ok) {
            const text = await res.text().catch(() => '');
            const err = new Error(`${method} ${path} failed: ${res.status} ${text}`);
            err.status = res.status;
            throw err;
        }
        return res.json().catch(() => ({}));
    } finally {
        if (timer) clearTimeout(timer);
    }
}

async function createSession() {
    const data = await _fetch('POST', '/agent/session');
    if (!data.session_id) throw new Error('createSession: missing session_id');
    return data.session_id;
}

async function setSessionMetadata(sessionId, metadata) {
    return _fetch('POST', `/agent/session/${encodeURIComponent(sessionId)}/metadata`, metadata);
}

async function postStep({ sessionId, userMessage, source, agentMode = 'remote' }) {
    // Do NOT timeout the step request. Agent steps can chain many tool calls
    // before returning; the bridge tracks completion via the WebSocket
    // `done` event, not by the HTTP response.
    const res = await fetch(`${BACKEND_URL}/agent/step`, {
        method: 'POST',
        headers: _headers(),
        body: JSON.stringify({
            session_id: sessionId,
            user_message: userMessage || '',
            base64_screenshot: '',
            agent_mode: agentMode,
            source,
        }),
    });
    if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`postStep failed: ${res.status} ${text}`);
    }
    return res.json().catch(() => ({}));
}

// Best-effort stop of any in-flight step on the given session. Used by the
// bridge when an allowlisted user issues `/new` mid-turn — we want the
// model loop on the OLD session to bail out instead of continuing to act
// on whatever task the user just abandoned.
async function stopSession(sessionId, { timeoutMs = 5_000 } = {}) {
    const controller = new AbortController();
    const timer = timeoutMs > 0 ? setTimeout(() => controller.abort(), timeoutMs) : null;
    try {
        const res = await fetch(`${BACKEND_URL}/agent/stop`, {
            method: 'POST',
            headers: _headers(),
            body: JSON.stringify({ session_id: sessionId }),
            signal: controller.signal,
        });
        if (!res.ok) {
            const text = await res.text().catch(() => '');
            throw new Error(`stopSession failed: ${res.status} ${text}`);
        }
        return res.json().catch(() => ({}));
    } finally {
        if (timer) clearTimeout(timer);
    }
}

// Returns { mime, buffer } for the most recent screenshot on the session,
// or null when none is cached. Used by the messaging bridge to attach a
// current desktop frame to outbound replies on WhatsApp.
async function fetchLatestScreenshot(sessionId, { timeoutMs = 10_000 } = {}) {
    const controller = new AbortController();
    const timer = timeoutMs > 0 ? setTimeout(() => controller.abort(), timeoutMs) : null;
    try {
        const id = encodeURIComponent(sessionId);
        const res = await fetch(`${BACKEND_URL}/sessions/${id}/latest_screenshot`, {
            method: 'GET',
            headers: { 'X-Emu-Token': readToken() },
            signal: controller.signal,
        });
        if (res.status === 404) return null;
        if (!res.ok) {
            throw new Error(`latest_screenshot failed: ${res.status}`);
        }
        const mime = res.headers.get('content-type') || 'image/png';
        const arrBuf = await res.arrayBuffer();
        return { mime, buffer: Buffer.from(arrBuf) };
    } finally {
        if (timer) clearTimeout(timer);
    }
}

// connectStream(sessionId, handler) → { close }
// handler({ type, message, ... }) receives every WS frame for the session.
// Reconnects automatically with exponential backoff (capped at 30s).
function connectStream(sessionId, handler, logger) {
    let ws = null;
    let closed = false;
    let backoff = 1000;
    const id = encodeURIComponent(sessionId);

    function open() {
        if (closed) return;
        const token = readToken();
        ws = new WebSocket(`${WS_URL}/ws/${id}?token=${encodeURIComponent(token)}`);

        ws.on('open', () => {
            backoff = 1000;
            if (logger) logger.info('ws-connected', { sessionId });
        });
        ws.on('message', (raw) => {
            let payload;
            try { payload = JSON.parse(raw.toString('utf8')); } catch { return; }
            try { handler(payload); } catch (err) {
                if (logger) logger.error('ws-handler-threw', { error: err.message });
            }
        });
        ws.on('close', (code, reason) => {
            if (logger) logger.warn('ws-closed', { code, reason: String(reason || '') });
            if (closed) return;
            setTimeout(open, backoff);
            backoff = Math.min(30_000, backoff * 2);
        });
        ws.on('error', (err) => {
            if (logger) logger.warn('ws-error', { error: err.message });
            // 'close' will follow; let it drive reconnection.
        });
    }

    open();

    return {
        close() {
            closed = true;
            try { ws && ws.close(); } catch (_) { /* ignore */ }
        },
    };
}

module.exports = {
    createSession,
    setSessionMetadata,
    postStep,
    stopSession,
    fetchLatestScreenshot,
    connectStream,
    BACKEND_URL,
};
