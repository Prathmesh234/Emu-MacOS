// services/websocket.js — WebSocket connection management
//
// Messages are queued and processed one-at-a-time so that the async
// handler (executeAction → continueLoop) finishes before the next
// WS message is handled.  This prevents race conditions where two
// handlers manipulate the DOM and psProcess concurrently.

const store = require('../state/store');

const WS_URL = 'ws://127.0.0.1:8000';

let onMessageHandler = null;
let _closing = false;
// The socket the app currently wants live. initWebSocket supersedes any
// previous one so a session switch can't leave a second socket open.
let _activeWs = null;

// ── Serial message queue ──────────────────────────────────────────────
const _queue = [];
let _processing = false;

async function _processQueue() {
    if (_processing) return;        // another call is already draining
    _processing = true;

    while (_queue.length > 0) {
        const data = _queue.shift();
        try {
            if (onMessageHandler) await onMessageHandler(data);
        } catch (err) {
            console.error('[ws] handler error:', err);
        }
    }

    _processing = false;
}

// ── Public API ────────────────────────────────────────────────────────
function initWebSocket(sessionId) {
    // Supersede any existing connection. Without this, switching sessions
    // (initSession → continuePastSession both call initWebSocket) would leak
    // the previous socket: it keeps pushing into the shared _queue — so old-
    // session events get handled as the current session — and its onclose
    // reconnect loop resurrects the abandoned session forever. Detach the
    // handlers first so closing it can't enqueue a spurious 'connection_closed'
    // or schedule a reconnect.
    if (_activeWs) {
        _activeWs.onopen = _activeWs.onmessage = _activeWs.onerror = _activeWs.onclose = null;
        try { _activeWs.close(); } catch (_) {}
        _activeWs = null;
    }
    // A fresh connection is explicitly wanted, so clear any shutdown flag a
    // prior closeWebSocket() left set.
    _closing = false;

    const { readAuthToken } = require('../emu/root');
    const token = readAuthToken();
    const ws = new WebSocket(`${WS_URL}/ws/${sessionId}?token=${encodeURIComponent(token)}`);
    _activeWs = ws;

    ws.onopen = () => {
        console.log('[ws] connected');
        _queue.push({ type: 'connection_open' });
        _processQueue();
    };

    ws.onclose = () => {
        // Ignore closes from a superseded socket or during shutdown.
        if (_closing || ws !== _activeWs) return;
        console.log('[ws] closed — reconnecting in 2s');
        _queue.push({ type: 'connection_closed' });
        _processQueue();
        setTimeout(() => {
            if (!_closing && ws === _activeWs) initWebSocket(sessionId);
        }, 2000);
    };

    ws.onerror = (e) => console.warn('[ws] error', e);

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            _queue.push(data);
            _processQueue();          // fire-and-forget; queue serialises
        } catch (e) {
            console.warn('[ws] bad JSON:', event.data);
        }
    };

    store.setWebSocket(ws);
}

function setMessageHandler(handler) {
    onMessageHandler = handler;
}

function closeWebSocket() {
    _closing = true;
    const ws = store.state.ws;
    if (ws) {
        try { ws.close(); } catch (_) {}
    }
}

module.exports = { initWebSocket, setMessageHandler, closeWebSocket };
