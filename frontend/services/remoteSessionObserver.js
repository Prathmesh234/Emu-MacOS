// services/remoteSessionObserver.js — passive WS observer for the
// singleton remote-control session created by the messaging bridge.
//
// The desktop UI already keeps a primary WebSocket against the user's
// local session id (services/websocket.js, dispatched into Chat.js's
// handleWsMessage). For WhatsApp / iMessage parity we want activity on
// the remote-control session to surface in the *same* History sidebar
// as a local session would — pulsing dot, live preview, refreshed list.
//
// This module owns the SECOND WebSocket and ONLY reports lifecycle
// events back to the caller. It deliberately does NOT:
//   - dispatch actions (those run server-side via emu-cua-driver)
//   - mutate store.state.ws (that belongs to the user's local session)
//   - render anything (HistoryPanel + Chat.js do that)
//
// Public API:
//   start({ onActivity, onStateChange }, logger?) → { stop, sessionId }
//   onActivity({ type, sessionId, preview }) — every step / done / etc.
//   onStateChange({ live: bool, sessionId })  — fires when the session
//       transitions between idle and "currently generating".

const api = require('./api');
const { readAuthToken } = require('../emu/root');

const WS_URL = 'ws://127.0.0.1:8000';

// How long after the last event we consider the session "idle". A bit
// generous because backend step events can be sparse during long tool
// chains (e.g. waiting on an Electron action round-trip).
const LIVE_IDLE_GRACE_MS = 8_000;

// Retry cadence when no remote session exists yet (bridge not running,
// allowlist empty, never paired). We keep retrying so the desktop UI
// picks up the session as soon as the user finishes pairing WhatsApp
// without requiring an app restart.
const DISCOVERY_RETRY_MS = 15_000;

// Cadence at which we re-check the bridge's `remote_session.json` for a
// rotated session id. When an allowlisted user sends `/new`, the bridge
// mints a fresh session and demotes the previous one — but our WebSocket
// is still pointed at the old id and will go silent forever. This poll
// lets the desktop UI hot-swap onto the new id without an app restart.
const ROTATION_POLL_MS = 15_000;

function _stripPlatformPrefix(text) {
    // Inbound bridge messages get a "[whatsapp:+15551234567] ..." prefix.
    // When we surface them in the desktop sidebar we want the human text
    // only, so strip the routing tag for display.
    return String(text || '').replace(/^\[[a-z]+:[^\]]+\]\s*/i, '').trim();
}

function _previewFromEvent(payload) {
    if (!payload || typeof payload !== 'object') return '';
    if (payload.type === 'done')   return _stripPlatformPrefix(payload.message);
    if (payload.type === 'status') return _stripPlatformPrefix(payload.message);
    if (payload.type === 'error')  return `Error: ${payload.message || 'unknown'}`;
    if (payload.type === 'step') {
        const reasoning = payload.reasoning_content || payload.final_message || '';
        return _stripPlatformPrefix(reasoning).slice(0, 140);
    }
    return '';
}

function _isLiveEvent(payload) {
    if (!payload || typeof payload !== 'object') return false;
    return payload.type === 'step' || payload.type === 'status';
}

function _isTerminalEvent(payload) {
    if (!payload || typeof payload !== 'object') return false;
    return payload.type === 'done'
        || payload.type === 'stopped'
        || payload.type === 'error';
}

function start({ onActivity, onStateChange } = {}, logger = console) {
    let ws = null;
    let stopped = false;
    let sessionId = null;
    let live = false;
    let idleTimer = null;
    let backoff = 1000;
    let retryTimer = null;
    let rotationTimer = null;

    const _emitActivity = (type, payload) => {
        if (!onActivity) return;
        try {
            onActivity({
                type,
                sessionId,
                preview: _previewFromEvent(payload),
                raw: payload,
            });
        } catch (err) {
            logger.warn('[remote-observer] onActivity threw:', err.message);
        }
    };

    const _setLive = (next) => {
        if (next === live) return;
        live = next;
        if (onStateChange) {
            try {
                onStateChange({ live, sessionId });
            } catch (err) {
                logger.warn('[remote-observer] onStateChange threw:', err.message);
            }
        }
    };

    const _bumpIdleTimer = () => {
        if (idleTimer) clearTimeout(idleTimer);
        idleTimer = setTimeout(() => { _setLive(false); }, LIVE_IDLE_GRACE_MS);
    };

    const _connect = (id) => {
        if (stopped) return;
        const token = readAuthToken();
        const url = `${WS_URL}/ws/${encodeURIComponent(id)}?token=${encodeURIComponent(token)}`;
        try {
            ws = new WebSocket(url);
        } catch (err) {
            logger.warn('[remote-observer] ws-construct-failed:', err.message);
            return;
        }

        ws.onopen = () => {
            backoff = 1000;
            logger.log('[remote-observer] connected', id);
        };

        ws.onmessage = (event) => {
            let payload;
            try { payload = JSON.parse(event.data); } catch { return; }
            _emitActivity(payload.type || 'unknown', payload);
            if (_isLiveEvent(payload)) {
                _setLive(true);
                _bumpIdleTimer();
            } else if (_isTerminalEvent(payload)) {
                _setLive(false);
                if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
            }
        };

        ws.onerror = () => { /* close will follow; we reconnect there */ };

        ws.onclose = () => {
            if (stopped) return;
            const delay = backoff;
            backoff = Math.min(30_000, backoff * 2);
            setTimeout(() => _connect(sessionId || id), delay);
        };
    };

    // Tear down the current WS (without triggering a reconnect) and open
    // a new one against `nextId`. Used when the bridge rotates the
    // singleton session via /new — we don't want the user to see a
    // forever-stale "Remote control" pin after the rotation.
    const _swapTo = (nextId) => {
        if (!nextId || nextId === sessionId) return;
        const previousId = sessionId;
        logger.log('[remote-observer] rotation detected', { previous: previousId, next: nextId });
        // Notify the caller that the previous session is no longer live
        // BEFORE we change sessionId, so HistoryPanel.setLive uses the
        // right id to clear the badge on the old pinned item.
        if (live) _setLive(false);
        if (onStateChange && previousId) {
            try { onStateChange({ live: false, sessionId: previousId, rotated: true }); }
            catch (err) { logger.warn('[remote-observer] onStateChange threw:', err.message); }
        }
        sessionId = nextId;
        backoff = 1000;
        if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
        // Close the old WS; its `onclose` handler is a no-op because we
        // null out `ws` first so the reconnect targets the NEW id.
        const oldWs = ws;
        ws = null;
        try { oldWs && oldWs.close(); } catch (_) { /* ignore */ }
        _connect(nextId);
    };

    const _pollRotation = async () => {
        if (stopped) return;
        try {
            const current = await api.fetchRemoteSessionId();
            if (current && current !== sessionId) _swapTo(current);
        } catch (_) {
            // Backend unreachable; the regular WS reconnect logic and the
            // next poll tick will recover.
        }
        if (!stopped) rotationTimer = setTimeout(_pollRotation, ROTATION_POLL_MS);
    };

    const _discoverThenConnect = async () => {
        if (stopped) return;
        try {
            sessionId = await api.fetchRemoteSessionId();
        } catch (err) {
            sessionId = null;
        }
        if (stopped) return;
        if (!sessionId) {
            // Bridge hasn't provisioned a session yet (no allowlist, never
            // paired). Retry on a slow cadence so the user doesn't have to
            // restart the desktop app after editing allowlist.json.
            retryTimer = setTimeout(_discoverThenConnect, DISCOVERY_RETRY_MS);
            return;
        }
        _connect(sessionId);
        // Begin watching for `/new` rotations now that we have an initial id.
        rotationTimer = setTimeout(_pollRotation, ROTATION_POLL_MS);
    };

    _discoverThenConnect();

    return {
        stop() {
            stopped = true;
            if (idleTimer) clearTimeout(idleTimer);
            if (retryTimer) clearTimeout(retryTimer);
            if (rotationTimer) clearTimeout(rotationTimer);
            try { ws && ws.close(); } catch (_) { /* ignore */ }
        },
        get sessionId() { return sessionId; },
        get live() { return live; },
    };
}

module.exports = { start };
