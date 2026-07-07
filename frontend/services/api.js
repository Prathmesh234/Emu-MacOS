// services/api.js — HTTP API calls to backend

const fs = require('fs');
const { authTokenPath } = require('../emu/root');

const BACKEND_URL = 'http://127.0.0.1:8000';

// Default per-request timeout. Long enough for slow LLM steps but bounded
// so a hung backend doesn't leave fetches pending forever.
const DEFAULT_TIMEOUT_MS = 30_000;

// Auth token path resolves via EMU_ROOT so it works in both source-checkout
// and packaged-DMG layouts.
const TOKEN_PATH = authTokenPath();
let pendingProviderSettingsSave = null;
let providerSettingsSaveQueue = Promise.resolve();

function getToken() {
    try {
        return fs.readFileSync(TOKEN_PATH, 'utf8').trim();
    } catch {
        return '';
    }
}

function authHeaders(extra = {}) {
    return { 'Content-Type': 'application/json', 'X-Emu-Token': getToken(), ...extra };
}

// fetch() wrapper that aborts after `timeoutMs` and surfaces a consistent
// error message. Pass timeoutMs=0/null for long-lived requests whose lifecycle
// is driven by the websocket stream instead of the HTTP response.
async function fetchWithTimeout(url, opts = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
    if (timeoutMs == null || timeoutMs <= 0) {
        return fetch(url, opts);
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await fetch(url, { ...opts, signal: controller.signal });
    } catch (err) {
        if (err && err.name === 'AbortError') {
            throw new Error(`Request to ${url} timed out after ${timeoutMs}ms`);
        }
        throw err;
    } finally {
        clearTimeout(timer);
    }
}

async function createSession() {
    const res = await fetchWithTimeout(`${BACKEND_URL}/agent/session`, {
        method: 'POST',
        headers: authHeaders(),
    }, 10_000);
    if (!res.ok) {
        throw new Error(`Session creation failed: ${res.status} ${res.statusText}`);
    }
    const data = await res.json();
    if (!data.session_id) {
        throw new Error('Session response missing session_id');
    }
    return data.session_id;
}

async function postStep({ sessionId, userMessage, base64Screenshot, agentMode }) {
    if (pendingProviderSettingsSave) {
        // Wait for an in-flight settings save to land so the step uses the new
        // provider/model — but don't let a failed save reject the send itself,
        // or the user's message would silently never post under a confusing
        // "Failed to save provider settings" error.
        await pendingProviderSettingsSave.catch(() => {});
    }
    // Do not abort agent steps from the renderer. Coworker mode can run long
    // server-side tool chains while progress streams over WebSocket; aborting
    // this fetch only drops the UI back to "send" while the backend continues.
    const res = await fetchWithTimeout(`${BACKEND_URL}/agent/step`, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({
            session_id:        sessionId,
            user_message:      userMessage || '',
            base64_screenshot: base64Screenshot || '',
            agent_mode:        agentMode || 'coworker',
        }),
    }, 0);
    if (!res.ok) {
        const err = new Error(`Agent step failed: ${res.status} ${res.statusText}`);
        err.httpStatus = res.status;
        throw err;
    }
    return res;
}

async function notifyActionComplete({ sessionId, ipcChannel, success, error, output }) {
    return fetchWithTimeout(`${BACKEND_URL}/action/complete`, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({
            session_id: sessionId,
            ipc_channel: ipcChannel,
            success,
            error: error || null,
            output: output || null,
        }),
    }, 10_000);
}

async function stopAgent(sessionId) {
    return fetchWithTimeout(`${BACKEND_URL}/agent/stop`, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ session_id: sessionId }),
    }, 5_000);
}

async function compactContext(sessionId) {
    const res = await fetchWithTimeout(`${BACKEND_URL}/agent/compact`, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ session_id: sessionId }),
    }, 60_000);
    if (!res.ok) throw new Error(`Compact failed: ${res.status} ${res.statusText}`);
    return res.json();
}

async function fetchSessionHistory() {
    try {
        const res = await fetchWithTimeout(`${BACKEND_URL}/sessions/history`, {
            headers: authHeaders(),
        }, 10_000);
        if (!res.ok) return [];
        const data = await res.json();
        return data.sessions || [];
    } catch (err) {
        console.warn('[api] fetchSessionHistory failed:', err.message);
        return [];
    }
}

async function fetchSessionMessages(sessionId) {
    try {
        const res = await fetchWithTimeout(`${BACKEND_URL}/sessions/${encodeURIComponent(sessionId)}/messages`, {
            headers: authHeaders(),
        }, 10_000);
        if (!res.ok) return [];
        const data = await res.json();
        return data.messages || [];
    } catch (err) {
        console.warn('[api] fetchSessionMessages failed:', err.message);
        return [];
    }
}

async function continueSession(previousSessionId, agentMode = 'coworker') {
    const res = await fetchWithTimeout(`${BACKEND_URL}/agent/session/continue`, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({
            previous_session_id: previousSessionId,
            agent_mode: agentMode,
        }),
    }, 10_000);
    if (!res.ok) throw new Error(`Continue session failed: ${res.status} ${res.statusText}`);
    const data = await res.json();
    if (!data.session_id) throw new Error('Continue session response missing session_id');
    return data.session_id;
}

async function getProviderSettings() {
    const res = await fetchWithTimeout(`${BACKEND_URL}/settings/provider`, {
        headers: authHeaders(),
    }, 10_000);
    if (!res.ok) throw new Error(`Failed to get provider settings: ${res.status}`);
    return res.json();
}

async function getProviderModelOptions(provider) {
    const suffix = provider ? `?provider=${encodeURIComponent(provider)}` : '';
    const res = await fetchWithTimeout(`${BACKEND_URL}/settings/provider/models${suffix}`, {
        headers: authHeaders(),
    }, 10_000);
    if (!res.ok) throw new Error(`Failed to get provider models: ${res.status}`);
    return res.json();
}

async function saveProviderSettings({ provider, model, apiKey }) {
    const request = providerSettingsSaveQueue.catch(() => {}).then(async () => {
        const res = await fetchWithTimeout(`${BACKEND_URL}/settings/provider`, {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify({ provider, model, api_key: apiKey }),
        }, 10_000);
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `Failed to save provider settings: ${res.status}`);
        return data;
    });

    pendingProviderSettingsSave = request;
    providerSettingsSaveQueue = request.catch(() => {});
    try {
        return await request;
    } finally {
        if (pendingProviderSettingsSave === request) {
            pendingProviderSettingsSave = null;
        }
    }
}

// Returns the singleton remote-control session id provisioned by the
// messaging bridge, or null when the bridge hasn't run yet (no allowlist
// entries, never paired, etc.). The desktop UI uses this to open a
// passive observer WebSocket so WhatsApp / iMessage activity shows up in
// the History sidebar in real time.
async function fetchRemoteSessionId() {
    try {
        const res = await fetchWithTimeout(`${BACKEND_URL}/messaging/remote_session_id`, {
            headers: authHeaders(),
        }, 5_000);
        if (!res.ok) return null;
        const data = await res.json();
        return data.session_id || null;
    } catch (err) {
        console.warn('[api] fetchRemoteSessionId failed:', err.message);
        return null;
    }
}

module.exports = { BACKEND_URL, createSession, continueSession, postStep, notifyActionComplete, stopAgent, compactContext, fetchSessionHistory, fetchSessionMessages, fetchRemoteSessionId, getProviderSettings, getProviderModelOptions, saveProviderSettings };
