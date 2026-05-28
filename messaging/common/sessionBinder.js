// messaging/common/sessionBinder.js — singleton "remote control" session
//
// The whole point of the messaging bridge is that all inbound text from
// WhatsApp + iMessage funnels into ONE session that grows over time
// instead of spawning a new session per message. We persist the chosen
// session id under .emu/messaging/remote_session.json so that across
// app restarts the same id is reused — the History sidebar then shows
// one continuous "Remote control" conversation.
//
// Allowlisted users can rotate the session at any time by sending a
// "/new" / "/reset" / "/clear" message; rotate() mints a fresh session
// id, archives the previous one's pinned label (so it falls back into
// the regular date groups), and updates the cached + on-disk pointer.

const fs = require('fs');
const { remoteSessionFilePath } = require('./paths');
const agentClient = require('./agentClient');

const LABEL = 'Remote control (WhatsApp & iMessage)';
const KIND  = 'remote_control';

let _cached = null;

function _readFile() {
    const path = remoteSessionFilePath();
    if (!fs.existsSync(path)) return null;
    try {
        const data = JSON.parse(fs.readFileSync(path, 'utf8'));
        return typeof data.session_id === 'string' && data.session_id
            ? data.session_id
            : null;
    } catch {
        return null;
    }
}

function _writeFile(sessionId) {
    const path = remoteSessionFilePath();
    const body = {
        session_id: sessionId,
        kind: KIND,
        label: LABEL,
        created_at: new Date().toISOString(),
    };
    fs.writeFileSync(path, JSON.stringify(body, null, 2), { mode: 0o600 });
    try { fs.chmodSync(path, 0o600); } catch (_) { /* ignore */ }
}

// Returns the remote-control session id, creating + tagging it once on
// first call. Idempotent and concurrency-safe within a single process.
let _provisioning = null;
async function getRemoteSessionId(logger) {
    if (_cached) return _cached;
    const fromDisk = _readFile();
    if (fromDisk) {
        _cached = fromDisk;
        if (logger) logger.info('reused-remote-session', { sessionId: fromDisk });
        // Best-effort: re-stamp metadata on every start so a hand-edited
        // sessions/<id>/metadata.json that lost the kind tag heals itself.
        try {
            await agentClient.setSessionMetadata(fromDisk, { kind: KIND, label: LABEL });
        } catch (err) {
            if (logger) logger.warn('metadata-restamp-failed', { error: err.message });
        }
        return fromDisk;
    }
    if (!_provisioning) {
        _provisioning = (async () => {
            const id = await agentClient.createSession();
            await agentClient.setSessionMetadata(id, { kind: KIND, label: LABEL });
            _writeFile(id);
            _cached = id;
            if (logger) logger.info('provisioned-remote-session', { sessionId: id });
            return id;
        })();
    }
    return _provisioning;
}

function currentId() {
    return _cached || _readFile();
}

// Mint a fresh remote-control session and demote the previous one. Returns
// the new session id. Safe to await; the caller is responsible for tearing
// down + reopening the WS stream against the new id.
async function rotate(logger) {
    const previous = _cached || _readFile();
    const id = await agentClient.createSession();
    await agentClient.setSessionMetadata(id, { kind: KIND, label: LABEL });
    _writeFile(id);
    _cached = id;
    if (logger) logger.info('rotated-remote-session', { previous, sessionId: id });
    if (previous && previous !== id) {
        // Demote the old session: clear its "remote_control" pin so it
        // drops back into the regular date groups in the sidebar. We
        // intentionally leave conversation.json intact so history is
        // preserved.
        try {
            await agentClient.setSessionMetadata(previous, { kind: '', label: '' });
        } catch (err) {
            if (logger) logger.warn('previous-demote-failed', {
                previous, error: err.message,
            });
        }
    }
    return id;
}

module.exports = { getRemoteSessionId, currentId, rotate, LABEL, KIND };
