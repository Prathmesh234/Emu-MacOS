// messaging/common/sessionBinder.js — singleton "remote control" session
//
// The whole point of the messaging bridge is that all inbound text from
// WhatsApp + iMessage funnels into ONE session that grows over time
// instead of spawning a new session per message. We persist the chosen
// session id under .emu/messaging/remote_session.json so that across
// app restarts the same id is reused — the History sidebar then shows
// one continuous "Remote control" conversation.

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

module.exports = { getRemoteSessionId, LABEL, KIND };
