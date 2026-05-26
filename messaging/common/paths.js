// messaging/common/paths.js — resolve .emu/messaging/* paths
//
// EMU_ROOT is published by the Electron main process before the runner is
// spawned, so packaged builds write under userData/.emu and dev runs write
// under <repo>/.emu — same model the backend and daemon use.

const fs = require('fs');
const path = require('path');

function emuRoot() {
    const fromEnv = (process.env.EMU_ROOT || '').trim();
    if (fromEnv) return path.resolve(fromEnv);
    // Source-checkout fallback: this file lives at <repo>/messaging/common/paths.js
    return path.resolve(__dirname, '..', '..', '.emu');
}

function messagingDir() {
    const dir = path.join(emuRoot(), 'messaging');
    fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
    try { fs.chmodSync(dir, 0o700); } catch (_) { /* best-effort */ }
    return dir;
}

function platformDir(platform) {
    const dir = path.join(messagingDir(), platform);
    fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
    try { fs.chmodSync(dir, 0o700); } catch (_) { /* best-effort */ }
    return dir;
}

function authTokenPath() {
    return path.join(emuRoot(), '.auth_token');
}

function allowlistPath() {
    return path.join(messagingDir(), 'allowlist.json');
}

function remoteSessionFilePath() {
    return path.join(messagingDir(), 'remote_session.json');
}

function droppedLogPath() {
    return path.join(messagingDir(), 'dropped.log');
}

function bridgeLogPath(platform) {
    return path.join(platformDir(platform), 'bridge.log');
}

module.exports = {
    emuRoot,
    messagingDir,
    platformDir,
    authTokenPath,
    allowlistPath,
    remoteSessionFilePath,
    droppedLogPath,
    bridgeLogPath,
};
