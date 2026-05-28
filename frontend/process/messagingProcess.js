/**
 * frontend/process/messagingProcess.js — Lifecycle for the messaging bridge runner.
 *
 * Spawns `messaging/runner.js` as a Node child of Electron main. The runner
 * supervises the WhatsApp + iMessage bridges and forwards inbound messages
 * to the local backend via the existing X-Emu-Token.
 *
 * The runner is spawned with a *scrubbed* env — only PATH/HOME/LANG/TERM
 * plus EMU_ROOT and EMU_AUTH_TOKEN. Provider API keys are never forwarded,
 * matching the Hermes child process model.
 *
 * The runner exits cleanly with code 0 if no allowlist is configured, so it
 * is always safe to spawn — first-time users will see a one-line
 * "no-allowlist-configured" log and the process simply stops.
 *
 * Opt-out: set EMU_DISABLE_MESSAGING=1 to skip spawning entirely.
 */

const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

let _child = null;
let _restartTimer = null;
let _restartAttempts = 0;
let _restartWindowStart = 0;
const RESTART_WINDOW_MS = 60_000;
const RESTART_MAX = 5;

function _resolveRunnerPath(app) {
    if (app && app.isPackaged) {
        // Packaged: messaging/ ships inside Resources/app.asar.unpacked or
        // app.asar (it's plain JS + a small native dep). Prefer the unpacked
        // location when present (electron-builder's default for native modules).
        const resources = process.resourcesPath;
        const candidates = [
            path.join(resources, 'app.asar.unpacked', 'messaging', 'runner.js'),
            path.join(resources, 'app', 'messaging', 'runner.js'),
            path.join(resources, 'messaging', 'runner.js'),
        ];
        for (const c of candidates) {
            if (fs.existsSync(c)) return c;
        }
        return null;
    }
    // Dev: <repo>/messaging/runner.js
    const repoRoot = path.resolve(__dirname, '..', '..');
    const dev = path.join(repoRoot, 'messaging', 'runner.js');
    return fs.existsSync(dev) ? dev : null;
}

function _scrubbedEnv({ emuRoot, authToken }) {
    // Allowlist of env vars the runner is permitted to see. NO provider
    // API keys, NO .env contents.
    const allowed = ['PATH', 'HOME', 'LANG', 'LC_ALL', 'TERM', 'USER', 'SHELL', 'TMPDIR'];
    const env = {};
    for (const key of allowed) {
        if (process.env[key] != null) env[key] = process.env[key];
    }
    if (emuRoot) env.EMU_ROOT = emuRoot;
    if (authToken) env.EMU_AUTH_TOKEN = authToken;
    // Propagate messaging-specific switches if the operator set them.
    for (const key of [
        'EMU_DISABLE_MESSAGING',
        'EMU_DISABLE_WHATSAPP',
        'EMU_DISABLE_IMESSAGE',
        'EMU_MESSAGING_AGENT_MODE',
        'EMU_MESSAGING_DRY_RUN',
        'EMU_MESSAGING_DEBUG',
        // Per-platform inbound mode (`bot` default, `self-chat` opt-in).
        // Self-chat mode lets the operator drive the agent from their own
        // personal account by messaging themselves; see
        // messaging/common/replyPrefix.js for the anti-loop layers.
        'EMU_MESSAGING_WHATSAPP_MODE',
        'EMU_MESSAGING_IMESSAGE_MODE',
        // Override the JIDs / handles that count as "the operator's own
        // self-chat" when self-chat mode is on. Optional; sensible defaults
        // are derived from sock.user (WhatsApp) / the allowlist (iMessage).
        'EMU_MESSAGING_WHATSAPP_SELF_JIDS',
        'EMU_MESSAGING_IMESSAGE_SELF_HANDLES',
        // Override the outbound reply prefix used for echo suppression.
        // Optional; defaults to "🤖 *Emu*\n────────\n" (see replyPrefix.js).
        'EMU_MESSAGING_REPLY_PREFIX',
        'IMESSAGE_DB',
    ]) {
        if (process.env[key] != null) env[key] = process.env[key];
    }
    return env;
}

function _shouldSpawn({ emuRoot }) {
    if (process.env.EMU_DISABLE_MESSAGING === '1') return false;
    if (process.platform !== 'darwin' && process.env.EMU_DISABLE_IMESSAGE !== '1') {
        // iMessage is darwin-only but WhatsApp works anywhere. Allow spawn
        // on other platforms; the runner will skip imessage internally.
    }
    // Don't spawn unless the user has configured an allowlist. This keeps
    // first-launch quiet (no bridges spinning up implicitly).
    try {
        const allowlistPath = path.join(emuRoot, 'messaging', 'allowlist.json');
        if (!fs.existsSync(allowlistPath)) return false;
        const raw = JSON.parse(fs.readFileSync(allowlistPath, 'utf8'));
        const total =
            (raw.whatsapp || []).length +
            (raw.imessage || []).length +
            (raw.discord || []).length;
        return total > 0;
    } catch (_) {
        return false;
    }
}

function start({ app, emuRoot, authToken }) {
    if (_child) return;
    if (!_shouldSpawn({ emuRoot })) {
        console.log('[messaging] not spawning — no allowlist configured (.emu/messaging/allowlist.json)');
        return;
    }

    const runnerPath = _resolveRunnerPath(app);
    if (!runnerPath) {
        console.warn('[messaging] runner.js not found — skipping');
        return;
    }

    // Wait (up to 30s) for the backend to write .emu/.auth_token. Without
    // it the runner's first /agent/session call would 401 and the bridge
    // would crash-loop until the backend caught up.
    _waitForAuthToken(emuRoot, 30_000).then((token) => {
        if (!token) {
            console.warn('[messaging] auth token never appeared — skipping spawn');
            return;
        }
        _spawnRunner({ app, emuRoot, authToken: authToken || token, runnerPath });
    });
}

async function _waitForAuthToken(emuRoot, timeoutMs) {
    const tokenPath = path.join(emuRoot, '.auth_token');
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        try {
            const t = fs.readFileSync(tokenPath, 'utf8').trim();
            if (t) return t;
        } catch (_) { /* not yet */ }
        await new Promise((r) => setTimeout(r, 250));
    }
    return null;
}

function _spawnRunner({ app, emuRoot, authToken, runnerPath }) {
    if (_child) return;
    const env = _scrubbedEnv({ emuRoot, authToken });
    const nodeBin = process.execPath; // Electron's bundled Node; works packaged or dev.

    console.log(`[messaging] spawning ${nodeBin} ${runnerPath}`);
    _child = spawn(nodeBin, [runnerPath], {
        cwd: path.dirname(runnerPath),
        env: {
            ...env,
            // Electron sets ELECTRON_RUN_AS_NODE so spawning Electron's
            // binary behaves like plain node (no app windows).
            ELECTRON_RUN_AS_NODE: '1',
        },
        stdio: ['ignore', 'pipe', 'pipe'],
    });

    _child.stdout.on('data', (d) => process.stdout.write(`[messaging] ${d}`));
    _child.stderr.on('data', (d) => process.stderr.write(`${d}`));

    _child.on('exit', (code, signal) => {
        console.warn(`[messaging] runner exited code=${code} signal=${signal}`);
        _child = null;
        // Code 0 means clean shutdown (e.g. no allowlist) — don't restart.
        if (code === 0) return;
        _scheduleRestart({ app, emuRoot, authToken });
    });
}

function _scheduleRestart(args) {
    const now = Date.now();
    if (now - _restartWindowStart > RESTART_WINDOW_MS) {
        _restartWindowStart = now;
        _restartAttempts = 0;
    }
    _restartAttempts += 1;
    if (_restartAttempts > RESTART_MAX) {
        console.error(`[messaging] giving up after ${RESTART_MAX} restarts in ${RESTART_WINDOW_MS}ms`);
        return;
    }
    const delay = Math.min(30_000, 1000 * Math.pow(2, _restartAttempts - 1));
    if (_restartTimer) clearTimeout(_restartTimer);
    _restartTimer = setTimeout(() => {
        _restartTimer = null;
        start(args);
    }, delay);
    _restartTimer.unref && _restartTimer.unref();
    console.warn(`[messaging] restarting in ${delay}ms (attempt ${_restartAttempts}/${RESTART_MAX})`);
}

function stop() {
    if (_restartTimer) {
        clearTimeout(_restartTimer);
        _restartTimer = null;
    }
    if (!_child) return;
    try {
        _child.kill('SIGTERM');
    } catch (_) { /* already gone */ }
    const pid = _child.pid;
    setTimeout(() => {
        try { process.kill(pid, 'SIGKILL'); } catch (_) {}
    }, 3000).unref();
    _child = null;
}

module.exports = { start, stop };
