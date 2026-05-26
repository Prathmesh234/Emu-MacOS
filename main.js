const { app, BrowserWindow, ipcMain, screen, shell, systemPreferences } = require('electron');
const path = require('path');
const psProcess = require('./frontend/process/psProcess');
const backendProcess = require('./frontend/process/backendProcess');
const emuCuaDriverProcess = require('./frontend/process/EmuCuaDriverProcess');
const daemonInstaller = require('./frontend/process/daemonInstaller');
const messagingProcess = require('./frontend/process/messagingProcess');
const { resolveEmuRoot } = require('./frontend/emu/root');
const { initEmu } = require('./frontend/emu');
const pkg = require('./package.json');
const { lockSessionDisplay, clearSessionLock } = require('./frontend/display/activeDisplay');

const BACKEND_URL = 'http://127.0.0.1:8000';

// Resolve .emu/ location BEFORE anything else runs. Publishing it via
// process.env means every child process (backend, daemon installer) and
// every renderer module inherits the same resolved path without having
// to re-derive it from __dirname (which is wrong inside app.asar).
const EMU_ROOT = resolveEmuRoot(app);
process.env.EMU_ROOT = EMU_ROOT;
console.log(`[emu] EMU_ROOT = ${EMU_ROOT}`);

let mainWindow;
let borderWindow;
let lastMissingPermissions = [];
const DEFAULT_DRIVER_MISSING_PERMISSIONS = ['accessibility', 'screen'];

function normalizeMissingPermissions(raw) {
  const items = Array.isArray(raw) ? raw : [];
  const result = [];
  for (const item of items) {
    const value = String(item || '').toLowerCase();
    let kind = null;
    if (value === 'accessibility' || value.includes('access')) {
      kind = 'accessibility';
    } else if (value === 'screen' || value.includes('screen')) {
      kind = 'screen';
    }
    if (kind && !result.includes(kind)) result.push(kind);
  }
  return result;
}

function missingPermissionsFromError(message) {
  const text = String(message || '');
  const missing = [];
  if (/accessibility/i.test(text)) missing.push('accessibility');
  if (/screen recording|screen/i.test(text)) missing.push('screen');
  if (missing.length) return missing;
  return /permission|not authorized|not authorised|unauthori[sz]ed|requires/i.test(text)
    ? DEFAULT_DRIVER_MISSING_PERMISSIONS
    : [];
}

function isDriverBinaryMissing(message) {
  return /binary not found|build\/install|install it from/i.test(String(message || ''));
}

function shouldPromptForDriverPermissions(message) {
  const text = String(message || '');
  if (isDriverBinaryMissing(text)) return false;
  return /permission|accessibility|screen recording|screen capture|not authorized|not authorised|unauthori[sz]ed|requires/i.test(text);
}

function emitPermissionsRequired(rawMissing) {
  const missing = normalizeMissingPermissions(rawMissing);
  if (!missing.length) return;

  lastMissingPermissions = missing;
  const payload = { missing };
  for (const win of BrowserWindow.getAllWindows()) {
    if (!win || win.isDestroyed()) continue;
    const send = () => {
      if (!win.isDestroyed()) {
        win.webContents.send('emu-cua:permissions-required', payload);
      }
    };
    if (win.webContents.isLoading()) {
      win.webContents.once('did-finish-load', send);
    } else {
      send();
    }
  }
}

function clearMissingPermissions() {
  lastMissingPermissions = [];
}

// ── Navigation lockdown ────────────────────────────────────────────────────
// The renderer runs with nodeIntegration enabled (see createWindow), so any
// successful off-origin navigation would inherit Node access and become host
// RCE. We allow only the bundled file:// pages and route every other URL
// through shell.openExternal after a strict scheme + host allowlist.

const ALLOWED_LOCAL_FILES = new Set(
  ['index.html', 'border.html'].map((f) =>
    require('url').pathToFileURL(path.join(__dirname, 'frontend', f)).href,
  ),
);

const EXTERNAL_URL_ALLOWLIST = [
  /^https:\/\/([a-z0-9-]+\.)*anthropic\.com(\/|$)/i,
  /^https:\/\/([a-z0-9-]+\.)*openai\.com(\/|$)/i,
  /^https:\/\/([a-z0-9-]+\.)*openrouter\.ai(\/|$)/i,
  /^https:\/\/([a-z0-9-]+\.)*baseten\.co(\/|$)/i,
  /^https:\/\/([a-z0-9-]+\.)*github\.com(\/|$)/i,
  /^https:\/\/([a-z0-9-]+\.)*apple\.com(\/|$)/i,
  /^x-apple\.systempreferences:/i,
];

function isLocalFrontendUrl(url) {
  if (!url) return false;
  if (ALLOWED_LOCAL_FILES.has(url)) return true;
  // about:blank is fine for one-off renderer-internal blank docs
  return url === 'about:blank';
}

function isAllowedExternal(url) {
  return EXTERNAL_URL_ALLOWLIST.some((re) => re.test(url));
}

function applyNavigationLockdown(win) {
  if (!win || !win.webContents) return;

  win.webContents.on('will-navigate', (event, url) => {
    if (isLocalFrontendUrl(url)) return;
    event.preventDefault();
    console.warn(`[security] blocked will-navigate → ${url}`);
    if (isAllowedExternal(url)) {
      shell.openExternal(url).catch(() => {});
    }
  });

  win.webContents.setWindowOpenHandler(({ url }) => {
    if (isAllowedExternal(url)) {
      shell.openExternal(url).catch(() => {});
    } else {
      console.warn(`[security] blocked window.open → ${url}`);
    }
    return { action: 'deny' };
  });

  // Refuse permission requests we never need (camera/mic/geolocation/etc.)
  win.webContents.session.setPermissionRequestHandler((_wc, permission, callback) => {
    const allowed = new Set(['clipboard-read', 'clipboard-sanitized-write']);
    callback(allowed.has(permission));
  });
}

// ── App lifecycle ──────────────────────────────────────────────────────────
function createWindow() {
  const primaryDisplay = screen.getPrimaryDisplay();
  const wa = primaryDisplay.workArea;

  // Larger default for the Design System v1 layout (chrome + body + composer
   // needs room; sidebar is 240px when open).
  const DEFAULT_W = 1040;
  const DEFAULT_H = 760;
  mainWindow = new BrowserWindow({
    width:  DEFAULT_W,
    height: DEFAULT_H,
    x: wa.x + Math.round((wa.width  - DEFAULT_W) / 2),
    y: wa.y + Math.round((wa.height - DEFAULT_H) / 2),
    minWidth: 420,
    minHeight: 440,
    frame: false,
    transparent: true,
    webPreferences: {
      // TODO: Security — migrate to nodeIntegration:false + contextIsolation:true
      // Currently required because the renderer uses CommonJS require() throughout.
      // Fixing this requires adding a bundler (esbuild/webpack) to resolve modules
      // at build time, then routing all IPC through the preload.js bridge.
      nodeIntegration: true,
      contextIsolation: false,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'frontend', 'index.html'));
  applyNavigationLockdown(mainWindow);
  mainWindow.webContents.on('did-finish-load', () => {
    if (lastMissingPermissions.length) {
      mainWindow.webContents.send('emu-cua:permissions-required', {
        missing: lastMissingPermissions,
      });
    }
  });

  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }
}

app.whenReady().then(() => {
  // Bootstrap .emu/ folder (idempotent — safe on every launch)
  initEmu(pkg.version);

  // Set up IPC for the desktop border
  // Border is shown only while the agent is executing.
  function ensureBorderWindow() {
    if (!borderWindow) {
      // Created with placeholder bounds; repositioned to active display before each show.
      borderWindow = new BrowserWindow({
        x: 0, y: 0, width: 100, height: 100,
        transparent: true,
        frame: false,
        alwaysOnTop: true,
        skipTaskbar: true,
        hasShadow: false,
        focusable: false,
        webPreferences: { nodeIntegration: true, contextIsolation: false, preload: path.join(__dirname, 'preload.js') }
      });
      borderWindow.setIgnoreMouseEvents(true, { forward: true });
      borderWindow.setAlwaysOnTop(true, 'screen-saver');
      borderWindow.setVisibleOnAllWorkspaces(true);
      borderWindow.loadFile(path.join(__dirname, 'frontend', 'border.html'));
      applyNavigationLockdown(borderWindow);
    }
    return borderWindow;
  }

  function showBorderOnActiveDisplay() {
    const display = lockSessionDisplay(screen, mainWindow);
    const bw = ensureBorderWindow();
    bw.setBounds(display.bounds);
    bw.show();
  }

  ipcMain.on('set-border', (event, visible) => {
    if (visible) {
      showBorderOnActiveDisplay();
    } else {
      clearSessionLock();
      if (borderWindow) borderWindow.hide();
    }
  });

  ipcMain.on('set-generating', (event, generating) => {
    // set-generating mirrors set-border; delegate to the same handler
    // so both signals manage the session lock consistently.
    if (generating) {
      showBorderOnActiveDisplay();
    } else {
      if (borderWindow) borderWindow.hide();
      // Do NOT clear the session lock here — set-border:false will do it.
      // Clearing independently could race with an in-flight set-border:true.
    }
  });

  // Register IPC handlers AFTER app is ready (desktopCapturer + screen need this)
  const { registerAll } = require('./frontend/actions');
  registerAll(ipcMain, {
      BACKEND_URL,
      screen,
      getMainWindow: () => mainWindow
  });

  emuCuaDriverProcess.configure({ app });
  require('./frontend/cua-driver-commands').registerAll(ipcMain, {
    callTool: (name, args) => emuCuaDriverProcess.callTool(name, args, { app }),
    onPermissionsRequired: emitPermissionsRequired,
  });

  const daemonInstallPromise = daemonInstaller.maybeInstall({ app, emuRoot: EMU_ROOT });

  // Install/repair the memory LaunchAgent first. The installer also removes
  // the legacy driver LaunchAgent; emu-cua-driver now runs as an Emu-owned
  // child so the permission flow remains without macOS background-item banners.
  Promise.resolve(daemonInstallPromise)
    .catch((err) => {
      console.warn(`[daemon-install] install error: ${err?.message || err}`);
    })
    .then(() => emuCuaDriverProcess.ensureStarted({ app, forcePermissionCheck: true }))
    .then((result) => {
      if (result?.success) {
        const missing = result?.permissions?.missing || [];
        if (missing.length) {
          console.warn(`[emu-cua-driver] startup ok, missing permissions: ${missing.join(', ')}`);
          emitPermissionsRequired(missing);
        } else {
          clearMissingPermissions();
          console.log(`[emu-cua-driver] startup ok (pid=${result.pid})`);
        }
      } else {
        console.warn(`[emu-cua-driver] startup deferred: ${result?.error || 'unknown'}`);
        if (result?.permissionsRequired || shouldPromptForDriverPermissions(result?.error)) {
          const missing = normalizeMissingPermissions(result?.missing || result?.permissions?.missing || []);
          emitPermissionsRequired(missing.length ? missing : missingPermissionsFromError(result?.error));
        }
      }
    })
    .catch((err) => {
      console.warn(`[emu-cua-driver] startup error: ${err?.message || err}`);
      if (shouldPromptForDriverPermissions(err?.message || err)) {
        emitPermissionsRequired(missingPermissionsFromError(err?.message || err));
      }
    });

  ipcMain.handle('permissions:status', async () => {
    if (process.platform !== 'darwin') {
      return {
        platform: process.platform,
        screenRecording: 'unsupported',
        accessibility: 'unsupported',
      };
    }

    let screenRecording = 'unknown';
    try {
      screenRecording = systemPreferences.getMediaAccessStatus('screen');
    } catch (_) {}

    let accessibility = 'unknown';
    try {
      accessibility = systemPreferences.isTrustedAccessibilityClient(false) ? 'granted' : 'denied';
    } catch (_) {}

    return { platform: process.platform, screenRecording, accessibility };
  });

  ipcMain.handle('permissions:open', async (_event, pane) => {
    if (process.platform !== 'darwin') return { success: false, error: 'not macOS' };
    const urls = {
      screen: 'x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture',
      accessibility: 'x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility',
    };
    const url = urls[pane] || urls.screen;
    try {
      await shell.openExternal(url);
      return { success: true };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  ipcMain.handle('emu-cua:recheck-permissions', async () => {
    try {
      const result = await emuCuaDriverProcess.recheckPermissions({ app });
      const missing = normalizeMissingPermissions(result?.missing || []);
      if (missing.length) {
        emitPermissionsRequired(missing);
      } else {
        clearMissingPermissions();
      }
      return { success: true, ...result, missing };
    } catch (err) {
      const shouldPrompt = shouldPromptForDriverPermissions(err?.message || err);
      const missing = shouldPrompt ? missingPermissionsFromError(err?.message) : [];
      if (shouldPrompt) {
        emitPermissionsRequired(missing);
      }
      return {
        success: false,
        granted: false,
        permissionsRequired: shouldPrompt,
        missing,
        error: err?.message || String(err),
      };
    }
  });

  ipcMain.handle('daemon:status', async () => daemonInstaller.getStatus({ app, emuRoot: EMU_ROOT }));
  ipcMain.handle('daemon:repair', async () => daemonInstaller.repair({ app, emuRoot: EMU_ROOT }));

  psProcess.start();
  // Spawn the backend (dev: backend.sh; packaged: frozen binary).
  // The backend writes .emu/.auth_token on startup; the renderer reads
  // it for every HTTP/WS call. See frontend/services/api.js.
  backendProcess.start({ app, emuRoot: EMU_ROOT });
  // Spawn the inbound messaging bridge runner (WhatsApp + iMessage).
  // No-ops if the user hasn't created .emu/messaging/allowlist.json or
  // EMU_DISABLE_MESSAGING=1 is set. Waits for the backend's auth token to
  // appear before launching, so it never races backend startup.
  messagingProcess.start({ app, emuRoot: EMU_ROOT });
  // Coworker execution talks to the emu-cua-driver daemon socket from the
  // backend rather than through renderer IPC.
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('will-quit', () => {
  // Close WebSocket first to prevent reconnect loop, then kill the shell process
  try { require('./frontend/services/websocket').closeWebSocket(); } catch (_) {}
  psProcess.stop();
  messagingProcess.stop();
  backendProcess.stop();
  emuCuaDriverProcess.stop();
});
