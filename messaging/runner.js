// messaging/runner.js — bridge supervisor
//
// One Node process owns all messaging bridges. Lifecycle is owned by
// Electron main (frontend/process/messagingProcess.js). This file:
//
//   1. Validates the env (EMU_ROOT, EMU_AUTH_TOKEN should be present).
//   2. Loads the allowlist; exits cleanly if nothing is configured.
//   3. Boots the Dispatcher (binds the singleton remote-control session).
//   4. Starts each enabled bridge (WhatsApp, iMessage).
//   5. Stops everything cleanly on SIGTERM/SIGINT.
//
// CLI:
//   node runner.js              — normal run
//   node runner.js --selfcheck  — print configuration summary and exit 0
//                                  (works without `npm install` so first-time
//                                  operators can verify env + allowlist)

const { makeLogger } = require('./common/log');
const allowlist = require('./common/allowlist');
const silenceLibsignal = require('./common/silenceLibsignal');

// Filter Baileys/libsignal's raw `console.log` SessionEntry dumps before any
// bridge module loads. Idempotent; safe under --selfcheck (no-op without
// Baileys present).
silenceLibsignal.install();

const logger = makeLogger('runner');

function _summary() {
    const a = allowlist.loadAllowlist();
    return {
        emuRoot: process.env.EMU_ROOT || '<derived>',
        // Hard-pinned to 'remote' (see Dispatcher constructor); env is
        // surfaced here purely for the --selfcheck summary so operators
        // can see what they tried to configure vs. what actually runs.
        agentMode: 'remote (forced for messaging bridges)',
        agentModeEnvRequested: process.env.EMU_MESSAGING_AGENT_MODE || '<unset>',
        dryRun: process.env.EMU_MESSAGING_DRY_RUN === '1',
        whatsappAllowlistSize: a.whatsapp.length,
        imessageAllowlistSize: a.imessage.length,
        whatsappDisabled: process.env.EMU_DISABLE_WHATSAPP === '1',
        imessageDisabled: process.env.EMU_DISABLE_IMESSAGE === '1',
        // Per-platform mode surfaces in --selfcheck so operators can
        // confirm self-chat is actually engaged before staring at logs.
        whatsappMode: (process.env.EMU_MESSAGING_WHATSAPP_MODE || 'bot').toLowerCase(),
        imessageMode: (process.env.EMU_MESSAGING_IMESSAGE_MODE || 'bot').toLowerCase(),
    };
}

async function main() {
    if (process.argv.includes('--selfcheck')) {
        // Intentionally avoid requiring the dispatcher / bridge modules here
        // so this command works on a fresh checkout before `npm install`.
        logger.info('selfcheck', _summary());
        process.exit(0);
        return;
    }

    if (process.env.EMU_DISABLE_MESSAGING === '1') {
        logger.info('disabled-by-env');
        process.exit(0);
        return;
    }

    if (!allowlist.anyConfigured()) {
        logger.info('no-allowlist-configured-idle-exit');
        process.exit(0);
        return;
    }

    // Lazy-load to keep `--selfcheck` dep-free.
    const { Dispatcher } = require('./common/dispatcher');
    const dispatcher = new Dispatcher(logger);
    try {
        await dispatcher.start();
    } catch (err) {
        logger.error('dispatcher-start-failed', { error: err.message });
        process.exit(2);
        return;
    }

    const bridges = [];

    if (process.env.EMU_DISABLE_WHATSAPP !== '1') {
        try {
            const { startWhatsApp } = require('./whatsapp');
            const handle = await startWhatsApp({ dispatcher });
            bridges.push(handle);
        } catch (err) {
            logger.error('whatsapp-start-failed', { error: err.message });
        }
    }

    if (process.env.EMU_DISABLE_IMESSAGE !== '1' && process.platform === 'darwin') {
        try {
            const { startIMessage } = require('./imessage');
            const handle = await startIMessage({ dispatcher });
            bridges.push(handle);
        } catch (err) {
            logger.error('imessage-start-failed', { error: err.message });
        }
    }

    let shuttingDown = false;
    function shutdown(reason) {
        if (shuttingDown) return;
        shuttingDown = true;
        logger.info('shutting-down', { reason });
        for (const b of bridges) {
            try { b && b.stop && b.stop(); } catch (_) { /* ignore */ }
        }
        try { dispatcher.stop(); } catch (_) { /* ignore */ }
        setTimeout(() => process.exit(0), 500).unref();
    }

    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT',  () => shutdown('SIGINT'));
    process.on('uncaughtException', (err) => {
        logger.error('uncaught-exception', { error: err.message, stack: err.stack });
    });
    process.on('unhandledRejection', (err) => {
        logger.error('unhandled-rejection', { error: err && err.message ? err.message : String(err) });
    });

    logger.info('runner-up', _summary());
}

main().catch((err) => {
    logger.error('runner-fatal', { error: err.message, stack: err.stack });
    process.exit(1);
});
