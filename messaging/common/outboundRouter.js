// messaging/common/outboundRouter.js — route outbound replies to the right chat
//
// The remote-control session is a single shared inbox: multiple senders
// (WhatsApp Alice, iMessage Bob, …) interleave into it. While the agent
// is mid-turn for one sender, every outbound reply that comes off the
// /ws/<remote_id> stream must land in THAT sender's chat — not the
// previous one. This is the bookkeeping layer.
//
// The pattern is "one active route at a time", paired with the per-handle
// FIFO queue in rateLimit.js. The runner sets the route immediately
// before posting /agent/step and clears it on the matching `done` /
// `stopped` / `error` event.
//
// Sender contract:
//   registerSender(platform, {
//       sendText:  async (handle, text)          → required
//       sendImage: async (handle, image, caption) → optional
//                  // image: { mime: 'image/png', buffer: Buffer }
//   })
// Bridges that don't natively support image attachments (e.g. iMessage)
// simply omit sendImage; the router will skip image sends silently.

const senders = new Map(); // platform -> { sendText, sendImage? }
let active = null;         // { platform, handle, id }
let counter = 0;

function registerSender(platform, handlers) {
    // Back-compat: a bare function is treated as sendText only.
    if (typeof handlers === 'function') {
        senders.set(platform, { sendText: handlers });
        return;
    }
    if (!handlers || typeof handlers.sendText !== 'function') {
        throw new Error(`registerSender(${platform}): sendText is required`);
    }
    senders.set(platform, {
        sendText: handlers.sendText,
        sendImage: typeof handlers.sendImage === 'function' ? handlers.sendImage : null,
    });
}

function unregisterSender(platform) {
    senders.delete(platform);
}

function supportsImage(platform) {
    const h = senders.get(platform);
    return !!(h && h.sendImage);
}

function beginTurn(platform, handle) {
    counter += 1;
    active = { platform, handle, id: counter };
    return active.id;
}

function endTurn(id) {
    if (active && active.id === id) {
        const finished = active;
        active = null;
        return finished;
    }
    return null;
}

function activeTurn() {
    return active;
}

async function sendToActive(text, logger) {
    if (!active) {
        if (logger) logger.warn('outbound-without-active-route', { text: text?.slice?.(0, 80) });
        return false;
    }
    return _sendTextVia(active.platform, active.handle, text, logger);
}

async function sendTo(platform, handle, text, logger) {
    return _sendTextVia(platform, handle, text, logger);
}

async function sendImageToActive(image, caption, logger) {
    if (!active) {
        if (logger) logger.warn('outbound-image-without-active-route');
        return false;
    }
    return _sendImageVia(active.platform, active.handle, image, caption, logger);
}

async function _sendTextVia(platform, handle, text, logger) {
    const sender = senders.get(platform);
    if (!sender) {
        if (logger) logger.warn('no-sender-for-platform', { platform });
        return false;
    }
    try {
        await sender.sendText(handle, text);
        return true;
    } catch (err) {
        if (logger) logger.error('send-failed', {
            platform,
            error: err.message,
        });
        return false;
    }
}

async function _sendImageVia(platform, handle, image, caption, logger) {
    const sender = senders.get(platform);
    if (!sender) {
        if (logger) logger.warn('no-sender-for-platform', { platform });
        return false;
    }
    if (!sender.sendImage) {
        if (logger) logger.debug('platform-no-image-support', { platform });
        return false;
    }
    if (!image || !image.buffer || !Buffer.isBuffer(image.buffer)) {
        if (logger) logger.warn('image-payload-invalid', { platform });
        return false;
    }
    try {
        await sender.sendImage(handle, image, caption || '');
        return true;
    } catch (err) {
        if (logger) logger.error('send-image-failed', {
            platform,
            error: err.message,
        });
        return false;
    }
}

module.exports = {
    registerSender,
    unregisterSender,
    supportsImage,
    beginTurn,
    endTurn,
    activeTurn,
    sendToActive,
    sendImageToActive,
    sendTo,
};
