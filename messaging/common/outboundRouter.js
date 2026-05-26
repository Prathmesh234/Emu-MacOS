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

const senders = new Map(); // platform -> sendFn(handle, text)
let active = null;         // { platform, handle, id }
let counter = 0;

function registerSender(platform, sendFn) {
    senders.set(platform, sendFn);
}

function unregisterSender(platform) {
    senders.delete(platform);
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
    return _sendVia(active.platform, active.handle, text, logger);
}

async function sendTo(platform, handle, text, logger) {
    return _sendVia(platform, handle, text, logger);
}

async function _sendVia(platform, handle, text, logger) {
    const sender = senders.get(platform);
    if (!sender) {
        if (logger) logger.warn('no-sender-for-platform', { platform });
        return false;
    }
    try {
        await sender(handle, text);
        return true;
    } catch (err) {
        if (logger) logger.error('send-failed', {
            platform,
            error: err.message,
        });
        return false;
    }
}

module.exports = {
    registerSender,
    unregisterSender,
    beginTurn,
    endTurn,
    activeTurn,
    sendToActive,
    sendTo,
};
