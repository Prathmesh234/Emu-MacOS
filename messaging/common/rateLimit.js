// messaging/common/rateLimit.js — per-handle FIFO queue
//
// "Queue, don't drop" — losing user intent is worse than a few seconds of
// delay. Each (platform, handle) pair gets its own promise chain; one
// inbound message blocks subsequent ones from the same handle until it
// finishes. Different handles run in parallel as far as queueing is
// concerned (the outbound router still serializes the actual agent turn).
//
// A hard cap per handle (default 16) prevents a misbehaving sender from
// growing the queue without bound; once exceeded we drop with a single
// canned reply.

const MAX_QUEUED = 16;

const chains = new Map();     // key -> Promise
const lengths = new Map();    // key -> queued depth

function _key(platform, handle) {
    return `${platform}::${handle}`;
}

function depth(platform, handle) {
    return lengths.get(_key(platform, handle)) || 0;
}

// schedule(platform, handle, work) → Promise<{ accepted, result }>.
// If `accepted` is false, the message was dropped because the per-handle
// queue is saturated; caller should send the user a canned "overwhelmed"
// reply.
async function schedule(platform, handle, work) {
    const key = _key(platform, handle);
    const queued = lengths.get(key) || 0;
    if (queued >= MAX_QUEUED) {
        return { accepted: false, result: null };
    }
    lengths.set(key, queued + 1);

    const prev = chains.get(key) || Promise.resolve();
    let release;
    const next = new Promise((resolve) => { release = resolve; });
    chains.set(key, prev.then(() => next));

    try {
        await prev;
        const result = await work();
        return { accepted: true, result };
    } finally {
        release();
        const remaining = (lengths.get(key) || 1) - 1;
        if (remaining <= 0) {
            lengths.delete(key);
            chains.delete(key);
        } else {
            lengths.set(key, remaining);
        }
    }
}

module.exports = { schedule, depth, MAX_QUEUED };
