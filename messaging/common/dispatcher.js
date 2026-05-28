// messaging/common/dispatcher.js — inbound → /agent/step → outbound reply
//
// One dispatcher process-wide. Each bridge calls dispatcher.handleInbound
// with the normalized message; the dispatcher enforces the allowlist,
// holds the per-handle queue, sets the active outbound route, posts to
// /agent/step, and tears the route down when the matching `done` /
// `stopped` / `error` arrives on the shared WebSocket.
//
// Outbound parity with the desktop UI:
//   - `done` / `assistant_text` / `final` → forwarded as the canonical
//     reply text.
//   - `step` events with a model-authored `reasoning_content` or a tool
//     action are condensed into a one-line progress ping so off-host
//     users see the agent is alive (throttled to one ping per 4s per
//     turn so fast tool chains don't spam the chat).
//   - On turn-end we fetch the latest cached screenshot from the backend
//     and attach it to the final reply when the platform supports image
//     attachments (WhatsApp yes, iMessage no — routes accordingly).
//   - Slash commands (`/new`, `/reset`, `/clear`, `/help`) handled
//     in-band so allowlisted users can rotate the singleton remote-
//     control session without operator intervention.

const allowlist = require('./allowlist');
const rateLimit = require('./rateLimit');
const outboundRouter = require('./outboundRouter');
const sessionBinder = require('./sessionBinder');
const agentClient = require('./agentClient');
const sanitizer = require('./sanitizer');
const { logDropped } = require('./log');

const OVERWHELMED_REPLY = "I'm overwhelmed right now — try again in a minute.";

// Minimum gap between live progress pings inside a single turn. Stops
// us from carpet-bombing the chat during fast cua_* tool chains.
const STEP_PING_THROTTLE_MS = 4_000;

// Max chars for a synthesized progress ping; we want WhatsApp-friendly
// one-liners, not paragraphs.
const STEP_PING_MAX_LEN = 220;

const SLASH_HELP = (
    'Commands:\n' +
    '/new — start a fresh session (clears context)\n' +
    '/help — show this list'
);

// Track which message types from the WS stream count as outbound text.
// We forward final/assistant text and the done card's message; status
// pings, tool-event chatter, and screenshot pushes stay local.
function _isOutboundText(payload) {
    if (!payload || typeof payload !== 'object') return false;
    if (payload.type === 'done') return true;
    if (payload.type === 'assistant_text') return true;
    if (payload.type === 'final' || payload.type === 'final_message') return true;
    if (payload.type === 'log') {
        const msg = String(payload.message || '');
        // Skip [user] echoes (we already saw the user message) and
        // routing tags. Forward [assistant] DONE messages.
        if (msg.startsWith('[assistant]') && msg.includes('DONE')) return true;
    }
    return false;
}

function _extractText(payload) {
    if (!payload) return '';
    if (typeof payload.final_message === 'string' && payload.final_message) return payload.final_message;
    if (typeof payload.message === 'string' && payload.message) return payload.message;
    if (typeof payload.text === 'string' && payload.text) return payload.text;
    return '';
}

function _isTurnEndEvent(payload) {
    if (!payload || typeof payload !== 'object') return false;
    return ['done', 'stopped', 'error', 'final', 'final_message'].includes(payload.type);
}

// Turn a backend `step` event into a short progress ping suitable for
// WhatsApp / iMessage. Returns '' to indicate "skip this step".
//
// Priority:
//   1. The model's free-text `reasoning_content` (already a sentence).
//   2. A glyph + tool name derived from the structured action payload.
function _synthesizeStepPing(payload) {
    if (!payload || payload.type !== 'step') return '';
    if (payload.done) return '';   // the matching `done` event carries the real reply

    const reasoning = String(payload.reasoning_content || '').trim();
    if (reasoning) {
        // Take just the first sentence so we don't dump the model's
        // multi-paragraph plan to chat.
        const firstSentence = reasoning.split(/(?<=[.!?])\s/)[0] || reasoning;
        return firstSentence.slice(0, STEP_PING_MAX_LEN);
    }

    const action = payload.action || {};
    const tool = String(action.type || action.tool || '').toLowerCase();
    if (!tool) return '';

    // Glyphs intentionally minimal — most platforms render them, none
    // are critical for comprehension.
    const labels = {
        cua_click:       '🖱️ clicking',
        cua_doubleclick: '🖱️ double-clicking',
        cua_drag:        '↔️ dragging',
        cua_keypress:    '⌨️ pressing keys',
        cua_type:        '⌨️ typing',
        cua_scroll:      '🖱️ scrolling',
        cua_screenshot:  '',           // way too noisy to forward
        screenshot:      '',
        cua_open_app:    '🚀 opening app',
        shell_exec:      '🔧 running shell command',
        update_plan:     '📝 updating plan',
        read_plan:       '',
        done:            '',
    };
    return labels[tool] || '';
}

class Dispatcher {
    constructor(logger) {
        this.logger = logger;
        this.sessionId = null;
        this.stream = null;
        this.dryRun = process.env.EMU_MESSAGING_DRY_RUN === '1';
        this.agentMode = (process.env.EMU_MESSAGING_AGENT_MODE || 'coworker').trim();
        if (!['coworker', 'remote'].includes(this.agentMode)) {
            this.logger.warn('invalid-agent-mode-falling-back', { value: this.agentMode });
            this.agentMode = 'coworker';
        }
        // Per-turn throttle/state for step-progress pings. Reset every
        // turn-end so the next inbound message starts fresh.
        this._lastStepPingAt = 0;
        this._lastFinalText = '';
    }

    async start() {
        this.sessionId = await sessionBinder.getRemoteSessionId(this.logger);
        this.logger.info('dispatcher-bound', {
            sessionId: this.sessionId,
            agentMode: this.agentMode,
            dryRun: this.dryRun,
        });
        this._openStream();
    }

    _openStream() {
        // Long-lived WS stream so we receive every outbound text the agent
        // produces for the remote-control session.
        this.stream = agentClient.connectStream(this.sessionId, (payload) => {
            this._onAgentEvent(payload);
        }, this.logger);
    }

    stop() {
        if (this.stream) {
            try { this.stream.close(); } catch (_) { /* ignore */ }
            this.stream = null;
        }
    }

    registerSender(platform, sendFn) {
        outboundRouter.registerSender(platform, sendFn);
    }

    // platform: "whatsapp" | "imessage"
    // handle:   normalized phone/email
    // text:     raw user message
    async handleInbound({ platform, handle, text, meta }) {
        const trimmed = String(text || '').trim();
        if (!trimmed) return;

        if (!allowlist.isAllowed(platform, handle)) {
            this.logger.info('dropped-not-allowlisted', { platform, handle });
            logDropped(platform, handle, 'not-allowlisted');
            // Intentionally do NOT reply; an unsolicited reply would
            // confirm to spammers that this number is live.
            return;
        }

        // Slash commands run OUTSIDE the per-handle queue + rate limit
        // so they can't be back-pressured behind a stuck turn. They
        // don't talk to the model — they only mutate bridge-side state.
        if (trimmed.startsWith('/')) {
            await this._handleSlashCommand({ platform, handle, text: trimmed });
            return;
        }

        if (rateLimit.depth(platform, handle) >= rateLimit.MAX_QUEUED) {
            this.logger.warn('queue-saturated', { platform, handle });
            logDropped(platform, handle, 'queue-saturated');
            try {
                await outboundRouter.sendTo(platform, handle, OVERWHELMED_REPLY, this.logger);
            } catch (_) { /* ignore */ }
            return;
        }

        await rateLimit.schedule(platform, handle, async () => {
            const turnId = outboundRouter.beginTurn(platform, handle);
            this._lastStepPingAt = 0;
            this._lastFinalText = '';
            this.logger.info('inbound-queued', {
                platform, handle,
                preview: trimmed.slice(0, 80),
                turnId,
            });
            try {
                if (this.dryRun) {
                    this.logger.info('dry-run-skip-post', { platform, handle });
                    return;
                }
                const prefixed = `[${platform}:${handle}] ${trimmed}`;
                await agentClient.postStep({
                    sessionId: this.sessionId,
                    userMessage: prefixed,
                    source: platform,
                    agentMode: this.agentMode,
                });
            } catch (err) {
                this.logger.error('agent-step-failed', {
                    platform, handle, error: err.message,
                });
                // Surface a short error to the user so they're not
                // wondering if the message disappeared.
                await outboundRouter.sendToActive(
                    `Sorry — the agent backend errored: ${err.message}`,
                    this.logger,
                );
                outboundRouter.endTurn(turnId);
                return;
            }
            // Wait for the turn-end event on the WS so the next queued
            // message for this handle doesn't pile in before the agent
            // finishes replying. _onAgentEvent resolves _turnDeadline.
            await this._awaitTurnEnd(turnId);
        });
    }

    // Slash command dispatcher. Kept tiny on purpose; it would be easy to
    // turn this into a permission surface (every command is effectively
    // "execute as the operator"), so we only ship the commands we want
    // allowlisted users to have access to.
    async _handleSlashCommand({ platform, handle, text }) {
        const [cmdRaw, ...rest] = text.split(/\s+/);
        const cmd = cmdRaw.toLowerCase();
        const args = rest.join(' ').trim();

        const reply = async (msg) => {
            try {
                await outboundRouter.sendTo(platform, handle, msg, this.logger);
            } catch (err) {
                this.logger.error('slash-reply-failed', { cmd, error: err.message });
            }
        };

        switch (cmd) {
            case '/new':
            case '/reset':
            case '/clear': {
                this.logger.info('slash-rotate', { platform, handle, cmd });
                const previousId = this.sessionId;
                try {
                    // 1. Best-effort: stop any in-flight model loop on the
                    //    OLD session so the agent doesn't keep clicking
                    //    things from the abandoned task. We don't await
                    //    forever — if the backend is wedged, rotate anyway.
                    if (previousId) {
                        try {
                            await agentClient.stopSession(previousId);
                        } catch (err) {
                            this.logger.warn('rotate-stop-old-failed', {
                                previous: previousId, error: err.message,
                            });
                        }
                    }
                    // 2. Tear down WS against the old id BEFORE rotating so
                    //    we don't race a stale `done` from a previous turn.
                    if (this.stream) {
                        try { this.stream.close(); } catch (_) { /* ignore */ }
                        this.stream = null;
                    }
                    // 3. Release any pending turn-end promise. If `/new`
                    //    arrived mid-turn, the WS we'd have heard `done`
                    //    on is now closed — without this the SAME handle's
                    //    next message would block for the 5-minute safety
                    //    net inside _awaitTurnEnd.
                    const active = outboundRouter.activeTurn();
                    if (active) {
                        if (this._pendingTurnResolver) {
                            this._pendingTurnResolver(active.id);
                        }
                        outboundRouter.endTurn(active.id);
                    }
                    this._lastStepPingAt = 0;
                    this._lastFinalText = '';
                    // 4. Mint + bind the new session id.
                    const newId = await sessionBinder.rotate(this.logger);
                    this.sessionId = newId;
                    this._openStream();
                    await reply('Started a fresh session — previous context cleared.');
                } catch (err) {
                    this.logger.error('rotate-failed', { error: err.message });
                    await reply(`Couldn't rotate the session: ${err.message}`);
                }
                return;
            }
            case '/help':
            case '/?': {
                await reply(SLASH_HELP);
                return;
            }
            default: {
                await reply(
                    `Unknown command: ${cmd}. Try /help for the list.\n` +
                    `If you meant to send "${text}" as a task, prefix with a letter or punctuation.`,
                );
                return;
            }
        }
    }

    _awaitTurnEnd(turnId) {
        return new Promise((resolve) => {
            this._pendingTurnResolver = (id) => {
                if (id === turnId) {
                    this._pendingTurnResolver = null;
                    resolve();
                }
            };
            // Safety net: if the WS misses an event, release the queue
            // after 5 minutes so the bridge can't deadlock.
            setTimeout(() => {
                if (this._pendingTurnResolver) {
                    this.logger.warn('turn-end-timeout-releasing-queue', { turnId });
                    this._pendingTurnResolver = null;
                    resolve();
                }
            }, 5 * 60 * 1000).unref();
        });
    }

    async _onAgentEvent(payload) {
        // Live progress ping: a short note while the agent is mid-turn so
        // off-host users see activity instead of dead silence. Throttled
        // and only when there's an active outbound route (i.e. a turn we
        // kicked off, not events from the desktop UI's own activity on
        // this same shared session).
        if (payload && payload.type === 'step' && outboundRouter.activeTurn()) {
            const now = Date.now();
            if (now - this._lastStepPingAt >= STEP_PING_THROTTLE_MS) {
                const ping = _synthesizeStepPing(payload);
                if (ping) {
                    const text = sanitizer.sanitize(ping);
                    if (text) {
                        this._lastStepPingAt = now;
                        await outboundRouter.sendToActive(text, this.logger);
                    }
                }
            }
        }

        if (_isOutboundText(payload)) {
            const text = sanitizer.sanitize(_extractText(payload));
            if (text) {
                this._lastFinalText = text;
                // For `done` we defer the text send until after we've
                // pulled the screenshot, so it lands as the caption on
                // the image attachment (single notification on phones).
                if (payload.type !== 'done') {
                    const sent = await outboundRouter.sendToActive(text, this.logger);
                    if (!sent) this.logger.warn('outbound-not-routed', { type: payload.type });
                }
            }
        }

        if (_isTurnEndEvent(payload)) {
            const finished = outboundRouter.activeTurn();
            if (finished) {
                // On `done`, try to attach the latest screenshot to the
                // final reply when the platform supports images. We do
                // this BEFORE endTurn so the route is still bound.
                if (payload.type === 'done' && outboundRouter.supportsImage(finished.platform)) {
                    try {
                        const shot = await agentClient.fetchLatestScreenshot(this.sessionId);
                        if (shot) {
                            const caption = this._lastFinalText || _extractText(payload) || '';
                            const ok = await outboundRouter.sendImageToActive(
                                shot,
                                caption,
                                this.logger,
                            );
                            if (!ok && this._lastFinalText) {
                                // Image route failed (sender error etc.) — fall
                                // back to text so the user still gets the reply.
                                await outboundRouter.sendToActive(this._lastFinalText, this.logger);
                            }
                        } else if (this._lastFinalText) {
                            // No screenshot cached — just send the text reply.
                            await outboundRouter.sendToActive(this._lastFinalText, this.logger);
                        }
                    } catch (err) {
                        this.logger.warn('screenshot-attach-failed', { error: err.message });
                        if (this._lastFinalText) {
                            await outboundRouter.sendToActive(this._lastFinalText, this.logger);
                        }
                    }
                } else if (payload.type === 'done' && this._lastFinalText) {
                    // Platform doesn't support images (e.g. iMessage) —
                    // send the final reply text we deferred earlier.
                    await outboundRouter.sendToActive(this._lastFinalText, this.logger);
                }

                outboundRouter.endTurn(finished.id);
                if (this._pendingTurnResolver) this._pendingTurnResolver(finished.id);
            }
            this._lastStepPingAt = 0;
            this._lastFinalText = '';
        }
    }
}

module.exports = { Dispatcher };
