// messaging/common/dispatcher.js — inbound → /agent/step → outbound reply
//
// One dispatcher process-wide. Each bridge calls dispatcher.handleInbound
// with the normalized message; the dispatcher enforces the allowlist,
// holds the per-handle queue, sets the active outbound route, posts to
// /agent/step, and tears the route down when the matching `done` /
// `stopped` / `error` arrives on the shared WebSocket.

const allowlist = require('./allowlist');
const rateLimit = require('./rateLimit');
const outboundRouter = require('./outboundRouter');
const sessionBinder = require('./sessionBinder');
const agentClient = require('./agentClient');
const sanitizer = require('./sanitizer');
const { logDropped } = require('./log');

const OVERWHELMED_REPLY = "I'm overwhelmed right now — try again in a minute.";

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
    }

    async start() {
        this.sessionId = await sessionBinder.getRemoteSessionId(this.logger);
        this.logger.info('dispatcher-bound', {
            sessionId: this.sessionId,
            agentMode: this.agentMode,
            dryRun: this.dryRun,
        });
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
        if (_isOutboundText(payload)) {
            const text = sanitizer.sanitize(_extractText(payload));
            if (text) {
                const sent = await outboundRouter.sendToActive(text, this.logger);
                if (!sent) this.logger.warn('outbound-not-routed', { type: payload.type });
            }
        }
        if (_isTurnEndEvent(payload)) {
            const finished = outboundRouter.activeTurn();
            if (finished) {
                outboundRouter.endTurn(finished.id);
                if (this._pendingTurnResolver) this._pendingTurnResolver(finished.id);
            }
        }
    }
}

module.exports = { Dispatcher };
