// components/conversation/PastSessionRenderer.js
//
// Renders a read-only view of a past session's messages into a chat
// wrapper element. Extracted verbatim from pages/Chat.js (loadPastSession's
// render block) so the main page module stays focused on live state.
//
// Inputs:
//   chatWrapper     — the .chat-wrapper element to append into (already cleared)
//   messages        — array from /sessions/{id}/messages
//   addMessage(role, content)
//                   — page-level helper that creates user/assistant turn DOM
//                     and returns the mount point (used for the user-message path)
//
// Past-session replay mirrors the live session UI: user messages are shown
// fully, while actions/tools are displayed as the same compact trace lines
// users see while the agent is running.

const { TurnEmu } = require('./TurnEmu');
const { StepCard, FileCard, SkillCard } = require('../index');
const { formatToolTrace } = require('../../services/traceLabels');

// Messages dispatched by the messaging/ bridge are prefixed with their
// origin (e.g. "[whatsapp:+15551234567] open Safari and …"). Past-session
// replay strips the prefix and renders it as a small origin chip above
// the message so the conversation reads naturally.
const ORIGIN_RE = /^\[(whatsapp|imessage|discord):([^\]]+)\]\s*/;

function _extractOrigin(content) {
    const raw = String(content || '');
    const m = raw.match(ORIGIN_RE);
    if (!m) return { content: raw, origin: null };
    return {
        content: raw.slice(m[0].length),
        origin: { platform: m[1], handle: m[2] },
    };
}

function _appendOriginChip(turnEl, origin) {
    if (!turnEl || !origin) return;
    const chip = document.createElement('div');
    chip.className = `turn-origin-chip turn-origin-${origin.platform}`;
    const icon = origin.platform === 'imessage' ? '💬' : '📱';
    chip.textContent = `${icon} ${origin.platform} · ${origin.handle}`;
    chip.title = `via ${origin.platform}`;
    // Insert below the "You" label, above the body text.
    const label = turnEl.querySelector('.turn-label');
    if (label && label.nextSibling) {
        turnEl.insertBefore(chip, label.nextSibling);
    } else {
        turnEl.appendChild(chip);
    }
}

function renderPastSession(chatWrapper, messages, addMessage) {
    if (!messages || messages.length === 0) return;

    let stepNum = 0;
    let currentAssistantBubble = null;
    let currentStepContainer = null;

    function ensureAssistantBubble() {
        if (!currentAssistantBubble) {
            const turn = TurnEmu('');
            currentAssistantBubble = turn.element;
            currentStepContainer  = turn.body;
            chatWrapper.appendChild(currentAssistantBubble);
        }
        return currentStepContainer;
    }

    function flushAssistantBubble() {
        currentAssistantBubble = null;
        currentStepContainer = null;
    }

    function shouldRenderToolMessage(meta, content) {
        const status = meta.status || '';
        if (status === 'started' || status === 'rejected') return true;
        if ((content || '').includes('[TOOL REJECTED]')) return true;
        if ((content || '').startsWith('[tool:start]')) return true;
        if ((content || '').startsWith('[tool]')) return false;
        // Older logs may not have status metadata; render those rather than
        // hiding potentially important trace entries.
        return !status;
    }

    function appendPlainMessage(container, content) {
        if (!content || !content.trim()) return;
        const textEl = document.createElement('div');
        textEl.className = 'turn-text';
        textEl.textContent = content;
        container.appendChild(textEl);
    }

    function renderToolTrace(meta, content) {
        if (!shouldRenderToolMessage(meta, content)) return;

        const container = ensureAssistantBubble();
        stepNum++;

        const toolName = meta.tool_name || toolNameFromContent(content) || '';
        const rejected = meta.status === 'rejected' || content.includes('[TOOL REJECTED]');

        if (toolName === 'use_skill' && !rejected) {
            let skillName = 'Unknown';
            try {
                const parsed = JSON.parse(meta.args || '{}');
                skillName = parsed.skill_name || 'Unknown';
            } catch (_) {}
            const skillCard = SkillCard(skillName);
            container.appendChild(skillCard.element);
            return;
        }

        if (toolName === 'write_session_file' && !rejected) {
            let filename = 'file';
            try {
                const parsed = JSON.parse(meta.args || '{}');
                filename = parsed.filename || 'file';
            } catch (_) {}
            const fileCard = FileCard(filename, 'created', null);
            container.appendChild(fileCard.element);
            return;
        }

        const toolWrap = document.createElement('div');
        toolWrap.className = rejected ? 'trace resolved trace-error' : 'trace resolved';
        toolWrap.textContent = formatToolTrace(toolName || 'tool', meta.args || argsFromContent(content) || '{}', { ok: !rejected });
        container.appendChild(toolWrap);
    }

    function toolNameFromContent(content) {
        const match = String(content || '').match(/^\[tool(?::start)?\]\s+([^(]+)\(/);
        return match ? match[1].trim() : '';
    }

    function argsFromContent(content) {
        const match = String(content || '').match(/^\[tool(?::start)?\]\s+[^(]+\((.*)\)(?:\s+→.*)?$/s);
        return match ? match[1] : '';
    }

    messages.forEach(msg => {
        const role = msg.role;
        const content = msg.content || '';
        const meta = msg.metadata || {};

        // Skip screenshot entries
        if (content === '<screenshot>') return;

        if (role === 'user') {
            flushAssistantBubble();
            const { content: cleanContent, origin } = _extractOrigin(content);
            const userTurnEl = addMessage('user', cleanContent);
            if (origin) _appendOriginChip(userTurnEl, origin);
            return;
        }

        if (role === 'assistant') {
            // Done message — render as DoneCard or plain text
            const finalMsg = meta.final_message || content.replace(/^DONE\s*—?\s*/, '');
            if (finalMsg) {
                const container = ensureAssistantBubble();
                stepNum++;
                const doneCard = StepCard({
                    action: { type: 'done' },
                    done: true,
                    final_message: finalMsg,
                    confidence: 1.0,
                }, stepNum);
                container.appendChild(doneCard.element);
            }
            flushAssistantBubble();
            return;
        }

        if (role === 'tool') {
            renderToolTrace(meta, content);
            return;
        }

        if (content.startsWith('[tool:start]')) {
            renderToolTrace({ ...meta, status: meta.status || 'started' }, content);
            return;
        }

        if (content.startsWith('[tool]')) {
            if (content.includes('[TOOL REJECTED]')) {
                renderToolTrace({ ...meta, status: meta.status || 'rejected' }, content);
            }
            return;
        }

        if (role === 'action') {
            const container = ensureAssistantBubble();
            stepNum++;

            const actionType = meta.action_type || content.split(/\s+/)[0] || 'unknown';
            const confidence = meta.confidence != null ? meta.confidence : null;
            const reasoning = meta.reasoning || '';
            const actionPayload = meta.action || { type: actionType };

            const stepCard = StepCard({
                action: actionPayload,
                done: false,
                confidence: confidence,
                reasoning_content: reasoning,
            }, stepNum);

            // Past sessions are fully resolved: hide the blinking caret
            // and drop the pending "…" badge. The trace text alone
            // communicates what the agent did.
            stepCard.element.classList.add('resolved');
            const badge = stepCard.element.querySelector('.step-action-status');
            if (badge) badge.remove();

            container.appendChild(stepCard.element);
            return;
        }

        if (content) {
            appendPlainMessage(ensureAssistantBubble(), content);
        }
    });
}

module.exports = { renderPastSession };
