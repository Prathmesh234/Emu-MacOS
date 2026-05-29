// pages/Chat.js — Chat page component
//
// Owns the chat UI: message list, input, status bar,
// WebSocket message handling, and action execution.
//
// Design change (Emu Design System v1 refactor — phases 1-3):
//   - Header      → MacWindow (chrome bar) + WindowHeader (Emu + status pill)
//   - ChatInput   → Composer (borderless, serif, hint row)
//   - Layout      → mac-chrome / mac-content / mac-main structure
//   All WebSocket, IPC, action dispatch, and API logic is unchanged.

const { ipcRenderer } = require('electron');
const { StepCard, DoneCard, ErrorCard, PlanCard, FileCard, SkillCard, HistoryPanel } = require('../components');
const { Greeting } = require('../components/conversation/Greeting');
const { Settings } = require('../components/frames/Settings');
const { TurnYou }      = require('../components/conversation/TurnYou');
const { TurnEmu }      = require('../components/conversation/TurnEmu');
const { renderPastSession } = require('../components/conversation/PastSessionRenderer');
const { MacWindow }    = require('../components/chrome/MacWindow');
const { WindowHeader } = require('../components/chrome/WindowHeader');
const { Composer }     = require('../components/chrome/Composer');
const { PermissionsCard } = require('../components/chrome/PermissionsCard');
const { renderMarkdown } = require('../components/markdown');
const { createEmuRunner } = require('../components/EmuRunner');
const { captureScreenshot, fullCapture } = require('../actions');
const { executeAction } = require('../actions/executor');
const { captureForStep } = require('../services/captureForStep');
const { createWindowManager } = require('../services/windowManager');
const { formatToolTrace, hasDedicatedToolEvent } = require('../services/traceLabels');
const store = require('../state/store');
const api = require('../services/api');
const { initWebSocket, setMessageHandler } = require('../services/websocket');
const remoteSessionObserver = require('../services/remoteSessionObserver');

// ── DOM refs (populated in mount) ────────────────────────────────────────

let chatContainer, chatWrapper, chatInput, header, historyPanel, winHeader;
let _historyPanelOpen = false;
let _viewingPastSession = false;
let _pastSessionId = null;
let _permissionsCard = null;
let _permissionsInitialCheckDone = false;
const DEFAULT_COWORKER_MISSING_PERMISSIONS = ['accessibility', 'screen'];
const PERMISSIONS_CARD_SUPPRESSED_KEY = 'emu-coworker-permissions-card-suppressed';

function isPermissionsCardSuppressed() {
    try {
        return localStorage.getItem(PERMISSIONS_CARD_SUPPRESSED_KEY) === '1';
    } catch (_) {
        return false;
    }
}

function suppressPermissionsCard() {
    try {
        localStorage.setItem(PERMISSIONS_CARD_SUPPRESSED_KEY, '1');
    } catch (_) { /* ignore storage failures */ }
}

// ── Helpers ──────────────────────────────────────────────────────────────

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
const COWORKER_SIDE_PANEL_DELAY_MS = 1500;

// Chain length threshold: auto-compact when the backend context chain exceeds this.
const COMPACT_THRESHOLD = 100;

// Generation counter: incremented on each new respond() call and on stop.
// Used to detect stale WS messages from a previous generation cycle so
// they don't accidentally execute actions after a stop or new task start.
let _generationId = 0;
let _terminalGenerationId = null;

/** Scroll chat to bottom after the browser has laid out new content. */
function scrollToBottom() {
    requestAnimationFrame(() => {
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    });
}

// Build a one-line trace label for coworker-mode driver calls. Strips the
// `cua_` prefix and only surfaces the args the user actually cares about
// (coords, element_index, key, text), keeping pid/window_id off the line so
// it stays readable when several calls fire in a row.
function _formatCuaCall(tool, argsJson, ok) {
    return formatToolTrace(tool, argsJson, { ok });
}

function syncGeneratingUI(generating) {
    store.setGenerating(generating);
    // Border glow only while agent is executing
    ipcRenderer.send('set-border', generating);
    // Design change: the working state is communicated by the window-header
    // status pill ("working" + pulsing dot). No need for a redundant hint
    // above the composer — leave it clean.
    chatInput.setMode(generating ? 'stop' : 'send');
    if (chatInput.setModelDisabled) chatInput.setModelDisabled(generating);
    chatInput.setTooltip('');
    // Update the window header status pill and lock mode during generation.
    if (winHeader) {
        winHeader.setStatus(generating ? 'working' : 'ready', generating);
        winHeader.setModeDisabled(generating);
    }
    // Disable the auto-mode toggle mid-generation
    if (header) header.setToggleDisabled(generating);
    // When generation ends, remove any lingering EmuRunner indicators so
    // they don't keep animating below a finished turn.
    if (!generating) {
        const el = store.state.currentAssistantEl;
        if (el) {
            el.querySelectorAll('.typing').forEach((t) => t.remove());
        }
    }
}

function markGenerationActive(reason = '') {
    _terminalGenerationId = null;
    if (reason) console.log(`[generation] active: ${reason} gen=${_generationId}`);
    syncGeneratingUI(true);
}

function markGenerationTerminal(reason = '') {
    _terminalGenerationId = _generationId;
    if (reason) console.log(`[generation] terminal: ${reason} gen=${_generationId}`);
    syncGeneratingUI(false);
}

function hasActiveGeneration() {
    return (
        _generationId > 0 &&
        _terminalGenerationId !== _generationId &&
        !store.state.isStopped
    );
}

function reassertGeneratingUI(reason = '') {
    if (!hasActiveGeneration()) return;
    const buttonLooksStopped = chatInput?.sendBtn?.textContent === 'stop';
    if (!store.state.isGenerating || !buttonLooksStopped) {
        console.warn(`[generation] reasserting Stop UI (${reason})`);
    }
    syncGeneratingUI(true);
}

function handlePostStepFailure(err, contextEl, source) {
    console.error(`[${source}] POST failed:`, err);
    const canStillReceiveStream =
        !err?.httpStatus &&
        store.state.ws &&
        store.state.ws.readyState === WebSocket.OPEN &&
        hasActiveGeneration();
    if (canStillReceiveStream) {
        showStatus('Still waiting for Emu…');
        reassertGeneratingUI(`${source}:post-failed`);
        return;
    }
    if (contextEl) {
        contextEl.textContent = `Backend unreachable: ${err.message}`;
    } else {
        showStatus(`Backend error: ${err.message}`);
    }
    markGenerationTerminal(`${source}:post-failed`);
}

// Re-attach a fresh EmuRunner ("the bird") at the bottom of the assistant
// turn so the user always sees the agent is thinking, even between steps.
// Called after each step/tool_event renders while generation is ongoing.
// The next step/tool_event removes any existing .typing before appending
// its card, so the runner naturally moves to the trailing position.
function ensureTypingIndicator(el) {
    if (!el) return;
    if (!store.state.isGenerating && !hasActiveGeneration()) return;
    if (store.state.isStopped) return;
    el.querySelectorAll('.typing').forEach((t) => t.remove());
    el.appendChild(createEmuRunner());
}

// ── Window helpers ───────────────────────────────────────────────────────
// Backed by services/windowManager.js — initialised after `header` is
// created in mount(). Until then, calls would no-op; nothing in the
// startup path uses these before mount() runs.
let _winMgr = null;
async function toggleWindow()      { if (_winMgr) return _winMgr.toggleWindow(); }
async function moveToSidePanel()   { if (_winMgr) return _winMgr.moveToSidePanel(); }
async function moveToCentered()    { if (_winMgr) return _winMgr.moveToCentered(); }
async function minimizeWindow()    { if (_winMgr) return _winMgr.minimizeWindow(); }
async function maximizeWindow()    { if (_winMgr) return _winMgr.maximizeWindow(); }
async function expandOrMaximizeWindow() {
    if (store.state.isSidePanel) return moveToCentered();
    return maximizeWindow();
}

async function moveCoworkerToSidePanelAfterDelay(generationId) {
    if (store.state.agentMode !== 'coworker' || store.state.isSidePanel) return true;

    await sleep(COWORKER_SIDE_PANEL_DELAY_MS);
    if (
        generationId !== _generationId ||
        store.state.isStopped ||
        (!store.state.isGenerating && !hasActiveGeneration())
    ) {
        return false;
    }

    try {
        await moveToSidePanel();
    } catch (err) {
        console.warn('[window] delayed coworker side-panel move failed:', err.message);
    }
    return true;
}

// ── Rendering ────────────────────────────────────────────────────────────

// Design change (Phase 5): replaced old EmptyState (emu SVG + "Hey I'm Emu")
// with the Idle greeting from the handoff design.
function showEmpty() {
    const greeting = Greeting();
    chatWrapper.appendChild(greeting.element);
}

// Design change (Phase 4): replaced Message('user'/'assistant') bubbles
// with TurnYou / TurnEmu blocks matching the handoff design.
// Returns: user → turn element; assistant → turn.body (mount point for trace lines).
function addMessage(role, content, index) {
    const greeting = chatWrapper.querySelector('.idle-greeting');
    if (greeting) greeting.remove();

    if (role === 'user') {
        const turn = TurnYou(content);
        chatWrapper.appendChild(turn.element);
        scrollToBottom();
        return turn.element;
    } else {
        const turn = TurnEmu(content);
        chatWrapper.appendChild(turn.element);
        scrollToBottom();
        return turn.body;
    }
}

function ensureAssistantBody() {
    const existing = store.state.currentAssistantEl;
    if (existing && existing.isConnected) return existing;

    const chat = store.getCurrentChat();
    if (!chat) return null;

    store.pushMessage(chat.id, { role: 'assistant', content: '', stepCount: 0 });
    const body = addMessage('assistant', '', chat.messages.length - 1);
    store.setAssistantEl(body);
    store.state.stepContainer = body;
    store.state.stepCount = store.state.stepCount || 0;
    return body;
}

// Design change (Phase 4): status messages now drive the WindowHeader pill
// instead of appending a visible bubble to the chat area. This matches the
// design's quiet aesthetic — status lives in the chrome, not the conversation.
function statusLooksLive(text) {
    return !/(failed|error|lost|unreachable|paused|stopped|rejected|denied)/i.test(String(text || ''));
}

function showStatus(text, live = statusLooksLive(text)) {
    if (winHeader) winHeader.setStatus(text, live);
}

function updateStatus(text, live = statusLooksLive(text)) {
    if (winHeader) winHeader.setStatus(text, live);
}

function removeStatus() {
    // Revert to the live/idle state that syncGeneratingUI will correct on next call
    const generating = store.state.isGenerating || hasActiveGeneration();
    if (winHeader) winHeader.setStatus(generating ? 'working' : 'ready', generating);
}

function invokeIpc(channel, payload) {
    if (window.electronAPI && typeof window.electronAPI.invoke === 'function') {
        return window.electronAPI.invoke(channel, payload);
    }
    return ipcRenderer.invoke(channel, payload);
}

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

function missingPermissionsFromText(text) {
    const value = String(text || '');
    const missing = [];
    if (/accessibility/i.test(value)) missing.push('accessibility');
    if (/screen recording|screen capture|screen/i.test(value)) missing.push('screen');
    return missing;
}

function isDriverBinaryMissing(text) {
    return /binary not found|build\/install|install it from/i.test(String(text || ''));
}

function shouldShowPermissionsForDriverFailure(resultOrError) {
    const explicit = normalizeMissingPermissions(
        resultOrError?.missing || resultOrError?.permissions?.missing || []
    );
    if (explicit.length) return true;

    const text = typeof resultOrError === 'string'
        ? resultOrError
        : [
            resultOrError?.error,
            resultOrError?.output,
            resultOrError?.message,
        ].filter(Boolean).join('\n');
    if (isDriverBinaryMissing(text)) return false;
    if (resultOrError?.permissionsRequired) return true;
    return /permission|accessibility|screen recording|screen capture|not authorized|not authorised|unauthori[sz]ed|requires/i.test(text);
}

function missingPermissionsFromDriverFailure(resultOrError) {
    const explicit = normalizeMissingPermissions(
        resultOrError?.missing || resultOrError?.permissions?.missing || []
    );
    if (explicit.length) return explicit;

    const text = typeof resultOrError === 'string'
        ? resultOrError
        : [
            resultOrError?.error,
            resultOrError?.output,
            resultOrError?.message,
        ].filter(Boolean).join('\n');
    const inferred = missingPermissionsFromText(text);
    if (inferred.length) return inferred;
    return shouldShowPermissionsForDriverFailure(resultOrError)
        ? DEFAULT_COWORKER_MISSING_PERMISSIONS
        : [];
}

function showPermissionsCard(missing, { fromInitialCheck = false } = {}) {
    if (isPermissionsCardSuppressed()) return;

    missing = normalizeMissingPermissions(missing);
    if (missing.length === 0) {
        if (_permissionsCard) _permissionsCard.update([]);
        suppressPermissionsCard();
        return;
    }

    if (_permissionsInitialCheckDone && !_permissionsCard && !fromInitialCheck) {
        return;
    }

    if (_permissionsCard) {
        _permissionsCard.update(missing);
        return;
    }

    _permissionsCard = PermissionsCard({
        missing,
        onAllow: (kind) => invokeIpc('permissions:open', kind),
        onRecheck: () => invokeIpc('emu-cua:recheck-permissions'),
        onDismiss: () => {
            suppressPermissionsCard();
            _permissionsCard = null;
        },
    });
    document.body.appendChild(_permissionsCard.element);
}

async function checkCoworkerPermissionsOnMount() {
    if (isPermissionsCardSuppressed()) {
        _permissionsInitialCheckDone = true;
        return;
    }

    try {
        const result = await invokeIpc('emu-cua:recheck-permissions');
        const missing = missingPermissionsFromDriverFailure(result);
        if (_permissionsCard) {
            _permissionsCard.update(missing);
        } else if (missing.length) {
            showPermissionsCard(missing, { fromInitialCheck: true });
        } else {
            suppressPermissionsCard();
        }
    } catch (err) {
        const missing = missingPermissionsFromDriverFailure(err);
        if (missing.length) {
            showPermissionsCard(missing, { fromInitialCheck: true });
        } else {
            console.warn('[permissions] coworker permission check failed:', err?.message || err);
            suppressPermissionsCard();
        }
    } finally {
        _permissionsInitialCheckDone = true;
    }
}

function subscribePermissionsRequired() {
    const handler = (payload) => {
        if (_permissionsInitialCheckDone || isPermissionsCardSuppressed()) return;
        showPermissionsCard(payload?.missing || []);
    };

    if (window.electronAPI && typeof window.electronAPI.on === 'function') {
        window.electronAPI.on('emu-cua:permissions-required', handler);
        return;
    }

    ipcRenderer.on('emu-cua:permissions-required', (_event, payload) => handler(payload));
}

// ── Chat selection ───────────────────────────────────────────────────────

function selectChat(id) {
    store.setCurrentChat(id);
    const chat = store.getChat(id);

    chatWrapper.innerHTML = '';

    if (chat && chat.messages.length > 0) {
        chat.messages.forEach((msg, i) => addMessage(msg.role, msg.content, i));
    } else {
        showEmpty(); // shows Idle greeting
    }

    chatInput.textarea.focus();
}

function newChat() {
    const id = store.createChat();
    _viewingPastSession = false;
    _pastSessionId = null;
    resetLiveTurnState();
    enableInput();
    selectChat(id);
    if (historyPanel) historyPanel.setActive(null);
}

function resetLiveTurnState() {
    store.state.currentAssistantEl = null;
    store.state.stepContainer = null;
    store.state.stepCount = 0;
}

// ── History panel ────────────────────────────────────────────────────────

function toggleHistoryPanel() {
    _historyPanelOpen = !_historyPanelOpen;
    const panel = historyPanel ? historyPanel.element : null;
    if (panel) {
        panel.classList.toggle('open', _historyPanelOpen);
    }
    if (_historyPanelOpen) {
        refreshHistory();
    }
}

function closeHistoryPanel() {
    if (!_historyPanelOpen) return;
    _historyPanelOpen = false;
    const panel = historyPanel ? historyPanel.element : null;
    if (panel) panel.classList.remove('open');
}

async function refreshHistory() {
    try {
        const sessions = await api.fetchSessionHistory();
        if (historyPanel) historyPanel.populate(sessions);
        // Re-apply live state to whichever pinned item just got
        // re-rendered (populate() wipes and rebuilds DOM).
        if (historyPanel && _remoteObserver && _remoteObserver.sessionId) {
            historyPanel.setLive(_remoteObserver.sessionId, _remoteObserver.live);
        }
    } catch (err) {
        console.warn('[history] failed to load:', err.message);
    }
}

// Debounced sidebar refresh, used by the remote-session observer so
// bursty step events don't trigger a /sessions/history call per event.
let _refreshHistoryTimer = null;
function refreshHistorySoon() {
    if (_refreshHistoryTimer) return;
    _refreshHistoryTimer = setTimeout(() => {
        _refreshHistoryTimer = null;
        // Only fetch when the sidebar is open (or about to be) — otherwise
        // we'd be doing pointless network work for an invisible panel.
        if (_historyPanelOpen) refreshHistory();
    }, 750);
}

// Passive observer for the messaging bridge's singleton remote-control
// session. Started in initSession() once the local session is up.
let _remoteObserver = null;

// Debounced refetch+re-render of the remote-control session, used when
// the user is currently viewing that session and the bridge produces
// new activity on a SECOND WebSocket the chat view doesn't listen to.
// Without this the user sees their inbound WhatsApp/iMessage message
// (loaded once when they opened the pin) but never the assistant reply
// that lands seconds later via the remote-observer WS.
//
// We refetch /sessions/{id}/messages and replay through renderPastSession
// rather than threading raw step events into handleWsMessage — the
// latter is tightly coupled to live-generation state for the LOCAL
// session and would mis-attribute coworker_target, generation IDs, etc.
let _remoteRerenderTimer = null;
function refreshRemoteSessionViewSoon(remoteId) {
    if (!remoteId) return;
    if (!_viewingPastSession || _pastSessionId !== remoteId) return;
    if (_remoteRerenderTimer) return;
    _remoteRerenderTimer = setTimeout(async () => {
        _remoteRerenderTimer = null;
        // Re-check on the trailing edge: the user may have navigated
        // away (or to a different session) during the debounce window.
        if (!_viewingPastSession || _pastSessionId !== remoteId) return;
        try {
            const messages = await api.fetchSessionMessages(remoteId);
            if (!messages || messages.length === 0) return;
            // Preserve scroll-at-bottom feel: if the user was already
            // pinned to the bottom, snap back after re-render.
            const wasAtBottom = chatContainer
                ? (chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight) < 32
                : true;
            store.hydrateSessionChat(remoteId, messages);
            chatWrapper.innerHTML = '';
            renderPastSession(chatWrapper, messages, addMessage);
            if (wasAtBottom) scrollToBottom();
        } catch (err) {
            console.warn('[remote-observer] re-render failed:', err.message);
        }
    }, 400);
}

async function loadPastSession(sessionId, prefetchedMessages = null) {
    try {
        const messages = prefetchedMessages !== null
            ? prefetchedMessages
            : await api.fetchSessionMessages(sessionId);

        // Set "viewing" state even when there are no messages yet — the
        // remote-control session can be opened from the sidebar BEFORE
        // the bridge has processed its first inbound message, and we
        // still need refreshRemoteSessionViewSoon() to fire when activity
        // arrives. Bailing here previously left the chat view stuck on
        // the local session while the user thought they'd switched.
        _viewingPastSession = true;
        _pastSessionId = sessionId || null;
        if (sessionId) {
            store.hydrateSessionChat(sessionId, messages || []);
            if (historyPanel) historyPanel.setActive(sessionId);
        }
        resetLiveTurnState();
        // Keep the composer visually identical to a fresh session — typing
        // and clicking send in a past session transparently continues it
        // (see sendMessage).
        enableInput();

        chatWrapper.innerHTML = '';

        // Delegate the (purely-presentational) rendering of past messages
        // to PastSessionRenderer so this module stays focused on live
        // session state. Behavior is byte-for-byte identical.
        if (messages && messages.length > 0) {
            renderPastSession(chatWrapper, messages, addMessage);
            scrollToBottom();
        }
    } catch (err) {
        console.warn('[history] failed to load session:', err.message);
    }
}

async function continuePastSession(oldSessionId) {
    try {
        closeHistoryPanel();

        const messages = await api.fetchSessionMessages(oldSessionId);

        const sessionId = await api.continueSession(oldSessionId, store.state.agentMode);

        if (messages && messages.length > 0) {
            await loadPastSession(sessionId, messages);
        } else {
            store.hydrateSessionChat(sessionId, []);
            resetLiveTurnState();
        }

        _viewingPastSession = false;
        _pastSessionId = null;
        store.setSession(sessionId);
        initWebSocket(sessionId);
        if (historyPanel) historyPanel.setActive(sessionId);
        enableInput();
        refreshHistory();
        return true;
    } catch (err) {
        console.warn('[continuePastSession] failed:', err.message);
        showStatus('Could not continue that session.');
        setTimeout(() => {
            if (!hasActiveGeneration()) removeStatus();
        }, 2500);
        enableInput();
        return false;
    }
}

function disableInput() {
    const container = chatInput.element;
    container.classList.add('composer-disabled');
    chatInput.textarea.disabled = true;
    chatInput.textarea.placeholder = '';
    chatInput.sendBtn.disabled = true;

    // Add overlay if not present
    if (!container.querySelector('.composer-overlay')) {
        const overlay = document.createElement('div');
        overlay.className = 'composer-overlay';
        overlay.textContent = 'Viewing past session — click ↩ in the sidebar to continue';
        container.appendChild(overlay);
    }
}

function enableInput() {
    const container = chatInput.element;
    container.classList.remove('composer-disabled');
    chatInput.textarea.disabled = false;
    chatInput.textarea.placeholder = 'Ask anything…';
    if (hasActiveGeneration()) {
        chatInput.setMode('stop');
        chatInput.sendBtn.disabled = false;
    } else if (!store.state.isGenerating) {
        chatInput.setMode('send');
        chatInput.sendBtn.disabled = !chatInput.textarea.value.trim();
    }

    const overlay = container.querySelector('.composer-overlay');
    if (overlay) overlay.remove();
}

// ── Message actions ──────────────────────────────────────────────────────
//
// editMessage / regenerate were removed during the Emu Design System v1
// refactor: the new TurnYou / TurnEmu blocks have no edit or regenerate
// affordance, and the old implementations queried `.message` (a class that
// no longer exists). If those affordances ever come back, they should
// query `.turn-you` / `.turn-emu` and use store.truncateMessages.

// ── Send / respond / stop ────────────────────────────────────────────────

async function stopAgent() {
    if (!store.state.isGenerating && !hasActiveGeneration()) return;

    console.log('[stopAgent] user requested stop');
    store.setStopped(true);
    // Bump generation ID so any in-flight WS messages from the old
    // generation are ignored — prevents stale actions from executing
    _generationId++;

    renderStoppedState();
    markGenerationTerminal('user-stop');

    // Notify backend — adds STOP to the context chain
    // The session itself is NOT destroyed — user can continue chatting
    api.stopAgent(store.state.sessionId)
        .catch((err) => console.warn('[stopAgent] backend call failed:', err.message));
}

function renderStoppedState() {
    // Show a stopped step card in the UI
    const { state } = store;
    store.state.stepCount = (store.state.stepCount || 0) + 1;
    const stepNum = store.state.stepCount;

    const stoppedCard = StepCard({
        action: { type: 'done' },
        done: true,
        final_message: 'Stopped by user',
        confidence: 1.0,
    }, stepNum);

    if (state.currentAssistantEl) {
        let container = state.stepContainer;
        if (container && !container.parentNode) {
            state.currentAssistantEl.appendChild(container);
        }
        if (container) {
            container.appendChild(stoppedCard.element);
        } else {
            state.currentAssistantEl.appendChild(stoppedCard.element);
        }
    }
    scrollToBottom();

    // Add visible STOP message in chat
    const chat = store.getCurrentChat();
    if (chat) {
        store.pushMessage(chat.id, { role: 'user', content: 'STOP' });
        addMessage('user', 'STOP', chat.messages.length - 1);
    }
}

async function sendMessage() {
    const text = chatInput.textarea.value.trim();
    if (!text) return;

    // Sending from a past-session view transparently continues that session.
    // Preserve the typed text, perform the continue flow, then fall through
    // to the normal send path against the freshly-minted session.
    if (_viewingPastSession) {
        const pendingText = text;
        const oldId = _pastSessionId;
        chatInput.textarea.value = '';
        chatInput.sendBtn.disabled = true;
        if (oldId) {
            const continued = await continuePastSession(oldId);
            if (!continued) {
                chatInput.textarea.value = pendingText;
                chatInput.sendBtn.disabled = false;
                return;
            }
        } else {
            _viewingPastSession = false;
        }
        chatInput.textarea.value = pendingText;
        chatInput.sendBtn.disabled = false;
        // Fall through with _viewingPastSession now false
    }

    // If currently generating, stop the current task first, then send the new message
    if (store.state.isGenerating || hasActiveGeneration()) {
        // Grab the text before stopping (stop clears state)
        const pendingText = text;
        await stopAgent();
        // Wait for stop to settle
        await sleep(300);
        // Now send the new message by putting it back and recursing
        chatInput.textarea.value = pendingText;
        chatInput.sendBtn.disabled = false;
        // Fall through to send logic below (isGenerating is now false)
    }

    let finalText = chatInput.textarea.value.trim();
    if (!finalText) return;

    // If user is refining a plan, prefix the message
    if (store.state.pendingPlanRefine) {
        finalText = `User wants to refine the plan: ${finalText}`;
        store.state.pendingPlanRefine = false;
        chatInput.textarea.placeholder = 'Ask anything...';
    }

    const chat = store.getCurrentChat();
    if (!chat) return;

    // Add user message
    store.pushMessage(chat.id, { role: 'user', content: finalText });
    addMessage('user', finalText, chat.messages.length - 1);
    store.updateChatPreview(chat.id, finalText);

    // Clear input
    chatInput.textarea.value = '';
    chatInput.textarea.style.height = 'auto';
    chatInput.sendBtn.disabled = true;

    // Wait for session
    if (!store.state.sessionId) {
        showStatus('Connecting to backend...');
        await new Promise(resolve => {
            const check = setInterval(() => {
                if (store.state.sessionId) { clearInterval(check); resolve(); }
            }, 100);
        });
        removeStatus();
    }

    // Don't move to side panel here — only move when a vision/action step arrives
    // (see handleWsMessage 'step' case)

    // Debug: full-capture
    if (/\bfull[-\s]?capture\b/i.test(finalText)) {
        showStatus('Full capture (panel excluded)...');
        const result = await fullCapture();
        removeStatus();
        const msg = result.success
            ? `Full capture saved: ${result.filename} (${result.width}\u00d7${result.height})`
            : `Full capture failed: ${result.error}`;
        store.pushMessage(chat.id, { role: 'assistant', content: msg });
        addMessage('assistant', msg, chat.messages.length - 1);
        return;
    }

    // Send text only — model will request a screenshot if it needs one
    await respond(chat);
}

async function respond(chat, base64Screenshot = null) {
    store.setStopped(false);
    _generationId++;
    const thisGenId = _generationId;
    store.state._generationId = thisGenId;
    markGenerationActive('respond');

    const lastUserMsg = store.getLastUserMessage(chat);

    store.pushMessage(chat.id, { role: 'assistant', content: '', stepCount: 0 });
    // Design change (Phase 4): contentEl is now TurnEmu.body — the direct
    // mount point for trace lines. No separate step-container div needed.
    const contentEl = addMessage('assistant', '', chat.messages.length - 1);
    contentEl.appendChild(createEmuRunner());

    store.setAssistantEl(contentEl);
    store.state.stepContainer = contentEl; // body IS the container
    store.state.stepCount = 0;

    try {
        const canContinue = await moveCoworkerToSidePanelAfterDelay(thisGenId);
        if (!canContinue) return;

        await api.postStep({
            sessionId:        store.state.sessionId,
            userMessage:      base64Screenshot ? '' : lastUserMsg,
            base64Screenshot: base64Screenshot || '',
            agentMode:        store.state.agentMode,
        });
    } catch (err) {
        handlePostStepFailure(err, contentEl, 'respond');
    }
}

/**
 * Continue the agent loop: capture screenshot and send to backend.
 * Called after the model requests a screenshot or after executing an action.
 */
async function continueLoop() {
    const loopGenId = _generationId;
    if (store.state.isStopped) {
        console.log('[continueLoop] stopped by user — bailing out');
        return;
    }
    // If generation changed since this continueLoop was called, bail
    if (loopGenId !== _generationId) {
        console.log('[continueLoop] generation changed — bailing out');
        return;
    }
    const chat = store.getCurrentChat();
    if (!chat) { console.warn('[continueLoop] no current chat'); return; }

    console.log('[continueLoop] capturing screenshot...');
    showStatus('Capturing screen...');
    const screenshot = await captureForStep();
    removeStatus();

    if (!screenshot.success) {
        console.error('[continueLoop] screenshot failed:', screenshot.error);
        showStatus('Screenshot failed: ' + (screenshot.error || 'unknown'));
        await sleep(1500);
        removeStatus();
        markGenerationTerminal('screenshot-failed');
        return;
    }

    const sizeKB = Math.round(screenshot.base64.length / 1024);
    console.log(`[continueLoop] screenshot OK (${sizeKB} KB) — posting to backend`);

    // Design change (Phase 8 polish): no inline screenshot card anymore.
    // The window-header status pill ("Capturing screen…") is the visible
    // feedback. Showing a thumbnail breaks the prose trace flow.
    store.state.stepCount = (store.state.stepCount || 0) + 1;

    const { state } = store;
    if (state.currentAssistantEl) {
        let container = state.stepContainer;
        if (container && !container.parentNode) {
            state.currentAssistantEl.appendChild(container);
        }
    }
    scrollToBottom();

    // Send screenshot to backend (no new assistant message bubble — reuse current)
    try {
        // Auto-compact if context chain is bloating
        const chainLen = store.state.lastChainLength || 0;
        if (chainLen >= COMPACT_THRESHOLD) {
            console.log(`[continueLoop] chain_length=${chainLen} >= ${COMPACT_THRESHOLD} — auto-compacting`);
            showStatus('Compacting context (summarising history)...');
            try {
                const result = await api.compactContext(store.state.sessionId);
                console.log(`[continueLoop] compact result:`, result);
                if (result.status === 'compacted') {
                    updateStatus(`Compacted: ${result.previous_length} → ${result.new_length} messages`);
                    store.state.lastChainLength = result.new_length;
                    await sleep(1000);
                }
            } catch (compactErr) {
                console.warn('[continueLoop] compact failed:', compactErr.message);
            }
            removeStatus();
        }

        await api.postStep({
            sessionId:        store.state.sessionId,
            userMessage:      '',
            base64Screenshot: screenshot.base64,
            agentMode:        store.state.agentMode,
        });
        console.log('[continueLoop] POST completed');
    } catch (err) {
        handlePostStepFailure(err, null, 'continueLoop');
    }
}

// ── WebSocket handler ────────────────────────────────────────────────────

async function handleWsMessage(data) {
    const { state } = store;
    // Capture the generation ID at the time this message was received
    const msgGenId = _generationId;

    switch (data.type) {
        case 'status':
            // Status updates logged to console only — no visible card
            console.log(`[status] ${data.message}`);
            reassertGeneratingUI('status');
            break;

        case 'connection_open':
            if (store.state.isGenerating) {
                removeStatus();
            }
            reassertGeneratingUI('connection_open');
            break;

        case 'connection_closed':
            if (store.state.isGenerating) {
                showStatus('Connection lost — reconnecting…');
            }
            reassertGeneratingUI('connection_closed');
            break;

        case 'log': {
            if (msgGenId !== _generationId) break;
            reassertGeneratingUI('log');
            if (!store.state.isGenerating) break;
            const message = data.message || '';
            if (!message.startsWith('[tool]')) break;
            if (!message.includes('[TOOL REJECTED]')) break;

            const root = ensureAssistantBody();
            if (!root) break;
            const typing = root.querySelector('.typing');
            if (typing) typing.remove();

            const wrap = document.createElement('div');
            wrap.className = 'trace resolved trace-error';
            const match = message.match(/^\[tool\]\s+([^(]+)\((.*?)\)\s+→/);
            wrap.textContent = match
                ? formatToolTrace(match[1], match[2], { ok: false })
                : 'Tool rejected';
            root.appendChild(wrap);
            scrollToBottom();
            break;
        }

        case 'step': {
            // If the generation has changed since this message was queued,
            // this is a stale response from a previous generation — ignore it
            if (msgGenId !== _generationId) {
                console.log(`[step] stale message (gen ${msgGenId} vs current ${_generationId}) — ignoring`);
                break;
            }
            if (!data.done) {
                reassertGeneratingUI('step');
            }
            removeStatus();

            // Track chain length for auto-compact
            if (data.chain_length != null) {
                store.state.lastChainLength = data.chain_length;
            }

            // PLAN §6.5: persist coworker (pid, window_id) so captureForStep
            // can pull the target-window screenshot on the next turn. Backend
            // sends this key in coworker mode (null when no target yet);
            // sets to null to clear, omits in remote mode.
            if ('coworker_target' in data) {
                store.setCoworkerTarget(data.coworker_target);
            }

            // Window placement per step:
            //   Once the window has moved to the side panel, never return it
            //   to centered automatically. The user controls expansion via
            //   the top green expand button only.
            if (!data.done) {
                await moveToSidePanel();
            }

            // Increment step counter
            store.state.stepCount = (store.state.stepCount || 0) + 1;
            const stepNum = store.state.stepCount;

            const stepCard = StepCard(data, stepNum);

            if (state.currentAssistantEl) {
                // Remove typing indicator on first step
                const typing = state.currentAssistantEl.querySelector('.typing');
                if (typing) typing.remove();

                // Ensure step container exists
                let container = state.stepContainer;
                if (container && !container.parentNode) {
                    state.currentAssistantEl.appendChild(container);
                }
                if (container) {
                    container.appendChild(stepCard.element);
                } else {
                    state.currentAssistantEl.appendChild(stepCard.element);
                }
            }

            if (state.currentChat) {
                const last = state.currentChat.messages[state.currentChat.messages.length - 1];
                last.content = data.reasoning || data.final_message || '';
                last.stepData = data;
                last.stepCount = stepNum;
            }

            scrollToBottom();

            if (data.done) {
                markGenerationTerminal('step-done');
                break;
            }

            // Bail out if user stopped
            if (store.state.isStopped) {
                console.log('[step] user stopped — not executing action');
                markGenerationTerminal('step-stopped');
                break;
            }

            // Don't re-attach the running emu between steps — it looked goofy
            // when actions were flowing fast (screenshot → click → screenshot).
            // The initial runner before step 1 stays; subsequent steps just
            // animate one after another for a clean flow.
            if (data.action) {
                try {
                    if (data.action.type === 'screenshot') {
                        console.log('[step] model requested screenshot — calling continueLoop');
                        await continueLoop();
                    } else {
                        console.log(`[step] executing action: ${data.action.type}`);
                        await executeAction(data.action, stepCard.element);
                        console.log('[step] action executed — sleeping 500ms');
                        await sleep(500);
                        console.log('[step] calling continueLoop');
                        await continueLoop();
                        console.log('[step] continueLoop done');
                    }
                } catch (err) {
                    console.error('[step] loop error:', err);
                    showStatus(`Agent error: ${err.message}`);
                    await sleep(2000);
                    removeStatus();
                    markGenerationTerminal('action-loop-error');
                }
            } else {
                showStatus('Agent paused: no action returned.');
                markGenerationTerminal('step-without-action');
            }
            break;
        }

        case 'done': {
            removeStatus();
            const msg = data.message || 'Task complete.';

            // If steps already rendered (stepCount > 0), the last StepCard
            // with done:true already shows the final_message — skip duplicate.
            const alreadyHandledByStep = (store.state.stepCount || 0) > 0;

            if (state.currentAssistantEl) {
                const typing = state.currentAssistantEl.querySelector('.typing');
                if (typing) typing.remove();

                if (!alreadyHandledByStep) {
                    // No steps ran — conversational reply (bootstrap, chat).
                    // Render as body text inside the TurnEmu block.
                    state.currentAssistantEl.innerHTML = '';
                    const textEl = document.createElement('div');
                    textEl.className = 'turn-text';
                    renderMarkdown(textEl, msg);
                    state.currentAssistantEl.appendChild(textEl);
                }
            }
            if (state.currentChat) {
                state.currentChat.messages[state.currentChat.messages.length - 1].content = msg;
            }
            markGenerationTerminal('done-event');
            scrollToBottom();
            break;
        }

        case 'stopped':
            removeStatus();
            console.log('[ws] received stopped from backend');
            markGenerationTerminal('stopped-event');
            break;

        case 'plan_review_request':
        case 'plan_review': {
            if (msgGenId !== _generationId) break;
            removeStatus();
            if (!store.state.autoMode) {
                // Plan review is a real pause: the backend has returned and
                // is waiting for user approval/refinement, so the composer
                // should not remain in Stop/working mode.
                markGenerationTerminal('plan-review');
            }

            const root = ensureAssistantBody();
            if (root) {
                const typing = root.querySelector('.typing');
                if (typing) typing.remove();

                let container = state.stepContainer;
                if (container && !container.parentNode) {
                    root.appendChild(container);
                }

                const planCard = PlanCard(data.content);

                // In auto mode, auto-accept the plan without user confirmation
                if (store.state.autoMode) {
                    markGenerationActive('plan-auto-accepted');
                    planCard.acceptBtn.disabled = true;
                    planCard.refineBtn.disabled = true;
                    planCard.acceptBtn.classList.add('chosen');

                    if (container) {
                        container.appendChild(planCard.element);
                    } else {
                        root.appendChild(planCard.element);
                    }
                    scrollToBottom();

                    showStatus('Plan auto-accepted — starting execution...');
                    api.postStep({
                        sessionId: store.state.sessionId,
                        userMessage: '[PLAN APPROVED] The user has accepted the plan. Proceed with execution — take a screenshot to orient yourself and begin from step 1.',
                        base64Screenshot: '',
                        agentMode: store.state.agentMode,
                    }).catch((err) => {
                        console.error('[plan_review] auto-accept failed:', err);
                        removeStatus();
                        handlePostStepFailure(err, null, 'plan_review:auto_accept');
                    });
                    break;
                }

                planCard.acceptBtn.addEventListener('click', async () => {
                    planCard.acceptBtn.disabled = true;
                    planCard.refineBtn.disabled = true;
                    planCard.acceptBtn.classList.add('chosen');

                    // User accepted — inject approval into context and resume
                    store.setStopped(false);
                    markGenerationActive('plan-accepted');
                    showStatus('Plan accepted — starting execution...');
                    try {
                        await api.postStep({
                            sessionId: store.state.sessionId,
                            userMessage: '[PLAN APPROVED] The user has accepted the plan. Proceed with execution — take a screenshot to orient yourself and begin from step 1.',
                            base64Screenshot: '',
                            agentMode: store.state.agentMode,
                        });
                    } catch (err) {
                        console.error('[plan_review] accept failed:', err);
                        removeStatus();
                        handlePostStepFailure(err, null, 'plan_review:accept');
                    }
                }, { once: true });

                planCard.refineBtn.addEventListener('click', async () => {
                    planCard.acceptBtn.disabled = true;
                    planCard.refineBtn.disabled = true;
                    planCard.refineBtn.classList.add('chosen');

                    // Pause — let user type refinement
                    markGenerationTerminal('plan-refine');
                    store.state.pendingPlanRefine = true;
                    chatInput.textarea.placeholder = 'Describe how to refine the plan…';
                    chatInput.textarea.focus();
                }, { once: true });

                if (container) {
                    container.appendChild(planCard.element);
                } else {
                    root.appendChild(planCard.element);
                }
            }
            scrollToBottom();
            break;
        }

        case 'tool_event': {
            if (msgGenId !== _generationId) break;
            reassertGeneratingUI('tool_event');
            removeStatus();

            const root = ensureAssistantBody();
            if (root) {
                const typing = root.querySelector('.typing');
                if (typing) typing.remove();

                let container = state.stepContainer;
                if (container && !container.parentNode) {
                    root.appendChild(container);
                }

                if (data.event === 'file_written') {
                    const fileCard = FileCard(data.filename, data.action, data.filepath);
                    if (container) {
                        container.appendChild(fileCard.element);
                    } else {
                        root.appendChild(fileCard.element);
                    }
                } else if (data.event === 'skill_used') {
                    const skillCard = SkillCard(data.skill_name);
                    if (container) {
                        container.appendChild(skillCard.element);
                    } else {
                        root.appendChild(skillCard.element);
                    }
                } else if (data.event === 'hermes_invoked') {
                    const toolWrap = document.createElement('div');
                    toolWrap.className = 'trace resolved hermes-trace';
                    toolWrap.dataset.hermes = 'running';
                    toolWrap.textContent = 'Launch Hermes';
                    if (container) {
                        container.appendChild(toolWrap);
                    } else {
                        root.appendChild(toolWrap);
                    }
                } else if (data.event === 'hermes_done') {
                    // Update the most recent still-running hermes trace card
                    // (if any) with the final status. Keeps the UI to a
                    // single line per Hermes invocation.
                    const root = state.currentAssistantEl || document;
                    const cards = root.querySelectorAll('.hermes-trace[data-hermes="running"]');
                    const card = cards.length ? cards[cards.length - 1] : null;
                    const status = data.status || 'completed';
                    const label = status === 'completed'
                        ? 'Hermes finished'
                        : `Hermes ${status}`;
                    if (card) {
                        card.dataset.hermes = status;
                        card.textContent = label;
                    } else {
                        // No matching start card (race / replay) — append a
                        // standalone done line so the user still sees it.
                        const wrap = document.createElement('div');
                        wrap.className = 'trace resolved hermes-trace';
                        wrap.dataset.hermes = status;
                        wrap.textContent = label;
                        if (container) container.appendChild(wrap);
                        else root.appendChild(wrap);
                    }
                } else if (data.event === 'tool_call_started') {
                    if (hasDedicatedToolEvent(data.tool)) break;
                    const wrap = document.createElement('div');
                    wrap.className = 'trace resolved';
                    wrap.textContent = formatToolTrace(data.tool, data.args);
                    if (container) container.appendChild(wrap);
                    else root.appendChild(wrap);
                } else if (data.event === 'cua_driver_call' || data.event === 'raise_app') {
                    if (data.permissions_required || data.missing_permissions) {
                        showPermissionsCard(data.missing_permissions || data.missing || []);
                    }
                    // Coworker-mode tool calls (cua_click, cua_type_text,
                    // raise_app, etc). Show one trace line per call so the
                    // user can see what the agent is actually doing — same
                    // pattern remote mode uses for click/navigate/etc.
                    const wrap = document.createElement('div');
                    wrap.className = 'trace resolved';
                    const label = data.event === 'raise_app'
                        ? formatToolTrace('raise_app', { app_name: data.app_name })
                        : _formatCuaCall(data.tool, data.args, data.ok);
                    wrap.textContent = label;
                    if (container) container.appendChild(wrap);
                    else root.appendChild(wrap);
                }
            }
            // Don't re-show the running emu between tool events either —
            // the trace lines (file_written / skill_used / hermes traces)
            // are enough visual feedback. Keeps the action flow clean.
            scrollToBottom();
            break;
        }

        case 'error':
            removeStatus();
            if (state.currentAssistantEl) {
                const typing = state.currentAssistantEl.querySelector('.typing');
                if (typing) typing.remove();

                const errCard = ErrorCard(data.message);
                let container = state.stepContainer;
                if (container && container.parentNode) {
                    container.appendChild(errCard.element);
                } else {
                    state.currentAssistantEl.appendChild(errCard.element);
                }
            }
            markGenerationTerminal('error-event');
            break;
    }
}

// ── Action execution ─────────────────────────────────────────────────────
// `executeAction` lives in actions/executor.js — imported at the top.
// The page calls it directly inside the WS step-handler.

// ── Session bootstrap ────────────────────────────────────────────────────

async function initSession() {
    console.log('[session] attempting to create session...');
    try {
        const id = await api.createSession();
        if (!id) {
            throw new Error('createSession returned empty id');
        }
        store.setSession(id);
        initWebSocket(id);
        console.log('[session] created successfully:', id);
        // Backend is up — load session history for the sidebar
        refreshHistory();
        // Start the passive observer for the messaging bridge's remote-
        // control session. This is what makes WhatsApp / iMessage activity
        // surface in the desktop sidebar in real time, matching the parity
        // promise: a session driven by a phone message looks indistinguishable
        // from one started locally.
        if (!_remoteObserver) {
            _remoteObserver = remoteSessionObserver.start({
                onActivity: ({ sessionId: remoteId, preview }) => {
                    if (!historyPanel) return;
                    if (preview) historyPanel.setLivePreview(remoteId, preview);
                    refreshHistorySoon();
                    // If the user is currently viewing this remote-control
                    // session, refetch + re-render so step pings, assistant
                    // replies, and turn-end cards land in the chat view in
                    // near-real-time (debounced inside the helper).
                    refreshRemoteSessionViewSoon(remoteId);
                },
                onStateChange: ({ live, sessionId: remoteId, rotated }) => {
                    if (!historyPanel) return;
                    historyPanel.setLive(remoteId, live);
                    // On rotation OR on turn-end, pull the sidebar so the
                    // new pinned item appears (rotation) or `last_active`
                    // sort order updates (turn-end).
                    if (rotated || !live) refreshHistorySoon();
                    // Turn-end is the most important refresh trigger: the
                    // conversation.json on disk has just been finalized
                    // with the assistant reply we want to show.
                    if (!live) refreshRemoteSessionViewSoon(remoteId);
                },
            });
        }
    } catch (err) {
        console.warn('[session] failed:', err.message);
        console.log('[session] retrying in 2s... (is backend running on localhost:8000?)');
        setTimeout(initSession, 2000);
    }
}

// ── Settings modal ───────────────────────────────────────────────────────

function openSettings() {
    const settings = Settings({});
    document.body.appendChild(settings.element);
}

// ── Mount ────────────────────────────────────────────────────────────────

function mount(appEl) {
    // Wire up WS handler
    setMessageHandler(handleWsMessage);
    subscribePermissionsRequired();

    // ── MacWindow chrome bar (traffic lights + title + actions) ──────────
    // `header` keeps the same variable name so all existing callers
    // (setExpandVisible, setToggleDisabled, setCompact) work unchanged.
    header = MacWindow({
        onMaximize:        expandOrMaximizeWindow,
        onMinimize:        minimizeWindow,
        onClose:           () => window.close(),
        onNewTask:         newChat,
        onOpenSettings:    openSettings,
    });

    appEl.appendChild(header.chromeEl);

    // Window manager depends on `header` — wire it up immediately so
    // the toggle/move/min/max wrappers above start delegating.
    _winMgr = createWindowManager(header);

    // ── mac-content (sidebar + main, flex row) ────────────────────────────
    appEl.appendChild(header.contentEl);

    // ── History sidebar ───────────────────────────────────────────────────
    historyPanel = HistoryPanel({
        onNewChat:         () => newChat(),
        onSelectSession:   (sid) => { closeHistoryPanel(); loadPastSession(sid); },
        onContinueSession: (sid) => continuePastSession(sid),
        onToggle:          () => toggleHistoryPanel(),
    });
    header.contentEl.appendChild(historyPanel.element);

    // ── Main column (window-header + chat body + composer) ───────────────
    const macMain = document.createElement('div');
    macMain.className = 'mac-main';
    header.contentEl.appendChild(macMain);

    // Clicking anywhere in the main column collapses the sessions panel
    // back to its original hidden state if it is currently open.
    // Exclude the sidebar toggle button itself so the subsequent click
    // doesn't re-open what we just closed.
    const _onMacMainMouseDown = (e) => {
        if (!_historyPanelOpen) return;
        if (e.target.closest && e.target.closest('.window-header-sidebar-btn')) return;
        closeHistoryPanel();
    };
    macMain.addEventListener('mousedown', _onMacMainMouseDown, true);

    // Window header: "Emu" mark + sidebar toggle + status pill
    winHeader = WindowHeader({ onToggleSidebar: toggleHistoryPanel });
    macMain.appendChild(winHeader.element);

    // Chat body (scrollable)
    chatContainer = document.createElement('div');
    chatContainer.className = 'chat-container';
    chatWrapper = document.createElement('div');
    chatWrapper.className = 'chat-wrapper';
    chatContainer.appendChild(chatWrapper);
    macMain.appendChild(chatContainer);

    // Auto-scroll whenever new content is added or elements resize
    const observer = new MutationObserver(() => scrollToBottom());
    observer.observe(chatWrapper, { childList: true, subtree: true, attributes: true });

    // Composer (drop-in replacement for ChatInput — same .textarea / .sendBtn / .setMode / .setTooltip API)
    chatInput = Composer(sendMessage, stopAgent);
    macMain.appendChild(chatInput.element);

    // Keyboard shortcuts — use addEventListener so we don't clobber any
    // other handler that may have set document.onkeydown.
    const _onChatKeyDown = (e) => {
        if (e.ctrlKey && e.shiftKey && e.key === 'N') {
            e.preventDefault();
            newChat();
        }
    };
    document.addEventListener('keydown', _onChatKeyDown);

    // Border glow is driven by syncGeneratingUI — starts hidden
    ipcRenderer.send('set-border', false);

    newChat();
    chatInput.textarea.focus();
    initSession();
    setTimeout(checkCoworkerPermissionsOnMount, 250);
}

module.exports = { mount, newChat, selectChat };
