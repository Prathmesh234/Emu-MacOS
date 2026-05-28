// HistoryPanel — sessions sidebar
//
// Part of the Emu Design System v1 refactor (see FRONTEND_REDESIGN.md).
// Design source: Emu-handoff.zip → project/frames/frames-a.jsx > F_WorkingSidebar
//
// Design change: replaces the collapsible-strip + hamburger approach with a
// clean 240px sidebar: "Emu" mark → "+ new session" → sessions grouped by
// date (Today / Yesterday / Earlier) → user avatar footer.
//
// Preserved API (Chat.js unchanged):
//   { element, populate(sessions), setActive(sessionId) }
// Preserved class: .history-panel, .history-panel.open (CSS transition).

function HistoryPanel({ onNewChat, onSelectSession, onContinueSession, onToggle }) {
    const panel = document.createElement('div');
    panel.className = 'history-panel';

    // ── Inner wrapper (width: 240px; clipped by panel overflow) ──────────
    const inner = document.createElement('div');
    inner.className = 'history-panel-inner';

    // ── Header: "Emu" mark + close ×  ────────────────────────────────────
    const top = document.createElement('div');
    top.className = 'history-panel-top';

    const title = document.createElement('div');
    title.className = 'history-panel-title';
    title.textContent = 'Emu';
    top.appendChild(title);

    const closeBtn = document.createElement('button');
    closeBtn.className = 'history-panel-close';
    closeBtn.title = 'Close sidebar';
    closeBtn.textContent = '×';
    closeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (onToggle) onToggle();
    });
    top.appendChild(closeBtn);
    inner.appendChild(top);

    // ── New session button ────────────────────────────────────────────────
    const newBtn = document.createElement('button');
    newBtn.className = 'history-new-chat';
    newBtn.setAttribute('aria-label', 'New session');

    const plus = document.createElement('span');
    plus.className = 'history-new-chat-plus';
    plus.textContent = '+';

    const newLabel = document.createElement('span');
    newLabel.className = 'history-new-chat-label';
    newLabel.textContent = 'new session';

    newBtn.appendChild(plus);
    newBtn.appendChild(newLabel);
    newBtn.addEventListener('click', () => { if (onNewChat) onNewChat(); });
    inner.appendChild(newBtn);

    // ── Scrollable list ───────────────────────────────────────────────────
    const list = document.createElement('div');
    list.className = 'history-list';
    inner.appendChild(list);

    // ── User footer ───────────────────────────────────────────────────────
    const footer = document.createElement('div');
    footer.className = 'history-panel-footer';

    const username = _displayName();
    const initial  = username.charAt(0).toUpperCase();

    const avatar = document.createElement('div');
    avatar.className = 'history-panel-avatar';
    avatar.textContent = initial;
    footer.appendChild(avatar);

    const nameEl = document.createElement('span');
    nameEl.className = 'history-panel-username';
    nameEl.textContent = username;
    footer.appendChild(nameEl);

    inner.appendChild(footer);
    panel.appendChild(inner);

    // ── State ─────────────────────────────────────────────────────────────
    let _activeId = null;

    // Sessions currently generating in the background (i.e. driven by the
    // messaging bridge). Stored separately from _activeId so a remote
    // session can be "live" while the user is viewing a local session.
    const _liveIds = new Set();
    // Latest preview text per session, used by setLivePreview() to update
    // the sidebar item without re-running populate().
    const _previews = new Map();

    function setActive(sessionId) {
        _activeId = sessionId;
        list.querySelectorAll('.history-item').forEach(el => {
            const isActive = el.dataset.sessionId === sessionId;
            el.classList.toggle('active', isActive);
            // Show/hide the pulsing dot
            let dot = el.querySelector('.history-item-dot');
            if (isActive && !dot) {
                dot = document.createElement('span');
                dot.className = 'history-item-dot';
                el.insertBefore(dot, el.firstChild);
            } else if (!isActive && dot) {
                dot.remove();
            }
        });
    }

    // Mark a session as actively generating in the background. Adds a
    // pulsing "live" badge to the sidebar item (distinct from the active
    // pulsing dot used for the currently-viewed session). Used by the
    // remoteSessionObserver so WhatsApp / iMessage activity surfaces
    // even when the user is looking at a different session.
    function setLive(sessionId, isLive) {
        if (!sessionId) return;
        if (isLive) _liveIds.add(sessionId); else _liveIds.delete(sessionId);
        const el = list.querySelector(`.history-item[data-session-id="${CSS.escape(sessionId)}"]`);
        if (!el) return;
        el.classList.toggle('history-item-live', isLive);
        let badge = el.querySelector('.history-item-live-badge');
        if (isLive && !badge) {
            badge = document.createElement('span');
            badge.className = 'history-item-live-badge';
            badge.title = 'Generating now';
            badge.textContent = '●';
            el.appendChild(badge);
        } else if (!isLive && badge) {
            badge.remove();
        }
    }

    // Update the preview text shown on a sidebar item without rebuilding
    // the list. No-ops silently when the session isn't currently rendered
    // (e.g. user closed the sidebar before activity arrived).
    function setLivePreview(sessionId, text) {
        if (!sessionId || !text) return;
        _previews.set(sessionId, text);
        const el = list.querySelector(`.history-item[data-session-id="${CSS.escape(sessionId)}"]`);
        if (!el) return;
        const textEl = el.querySelector('.history-item-text');
        if (!textEl) return;
        // Pinned items keep their fixed label; only update the regular
        // preview text so we don't clobber "Remote control".
        if (el.classList.contains('history-item-pinned')) {
            // For pinned items, prepend a subdued live preview line below
            // the label by stashing it as a title — minimal DOM churn.
            el.title = text;
            return;
        }
        textEl.textContent = text;
    }

    function populate(sessions) {
        list.innerHTML = '';

        if (!sessions || sessions.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'history-empty';
            empty.textContent = 'No past sessions';
            list.appendChild(empty);
            return;
        }

        // The remote-control session (inbound from WhatsApp / iMessage) is
        // pinned to the top with a distinct label, regardless of date —
        // it's intentionally a single long-lived session, not a chat
        // bucket. All other sessions group by date as usual.
        const pinned = sessions.filter(s => s.kind === 'remote_control');
        const regular = sessions.filter(s => s.kind !== 'remote_control');

        if (pinned.length) {
            const groupEl = document.createElement('div');
            groupEl.className = 'history-group-label';
            groupEl.textContent = 'Remote';
            list.appendChild(groupEl);
            pinned.forEach(session => list.appendChild(_renderItem(session, { pinned: true })));
        }

        // Group sessions by date bucket
        const groups = _groupByDate(regular);

        for (const [label, items] of groups) {
            if (!items.length) continue;

            const groupEl = document.createElement('div');
            groupEl.className = 'history-group-label';
            groupEl.textContent = label;
            list.appendChild(groupEl);

            items.forEach(session => list.appendChild(_renderItem(session, { pinned: false })));
        }
    }

    function _renderItem(session, { pinned }) {
        const item = document.createElement('div');
        item.className = 'history-item' + (session.session_id === _activeId ? ' active' : '');
        if (pinned) item.classList.add('history-item-pinned');
        if (_liveIds.has(session.session_id)) item.classList.add('history-item-live');
        item.dataset.sessionId = session.session_id;
        item.setAttribute('role', 'button');
        item.tabIndex = 0;

        // Pulsing dot for active session
        if (session.session_id === _activeId) {
            const dot = document.createElement('span');
            dot.className = 'history-item-dot';
            item.appendChild(dot);
        }

        if (pinned) {
            const glyph = document.createElement('span');
            glyph.className = 'history-item-glyph';
            glyph.setAttribute('aria-hidden', 'true');
            glyph.textContent = '📱';
            item.appendChild(glyph);
        }

        const text = document.createElement('span');
        text.className = 'history-item-text';
        const cachedPreview = _previews.get(session.session_id);
        text.textContent = pinned
            ? (session.label || 'Remote control')
            : (cachedPreview || session.preview || 'Untitled');
        item.appendChild(text);

        if (pinned && cachedPreview) {
            item.title = cachedPreview;
        }

        // Live badge: a small pulsing dot at the right edge indicating
        // the backend is currently generating for this session. Used by
        // the remoteSessionObserver to surface WhatsApp / iMessage
        // activity in the sidebar.
        if (_liveIds.has(session.session_id)) {
            const badge = document.createElement('span');
            badge.className = 'history-item-live-badge';
            badge.title = 'Generating now';
            badge.textContent = '●';
            item.appendChild(badge);
        }

        const continueBtn = document.createElement('button');
        continueBtn.className = 'history-item-continue-btn';
        continueBtn.title = pinned
            ? 'Open the remote-control session'
            : 'Continue this session';
        continueBtn.textContent = '↩';
        continueBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (onContinueSession) onContinueSession(session.session_id);
        });
        item.appendChild(continueBtn);

        const selectSession = () => {
            setActive(session.session_id);
            if (onSelectSession) onSelectSession(session.session_id);
        };

        item.addEventListener('click', selectSession);
        item.addEventListener('keydown', (e) => {
            if (e.key !== 'Enter' && e.key !== ' ') return;
            e.preventDefault();
            selectSession();
        });
        return item;
    }

    return { element: panel, populate, setActive, setLive, setLivePreview };
}

// ── Date grouping helpers ─────────────────────────────────────────────────

function _groupByDate(sessions) {
    const now   = new Date();
    const today = _dayStart(now);
    const yesterday = new Date(today - 86400000);

    const groups = [
        ['Today',     []],
        ['Yesterday', []],
        ['Earlier',   []],
    ];

    sessions.forEach(s => {
        const t = s.last_active ? new Date(s.last_active * 1000) : new Date(0);
        const d = _dayStart(t);
        if (d >= today)                      groups[0][1].push(s);
        else if (d >= yesterday.getTime())   groups[1][1].push(s);
        else                                 groups[2][1].push(s);
    });

    return groups;
}

function _dayStart(date) {
    return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

function _displayName() {
    const raw = (typeof process !== 'undefined' && process.env && process.env.USER) || '';
    if (!raw) return 'you';
    return raw.charAt(0).toUpperCase() + raw.slice(1);
}

module.exports = { HistoryPanel };
