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
        text.textContent = pinned
            ? (session.label || 'Remote control')
            : (session.preview || 'Untitled');
        item.appendChild(text);

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

    return { element: panel, populate, setActive };
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
