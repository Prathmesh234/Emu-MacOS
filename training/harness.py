"""
training/harness.py

Builds the EXACT system prompt + workspace context that emu's backend would
inject at inference time, so trajectories in our SFT dataset are
harness-compatible — i.e. the model trains on the same prefix it will see
in production and won't collapse when it encounters real workspace files.

This pipeline is REMOTE-MODE ONLY. Coworker mode is trained separately on
its own dataset, so all coworker prompt/tool/workspace builders are
intentionally absent here.

We import emu's `build_system_prompt` and `tools_for_anthropic("remote")`
directly (single source of truth), ship dummy-but-realistic firmware files
(SOUL.md / AGENTS.md / IDENTITY.md / USER.md / MEMORY.md) plus a handful
of skill stubs, and rotate through a few personas so the dataset isn't
overfit to one user.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path

# ── import emu's real system_prompt builder ────────────────────────────────
# Load the module directly via importlib so we sidestep
# backend/utilities/__init__.py (which imports fastapi via connection.py)
# and backend/prompts/__init__.py. We only need build_system_prompt.
import importlib.util as _ilu
import types as _types

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = _REPO_ROOT / "backend"


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub `utilities` package and load its `paths` submodule directly,
# so `system_prompt.py`'s `from utilities.paths import ...` resolves
# without triggering utilities/__init__.py.
_utilities_pkg = _types.ModuleType("utilities")
_utilities_pkg.__path__ = [str(_BACKEND / "utilities")]
sys.modules["utilities"] = _utilities_pkg
_load_module("utilities.paths", _BACKEND / "utilities" / "paths.py")

_sp = _load_module("emu_system_prompt", _BACKEND / "prompts" / "system_prompt.py")
_at = _load_module("emu_agent_tools", _BACKEND / "providers" / "agent_tools.py")
build_system_prompt = _sp.build_system_prompt
_tools_for_anthropic = _at.tools_for_anthropic

# Tools we deliberately strip from the REMOTE-MODE training schema.
# `bring_app_frontmost` lives in production's always-on catalog but its
# description and behaviour are coworker-mode-specific (visible-by-default
# app activation around `cua_*` driver flows). The remote agent uses
# `raise_app` instead, so we never want the model to learn it for remote.
_REMOTE_TOOL_BLOCKLIST = {"bring_app_frontmost"}


def tools_for_remote() -> list[dict]:
    """Canonical REMOTE-MODE tool schema (Anthropic format) attached to
    every synth trajectory so the SFT pipeline trains against the exact
    tools Emu publishes at inference time. Coworker-only tools are
    deliberately excluded."""
    return [
        t for t in _tools_for_anthropic("remote")
        if t["name"] not in _REMOTE_TOOL_BLOCKLIST
    ]


# ───────────────────────────────────────────────────────────────────────────
# DUMMY FIRMWARE — same shape as real .emu/workspace/* files
# ───────────────────────────────────────────────────────────────────────────

SOUL_MD = """\
# SOUL.md — who Emu is

You are warm, direct, and competent. You speak like a senior teammate, not a
chatbot. You don't pad messages with apologies or hedging. You take action
and report results crisply. When you don't know, you say so and propose how
to find out. You respect the user's time.
"""

AGENTS_MD = """\
# AGENTS.md — operational rules

- One desktop action per turn. Always.
- Coordinates are normalized [0,1] — never raw pixels.
- raise_app FIRST: before ANY interaction with an app other than the one
  currently focused (click, type, scroll, drag, key_press, screenshot for
  that app), call raise_app(app_name="…") with the exact macOS app name.
  No-op if already focused — when in doubt, raise. Re-raise on every
  app switch.
- For complex tasks (3+ steps): call update_plan FIRST and wait for the
  next user turn ("[PLAN APPROVED]") before any desktop action.
- For information-gathering tasks: call write_session_file IMMEDIATELY when
  you see relevant data on screen. Do not rely on memory.
- Anti-loop: never repeat a failing action more than twice. Switch strategy
  (Spotlight, shell_exec, keyboard shortcut, different element).
- Focus safety: ensure the target app is focused before text/keyboard input.
- invoke_hermes is FIRE-AND-FORGET by default. After invoking, emit a
  `done` action whose final_message tells the user Hermes is running in
  the background. Only call check_hermes(job_id, wait_s=60) when the user
  explicitly said "wait" / "don't return until it's done", or later asks
  for a status update.
- Use shell_exec for safe file-backed work (find, cat, python3) under .emu
  only. Never curl/wget/ssh/sudo/rm -rf/kill/pkill/chmod/chown/launchctl/
  systemctl/mount/dd/pipe-to-shell/eval. For .emu files use the dedicated
  tools (read_plan, read_memory, read_session_file, list_session_files).
"""

IDENTITY_MD = """\
# IDENTITY.md — voice & tone

- Concise. Avoid filler ("Sure!", "Of course!", "I'd be happy to").
- Action over narration. Don't say "I'm going to click X" — just click it.
- Report outcomes, not intentions.
- When stuck, name the blocker and the next thing you'll try.
- No emojis unless the user uses them first.
"""

# ── A few personas to rotate through, so the dataset has user diversity ──
PERSONAS: list[dict] = [
    {
        "USER.md": """\
# USER.md
- Name: John Doe
- Role: Software engineer, infra team
- Timezone: America/Los_Angeles (PT)
- Primary apps: VS Code, Chrome, Terminal, Slack, Notion
- Working style: keyboard-first, prefers shell over GUI when feasible
- Pet peeves: confirmations, preview steps, anything that breaks flow
""",
        "MEMORY.md": """\
# MEMORY.md — long-term notes

- Default browser is Chrome (Profile 1, work account).
- Notes go in Notion under workspace "infra-notes".
- Always paste meeting notes into Notion under the matching meeting page,
  not into a new doc.
- Standup is daily at 09:30 PT in #team-infra Slack channel.
- For Excel/PPT generation, prefer invoke_hermes — it's faster and more
  accurate than driving Office UI.
""",
    },
    {
        "USER.md": """\
# USER.md
- Name: Jane Smith
- Role: Product manager
- Timezone: Europe/London (UK)
- Primary apps: Figma, Linear, Slack, Gmail, Sheets
- Working style: visual, lots of context-switching between docs
""",
        "MEMORY.md": """\
# MEMORY.md — long-term notes

- Linear team prefix: PROD.
- Roadmap doc lives in Notion → "Q3 roadmap (live)".
- Weekly review every Friday 16:00 UK — pulls metrics from Sheets
  "metrics-2026" tab "Weekly".
- Prefers screenshots saved to ~/Desktop/screens/ via Cmd+Shift+4.
""",
    },
    {
        "USER.md": """\
# USER.md
- Name: Jack Wilson
- Role: Data scientist
- Timezone: Asia/Tokyo (JST)
- Primary apps: VS Code, Jupyter, Chrome, Slack, iTerm
- Working style: notebook-driven, lots of pandas + matplotlib
""",
        "MEMORY.md": """\
# MEMORY.md — long-term notes

- Default Python env: ~/.venvs/ds (activate before running any script).
- Datasets cached in ~/data/.
- Plots are saved to ~/reports/<date>/ as PNG, then dropped into the
  weekly Notion report.
- For multi-file refactors or large pandas pipelines, use invoke_hermes.
""",
    },
]


# ── Skill catalog. Mirrors backend/skills/bundled/* (real emu skill names),
#    plus a handful of "user-authored" custom skills that an Emu user would
#    accumulate over time for their personal redundant tasks. The block
#    matches the format emit by skills.loader.format_skills_for_prompt().
_BUNDLED_SKILLS: list[tuple[str, str]] = [
    ("app-launcher",         "Open, switch, and manage apps via Spotlight / raise_app."),
    ("file-manager",         "Find, move, rename, copy files via Finder shortcuts and shell."),
    ("system-info",          "Inspect macOS system info — disk, CPU, network, installed apps."),
    ("web-search",           "Run a web search via the default browser address bar."),
    ("summarize",            "Summarize visible page / document content into session notes."),
    ("google-chrome",        "Drive Google Chrome — tabs, address bar, bookmarks, DevTools."),
    ("google-docs",          "Edit Google Docs — formatting, headings, comments, sharing."),
    ("google-sheets",        "Edit Google Sheets — cells, formulas, sorting, charts."),
    ("google-slides",        "Edit Google Slides — add slides, layouts, text, images."),
    ("google-drive",         "Manage Google Drive files — upload, share, move, search."),
    ("google-calendar",      "View and create Google Calendar events."),
    ("google-meet",          "Join / start Google Meet calls and toggle mic / camera."),
    ("google-vids",          "Edit Google Vids projects."),
    ("gmail",                "Read, search, compose, and reply to Gmail."),
    ("microsoft-word",       "Edit Word docs — formatting, styles, comments, track changes."),
    ("microsoft-excel",      "Edit Excel sheets — cells, formulas, pivot tables, charts."),
    ("microsoft-powerpoint", "Edit PowerPoint decks — slides, layouts, text, images."),
    ("microsoft-outlook",    "Read, search, compose Outlook mail and calendar."),
    ("microsoft-onenote",    "Capture and edit OneNote notes."),
    ("microsoft-teams",      "Read Teams chats, join calls, send messages."),
]

# User-authored skills accumulated over time. These look like the kind of
# small redundant-task helpers a user would write themselves.
_CUSTOM_SKILLS: list[tuple[str, str]] = [
    ("daily-standup-notes",  "Open the team standup Notion page and append today's notes."),
    ("vscode-open-repo",     "Open a repo in VS Code via Cmd+Space → Code → File›Open."),
    ("slack-dm",             "Send a Slack DM in the user's signed-in workspace."),
    ("screenshot-region",    "Capture a screen region to ~/Desktop/screens/ via Cmd+Shift+4."),
    ("finder-reveal",        "Reveal a file in Finder."),
    ("gh-pr-review",         "Open a GitHub PR and walk through file diffs in Files Changed tab."),
    ("linear-create-issue",  "Create a Linear issue with title, description, and team."),
    ("notion-append-page",   "Append a block to a known Notion page by URL or title."),
    ("vlc-play-file",        "Open a media file in VLC and toggle fullscreen."),
    ("libreoffice-open",     "Open .ods / .odt / .odp files in LibreOffice."),
]

_ALL_SKILLS = _BUNDLED_SKILLS + _CUSTOM_SKILLS

# Block emitted into the workspace context, matching the real format from
# backend/skills/loader.format_skills_for_prompt().
SKILLS_BLOCK = (
    "## Skills (mandatory)\n"
    "The following skills are available. If one matches the task, call\n"
    "use_skill(skill_name=...) BEFORE taking any desktop action.\n\n"
    + "\n".join(f"- {n} — {d}" for n, d in _ALL_SKILLS)
)


def get_all_skill_names() -> list[str]:
    """Public accessor used by synth.py to teach the synth LLM the skill vocabulary."""
    return [n for n, _ in _ALL_SKILLS]


def get_skills_catalog() -> list[tuple[str, str]]:
    return list(_ALL_SKILLS)


def get_persona_memory(persona_idx: int = 0) -> str:
    """Return the persona's MEMORY.md body — used by synth post-processing
    to substitute the <<PERSONA_MEMORY/>> token in read_memory tool_results."""
    return _persona_for_index(persona_idx)["MEMORY.md"]


def _persona_for_index(i: int) -> dict:
    return PERSONAS[i % len(PERSONAS)]


def build_workspace_context(persona_idx: int = 0) -> str:
    """Same shape as backend/workspace/reader.build_workspace_context()."""
    p = _persona_for_index(persona_idx)
    today = datetime.now().strftime("%Y-%m-%d")
    parts: list[str] = [
        "WORKSPACE CONTEXT (.emu/)",
        f"Date: {today}",
        "",
        SKILLS_BLOCK,
        "",
        "## SOUL.md",
        SOUL_MD,
        "",
        "## AGENTS.md",
        AGENTS_MD,
        "",
        "## IDENTITY.md",
        IDENTITY_MD,
        "",
        "## USER.md",
        p["USER.md"],
        "",
        "## MEMORY",
        p["MEMORY.md"],
        "",
    ]
    return "\n".join(parts)


def build_full_system_prompt(persona_idx: int = 0, session_id: str | None = None) -> str:
    """Build the complete system prompt the emu backend would send."""
    sid = session_id or str(uuid.uuid4())
    workspace_ctx = build_workspace_context(persona_idx)
    return build_system_prompt(
        workspace_context=workspace_ctx,
        session_id=sid,
        bootstrap_mode=False,
        device_details={
            "os_name": "macOS",
            "arch": "arm64",
            "screen_width": 1920,
            "screen_height": 1080,
            "scale_factor": 2,
        },
        use_omni_parser=False,
    )


__all__ = [
    "PERSONAS",
    "build_full_system_prompt",
    "build_workspace_context",
    "get_all_skill_names",
    "get_skills_catalog",
    "get_persona_memory",
    "tools_for_remote",
    "SKILLS_BLOCK",
]
