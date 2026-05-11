"""
training/synth.py

Take a REAL OSWorld agent trajectory (from data/real_trajs/, pulled via
dataset.py) and ask Claude to rewrite it as a synthetic EMU trajectory --
same task, same general action sequence, but in emu's exact format with
emu's scaffolding (raise_app, plan.md, read_memory, use_skill,
write_session_file, compact_context, invoke_hermes when appropriate).

This dataset targets EMU REMOTE MODE (the default desktop-automation agent
that observes via screenshots and emits one action per turn) — it does NOT
target coworker mode. None of the cua_*/coworker driver tools may appear.

Output: JSONL, one line per trajectory. Each line is harness-compatible
(emu remote-mode system prompt + persona stitched in at write time).

Usage:
    uv run python synth.py --count 10
    uv run python synth.py --count 10 --out data/synthetic/run1.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent / ".env")

from buffer import Buffer
from harness import (
    PERSONAS,
    build_full_system_prompt,
    get_skills_catalog,
    tools_for_remote,
)
from synth_utils import (
    extract_json,
    git_commit_and_push,
    relevant_skills_for,
    substitute_persona_memory,
    summarize_step,
)

# We access Claude Sonnet 4.6 through OpenRouter's chat.completions endpoint
# (OpenAI-compatible), and enable Anthropic prompt caching via explicit
# per-block `cache_control` breakpoints on the large, stable system prompt.
# See: https://openrouter.ai/docs/features/prompt-caching#anthropic-claude
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "").strip()
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "emu-training-synth").strip()
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "32768"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
# Retries for transient API failures (rate limits, 5xx, network blips).
MAX_RETRIES = int(os.getenv("SYNTH_MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.getenv("SYNTH_RETRY_BASE_DELAY", "4.0"))

SYNTH_DIR = Path(__file__).parent / "data" / "synthetic"
SYNTH_DIR.mkdir(parents=True, exist_ok=True)


# ── emu tool catalog (must match backend/providers/agent_tools.py for
#    REMOTE MODE — coworker-only tools are intentionally excluded) ────────────
EMU_TOOL_CATALOG = """\
EMU FUNCTION TOOLS (called via the model's native tool / function-calling API
— in Anthropic JSON these are tool_use / tool_result blocks. NEVER stringify
them as desktop-action JSON):

  Window management (REMOTE MODE):
    - raise_app(app_name)
        Bring a named macOS app to the foreground. MANDATORY before ANY
        interaction (click, type, scroll, drag, key_press, screenshot intended
        for that app) with any application other than the one currently
        focused. No-op if already focused. Pass the EXACT macOS app name
        ("Google Chrome", "Finder", "Visual Studio Code", "Microsoft Excel",
        "Slack", "Terminal", ...).

  Plan / session:
    - update_plan(content)            write/overwrite plan.md (3+ step tasks)
    - read_plan()                     re-read plan.md to re-orient
    - write_session_file(filename, content)  save scratchpad notes
    - read_session_file(filename)     read a scratchpad
    - list_session_files()
    - compact_context(focus?)         compress chain when getting long

  Memory & skills:
    - read_memory(target, date?)      target ∈ {long_term, preferences, daily_log}
    - use_skill(skill_name)           load a skill's full body (call when a
                                      listed skill matches the task)
    - create_skill(name, description, instructions, files?, overwrite?)
        Save a recurring USER-PERSONAL workflow as a reusable skill (e.g.
        "check Chase balance", "file weekly expenses in Concur"). Rare —
        only when the trajectory is teaching emu a personal repeatable flow.

  Shell:
    - shell_exec(command)
        FUNCTION TOOL (not a desktop action). Sandboxed: cwd is .emu, only
        paths under .emu allowed. BLOCKED: curl, wget, ssh, scp, nc, rsync,
        sudo, rm -rf, chmod, chown, kill, pkill, launchctl, systemctl,
        mount, mkfs, dd, pipe-to-shell (| bash), eval, source. 30s timeout,
        100 KB output cap. For .emu files prefer the dedicated tools
        (read_plan / read_memory / read_session_file).

  Hermes (heavy non-GUI delegation, async):
    - invoke_hermes(goal, context, file_paths?, output_target?, constraints?)
        Hand off code/shell-shaped work (build a .pptx, multi-sheet Excel,
        bulk file transforms, multi-file refactor, scripted research).
        Returns IMMEDIATELY with a job_id. DEFAULT FLOW = fire-and-forget:
        after invoking, emit a `done` desktop action whose final_message
        tells the user Hermes is running in the background. Do NOT call
        check_hermes in the same turn. Only chain check_hermes(job_id,
        wait_s=60) when the user explicitly said "wait" / "don't return
        until it's done".
        Hermes is BLIND to the screen — pass every transcript, name, date,
        URL, and decision into `context` / `file_paths`.
    - check_hermes(job_id, wait_s?)   only after user asks for a status
    - cancel_hermes(job_id)
    - list_hermes_jobs()

EMU DESKTOP ACTIONS — returned as ONE assistant text block whose text is raw
JSON (no markdown fences, no prose). Coordinates are normalized [0,1] floats:

  {"action": {"type": "screenshot"}, "done": false, "confidence": 0.9}
  {"action": {"type": "navigate_and_click",        "coordinates": {"x": 0.45, "y": 0.32}}, "done": false}
  {"action": {"type": "navigate_and_right_click",  "coordinates": {...}}, "done": false}
  {"action": {"type": "navigate_and_triple_click", "coordinates": {...}}, "done": false}
  {"action": {"type": "left_click"},   "done": false}   # at CURRENT cursor
  {"action": {"type": "right_click"},  "done": false}   # at CURRENT cursor
  {"action": {"type": "double_click"}, "done": false}   # at CURRENT cursor
  {"action": {"type": "triple_click"}, "done": false}   # at CURRENT cursor
  {"action": {"type": "mouse_move",   "coordinates": {...}}, "done": false}
  {"action": {"type": "type_text",    "text": "..."}, "done": false}
  {"action": {"type": "key_press",    "key": "enter"}, "done": false}
  {"action": {"type": "key_press",    "key": "l", "modifiers": ["cmd"]}, "done": false}
  {"action": {"type": "scroll",       "direction": "down", "amount": 5}, "done": false}
  {"action": {"type": "drag",         "coordinates": {...}, "end_coordinates": {...}}, "done": false}
  {"action": {"type": "wait",         "ms": 1000}, "done": false}
  {"action": {"type": "done"}, "done": true, "final_message": "..."}

  Modifiers: cmd, ctrl, alt, shift (use "cmd" — NEVER meta/super/win).
  Valid key names: enter, tab, escape, space, backspace, delete, insert,
    up, down, left, right, home, end, pageup, pagedown, f1–f12, a–z, 0–9.
  Prefer navigate_and_click over bare left_click unless the cursor is already
  on the target (e.g. immediately after mouse_move).
  `confidence` is OPTIONAL on non-done actions; `final_message` is REQUIRED
  on the terminating done action.
"""


SYNTH_SYSTEM = f"""\
You are a synthetic-data author. You will be given a REAL agent trajectory
that solved an OSWorld task (recorded actions + reasoning from a different
computer-use agent). Your job is to rewrite it as a trajectory that looks
exactly like the **Emu desktop automation agent in REMOTE MODE** produced
it -- same task, same overall action sequence, but in Emu's exact format
with Emu's scaffolding.

This data trains the REMOTE-MODE system prompt (the screenshot-driven
agent that emits one desktop-action JSON per turn). It is NOT for
coworker mode. Therefore:
  • You MUST NOT call any cua_*, list_running_apps, bring_app_frontmost,
    or other coworker-only tool.
  • You MUST NOT reference window_id / pid / AX element_index workflows.
  • Stick to the REMOTE function-tool catalog below + the screenshot-based
    desktop-action channel.

{EMU_TOOL_CATALOG}

ACTION TRANSLATION (from the real trajectory's "computer" tool calls to emu):
  left_click [x,y]             -> navigate_and_click   {{x,y normalized}}
  right_click [x,y]            -> navigate_and_right_click
  double_click [x,y]           -> navigate_and_click then double_click
  triple_click [x,y]           -> navigate_and_triple_click
  left_click_drag [a]->[b]     -> drag {{coordinates: a, end_coordinates: b}}
  type "..."                   -> type_text
  key "Return"                 -> key_press {{"key": "enter"}}
  key "ctrl+c"                 -> key_press {{"key": "c", "modifiers": ["ctrl"]}}
  scroll                       -> scroll
  screenshot                   -> screenshot
Coords: divide by screen size 1920x1080 unless trajectory says otherwise.
Round to 3 decimal places.

PLATFORM TRANSLATION: OSWorld ran on Ubuntu but Emu's personas run on
macOS. Translate keyboard shortcuts to their macOS-native equivalents
when rewriting (Ctrl→Cmd for typical app shortcuts like address bar,
copy/paste/save/find, app switching; keep Ctrl for terminal/Linux-only
contexts where Cmd has no analog). Use the macOS app names in raise_app
(see PLATFORM NOTE in the user prompt). Use "cmd" — never meta/super/win.

EMU SCAFFOLDING TO ADD (the real trajectory does NOT have these — weave in
~5-8 of them naturally; do not oversaturate, and only call each tool when
it's plausibly useful for THIS task):

  1. read_memory(target="long_term") at task start, before any desktop
     action — recall any user preferences relevant to the task.

  2. update_plan(content="...") for any 3+ step task, called BEFORE any
     desktop action. After update_plan the agent's turn ends; the next
     user turn will be the literal text "[PLAN APPROVED] The user has
     accepted the plan. Proceed with execution — take a screenshot to
     orient yourself and begin from step 1." Use that EXACT string
     verbatim — it is what production injects after every approved plan
     update. Only after that text appears does the agent take its first
     desktop action (typically a screenshot, then raise_app + work).
     As progress is made, mark steps [x] with another update_plan call
     once or twice; do not spam updates every step.

  3. raise_app(app_name="<exact macOS app name>") — MANDATORY before ANY
     interaction with an app other than the one currently focused. The
     real OSWorld trajectory was on Ubuntu and skipped this; in your
     emu rewrite it is REQUIRED. Specifically, call raise_app:
       • Right after [PLAN APPROVED] (or right after the initial planning
         block on simple 1-2 step tasks), before the first desktop action.
       • Whenever the trajectory switches between apps (Chrome ⇄ VS Code
         ⇄ Finder ⇄ Excel ⇄ Slack ⇄ Terminal, etc.).
     The macOS app name examples: "Google Chrome", "Finder", "Visual
     Studio Code", "Microsoft Excel", "Microsoft Word", "Microsoft
     PowerPoint", "Slack", "Terminal", "Mail". Map LibreOffice trajectories
     to the closest macOS analog ("Microsoft Excel" / "Microsoft Word" /
     "Microsoft PowerPoint") to match the persona's macOS environment.

  4. use_skill(skill_name="...") when the SKILL CATALOG in the user prompt
     contains a clearly-matching skill — call it AFTER read_memory and
     BEFORE the first desktop action. tool_result is a short plausible
     markdown body (1-2 short paragraphs) summarizing the skill's steps.

  5. write_session_file(filename, content) IMMEDIATELY when the trajectory
     gathers concrete data (URLs, names, prices, search results, meeting
     times). Don't wait until the end — use it as a scratchpad during the
     run. read_session_file when re-orienting after switching apps or
     before reporting results.

  6. compact_context(focus="...") if the real trajectory had ~20+ steps
     and you can mark a clean midpoint. Insert exactly once.

  7. shell_exec(command) is a FUNCTION TOOL (NOT a desktop action) for
     safe file-backed work the real trajectory achieved with many GUI
     clicks — find / cat / grep / python3 -c / ls under .emu only.
     NEVER curl, wget, ssh, sudo, rm -rf, kill, pkill, chmod, chown,
     launchctl, systemctl, mount, dd, pipe-to-shell, eval. The model
     calls it via tool_use; the response is a normal tool_result whose
     `content` is the captured stdout (or stderr on failure). Do NOT
     emit shell_exec as a desktop-action JSON — the validator rejects it
     with a hard error.

  8. invoke_hermes(goal, context, ...) for HEAVY non-GUI work the real
     trajectory accomplished with many GUI clicks (building a .pptx from
     scratch, multi-sheet Excel, multi-file refactor, bulk file
     transforms). DEFAULT FLOW (use this unless the OSWorld instruction
     explicitly says "wait until done"):
        a) invoke_hermes with full goal + every gathered fact in context.
        b) IMMEDIATELY emit a `done` desktop action with a final_message
           like "I've delegated the .pptx build to Hermes — it's running
           in the background. Ping me when you'd like a status update."
        c) DO NOT call check_hermes in the same trajectory.
     Only when the user instruction said "wait" / "don't return until
     done", chain check_hermes(job_id, wait_s=60) until completion before
     the final done.

  9. create_skill(name, description, instructions) — RARE. Only insert if
     the trajectory naturally taught a personal repeatable user-specific
     workflow worth saving (e.g. "file weekly expenses in Concur").
     Skip on generic tasks.

 10. ANTI-LOOP: if the real trajectory repeated a failing action, change
     strategy in your version (Spotlight via Cmd+Space, a different
     element, a keyboard shortcut, a shell_exec). Never repeat the same
     failing action more than twice.

 11. FOCUS SAFETY: never type or press keys while the Emu panel could be
     focused. Use raise_app + a screenshot to confirm the target app owns
     focus before type_text / key_press.

 12. NO SHELL FOR .emu FILES: never use shell_exec to read .emu/plan.md,
     .emu/MEMORY.md, or .emu session files — always use the dedicated
     read_plan / read_memory / read_session_file / list_session_files
     tools. shell_exec is for non-.emu inspection-shaped work only.

TOOL_RESULT FORMAT (these are the EXACT strings the production handlers
return — your tool_result `content` blocks must match these shapes so the
SFT distribution lines up with inference-time behavior):

  • read_memory(target="long_term") on success →
        "[MEMORY.md]\\n<<PERSONA_MEMORY/>>"
    where the literal token <<PERSONA_MEMORY/>> is replaced post-generation
    with the persona's real MEMORY.md. Do NOT invent the body — just emit
    the [MEMORY.md] header followed by the token on its own line.
    On empty → "MEMORY.md is empty or does not exist yet."

  • read_memory(target="preferences") → "[preferences.md]\\n<short body>"
    or "preferences.md is empty or does not exist yet."

  • read_memory(target="daily_log", date="YYYY-MM-DD") →
        "[Daily log for <date or 'today'>]\\n<short body>"
    or "No daily log found for <label>."

  • update_plan(content=...) → "Plan updated successfully."

  • read_plan() → "[YOUR PLAN]\\n<plan body>" or
        "No plan.md found for this session. You may need to create one."

  • use_skill(skill_name="X") → "[SKILL: X]\\n\\n<short plausible markdown
        body — 1-3 short paragraphs of steps/pitfalls>"
    On miss → "Skill 'X' not found. Available skills: <list>. ..."

  • create_skill(name=..., description=..., instructions=...) →
        "Skill '<slug>' created at <path>.\\nFiles written: SKILL.md.\\n
         It is now listed in available skills and can be loaded via
         use_skill('<slug>')."

  • write_session_file(filename="X.md", content=...) →
        "File 'X.md' written successfully."

  • read_session_file(filename="X.md") → "<file body>"
    or "File 'X.md' not found."

  • list_session_files() → "Session files: a.md, b.md"
    or "No files in current session. (plan.md and notes.md are considered
       system files)"

  • compact_context(focus=...) → a short "[Context compacted ...]" string;
    a one-line summary line is fine.

  • raise_app(app_name="X") on success →
        "X is raised - continue"
    On error → "ERROR: could not raise 'X'. ..." (rare in synth — only use
    if the trajectory needed an anti-loop pivot).

  • shell_exec(command="...") →
    A normal tool_result whose `content` is the command's captured stdout
    (truncated to 100 KB). On nonzero exit the content is the captured
    stderr instead. shell_exec is a FUNCTION TOOL, so you call it via a
    tool_use block and receive a tool_result like every other agent tool —
    do NOT emit it as a desktop-action JSON.

  • invoke_hermes(goal=..., context=..., ...) →
        "Hermes job started: job_id=`hermes-<8 hex>` (timeout 1800s).\\n
         Prompt saved to .emu/sessions/<sid>/hermes/task_brief_01.md.\\n\\n
         Hermes is running in the background — Emu is NOT blocked.\\n\\n
         DEFAULT BEHAVIOUR: end the current turn now. Do NOT call
         check_hermes in this turn. ..." (use a plausible 8-hex job_id)

  • check_hermes(job_id=...) when complete →
        "Hermes job <id> completed (exit 0). \\n<final stdout>\\n
         Output saved to <path>"
    Only emit if the trajectory actually waits for Hermes.

ACTION FEEDBACK (user turn AFTER an assistant desktop-action JSON):
  Production sends an IMAGE (the post-action screenshot) on success and
  a structured TEXT block only on failure / shell_exec. Match these
  shapes exactly:

  • SUCCESS (clicks, typing, scroll, drag, key_press, screenshot, mouse_move,
    wait): the user turn is a single placeholder text block:
        {{"role": "user", "content": [{{"type": "text", "text": "[screenshot]"}}]}}
    The literal token "[screenshot]" is the placeholder our training
    pipeline substitutes with a real image at SFT time. Do NOT add any
    other prose, do NOT echo the action, do NOT say "[ACTION OK]".

  • FAILURE: a single text block matching production's
    `interpret_action_error` shape:
        "[ACTION FAILED: <action_label>] <category> — <error>\\n<remediation>"
    where <action_label> is the action's `type` field (e.g. navigate_and_click)
    and <category> is one of:
       Permission denied — ...
       Target not found — ...
       Timed out after 30s — ...
       Shell process error — ...
       (or just the raw <error> for generic failures)
    Only insert failures when the real trajectory ACTUALLY had a failure
    or when an anti-loop pivot makes pedagogical sense.

  • DONE action's user-turn AFTER it: there is NONE. The trajectory ends.

OUTPUT FORMAT — RETURN ONE JSON OBJECT, NOTHING ELSE:

{{
  "task_id": "<the OSWorld task uuid>",
  "instruction": "<verbatim user instruction inferred from the real trajectory>",
  "messages": [
    {{"role": "user", "content": [{{"type": "text", "text": "<task instruction>"}}]}},
    {{"role": "assistant", "content": [
        {{"type": "text", "text": "<brief reasoning>"}},
        {{"type": "tool_use", "id": "toolu_001", "name": "read_memory",
         "input": {{"target": "long_term"}}}}
    ]}},
    {{"role": "user", "content": [
        {{"type": "tool_result", "tool_use_id": "toolu_001",
         "content": "[MEMORY.md]\\n<<PERSONA_MEMORY/>>"}}
    ]}},
    {{"role": "assistant", "content": [
        {{"type": "text", "text": "<reasoning>"}},
        {{"type": "tool_use", "id": "toolu_002", "name": "update_plan",
         "input": {{"content": "# Goal\\n...\\n# Steps\\n- [ ] step 1\\n- [ ] step 2\\n# Done when\\n..."}}}}
    ]}},
    {{"role": "user", "content": [
        {{"type": "tool_result", "tool_use_id": "toolu_002",
         "content": "Plan updated successfully."}},
        {{"type": "text", "text": "[PLAN APPROVED] The user has accepted the plan. Proceed with execution — take a screenshot to orient yourself and begin from step 1."}}
    ]}},
    {{"role": "assistant", "content": [
        {{"type": "text", "text": "<reasoning>"}},
        {{"type": "tool_use", "id": "toolu_003", "name": "raise_app",
         "input": {{"app_name": "Google Chrome"}}}}
    ]}},
    {{"role": "user", "content": [
        {{"type": "tool_result", "tool_use_id": "toolu_003",
         "content": "Google Chrome is raised - continue"}}
    ]}},
    ... continue alternating ...
    {{"role": "assistant", "content": [
        {{"type": "text", "text":
          "{{\\"action\\": {{\\"type\\": \\"navigate_and_click\\", \\"coordinates\\": {{\\"x\\": 0.225, \\"y\\": 0.218}}}}, \\"done\\": false, \\"confidence\\": 0.9}}"}}
    ]}},
    {{"role": "user", "content": [{{"type": "text", "text": "[screenshot]"}}]}},
    ...
    {{"role": "assistant", "content": [
        {{"type": "text", "text":
          "{{\\"action\\": {{\\"type\\": \\"done\\"}}, \\"done\\": true, \\"final_message\\": \\"...\\"}}"}}
    ]}}
  ]
}}

HARD RULES:
  - This is REMOTE MODE only. NEVER emit cua_*, list_running_apps,
    bring_app_frontmost, or any other coworker-only tool. NEVER reference
    pid/window_id/element_index.
  - CHANNELS: per assistant turn emit EITHER one tool_use OR one desktop
    action JSON — never both, never neither. Never put a tool name inside
    an action JSON; never wrap a desktop action as a tool_use.
  - Function tools (the catalog above, INCLUDING shell_exec) MUST be
    Anthropic tool_use / tool_result blocks. NEVER stringify them as
    desktop-action JSON.
  - Desktop actions MUST be a single assistant text block containing raw
    JSON (no fences, no commentary). NEVER as tool_use blocks.
  - Coordinates normalized [0,1], 3 decimals. Treat any value > 1.5 as a
    bug — the production validator rejects it as raw-pixel coordinates.
  - NEVER click / mouse_move to (0.0, 0.0) — the validator rejects this
    as a default/error coordinate.
  - NEVER emit two consecutive mouse_move actions targeting the same
    coordinates (within 0.01) — the validator treats it as a no-op.
  - Anti-loop: NEVER repeat the same action type more than 4 times
    consecutively (the validator rejects the 5th). screenshot, scroll,
    and done are exempt; everything else (click family, type_text,
    key_press, mouse_move, drag, wait) must change strategy by repeat 5.
  - wait actions: `ms` must be ≤ 30000 (30s cap).
  - One desktop action per assistant message. Each is followed by a user
    turn with the [screenshot] placeholder on success, or
    [ACTION FAILED: <type>] ... on failure. NEVER emit "[ACTION OK]".
  - update_plan must be followed by a user turn whose text is the EXACT
    production string "[PLAN APPROVED] The user has accepted the plan.
    Proceed with execution — take a screenshot to orient yourself and
    begin from step 1." before the agent emits any desktop action.
  - raise_app must precede the first desktop interaction with any app and
    must be re-issued whenever the agent switches apps.
  - Tool result `content` strings MUST match the TOOL_RESULT FORMAT shapes
    above verbatim (the [MEMORY.md], [SKILL: ...], "X is raised - continue",
    "Plan updated successfully." prefixes, etc.).
  - Use the literal token <<PERSONA_MEMORY/>> exactly once, inside the
    read_memory(target="long_term") tool_result. Post-processing fills it
    with the trajectory's persona MEMORY.md.
  - tool_use `id` strings must be unique per assistant message (toolu_001,
    toolu_002, ...).
  - Stay faithful to the real trajectory's action sequence — do NOT invent
    a different solution path. You may compress repetitive consecutive
    identical actions.
  - Final assistant message MUST be a desktop "done" action whose JSON
    includes "done": true and a final_message.
  - Output ONLY the JSON object. No markdown fences, no commentary.
"""


def build_user_prompt(real: dict) -> str:
    skills = relevant_skills_for(real.get("zip", ""))
    catalog = get_skills_catalog()
    desc_by_name = {n: d for n, d in catalog}
    skills_block = "\n".join(
        f"  - {n} -- {desc_by_name.get(n, '(custom user skill)')}" for n in skills
    )

    steps = real.get("steps", [])
    instruction = ""
    if steps:
        first_resp = steps[0].get("response", "") or ""
        raw = (steps[0].get("action", {}) or {}).get("raw_response", "") or ""
        for hay in (raw, first_resp):
            if "user wants" in hay.lower() or "user asked" in hay.lower():
                instruction = hay.strip().split("\n", 1)[0][:400]
                break

    step_lines = "\n".join(summarize_step(s) for s in steps)

    return f"""\
Rewrite this REAL OSWorld trajectory as an EMU REMOTE-MODE trajectory.

TASK ID: {real.get('task_id')}
SOURCE RUN (HF zip): {real.get('zip')}
RESULT (1=passed evaluator): {real.get('result')}
INSTRUCTION (inferred from agent's first reasoning, refine if needed):
{instruction or '(infer from action sequence below)'}

REAL ACTION SEQUENCE ({len(steps)} steps):
{step_lines}

SKILL CATALOG (call use_skill on a matching one early — if none clearly
fits the task, omit use_skill rather than forcing it):
{skills_block}

PLATFORM NOTE: the OSWorld run was on Ubuntu, but Emu (the agent we are
producing data for) runs on macOS. Use the EXACT macOS app name in
raise_app calls, e.g. map "libreoffice_calc" → "Microsoft Excel",
"libreoffice_writer" → "Microsoft Word", "libreoffice_impress" →
"Microsoft PowerPoint", "thunderbird" → "Mail", "chrome" →
"Google Chrome", "vscode" → "Visual Studio Code", "vlc" → "VLC", "gimp"
→ "GIMP", "files"/"nautilus" → "Finder", "terminal" → "Terminal".

Now produce the JSON emu trajectory object as specified.
"""


def generate_one(client: OpenAI, real: dict) -> dict:
    # Anthropic prompt caching via OpenRouter: mark the large, stable SYNTH_SYSTEM
    # block with an explicit ephemeral cache breakpoint. The per-trajectory user
    # prompt is left uncached (it changes every request). 5-minute TTL is fine
    # for a tight generation loop; bump to "1h" if running long jobs.
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": SYNTH_SYSTEM,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {"role": "user", "content": build_user_prompt(real)},
                ],
            )
            break
        except Exception as e:  # noqa: BLE001 — broad on purpose, classify by message.
            msg = str(e).lower()
            transient = (
                "429" in msg
                or "rate limit" in msg
                or "timeout" in msg
                or "timed out" in msg
                or "temporarily" in msg
                or any(code in msg for code in ("500", "502", "503", "504"))
            )
            last_err = e
            if not transient or attempt == MAX_RETRIES:
                raise
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"[synth] transient error (attempt {attempt}/{MAX_RETRIES}): {e}"
                  f" — retrying in {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)
    else:  # pragma: no cover — loop always either breaks or raises.
        raise last_err if last_err else RuntimeError("generate_one: unreachable")

    # Loud failure on truncation / refusal: silently parsing a half JSON object
    # is what produced phantom "failed" entries before. Surface the reason.
    finish_reason = None
    if resp.choices:
        finish_reason = getattr(resp.choices[0], "finish_reason", None)
    if finish_reason and finish_reason not in ("stop", "end_turn", "tool_calls"):
        raise RuntimeError(
            f"model stopped with finish_reason={finish_reason!r} "
            f"(MAX_TOKENS={MAX_TOKENS}); raise MAX_TOKENS env var or shorten input"
        )

    text = resp.choices[0].message.content or ""
    traj = extract_json(text)

    # OSWorld identifiers. `task_id` is the per-example UUID OSWorld uses in
    # its public registry (evaluation_examples/examples/<app>/<UUID>.json) and
    # it is also the namespace HF uses inside the verified-trajs zip
    # (.../trajectories/<UUID>/step_*.png). We re-surface it at the top level
    # of every generated record AND inside `_meta.osworld` so downstream SFT
    # code can locate the original screenshots regardless of nesting.
    osworld_task_id = real.get("task_id")
    osworld_zip = real.get("zip")
    traj["task_id"] = osworld_task_id
    traj["osworld_task_id"] = osworld_task_id
    usage = getattr(resp, "usage", None)
    cache_info: dict = {}
    if usage is not None:
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            # `prompt_tokens_details` is a pydantic model on openai>=1.51; coerce to dict.
            as_dict = details.model_dump() if hasattr(details, "model_dump") else dict(details)
            cache_info = {
                "cached_tokens": as_dict.get("cached_tokens", 0),
                "cache_write_tokens": as_dict.get("cache_write_tokens", 0),
            }
    # Pointer back to the OSWorld benchmark example and to the HF zip that
    # holds the original screenshots. `screenshot_path_glob` is the in-zip
    # path pattern (any entry containing the UUID); _extract_one() in
    # dataset.py currently strips images on download, so to recover them at
    # SFT time you re-open the zip and read entries matching this glob.
    osworld_meta = {
        "task_id": osworld_task_id,
        "benchmark": "osworld",
        "example_registry_path": (
            f"evaluation_examples/examples/*/{osworld_task_id}.json"
            if osworld_task_id else None
        ),
        "source_zip": osworld_zip,
        "source_zip_hf_url": (
            f"https://huggingface.co/datasets/xlangai/ubuntu_osworld_verified_trajs/"
            f"resolve/main/{osworld_zip}.zip" if osworld_zip else None
        ),
        "screenshot_path_glob": (
            f"*{osworld_task_id}*.png" if osworld_task_id else None
        ),
        "source_steps": len(real.get("steps", [])),
        "source_result": real.get("result"),
    }
    traj["_meta"] = {
        "model": OPENROUTER_MODEL,
        "provider": "openrouter",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "osworld": osworld_meta,
        # legacy / convenience aliases kept for the previous schema:
        "source_zip": osworld_zip,
        "source_steps": osworld_meta["source_steps"],
        "source_result": osworld_meta["source_result"],
        "cache": cache_info,
    }
    return traj


def attach_harness_prompt(traj: dict, persona_idx: int) -> dict:
    traj["system"] = build_full_system_prompt(persona_idx=persona_idx)
    traj["tools"] = tools_for_remote()
    substitutions = substitute_persona_memory(traj.get("messages", []), persona_idx)
    traj.setdefault("_meta", {})
    traj["_meta"]["persona_idx"] = persona_idx
    # Defensive: a malformed PERSONAS entry must not discard an otherwise valid
    # trajectory. Fall back to a generic label if the USER.md header is unusual.
    persona_name = f"persona_{persona_idx}"
    try:
        lines = PERSONAS[persona_idx % len(PERSONAS)]["USER.md"].splitlines()
        if len(lines) >= 2 and ":" in lines[1]:
            persona_name = lines[1].split(":", 1)[-1].strip() or persona_name
    except Exception:  # noqa: BLE001
        pass
    traj["_meta"]["persona_name"] = persona_name
    traj["_meta"]["persona_memory_substitutions"] = substitutions
    return traj


def _run_batch(client: OpenAI, buf: Buffer, out_path: Path, target: int,
               start_persona_idx: int) -> int:
    """Generate up to `target` synthetic trajectories into `out_path`.
    Returns the number successfully written.
    """
    written = 0
    with open(out_path, "a", encoding="utf-8") as f:
        pbar = tqdm(total=target, desc=f"batch -> {out_path.name}")
        while written < target:
            entry = buf.next()
            if entry is None:
                print("[synth] no pending real trajectories -- run dataset.py fetch first",
                      file=sys.stderr)
                break
            try:
                traj = generate_one(client, entry["real"])
                traj = attach_harness_prompt(traj,
                                             persona_idx=start_persona_idx + written)
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")
                f.flush()
                buf.mark_done(entry["task_id"])
                written += 1
                pbar.update(1)
            except Exception as e:
                print(f"[synth] {entry['task_id']} failed: {e}", file=sys.stderr)
                buf.mark_failed(entry["task_id"], str(e))
                time.sleep(2)
        pbar.close()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10,
                    help="trajectories per batch (default: 10)")
    ap.add_argument("--batches", type=int, default=1,
                    help="number of batches to run; git commit + push after each")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path. Without --batches>1 a single file is written; "
                         "with multiple batches this is used as a stem and -bNN is appended")
    ap.add_argument("--no-git", action="store_true",
                    help="disable git commit/push between batches")
    ap.add_argument("--no-push", action="store_true",
                    help="commit but do not push (still runs git add/commit)")
    ap.add_argument("--git-remote", default="origin")
    ap.add_argument("--git-branch", default=None,
                    help="branch to push (defaults to the current branch)")
    args = ap.parse_args()

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set in training/.env", file=sys.stderr)
        sys.exit(1)

    default_headers: dict[str, str] = {}
    if OPENROUTER_SITE_URL:
        default_headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        default_headers["X-Title"] = OPENROUTER_APP_NAME
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers=default_headers or None,
    )
    buf = Buffer()
    added = buf.scan()
    if added:
        print(f"[synth] indexed {added} new real trajectories")

    run_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    base_out = args.out or (SYNTH_DIR / f"synth-{run_stamp}.jsonl")
    base_out.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    persona_cursor = 0
    for b in range(1, args.batches + 1):
        if args.batches == 1 and args.out is not None:
            out_path = base_out
        else:
            out_path = base_out.with_name(f"{base_out.stem}-b{b:02d}{base_out.suffix}")
        print(f"\n[synth] === batch {b}/{args.batches} -> {out_path} ===")
        n = _run_batch(client, buf, out_path, args.count, persona_cursor)
        persona_cursor += n
        total_written += n
        print(f"[synth] batch {b}: wrote {n} trajectories -> {out_path}")

        if not args.no_git:
            git_commit_and_push(
                out_path,
                batch_num=b,
                batch_count=n,
                push=not args.no_push,
                remote=args.git_remote,
                branch=args.git_branch,
            )

        if n == 0:
            print("[synth] batch produced 0 trajectories; stopping early.",
                  file=sys.stderr)
            break

    print(f"\n[synth] DONE -- {total_written} trajectories across {args.batches} batch(es)")


if __name__ == "__main__":
    main()
