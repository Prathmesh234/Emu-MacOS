"""
training/synth_utils.py

Pure helpers extracted from synth.py so the main module stays focused on
prompt construction and the OpenRouter call. Grouped by concern:

  - JSON extraction from LLM responses
  - OSWorld -> emu skill routing
  - Real-trajectory step summarization
  - Persona MEMORY.md token substitution
  - Git batch commit/push helpers

These functions are deliberately import-light (no openai/anthropic) so they
can be unit-tested in isolation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from harness import get_persona_memory


# ── JSON extraction ──────────────────────────────────────────────────────────
def strip_fences(s: str) -> str:
    """Remove a single leading/trailing ```...``` markdown fence, if present."""
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()


def extract_json(text: str) -> dict:
    """Best-effort JSON parse: strip fences, then fall back to slicing the
    outermost { ... } if the model added stray commentary.
    """
    text = strip_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


# ── skill routing ────────────────────────────────────────────────────────────
APP_TO_SKILLS: dict[str, list[str]] = {
    "chrome":              ["google-chrome", "web-search"],
    "vscode":              ["vscode-open-repo", "file-manager"],
    "libreoffice_calc":    ["libreoffice-open", "microsoft-excel"],
    "libreoffice_writer":  ["libreoffice-open", "microsoft-word"],
    "libreoffice_impress": ["libreoffice-open", "microsoft-powerpoint"],
    "thunderbird":         ["microsoft-outlook", "gmail"],
    "vlc":                 ["vlc-play-file"],
    "gimp":                ["file-manager", "app-launcher"],
    "os":                  ["app-launcher", "system-info", "file-manager"],
    "multi_apps":          ["app-launcher"],
}


def relevant_skills_for(zip_or_path: str) -> list[str]:
    """The HF zip stem / step paths embed the OSWorld app folder; sniff it
    to suggest a small skill catalog for the user prompt.
    """
    out: list[str] = []
    s = (zip_or_path or "").lower()
    for app, skills in APP_TO_SKILLS.items():
        if app in s:
            out.extend(sk for sk in skills if sk not in out)
    if "app-launcher" not in out:
        out.append("app-launcher")
    return out


# ── real-trajectory step summary ─────────────────────────────────────────────
def summarize_step(step: dict) -> str:
    """Compact one-line summary of a real-trajectory step for the prompt."""
    act = step.get("action", {}) or {}
    inp = act.get("input", {}) or {}
    name = inp.get("action") or act.get("name") or "?"
    parts = [f"#{step.get('step_num', '?')} {name}"]
    for k in ("coordinate", "start_coordinate", "text", "key", "scroll_direction"):
        if k in inp:
            v = inp[k]
            if isinstance(v, str) and len(v) > 80:
                v = v[:80] + "..."
            parts.append(f"{k}={v}")
    resp = (step.get("response") or "").strip().replace("\n", " ")
    if resp:
        parts.append(f"// {resp[:140]}")
    return " ".join(parts)


# ── persona MEMORY.md substitution ───────────────────────────────────────────
PERSONA_MEMORY_TOKEN = "<<PERSONA_MEMORY/>>"


def substitute_persona_memory(messages: list, persona_idx: int) -> int:
    """Replace the literal <<PERSONA_MEMORY/>> token in tool_result content
    with the persona's actual MEMORY.md body. Returns the count of
    substitutions made (typically 1, occasionally 0 if the model omitted it).
    """
    body = get_persona_memory(persona_idx).strip()
    n = 0
    for msg in messages or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            text = block.get("content")
            if isinstance(text, str) and PERSONA_MEMORY_TOKEN in text:
                block["content"] = text.replace(PERSONA_MEMORY_TOKEN, body)
                n += 1
            elif isinstance(text, list):
                for sub in text:
                    if isinstance(sub, dict) and isinstance(sub.get("text"), str) \
                            and PERSONA_MEMORY_TOKEN in sub["text"]:
                        sub["text"] = sub["text"].replace(PERSONA_MEMORY_TOKEN, body)
                        n += 1
    return n


# ── git batch helpers ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent


def run_git(args: Iterable[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *list(args)],
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def git_commit_and_push(out_path: Path, batch_num: int, batch_count: int,
                        push: bool, remote: str, branch: str | None) -> None:
    """Stage the synthetic-batch file, commit, and (optionally) push.

    Stages `out_path` plus the buffer state file so resumes work after pull.
    Skips silently if there is nothing to commit (e.g. zero successful trajs).
    """
    rel_out = out_path.resolve().relative_to(REPO_ROOT)
    buffer_state = (Path(__file__).parent / "data" / "state" / "buffer.json").resolve()
    paths_to_add = [str(rel_out)]
    if buffer_state.exists():
        paths_to_add.append(str(buffer_state.relative_to(REPO_ROOT)))

    add = run_git(["add", "-f", "--", *paths_to_add], check=False)
    if add.returncode != 0:
        print(f"[git] add failed: {add.stderr.strip()}", file=sys.stderr)
        return

    status = run_git(["status", "--porcelain", "--", *paths_to_add], check=False)
    if not status.stdout.strip():
        print(f"[git] batch {batch_num}: nothing to commit")
        return

    msg = f"synth: batch {batch_num} ({batch_count} trajectories) -> {rel_out.as_posix()}"
    commit = run_git(["commit", "-m", msg], check=False)
    if commit.returncode != 0:
        print(f"[git] commit failed: {commit.stderr.strip() or commit.stdout.strip()}",
              file=sys.stderr)
        return
    print(f"[git] committed batch {batch_num}: {msg}")

    if not push:
        return
    push_args = ["push", remote]
    if branch:
        push_args.append(branch)
    pushed = run_git(push_args, check=False)
    if pushed.returncode != 0:
        print(f"[git] push failed: {pushed.stderr.strip() or pushed.stdout.strip()}",
              file=sys.stderr)
    else:
        print(f"[git] pushed batch {batch_num} to {remote}"
              + (f"/{branch}" if branch else ""))
