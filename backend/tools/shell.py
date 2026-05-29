"""
tools/shell.py

Guardrailed `shell_exec` tool (backend function tool, NOT a desktop action).

Guardrails:
  1. cwd pinned to the .emu directory (via utilities.paths.get_emu_path).
  2. HOME env rewritten to .emu so `~/foo` lands inside .emu.
  3. Every absolute-path token in the command must resolve inside .emu, with
     a small allowlist for interpreter binaries (/bin, /usr/bin,
     /usr/local/bin, /opt/homebrew/bin). Relative paths resolve against cwd.
  4. Blocklist rejects network tools (curl/wget/ssh/scp/nc/rsync/ftp/telnet),
     privilege escalation (sudo/su), destructive commands (rm -rf, mkfs, dd,
     chmod/chown, kill/pkill/killall, launchctl/systemctl, mount/umount),
     pipe-to-shell (| bash), eval/source.
  5. 30s timeout, 100 KB combined output cap.

Defense-in-depth: interpreter '-c'/'-e'/'-f' invocations (python -c, node -e,
osascript -e, etc.) are now rejected because they hide real work from the
path-scope check. Agents that need to run a script must write it to a file
under .emu and execute it by path. For true FS sandboxing use sandbox-exec.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

from utilities.paths import get_emu_path


_TIMEOUT_S = 30
_MAX_OUTPUT_BYTES = 100 * 1024

_BLOCKED_TOKENS = {
    "curl", "wget", "nc", "ncat", "socat",
    "ssh", "scp", "sftp", "rsync", "ftp", "tftp", "telnet",
    "sudo", "su", "doas",
    "mkfs", "dd", "fdisk", "diskutil", "mount", "umount",
    "shutdown", "reboot", "halt", "poweroff",
    "launchctl", "systemctl", "service",
    "kill", "pkill", "killall",
    "chmod", "chown", "chgrp",
    "eval", "source",
}

_BLOCKED_PATTERNS = [
    re.compile(r"\brm\s+-[rfRF]{1,2}\b"),
    re.compile(r"\|\s*(bash|sh|zsh|ksh)\b"),
    re.compile(r"\b(bash|sh|zsh|ksh)\s+<\("),
    # Block writes to device nodes (e.g. `> /dev/sda`) while still allowing the
    # safe-and-common stderr/stdout discard idioms (`2>/dev/null`, `>/dev/null`,
    # `&>/dev/null`, `>/dev/stdout`, `>/dev/stderr`). Without the negative
    # lookahead, agents constantly got false positives for `cmd 2>/dev/null`.
    re.compile(r">\s*/dev/(?!null\b|stdout\b|stderr\b)"),
    re.compile(r"^\s*:\s*\(\s*\)\s*\{"),
]

# Interpreter "exec a string" forms that bypass the path/blocklist guards
# above by hiding the real work inside an argument string the validator can't
# inspect. Block them outright; agents that need to run code should write a
# script under .emu and execute it by file path so the path-scope check runs.
_INTERPRETER_EXEC_RE = re.compile(
    r"(?:^|[\s|;&(`])"                       # word boundary or shell separator
    r"(?:/[^\s]+/)?"                         # optional absolute path prefix
    r"(?:python3?|node|deno|bun|ruby|perl|php|lua|tclsh|"
    r"osascript|swift|awk|sed|"
    r"bash|sh|zsh|ksh|dash|fish)"
    r"(?:[0-9.]*)"                           # optional version suffix (python3.12)
    r"\s+(?:-[A-Za-z]*[ceEfRr])\b"           # -c / -e / -E / -f / -R / -r
)

_ALLOWED_BIN_PREFIXES = (
    "/bin/", "/sbin/", "/usr/bin/", "/usr/sbin/",
    "/usr/local/bin/", "/usr/local/sbin/",
    "/opt/homebrew/bin/", "/opt/homebrew/sbin/",
)


def _reject(reason: str) -> str:
    return (
        f"ERROR: shell_exec refused to run — {reason}\n\n"
        f"shell_exec guardrails:\n"
        f"  • cwd is pinned to .emu; only .emu paths are allowed.\n"
        f"  • Blocked: curl/wget/ssh/scp/nc, sudo/su, rm -rf, chmod/chown,\n"
        f"    kill/pkill/killall, launchctl/systemctl, mount/umount,\n"
        f"    mkfs/dd, pipe-to-shell (| bash), eval/source,\n"
        f"    interpreter -c/-e/-f forms (python -c, node -e, osascript -e,\n"
        f"    bash -c, awk -f, sed -f, …). Write a script under .emu and\n"
        f"    run it by path instead.\n"
        f"  • 30s timeout, 100 KB output cap.\n"
        f"Rewrite to stay within these rules, or use another tool "
        f"(read_session_file, write_session_file, read_memory, etc.)."
    )


def _tokens(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def _paths_in_scope(command: str, emu_dir: Path) -> tuple[bool, str]:
    emu_resolved = emu_dir.resolve()
    for tok in _tokens(command):
        expanded = os.path.expanduser(tok)
        if not expanded.startswith("/"):
            continue
        if any(expanded.startswith(p) for p in _ALLOWED_BIN_PREFIXES):
            continue
        try:
            resolved = Path(expanded).resolve()
        except (OSError, RuntimeError):
            return False, f"could not resolve path '{tok}'"
        try:
            resolved.relative_to(emu_resolved)
        except ValueError:
            return False, (
                f"path '{tok}' is outside the .emu directory "
                f"({emu_resolved}). shell_exec can only touch files under .emu."
            )
    return True, ""


def _violates_blocklist(command: str) -> tuple[bool, str]:
    lowered = command.lower()
    for pat in _BLOCKED_PATTERNS:
        if pat.search(lowered):
            return True, f"command matches blocked pattern ({pat.pattern!r})"
    if _INTERPRETER_EXEC_RE.search(lowered):
        return True, (
            "interpreter '-c' / '-e' / '-f' style invocations are blocked "
            "(python -c, node -e, ruby -e, perl -e, osascript -e, awk -f, "
            "sed -f, bash -c, etc.) — they hide the real command from the "
            "path-scope check and can read files outside .emu. Write the "
            "script to a file under .emu and execute it by path instead."
        )
    for t in _tokens(command):
        base = os.path.basename(t).lower()
        if base in _BLOCKED_TOKENS:
            return True, f"command uses blocked program '{base}'"
    return False, ""


def _violates_coworker_contract(command: str) -> tuple[bool, str]:
    lowered = command.lower()
    uses_osascript = any(os.path.basename(t).lower() == "osascript" for t in _tokens(command))
    if not uses_osascript:
        return False, ""

    if "tell application" in lowered or "system events" in lowered:
        return True, (
            "coworker mode forbids AppleScript/System Events app scripting. "
            "Use emu-cua-driver tools for native app automation; if the driver "
            "cannot access the target UI in the background, stop with a done "
            "message instead of bypassing the driver."
        )

    return True, (
        "coworker mode does not use osascript as an automation escape hatch. "
        "Use emu-cua-driver tools or a safe file-inspection command."
    )


def handle_shell_exec(command: str, agent_mode: str = "remote") -> str:
    command = (command or "").strip()
    if not command:
        return "ERROR: shell_exec requires a non-empty 'command'."
    if len(command) > 4000:
        return _reject("command is longer than 4000 chars")

    bad, reason = _violates_blocklist(command)
    if bad:
        return _reject(reason)

    if agent_mode == "coworker":
        bad, reason = _violates_coworker_contract(command)
        if bad:
            return _reject(reason)

    emu_dir = get_emu_path()
    emu_dir.mkdir(parents=True, exist_ok=True)

    ok, reason = _paths_in_scope(command, emu_dir)
    if not ok:
        return _reject(reason)

    safe_env = {
        "PATH": os.environ.get(
            "PATH", "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin",
        ),
        "HOME": str(emu_dir),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "en_US.UTF-8"),
        "EMU_DIR": str(emu_dir),
    }

    try:
        proc = subprocess.run(
            ["/bin/bash", "-c", command],
            cwd=str(emu_dir),
            env=safe_env,
            capture_output=True,
            timeout=_TIMEOUT_S,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: shell_exec timed out after {_TIMEOUT_S}s. Command: {command}"
    except Exception as e:
        return f"ERROR: shell_exec failed to launch: {e}"

    out = proc.stdout or ""
    err = proc.stderr or ""
    combined = out + (("\n" + err) if err else "")
    if len(combined) > _MAX_OUTPUT_BYTES:
        combined = combined[:_MAX_OUTPUT_BYTES] + f"\n…[truncated at {_MAX_OUTPUT_BYTES} bytes]"

    header = f"[shell_exec exit={proc.returncode} cwd={emu_dir}]"
    if not combined.strip():
        return f"{header}\n(no output)"
    return f"{header}\n{combined}"
