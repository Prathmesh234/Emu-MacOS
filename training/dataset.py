"""
training/dataset.py

Pull real OSWorld agent trajectories from Hugging Face
(xlangai/ubuntu_osworld_verified_trajs). These are real recorded action
sequences from running computer-use agents (Claude, GPT, etc.) on the
OSWorld benchmark — they're the skeletons that synth.py turns into
emu-format trajectories.

Streams individual files out of the multi-GB zips via HTTP range requests,
so we never download the full archives. Cache layout:

    data/real_trajs/<zip_stem>/<task_uuid>/
        traj.jsonl      one step per line (action + reasoning + reward)
        result.txt      "1" pass / "0" fail on the OSWorld evaluator
        runtime.log     env logs (ignored)
        _files.json     local manifest

DEFAULT FILTERS (synth.py only knows how to convert pyautogui actions today):
  * gemini-only  -> only zips whose name contains "gemini" are listed/fetched
  * pyautogui-only -> a trajectory is cached only if EVERY useful action is a
                      pyautogui call (no LibreOffice UNO / macro tool calls).
Use --all-models / --all-actions to bypass either filter.

Usage:
    uv run python dataset.py list-runs                      # gemini zips only
    uv run python dataset.py list-runs --all-models         # every zip
    uv run python dataset.py inspect-zip <zip_name>
    uv run python dataset.py fetch --limit 100              # gemini + pyautogui
    uv run python dataset.py fetch --zip <zip_name> --all-actions
    uv run python dataset.py show <task_id>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi
from remotezip import RemoteZip
from tqdm import tqdm

load_dotenv(Path(__file__).parent / ".env")

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_TRAJ_REPO = "xlangai/ubuntu_osworld_verified_trajs"

REAL_TRAJS_DIR = Path(__file__).parent / "data" / "real_trajs"
REAL_TRAJS_DIR.mkdir(parents=True, exist_ok=True)

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

# ---------- gemini + pyautogui filters ----------
# A zip is a "gemini" run if its filename contains the substring below
# (case-insensitive). The current OSWorld dump only ships one such zip
# (results_agent_s2_gemini_15steps.zip, ~1.5 GB, 369 trajectories).
GEMINI_ZIP_SUBSTRING = "gemini"

# Action classifier — kept in sync with the prior audit_real_trajs.py rules.
# A trajectory is "pure pyautogui" iff every useful step (i.e. ignoring
# DONE/FAIL/WAIT control tokens and bare time.sleep filler) calls
# pyautogui.* and no step uses the LibreOffice UNO bridge or a macro
# helper module — those are what synth.py cannot translate today.
_CONTROL_RE = re.compile(r"^\s*(DONE|FAIL|WAIT|done|fail|wait)\s*$")
_TIME_SLEEP_ONLY_RE = re.compile(
    r"^\s*import\s+time\s*;\s*time\.sleep\([^)]*\)\s*;?\s*$"
)
_PYAUTOGUI_RE = re.compile(
    r"pyautogui\.(?:click|doubleClick|rightClick|tripleClick|write|typewrite|"
    r"press|hotkey|keyDown|keyUp|moveTo|moveRel|dragTo|dragRel|mouseDown|"
    r"mouseUp|scroll|hscroll|vscroll|position|screenshot)"
)
_MACRO_RE = re.compile(
    r"\b(?:WriterTools|CalcTools|ImpressTools|DrawTools|BaseTools|MathTools|"
    r"AgentTools|agent\.(?:click|type|write|hotkey|press|drag|moveTo|scroll|"
    r"exit|done|fail|wait)|Agent\.exit|from\s+libreoffice_\w+\s+import)"
)
_UNO_RE = re.compile(r"\bimport\s+uno\b|com\.sun\.star\.")


def _classify_action(action: str) -> str:
    """One of: control | sleep_only | empty | pyautogui | macro | uno |
    pyautogui_plus | other."""
    if not action or not action.strip():
        return "empty"
    if _CONTROL_RE.match(action):
        return "control"
    if _TIME_SLEEP_ONLY_RE.match(action):
        return "sleep_only"
    has_py = bool(_PYAUTOGUI_RE.search(action))
    has_mc = bool(_MACRO_RE.search(action))
    has_uno = bool(_UNO_RE.search(action))
    if has_py and not (has_mc or has_uno):
        return "pyautogui"
    if has_mc and not has_py:
        return "macro"
    if has_uno and not has_py:
        return "uno"
    if has_py and (has_mc or has_uno):
        return "pyautogui_plus"
    return "other"


def is_pure_pyautogui(traj_bytes: bytes | str) -> bool:
    """True iff the trajectory has >=1 useful step and every useful step is
    a pure pyautogui call (no UNO, no macro tools)."""
    text = traj_bytes.decode("utf-8", errors="ignore") if isinstance(
        traj_bytes, (bytes, bytearray)) else traj_bytes
    useful = pyauto = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            step = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = _classify_action(str(step.get("action", "")))
        if kind in ("control", "sleep_only", "empty"):
            continue
        useful += 1
        if kind == "pyautogui":
            pyauto += 1
    return useful > 0 and pyauto == useful


def is_gemini_zip(zip_name: str) -> bool:
    return GEMINI_ZIP_SUBSTRING in zip_name.lower()


def _hf_session() -> requests.Session:
    s = requests.Session()
    if HF_TOKEN:
        s.headers["Authorization"] = f"Bearer {HF_TOKEN}"
    s.headers["User-Agent"] = "emu-training/0.1"
    return s


def _zip_url(zip_name: str) -> str:
    return f"https://huggingface.co/datasets/{HF_TRAJ_REPO}/resolve/main/{zip_name}"


def _open_zip(zip_name: str) -> RemoteZip:
    return RemoteZip(_zip_url(zip_name), session=_hf_session())


def list_runs(gemini_only: bool = True) -> list[dict]:
    api = HfApi(token=HF_TOKEN or None)
    info = api.repo_info(repo_id=HF_TRAJ_REPO, repo_type="dataset", files_metadata=True)
    out = [
        {"name": f.rfilename, "size_mb": (f.size or 0) // (1024 * 1024)}
        for f in info.siblings if f.rfilename.endswith(".zip")
    ]
    if gemini_only:
        out = [r for r in out if is_gemini_zip(r["name"])]
    out.sort(key=lambda r: r["size_mb"])
    return out


def inspect_zip(zip_name: str, n: int = 30) -> dict:
    with _open_zip(zip_name) as zf:
        names = zf.namelist()
    ext_count: dict[str, int] = {}
    for name in names:
        ext = Path(name).suffix.lower() or "(none)"
        ext_count[ext] = ext_count.get(ext, 0) + 1
    return {
        "zip": zip_name,
        "total_entries": len(names),
        "sample_entries": names[:n],
        "extensions": dict(sorted(ext_count.items(), key=lambda kv: -kv[1])),
    }


def _extract_one(zf: RemoteZip, zip_stem: str, task_id: str,
                 entries_by_id: dict[str, list[str]]) -> Path:
    out_dir = REAL_TRAJS_DIR / zip_stem / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        n for n in entries_by_id.get(task_id, [])
        if not n.lower().endswith((".png", ".jpg", ".jpeg", ".mp4"))
    ]
    pulled = []
    for entry in targets:
        try:
            data = zf.read(entry)
        except Exception as e:
            print(f"[hf] skip {entry}: {e}", file=sys.stderr)
            continue
        (out_dir / Path(entry).name).write_bytes(data)
        pulled.append(Path(entry).name)
    (out_dir / "_files.json").write_text(json.dumps(pulled, indent=2), encoding="utf-8")
    return out_dir


def fetch(zip_name: str | None = None, limit: int = 50,
          gemini_only: bool = True, pyautogui_only: bool = True) -> list[Path]:
    """Stream-fetch trajectories from HF, applying gemini + pyautogui filters.

    If `zip_name` is None, every gemini zip on the dataset is scanned (today
    that's just one). Trajectories that fail the pyautogui filter are
    skipped without writing anything to disk.
    """
    zips: list[str]
    if zip_name is not None:
        if gemini_only and not is_gemini_zip(zip_name):
            raise ValueError(
                f"--zip {zip_name!r} is not a gemini zip; pass --all-models "
                f"to fetch from non-gemini runs.")
        zips = [zip_name]
    else:
        zips = [r["name"] for r in list_runs(gemini_only=gemini_only)]
        if not zips:
            print("[hf] no zips matched current filters", file=sys.stderr)
            return []

    out: list[Path] = []
    for zname in zips:
        zip_stem = Path(zname).stem
        with _open_zip(zname) as zf:
            entries_by_id: dict[str, list[str]] = {}
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                m = _UUID_RE.search(name)
                if not m:
                    continue
                entries_by_id.setdefault(m.group(0), []).append(name)
            ids = sorted(entries_by_id)
            print(f"[hf] {zname}: {len(ids)} task IDs total"
                  + (f" (pyautogui filter on)" if pyautogui_only else ""))
            kept = skipped_filter = skipped_no_traj = 0
            for tid in tqdm(ids, desc=f"trajs:{zip_stem[:24]}"):
                if limit and kept >= limit:
                    break
                if pyautogui_only:
                    traj_entry = next(
                        (e for e in entries_by_id[tid]
                         if Path(e).name.lower() == "traj.jsonl"),
                        None,
                    )
                    if traj_entry is None:
                        skipped_no_traj += 1
                        continue
                    try:
                        traj_bytes = zf.read(traj_entry)
                    except Exception as e:
                        print(f"[hf] {tid}: read traj failed: {e}",
                              file=sys.stderr)
                        skipped_no_traj += 1
                        continue
                    if not is_pure_pyautogui(traj_bytes):
                        skipped_filter += 1
                        continue
                try:
                    out.append(_extract_one(zf, zip_stem, tid, entries_by_id))
                    kept += 1
                except Exception as e:
                    print(f"[hf] {tid}: {e}", file=sys.stderr)
            print(f"[hf] {zname}: kept={kept} "
                  f"skipped_pyautogui_filter={skipped_filter} "
                  f"skipped_no_traj={skipped_no_traj}")
    return out


def iter_cached() -> list[Path]:
    return [d for d in sorted(REAL_TRAJS_DIR.glob("*/*")) if d.is_dir()]


def load_traj(traj_dir: Path) -> dict:
    """Load one cached trajectory: {task_id, zip, steps, result}."""
    steps = []
    traj_path = traj_dir / "traj.jsonl"
    if traj_path.exists():
        for line in traj_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    result = None
    rp = traj_dir / "result.txt"
    if rp.exists():
        try:
            result = float(rp.read_text(encoding="utf-8").strip() or "0")
        except ValueError:
            result = None
    return {
        "task_id": traj_dir.name,
        "zip": traj_dir.parent.name,
        "steps": steps,
        "result": result,
    }


def show(task_id: str):
    for d in iter_cached():
        if d.name.startswith(task_id):
            t = load_traj(d)
            print(f"=== {d} ===")
            print(f"  result: {t['result']}   steps: {len(t['steps'])}")
            for s in t["steps"][:3]:
                act = s.get("action", {})
                print(f"  step {s.get('step_num')}: {act.get('input', act)}")
            return
    print(f"trajectory '{task_id}' not found", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lr = sub.add_parser("list-runs")
    lr.add_argument("--all-models", action="store_true",
                    help="list every zip, not just gemini runs")
    iz = sub.add_parser("inspect-zip"); iz.add_argument("zip_name"); iz.add_argument("--n", type=int, default=30)
    f = sub.add_parser("fetch")
    f.add_argument("--zip", dest="zip_name", default=None,
                   help="specific zip name; default = all gemini zips")
    f.add_argument("--limit", type=int, default=50,
                   help="max trajectories to keep per zip after filtering")
    f.add_argument("--all-models", action="store_true",
                   help="bypass the gemini-only zip filter")
    f.add_argument("--all-actions", action="store_true",
                   help="bypass the pure-pyautogui trajectory filter")
    s = sub.add_parser("show"); s.add_argument("task_id")
    args = ap.parse_args()

    if args.cmd == "list-runs":
        for r in list_runs(gemini_only=not args.all_models):
            print(f"{r['size_mb']:>6} MB  {r['name']}")
    elif args.cmd == "inspect-zip":
        print(json.dumps(inspect_zip(args.zip_name, args.n), indent=2))
    elif args.cmd == "fetch":
        paths = fetch(
            zip_name=args.zip_name,
            limit=args.limit,
            gemini_only=not args.all_models,
            pyautogui_only=not args.all_actions,
        )
        print(f"\n[hf] cached {len(paths)} trajectories -> {REAL_TRAJS_DIR}")
    elif args.cmd == "show":
        show(args.task_id)


if __name__ == "__main__":
    main()
