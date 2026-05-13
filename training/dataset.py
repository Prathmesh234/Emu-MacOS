"""
training/dataset.py

Pull real agent trajectories from Hugging Face for synth.py to rewrite.
Two source families are supported:

  (A) xlangai/ubuntu_osworld_verified_trajs — recorded agent runs on the
      OSWorld benchmark (Claude, GPT, Gemini, Qwen, …). Streamed straight
      out of multi-GB zips via HTTP range requests so we never download
      the full archive.

  (B) xlangai/computer-agent-arena — crowdsourced real-user tasks evaluated
      by many agents. A single 49 MB JSONL contains 4,641 trajectories,
      including ~502 Gemini ones — every row a distinct task_id, in the
      same pyautogui step shape Agent-S2 uses, so the existing synth.py
      translator works without changes. See DATA_COLLECTION.md.

Cache layout (identical for both sources):

    data/real_trajs/<source_stem>/<task_uuid>/
        traj.jsonl       one step per line (action + reasoning + reward)
        result.txt       "1" pass / "0" fail
        runtime.log      env logs (ignored, OSWorld only)
        _files.json      local manifest
        _arena_meta.json arena only — original instruction + model + correctness

DEFAULT FILTERS for OSWorld fetches (synth.py only knows how to convert
pyautogui actions today):
  * gemini-only  -> only zips whose name contains "gemini" are listed/fetched
  * pyautogui-only -> a trajectory is cached only if EVERY useful action is a
                      pyautogui call (no LibreOffice UNO / macro tool calls).
Use --all-models / --all-actions to bypass either filter.

Arena fetches are gemini-filtered at the row level (by `model` field) and
already use the pyautogui step shape, so the OSWorld filters do not apply.

Usage:
    uv run python dataset.py list-runs                      # gemini zips only
    uv run python dataset.py list-runs --all-models         # every zip
    uv run python dataset.py inspect-zip <zip_name>
    uv run python dataset.py fetch --limit 100              # gemini + pyautogui
    uv run python dataset.py fetch --zip <zip_name> --all-actions
    uv run python dataset.py arena --limit 200              # Computer Agent Arena, gemini-only
    uv run python dataset.py arena --limit 50 --model gemini-2.5-pro --passed-only
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

# Computer Agent Arena — same XLang lab as OSWorld, but crowdsourced
# real-user tasks instead of the fixed OSWorld benchmark. Single 49 MB
# JSONL contains 4,641 trajectories; 502 of them are Gemini, every one a
# distinct task. See DATA_COLLECTION.md for the full survey.
HF_ARENA_REPO = "xlangai/computer-agent-arena"
HF_ARENA_FILE = "agent_arena_data.jsonl"
ARENA_DIR_STEM = "agent-arena-gemini"

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


# ---------- Computer Agent Arena ingest ----------
# Arena rows look like:
#   {"task_id": "...", "instruction": "...", "human_eval_correctness": 0|1,
#    "model": "gemini/gemini-2.5-pro-exp-03-25 (base_agent)",
#    "traj": [{"index": 1, "image": "...", "value": {"thought": "...",
#              "code": "import pyautogui\npyautogui.click(160, 700)"}}, ...]}
# The `value.code` is the same pyautogui shape Agent-S2 uses, so we project
# each step into the dict shape `summarize_step()` already understands:
#   {"step_num": int, "action": <code>, "plan_code": <thought>, "done": bool}
ARENA_MODEL_ALIASES: dict[str, str] = {
    "gemini-2.5-pro": "gemini/gemini-2.5-pro-exp-03-25",
    "gemini-2.0-flash": "gemini/gemini-2.0-flash",
    "gemini-1.5-pro": "gemini/gemini-1.5-pro",
    "gemini-1.5-flash": "gemini/gemini-1.5-flash",
}


def _arena_url() -> str:
    return f"https://huggingface.co/datasets/{HF_ARENA_REPO}/resolve/main/{HF_ARENA_FILE}"


def _arena_step_to_agent_s2(step: dict, total_steps: int) -> dict | None:
    """Project one Arena step into the dict shape `summarize_step()` reads.

    Returns None for malformed entries so the caller can skip them.
    """
    if not isinstance(step, dict):
        return None
    idx = step.get("index")
    value = step.get("value") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {}
    code = (value.get("code") or "").strip()
    thought = (value.get("thought") or "").strip()
    if not code and not thought:
        return None
    return {
        "step_num": idx,
        "action": code,
        # `plan_code` is what summarize_step / build_user_prompt look at for
        # the agent's natural-language intent on Agent-S2 / pyautogui rows.
        "plan_code": thought,
        "done": isinstance(idx, int) and total_steps and idx >= total_steps,
    }


def _arena_select(model_filter: str, passed_only: bool):
    """Return a row predicate matching the requested filters."""
    needle = (model_filter or "").lower()

    def keep(row: dict) -> bool:
        m = (row.get("model") or "").lower()
        if needle and needle not in m:
            return False
        if passed_only:
            c = row.get("human_eval_correctness")
            try:
                if int(c) != 1:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    return keep


def _write_arena_traj(out_dir: Path, row: dict) -> bool:
    """Materialize one Arena row into the OSWorld-shaped cache directory.
    Returns True if a usable trajectory was written.
    """
    raw_steps = row.get("traj") or []
    if not isinstance(raw_steps, list) or not raw_steps:
        return False
    steps: list[dict] = []
    for s in raw_steps:
        projected = _arena_step_to_agent_s2(s, total_steps=len(raw_steps))
        if projected is not None:
            steps.append(projected)
    if not steps:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)

    # `traj.jsonl` — one projected step per line, exact shape consumed by
    # `dataset.load_traj()` -> `synth_utils.summarize_step()`.
    with (out_dir / "traj.jsonl").open("w", encoding="utf-8") as f:
        for s in steps:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    correctness = row.get("human_eval_correctness")
    try:
        result = "1" if int(correctness) == 1 else "0"
    except (TypeError, ValueError):
        result = "0"
    (out_dir / "result.txt").write_text(result, encoding="utf-8")

    # The original user instruction is the cleanest signal of intent — far
    # better than the agent's first-step thought. Stash it so synth.py can
    # pick it up via the existing "user asked" / "user wants" inference path
    # in `build_user_prompt`, AND keep a structured copy for future use.
    instruction = (row.get("instruction") or "").strip()
    if instruction and steps:
        steps[0]["plan_code"] = (
            f"User asked: {instruction}\n{steps[0].get('plan_code', '')}"
        ).strip()
        # Re-flush traj.jsonl so the prompt-side instruction inference works.
        with (out_dir / "traj.jsonl").open("w", encoding="utf-8") as f:
            for s in steps:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    (out_dir / "_arena_meta.json").write_text(
        json.dumps(
            {
                "source": HF_ARENA_REPO,
                "task_id": row.get("task_id"),
                "model": row.get("model"),
                "instruction": instruction,
                "human_eval_correctness": correctness,
                "n_steps": len(steps),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "_files.json").write_text(
        json.dumps(["traj.jsonl", "result.txt", "_arena_meta.json"], indent=2),
        encoding="utf-8",
    )
    return True


def fetch_arena(
    limit: int = 200,
    model: str = "gemini",
    passed_only: bool = False,
    skip_existing: bool = True,
) -> list[Path]:
    """Stream `xlangai/computer-agent-arena/agent_arena_data.jsonl` and
    materialize each matching row as a synth-ready trajectory under
    `data/real_trajs/agent-arena-gemini/<task_id>/`.

    `model` is matched as a case-insensitive substring against the row's
    `model` field (default "gemini" matches every Gemini variant). Pass an
    `ARENA_MODEL_ALIASES` key (e.g. "gemini-2.5-pro") for a specific
    Gemini, or any literal substring of the field. Set `passed_only=True`
    to keep only `human_eval_correctness == 1` rows.
    """
    needle = ARENA_MODEL_ALIASES.get(model.lower(), model)
    keep = _arena_select(needle, passed_only)

    out_root = REAL_TRAJS_DIR / ARENA_DIR_STEM
    out_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen_existing = skipped_filter = malformed = 0

    print(f"[arena] streaming {HF_ARENA_FILE} from {HF_ARENA_REPO}"
          f" (model~={needle!r}, passed_only={passed_only}, limit={limit})")
    sess = _hf_session()
    with sess.get(_arena_url(), stream=True, timeout=120) as resp:
        resp.raise_for_status()
        # `iter_lines` decodes each newline-delimited JSON record without
        # buffering the full 49 MB into memory.
        bar = tqdm(desc=f"arena:{needle[:20]}",
                   total=limit if limit else None, unit="traj")
        for line in resp.iter_lines(decode_unicode=True):
            if limit and len(written) >= limit:
                break
            if not line or not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not keep(row):
                skipped_filter += 1
                continue
            tid = row.get("task_id")
            if not tid:
                malformed += 1
                continue
            out_dir = out_root / tid
            if skip_existing and (out_dir / "traj.jsonl").exists():
                seen_existing += 1
                continue
            if _write_arena_traj(out_dir, row):
                written.append(out_dir)
                bar.update(1)
            else:
                malformed += 1
        bar.close()

    print(f"[arena] kept={len(written)} already_cached={seen_existing} "
          f"filtered_out={skipped_filter} malformed={malformed}")
    return written


# ---------- shared loaders ----------
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
    a = sub.add_parser(
        "arena",
        help="ingest gemini trajectories from xlangai/computer-agent-arena",
    )
    a.add_argument("--limit", type=int, default=200,
                   help="max trajectories to ingest (default 200, 0 = all)")
    a.add_argument("--model", default="gemini",
                   help="case-insensitive substring matched against the row's "
                        "`model` field; aliases: "
                        + ", ".join(sorted(ARENA_MODEL_ALIASES))
                        + " (default 'gemini' = every Gemini variant)")
    a.add_argument("--passed-only", action="store_true",
                   help="only keep rows where human_eval_correctness == 1")
    a.add_argument("--no-skip-existing", dest="skip_existing",
                   action="store_false", default=True,
                   help="re-write task dirs even if traj.jsonl already exists")
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
    elif args.cmd == "arena":
        paths = fetch_arena(
            limit=args.limit,
            model=args.model,
            passed_only=args.passed_only,
            skip_existing=args.skip_existing,
        )
        print(f"\n[arena] cached {len(paths)} trajectories -> "
              f"{REAL_TRAJS_DIR / ARENA_DIR_STEM}")
    elif args.cmd == "show":
        show(args.task_id)


if __name__ == "__main__":
    main()
