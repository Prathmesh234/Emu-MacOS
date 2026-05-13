"""One-shot helper: rewrite buffer paths to mac-local + fetch missing pending
real_traj data files from HF.

Run from training/:  uv run python _resume_helper.py
Safe to re-run — idempotent (skips already-fetched files).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import dataset
from tqdm import tqdm

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
TRAINING = Path(__file__).resolve().parent
STATE = TRAINING / "data" / "state" / "buffer.json"
REAL_TRAJS = TRAINING / "data" / "real_trajs"


def main() -> int:
    if not STATE.exists():
        print(f"ERROR: {STATE} not found", file=sys.stderr)
        return 1

    bak = STATE.with_suffix(".json.bak")
    shutil.copy2(STATE, bak)
    print(f"[backup] {STATE.name} -> {bak.name}")

    state = json.loads(STATE.read_text(encoding="utf-8"))
    print(f"[buffer] {len(state['trajs'])} entries")

    rewritten = 0
    for tid, meta in state["trajs"].items():
        new_path = str(REAL_TRAJS / meta["zip"] / tid)
        if meta["path"] != new_path:
            meta["path"] = new_path
            rewritten += 1
    print(f"[paths]  rewrote {rewritten} entries -> {REAL_TRAJS}/<zip>/<tid>")

    needed_by_zip: dict[str, set[str]] = {}
    for tid, meta in state["trajs"].items():
        if meta["status"] not in ("pending", "in_progress"):
            continue
        needed_by_zip.setdefault(meta["zip"], set()).add(tid)

    total_needed = sum(len(v) for v in needed_by_zip.values())
    print(f"[fetch]  need {total_needed} pending/in_progress traj files "
          f"across {len(needed_by_zip)} zip(s)")

    fetched = 0
    skipped_existing = 0
    fetch_errors: list[tuple[str, str]] = []

    for zip_stem, tids in needed_by_zip.items():
        zname = zip_stem + ".zip"
        print(f"[fetch]  opening {zname}  (need {len(tids)} entries)")
        with dataset._open_zip(zname) as zf:
            entries_by_id: dict[str, list[str]] = {}
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                m = UUID_RE.search(name)
                if m and m.group(0) in tids:
                    entries_by_id.setdefault(m.group(0), []).append(name)

            missing_in_zip = tids - set(entries_by_id.keys())
            if missing_in_zip:
                print(f"[fetch]  ⚠ {len(missing_in_zip)} tids not found in zip "
                      f"index — likely a different zip recorded them")
                for t in list(missing_in_zip)[:3]:
                    print(f"         {t}")

            for tid in tqdm(sorted(entries_by_id.keys()), desc=zip_stem[:32]):
                out_dir = REAL_TRAJS / zip_stem / tid
                traj_file = out_dir / "traj.jsonl"
                if traj_file.exists():
                    skipped_existing += 1
                    continue
                try:
                    dataset._extract_one(zf, zip_stem, tid, entries_by_id)
                    fetched += 1
                except Exception as e:  # noqa: BLE001
                    fetch_errors.append((tid, str(e)[:160]))

    print(f"[fetch]  done: fetched={fetched}, "
          f"already_on_disk={skipped_existing}, errors={len(fetch_errors)}")
    for tid, err in fetch_errors[:5]:
        print(f"         err {tid[:12]}: {err}")

    missing: list[str] = []
    for tid, meta in state["trajs"].items():
        if meta["status"] not in ("pending", "in_progress"):
            continue
        if not (Path(meta["path"]) / "traj.jsonl").exists():
            missing.append(tid)

    if missing:
        print(f"[verify] ⚠ {len(missing)} pending entries STILL missing local data")
        for tid in missing[:10]:
            print(f"         {tid[:12]}  expected: {state['trajs'][tid]['path']}")
        print(f"[verify] aborting WITHOUT saving rewritten buffer "
              f"(original preserved). Restore from {bak.name} if needed.")
        return 2

    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"[save]   buffer.json updated ({rewritten} paths + {fetched} new traj files)")
    print(f"[verify] ✓ all {total_needed} pending/in_progress entries have local data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
