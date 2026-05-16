"""Prefetch OSWorld screenshots for VLM SFT.

Reads data/synth/synth_trajectories.json, gathers (source_zip, task_id) pairs,
and streams PNGs from xlangai/ubuntu_osworld_verified_trajs (via remote-zip)
into data/real_trajs/<zip_stem>/<task_id>/*.png.

Arena trajectories (source_zip == 'agent-arena-gemini') are skipped — Arena
JSONL rows carry no screenshot bytes.

Run:
    uv run python prefetch_screenshots.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

# Reuse the HF-zip plumbing from dataset.py
from dataset import REAL_TRAJS_DIR, _UUID_RE, _open_zip

SYNTH_PATH = Path(__file__).parent / "data" / "synth" / "synth_trajectories.json"
ARENA_STEM = "agent-arena-gemini"


def _normalize_zip_name(source_zip: str) -> str:
    """Map source_zip metadata back to its canonical .zip filename."""
    if source_zip.endswith(".zip"):
        return source_zip
    return source_zip + ".zip"


def collect_targets() -> dict[str, set[str]]:
    """Returns {zip_filename: {task_id, ...}} for OSWorld-derived trajs only."""
    data = json.loads(SYNTH_PATH.read_text())
    targets: dict[str, set[str]] = defaultdict(set)
    skipped_arena = 0
    for item in data:
        meta = item.get("_meta", {})
        sz = meta.get("source_zip") or meta.get("osworld", {}).get("source_zip")
        tid = item.get("task_id")
        if not sz or not tid:
            continue
        if sz == ARENA_STEM:
            skipped_arena += 1
            continue
        targets[_normalize_zip_name(sz)].add(tid)
    print(
        f"[prefetch] {sum(len(v) for v in targets.values())} OSWorld task ids "
        f"across {len(targets)} zips (skipped {skipped_arena} arena trajs)",
        file=sys.stderr,
    )
    return targets


def fetch_pngs_for_zip(zip_name: str, task_ids: set[str]) -> tuple[int, int]:
    """Stream all PNGs for the requested task_ids from one HF zip.

    Returns (num_pngs_written, num_tasks_with_any_png).
    """
    zip_stem = Path(zip_name).stem
    written = 0
    tasks_seen: set[str] = set()
    with _open_zip(zip_name) as zf:
        png_by_task: dict[str, list[str]] = defaultdict(list)
        for entry in zf.namelist():
            if not entry.lower().endswith(".png"):
                continue
            m = _UUID_RE.search(entry)
            if not m:
                continue
            tid = m.group(0)
            if tid in task_ids:
                png_by_task[tid].append(entry)

        for tid, entries in tqdm(png_by_task.items(), desc=f"png:{zip_stem[:24]}"):
            out_dir = REAL_TRAJS_DIR / zip_stem / tid
            out_dir.mkdir(parents=True, exist_ok=True)
            # Sort so step ordering on disk matches in-zip ordering. The
            # in-zip names typically include a step index (step_0.png ...);
            # lexical sort matches numeric order when zero-padded, and is at
            # least deterministic when not.
            for entry in sorted(entries):
                out_path = out_dir / Path(entry).name
                if out_path.exists() and out_path.stat().st_size > 0:
                    continue
                try:
                    out_path.write_bytes(zf.read(entry))
                    written += 1
                except Exception as e:  # noqa: BLE001
                    print(f"[prefetch] skip {entry}: {e}", file=sys.stderr)
            # Manifest of PNG basenames in deterministic order. The training
            # loader maps the Nth '[screenshot]' placeholder to manifest[N].
            manifest = sorted(p.name for p in out_dir.glob("*.png"))
            (out_dir / "screenshots.json").write_text(json.dumps(manifest, indent=2))
            if manifest:
                tasks_seen.add(tid)
    return written, len(tasks_seen)


def main() -> None:
    targets = collect_targets()
    if not targets:
        print("[prefetch] no OSWorld targets — nothing to do.", file=sys.stderr)
        return
    grand_written = grand_tasks = 0
    for zip_name, task_ids in targets.items():
        try:
            w, t = fetch_pngs_for_zip(zip_name, task_ids)
        except Exception as e:  # noqa: BLE001
            print(f"[prefetch] {zip_name} failed: {e}", file=sys.stderr)
            continue
        grand_written += w
        grand_tasks += t
        print(f"[prefetch] {zip_name}: {w} PNGs, {t} tasks populated",
              file=sys.stderr)
    print(f"[prefetch] DONE — {grand_written} PNGs across {grand_tasks} tasks "
          f"under {REAL_TRAJS_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
