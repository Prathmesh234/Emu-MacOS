"""
training/buffer.py

Tiny disk-backed work queue over the cached real trajectories. Sits between
dataset.py (pulls real trajs from HF) and synth.py (turns each into an
emu-format synthetic trajectory).

Status per traj: pending | in_progress | done | failed.
State persisted to data/state/buffer.json so runs are resumable.

CLI:
    uv run python buffer.py status
    uv run python buffer.py refill           # re-scan data/real_trajs/
    uv run python buffer.py reset [--keep-done]
"""
from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Optional

import dataset

STATE_DIR = Path(__file__).parent / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "buffer.json"
_LOCK = threading.Lock()


def _load() -> dict:
    if not STATE_FILE.exists():
        return {"trajs": {}}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def _save(state: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


class Buffer:
    def __init__(self):
        with _LOCK:
            self.state = _load()
            # Auto-recover entries stranded by Ctrl+C / crash mid-generation.
            # `next()` only picks up `pending`, so without this any in-flight
            # entry would be skipped forever after a kill.
            recovered = 0
            for meta in self.state.get("trajs", {}).values():
                if meta.get("status") == "in_progress":
                    meta["status"] = "pending"
                    recovered += 1
            if recovered:
                self._flush()
                print(f"[buffer] recovered {recovered} stranded in_progress entr{'y' if recovered == 1 else 'ies'}")

    def _flush(self):
        _save(self.state)

    def scan(self) -> int:
        """Sync queue with files in data/real_trajs/. Returns # added."""
        added = 0
        for d in dataset.iter_cached():
            tid = d.name
            if tid in self.state["trajs"]:
                continue
            self.state["trajs"][tid] = {
                "path": str(d),
                "zip": d.parent.name,
                "status": "pending",
                "error": None,
            }
            added += 1
        if added:
            self._flush()
        return added

    def next(self) -> Optional[dict]:
        with _LOCK:
            for tid, meta in self.state["trajs"].items():
                if meta["status"] == "pending":
                    meta["status"] = "in_progress"
                    self._flush()
                    real = dataset.load_traj(Path(meta["path"]))
                    return {"task_id": tid, "zip": meta["zip"], "real": real}
            return None

    def mark_done(self, tid: str):
        with _LOCK:
            if tid in self.state["trajs"]:
                self.state["trajs"][tid]["status"] = "done"
                self.state["trajs"][tid]["error"] = None
                self._flush()

    def mark_failed(self, tid: str, error: str):
        with _LOCK:
            if tid in self.state["trajs"]:
                self.state["trajs"][tid]["status"] = "failed"
                self.state["trajs"][tid]["error"] = error[:500]
                self._flush()

    def status(self) -> dict:
        with _LOCK:
            counts = {"pending": 0, "in_progress": 0, "done": 0, "failed": 0}
            for t in self.state["trajs"].values():
                counts[t["status"]] = counts.get(t["status"], 0) + 1
            return {
                "total": len(self.state["trajs"]),
                "by_status": counts,
                "state_file": str(STATE_FILE),
            }

    def reset(self, keep_done: bool = False):
        with _LOCK:
            if keep_done:
                self.state["trajs"] = {
                    k: v for k, v in self.state["trajs"].items() if v["status"] == "done"
                }
            else:
                self.state["trajs"] = {}
            self._flush()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("refill")
    rs = sub.add_parser("reset"); rs.add_argument("--keep-done", action="store_true")
    args = ap.parse_args()

    buf = Buffer()
    if args.cmd == "status":
        print(json.dumps(buf.status(), indent=2))
    elif args.cmd == "refill":
        added = buf.scan()
        print(f"[buffer] added {added} new trajectories from disk")
        print(json.dumps(buf.status(), indent=2))
    elif args.cmd == "reset":
        buf.reset(keep_done=args.keep_done)
        print(json.dumps(buf.status(), indent=2))


if __name__ == "__main__":
    main()
