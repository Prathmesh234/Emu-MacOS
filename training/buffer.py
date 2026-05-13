"""
training/buffer.py

Tiny disk-backed work queue over the cached real trajectories. Sits between
dataset.py (pulls real trajs from HF) and synth.py (turns each into an
emu-format synthetic trajectory).

Status per traj: pending | in_progress | done | failed.
State persisted to data/state/buffer.json so runs are resumable.

Concurrency model
-----------------
Every mutating operation (`next`, `mark_done`, `mark_failed`, `scan`, `reset`)
runs under an OS-level file lock on `data/state/buffer.lock`. Inside the lock
we re-read `buffer.json` from disk, mutate the in-memory dict, and write back
atomically (tmp + replace). This makes it safe for multiple synth.py workers
in separate processes to share the same queue without picking duplicate
trajectories.

Stranded `in_progress` entries (from a Ctrl+C or crash) are only auto-recovered
when their `claimed_at` timestamp is older than `STALE_AFTER_SEC` (30 min by
default). That prevents a freshly-started worker from clobbering another live
worker's claim.

CLI:
    uv run python buffer.py status
    uv run python buffer.py refill                  # re-scan data/real_trajs/
    uv run python buffer.py recover [--max-age-sec N]  # force-recover stale claims
    uv run python buffer.py reset [--keep-done]
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import dataset

STATE_DIR = Path(__file__).parent / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "buffer.json"
LOCK_FILE = STATE_DIR / "buffer.lock"
STALE_AFTER_SEC = int(os.environ.get("BUFFER_STALE_AFTER_SEC", "1800"))
_THREAD_LOCK = threading.Lock()

try:
    import fcntl  # POSIX
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


@contextmanager
def _file_lock():
    """Cross-process exclusive lock around buffer.json mutations.

    Falls back to a thread lock only if fcntl is unavailable (e.g. on Windows),
    in which case cross-process safety is not guaranteed — emit a one-time
    warning when multiple workers might exist.
    """
    with _THREAD_LOCK:
        if not _HAS_FCNTL:
            yield
            return
        LOCK_FILE.touch(exist_ok=True)
        fh = open(LOCK_FILE, "r+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()


def _load() -> dict:
    if not STATE_FILE.exists():
        return {"trajs": {}}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def _save(state: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def _recover_stale(state: dict, max_age_sec: int) -> int:
    """Return entries to pending if their claimed_at is older than max_age_sec.

    Legacy entries without a `claimed_at` timestamp are *only* recovered when
    `max_age_sec == 0` (caller explicitly requested a full sweep, e.g. via the
    `buffer.py recover` CLI). The auto-recovery path on `Buffer.__init__` always
    passes a positive age, so it can't clobber claims made by a concurrently
    running pre-upgrade worker.
    """
    now = time.time()
    recovered = 0
    for meta in state.get("trajs", {}).values():
        if meta.get("status") != "in_progress":
            continue
        claimed_at = meta.get("claimed_at")
        if claimed_at is None:
            if max_age_sec <= 0:
                meta["status"] = "pending"
                recovered += 1
            continue
        if (now - float(claimed_at)) > max_age_sec:
            meta["status"] = "pending"
            meta.pop("claimed_at", None)
            recovered += 1
    return recovered


class Buffer:
    def __init__(self, recover_stale: bool = True, stale_after_sec: Optional[int] = None):
        """Load state. By default recovers in_progress entries older than
        `stale_after_sec` (defaults to `STALE_AFTER_SEC`, env-configurable).
        Pass `recover_stale=False` for read-only inspection.
        """
        age = stale_after_sec if stale_after_sec is not None else STALE_AFTER_SEC
        with _file_lock():
            self.state = _load()
            if recover_stale:
                recovered = _recover_stale(self.state, age)
                if recovered:
                    _save(self.state)
                    print(f"[buffer] recovered {recovered} stale in_progress entr{'y' if recovered == 1 else 'ies'} (>{age}s)")

    def _reload_locked(self):
        """Must be called while holding _file_lock()."""
        self.state = _load()

    def _flush(self):
        _save(self.state)

    def scan(self) -> int:
        """Sync queue with files in data/real_trajs/. Returns # added."""
        with _file_lock():
            self._reload_locked()
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
        with _file_lock():
            self._reload_locked()
            for tid, meta in self.state["trajs"].items():
                if meta["status"] == "pending":
                    meta["status"] = "in_progress"
                    meta["claimed_at"] = time.time()
                    meta["claimed_by"] = os.getpid()
                    self._flush()
                    real = dataset.load_traj(Path(meta["path"]))
                    return {"task_id": tid, "zip": meta["zip"], "real": real}
            return None

    def mark_done(self, tid: str):
        with _file_lock():
            self._reload_locked()
            if tid in self.state["trajs"]:
                entry = self.state["trajs"][tid]
                entry["status"] = "done"
                entry["error"] = None
                entry.pop("claimed_at", None)
                entry.pop("claimed_by", None)
                self._flush()

    def mark_failed(self, tid: str, error: str):
        with _file_lock():
            self._reload_locked()
            if tid in self.state["trajs"]:
                entry = self.state["trajs"][tid]
                entry["status"] = "failed"
                entry["error"] = error[:500]
                entry.pop("claimed_at", None)
                entry.pop("claimed_by", None)
                self._flush()

    def status(self) -> dict:
        with _file_lock():
            self._reload_locked()
            counts = {"pending": 0, "in_progress": 0, "done": 0, "failed": 0}
            for t in self.state["trajs"].values():
                counts[t["status"]] = counts.get(t["status"], 0) + 1
            return {
                "total": len(self.state["trajs"]),
                "by_status": counts,
                "state_file": str(STATE_FILE),
            }

    def reset(self, keep_done: bool = False):
        with _file_lock():
            self._reload_locked()
            if keep_done:
                self.state["trajs"] = {
                    k: v for k, v in self.state["trajs"].items() if v["status"] == "done"
                }
            else:
                self.state["trajs"] = {}
            self._flush()

    def force_recover(self, max_age_sec: int = 0) -> int:
        with _file_lock():
            self._reload_locked()
            recovered = _recover_stale(self.state, max_age_sec)
            if recovered:
                self._flush()
            return recovered


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("refill")
    rc = sub.add_parser("recover")
    rc.add_argument("--max-age-sec", type=int, default=0,
                    help="recover entries claimed more than N seconds ago (0 = recover all in_progress)")
    rs = sub.add_parser("reset"); rs.add_argument("--keep-done", action="store_true")
    args = ap.parse_args()

    if args.cmd == "status":
        buf = Buffer(recover_stale=False)
        print(json.dumps(buf.status(), indent=2))
    elif args.cmd == "refill":
        buf = Buffer()
        added = buf.scan()
        print(f"[buffer] added {added} new trajectories from disk")
        print(json.dumps(buf.status(), indent=2))
    elif args.cmd == "recover":
        buf = Buffer(recover_stale=False)
        n = buf.force_recover(max_age_sec=args.max_age_sec)
        print(f"[buffer] force-recovered {n} stale in_progress entr{'y' if n == 1 else 'ies'}")
        print(json.dumps(buf.status(), indent=2))
    elif args.cmd == "reset":
        buf = Buffer(recover_stale=False)
        buf.reset(keep_done=args.keep_done)
        print(json.dumps(buf.status(), indent=2))


if __name__ == "__main__":
    main()
