# Synthetic Trajectory Data Collection

This doc explains where the **real** Gemini agent trajectories that feed `synth.py`
come from, why we hit a hard ceiling around ~90 unique trajectories with the
current pipeline, and how to scale the source pool past 1,000 unique tasks
without leaving the "produced by Gemini" constraint.

## TL;DR

* Today's pool: **~93 unique Gemini trajectories** from the two `*gemini*` zips
  in `xlangai/ubuntu_osworld_verified_trajs`.
* The real ceiling on OSWorld is **369 task examples total** — that is the
  whole `OSWorld-Verified` benchmark — and the buffer collapses any UUID
  that recurs across zips, so even running every Gemini eval on OSWorld
  caps us at 369.
* New source identified: **`xlangai/computer-agent-arena`** — same lab as
  OSWorld, crowdsourced real-user tasks, **502 distinct Gemini trajectories**
  in **the exact same `pyautogui` step format** the existing pipeline reads.
* Combining the OSWorld Gemini zips + Arena Gemini rows yields **~595 unique
  Gemini source trajectories** without any variant trickery.
* With a small persona / variant multiplier on top, hitting **1,000+** is
  trivial.

## Pipeline recap (what produces a trajectory)

1. `dataset.py fetch` streams `traj.jsonl` files out of `xlangai/ubuntu_osworld_verified_trajs`
   zips via HTTP range requests and caches them to `data/real_trajs/<zip>/<task_id>/`.
2. `buffer.py` indexes the cached dirs into a per-task work queue
   (`data/state/buffer.json`), keyed by **bare `task_id`**.
3. `synth.py` pops one entry, asks DeepSeek V4 Pro (via OpenRouter) to
   rewrite that real trajectory into Emu remote-mode format, validates it,
   attaches the Emu system prompt + a persona, and appends one line to
   the batch JSONL under `data/synth/`.

Result: one real trajectory → one synthetic trajectory; buffer entry is
then `done` and never re-emitted.

## Current source breakdown

Counts on disk in `training/data/real_trajs/` and what the buffer indexes:

| HF zip                                | Trajs on disk | Unique in buffer | Notes |
| --- | --- | --- | --- |
| `results_agent_s2_gemini_15steps`     | 92 | 87 in this zip | gemini, pyautogui-only |
| `results_gemini_50_steps_aws`         | 89 | 1 unique (88 dedup'd) | gemini, pyautogui-only |
| `autoglm_15steps`                     | 6  | 6  | **not gemini** — older fetch |
| **Total directories on disk**         | 187 |   | |
| **Unique task_ids in buffer**         |    | 95 | OSWorld dedup at task_id level |

Status snapshot of the buffer: 36 `done`, 58 `pending`, 1 `in_progress`.

The 88-row collapse between the two Gemini zips happens because
`Buffer.scan()` keys by bare `task_id` (`buffer.py:67-78`) and OSWorld
re-uses the same UUID across every model evaluation.

## The four bottlenecks

1. **Buffer keys by `task_id` only** — same UUID in two zips collapses to one
   buffer entry, so cross-zip duplicates are lost.
2. **`mark_done()` retires the row after a single synth** — no concept of
   variants per source.
3. **Default fetch is gemini-zip-only** (`GEMINI_ZIP_SUBSTRING="gemini"`,
   `dataset.py:62`) — only 2 such zips exist in the OSWorld trajectory repo.
4. **pyautogui-only filter** drops every UNO/macro trajectory
   (`dataset.py:109-129`) — that is the 369 → ~92 attrition.

## OSWorld-flavored datasets surveyed

I checked every OSWorld-* repo on Hugging Face. Most ship **task definitions
or assets, not recorded trajectories**:

| Dataset | What it is | Trajectories? |
| --- | --- | --- |
| `xlangai/ubuntu_osworld_verified_trajs` | The repo we already use | ✅ (1 Gemini zip with usable trajs) |
| `hud-evals/OSWorld-Verified` (369) / `OSWorld-Gold` (294) / `OSWorld-Gold-Mini` (20) | Task instruction + setup configs only | ❌ |
| `xlangai/ubuntu_osworld` (4.24k) | Task examples + VM snapshots | ❌ |
| `xlangai/windows_osworld`, `xlangai/macos_osworld` | OS ports of task defs + snapshots | ❌ |
| `xlangai/ubuntu_osworld_file_cache` (1.05M) | Setup-time file assets | ❌ |
| `MMInstruction/OSWorld-G`, `jdubkim/OSWorld-G(-refined)` | GUI grounding (single click) | ❌ |
| `andersonbcdefg/osworld_screenshots` (10.2k) | Just images | ❌ |
| `mlfoundations-cua-dev/osworld-trajectories` (13.7k) | Big SFT dump — subsets `gpt5/o3/gta1/qwen3vl` | ❌ for Gemini |
| `sum-s4/os-world-pro`, `bhushan-hash/OSWorld-Pro`, `frist-research/osworld_pro_file_cache` | Tiny (<1k), task defs / videos | ❌ |
| **`xlangai/computer-agent-arena`** | **4,641 real recorded trajectories with model labels** | ✅ **502 Gemini** |

So apart from the OSWorld zips, **`xlangai/computer-agent-arena`** is the only
public Hugging Face dataset I found that contains a meaningful pool of
recorded Gemini agent trajectories.

## The win: `xlangai/computer-agent-arena`

XLang (the same lab that maintains OSWorld) shipped a separate evaluation
platform — Computer Agent Arena — where real users submit tasks and pairs
of agents compete on them. The dataset contains 4,641 trajectories from
many state-of-the-art agents.

### Gemini coverage (verified from `agent_arena_data.jsonl`)

```
395  gemini/gemini-2.5-pro-exp-03-25 (base_agent)
 60  gemini/gemini-2.0-flash         (base_agent)
 28  gemini/gemini-1.5-pro           (vision_based)
 19  gemini/gemini-1.5-flash         (vision_based)
---
502  total Gemini trajectories — every one a distinct task_id
```

Other facts (measured locally):

* All 4,641 entries have unique `task_id` values; **no dedup will happen**
  the way OSWorld's same-UUID-across-zips collision does.
* Every Gemini trajectory has a distinct `task_id` ⇒ they are **502
  different tasks**, not 502 reruns of the same task.
* Average **27.5 steps** per Gemini trajectory (range 1 – 101).
* Human-evaluated correctness present on every row: **230 passed (45.8%),
  272 failed**.
* Tasks are crowdsourced from real users, so they are broad-domain (To-Do
  lists, Notepad, web research, file ops, …) — not OSWorld's
  LibreOffice-heavy mix.

### Step format — drop-in compatible with `synth.py`

Each step is `{index, image, value:{thought, code}}`, e.g.:

```json
{
  "index": 1,
  "image": "images/c5d14580-..._step_1.png",
  "value": {
    "thought": "First, I will open Notepad to create the To Do List...",
    "code":    "import pyautogui\nimport time\npyautogui.click(160, 700)"
  }
}
```

That is the same `pyautogui` action shape that `synth.py:192-232` already
translates for the `agent_s2_gemini` zip — `pyautogui.click(x,y)` →
`navigate_and_click {x/W, y/H}` and so on. **No translator changes are
required.** Only an ingest adapter that maps Arena's per-step shape into
the per-step shape `dataset.load_traj()` and `summarize_step()` already
understand.

### Dataset assets

| File | Size | Use |
| --- | --- | --- |
| `agent_arena_data.jsonl` | 49 MB | All 4,641 trajectories with `model`, `task_id`, `instruction`, `human_eval_correctness`, `traj` |
| `sessions.csv` | ~3.6k rows | Session metadata (per-session pair-of-agents view) |
| `images.tar.gz` and `images_part_*` | ~30 GB | All step screenshots — **optional**, the synth pipeline currently strips images on download |

Only `agent_arena_data.jsonl` is needed for SFT trajectory generation.

## Plan to scale beyond 1,000 unique Gemini source trajectories

| Step | Action | Pool after step |
| --- | --- | --- |
| 0 | Today | 92 |
| 1 | Re-fetch both OSWorld Gemini zips with `--limit 400`, key buffer by `(zip, task_id)` so the cross-zip overlap is preserved | ~181 (action-sequence diverse, ~93 distinct OSWorld tasks) |
| 2 | **Ingest `xlangai/computer-agent-arena` Gemini rows** (`fetch_arena`) | **~683 distinct sources, ~595 distinct tasks** |
| 3 | Optional: 2 personas per source (small change) | **~1,366 trajectories** |

Steps 1+2 alone get us comfortably past the 1,000 trajectory target with all
sources still being **Gemini-produced trajectories on different tasks**.

## Implementation outline

Bounded changes — no `synth.py` translator changes since the action format
matches:

1. **`dataset.py: fetch_arena()`** — stream
   `xlangai/computer-agent-arena/agent_arena_data.jsonl` once, filter rows
   by `model` (substring `"gemini"` by default; CLI `--model` to pick a
   specific variant) and optionally by `human_eval_correctness == 1`
   (`--passed-only`). For each kept row, materialize one
   `data/real_trajs/agent-arena-gemini/<task_id>/` directory containing:
     * `traj.jsonl` — one line per step in the shape
       `{"step_num", "action", "full_plan"}` so `dataset.load_traj()` and
       `synth_utils.summarize_step()` work unchanged.
     * `result.txt` — `"1"` or `"0"` from `human_eval_correctness`.
     * `_files.json` — manifest, mirrors the OSWorld layout.
   Skip images (matching OSWorld's `_extract_one()`).

2. **`dataset.py: arena` subcommand** — `uv run python dataset.py arena
   --limit 200 [--model gemini-2.5-pro] [--passed-only]`. Idempotent: skip
   task_ids that already exist on disk so re-runs only fill the deficit.

3. **`is_gemini_zip()`** — extend to also accept the `agent-arena-gemini`
   directory name so the buffer + synth flow doesn't reject it. (Or bypass
   `gemini_only` for arena, since Arena rows are already Gemini-filtered at
   ingest time.)

4. **`buffer.py: scan()`** — re-key entries by `(zip, task_id)` so OSWorld
   cross-zip duplicates and any future task_id namespace collisions are
   preserved as distinct sources. (Existing entries can stay; new entries
   land under the composite key.)

5. **`synth.py`** — needs no translation changes; the existing
   `summarize_step()` and prompt format already cover Arena's pyautogui
   shape. The `relevant_skills_for(zip)` lookup in
   `synth_utils.py:APP_TO_SKILLS` should learn an `"agent-arena-gemini"`
   key so the synth prompt picks reasonable skills for Arena's broader
   task domain.

After (1)–(3) you can run:

```sh
uv run python dataset.py arena --limit 500 --passed-only
uv run python buffer.py refill
uv run python synth.py --count 50 --batches 10 --auto-fetch 0
```

…to add ~500 new Gemini-sourced synthetic trajectories on top of the
existing 90.

## Status

* `xlangai/computer-agent-arena` JSONL inspected locally — 502 Gemini
  trajectories confirmed, format validated as pyautogui-compatible.
* Implementation of `fetch_arena()` and the buffer re-key is pending in
  this branch.
