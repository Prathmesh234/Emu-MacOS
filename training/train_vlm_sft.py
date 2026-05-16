"""Vision SFT of Qwen3.5-397B-A17B on OSWorld synthetic trajectories.

Reads data/synth/synth_trajectories.json, replaces each '[screenshot]'
placeholder with the matching screenshot from data/real_trajs/<zip>/<task_id>/,
converts Anthropic-style content blocks to Qwen3-VL messages, and runs
LoRA SFT via tinker-cookbook.

Prerequisite: run `prefetch_screenshots.py` first to populate
data/real_trajs/.

Run:
    export TINKER_API_KEY=...
    uv run python train_vlm_sft.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chz
import tinker
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers import ImagePart, Message, TextPart, TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import (
    SupervisedDataset,
    SupervisedDatasetBuilder,
)

# ---- config knobs (edit here) ------------------------------------------------

MODEL_NAME = "Qwen/Qwen3.5-397B-A17B"     # user-confirmed VLM
RENDERER_NAME = "qwen3_vl_instruct"
LOG_PATH = "~/logs/emu-vlm-sft"
LEARNING_RATE = 1e-4
LORA_RANK = 32
BATCH_SIZE = 4                            # trajectories per batch
MAX_LENGTH = 16384                        # max tokens per example
NUM_EPOCHS = 1
SAVE_EVERY = 50
EVAL_EVERY = 25
EVAL_FRACTION = 0.05                      # 5% held out for eval

SYNTH_PATH = Path(__file__).parent / "data" / "synth" / "synth_trajectories.json"
REAL_TRAJS_DIR = Path(__file__).parent / "data" / "real_trajs"
ARENA_STEM = "agent-arena-gemini"

SCREENSHOT_TOKEN = "[screenshot]"

# ---- message conversion ------------------------------------------------------


@dataclass
class _ConvResult:
    messages: list[Message]
    n_placeholders: int
    n_images_attached: int


def _normalize_zip_stem(source_zip: str) -> str:
    return source_zip[:-4] if source_zip.endswith(".zip") else source_zip


def _list_screenshots(zip_stem: str, task_id: str) -> list[Path]:
    """Return ordered PNG paths for one task, [] if none on disk."""
    manifest = REAL_TRAJS_DIR / zip_stem / task_id / "screenshots.json"
    if not manifest.exists():
        return []
    names = json.loads(manifest.read_text())
    base = REAL_TRAJS_DIR / zip_stem / task_id
    return [base / n for n in names if (base / n).exists()]


def _serialize_tool_use(block: dict) -> str:
    payload = {"id": block.get("id"), "name": block.get("name"),
               "input": block.get("input")}
    return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"


def _serialize_tool_result(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, list):
        content = "".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return (f"<tool_result id={block.get('tool_use_id')!r}>\n"
            f"{content}\n</tool_result>")


def _serialize_desktop_action(block: dict) -> str:
    return json.dumps({"action": block.get("action"),
                       "done": block.get("done", False)},
                      ensure_ascii=False)


def _convert_messages(item: dict, screenshots: list[Path]) -> _ConvResult:
    """Anthropic-style messages → list[Message] for Qwen3-VL renderer.

    - assistant tool_use blocks → serialized text in the assistant turn
    - user tool_result blocks → serialized text in the user turn
    - '[screenshot]' tokens inside user text → split into TextPart/ImagePart,
      consuming `screenshots` in order

    A turn whose final content list is empty is dropped.
    """
    out: list[Message] = []
    img_cursor = 0
    n_placeholders = 0

    # Prepend system prompt as a system message if present.
    sys_prompt = item.get("system")
    if isinstance(sys_prompt, str) and sys_prompt.strip():
        out.append(Message(role="system",
                           content=[TextPart(type="text", text=sys_prompt)]))

    for msg in item.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, list):
            content = [{"type": "text", "text": str(content or "")}]
        parts: list[Any] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "")
                # Split around screenshot placeholders, attaching images.
                if SCREENSHOT_TOKEN in text:
                    chunks = text.split(SCREENSHOT_TOKEN)
                    for i, chunk in enumerate(chunks):
                        if chunk:
                            parts.append(TextPart(type="text", text=chunk))
                        if i < len(chunks) - 1:
                            n_placeholders += 1
                            if img_cursor < len(screenshots):
                                parts.append(ImagePart(
                                    type="image",
                                    image=screenshots[img_cursor].as_uri(),
                                ))
                                img_cursor += 1
                            else:
                                # No image to attach — leave a marker so the
                                # text remains grammatical.
                                parts.append(TextPart(
                                    type="text", text="[screenshot-missing]"))
                elif text:
                    parts.append(TextPart(type="text", text=text))
            elif btype == "tool_use":
                parts.append(TextPart(type="text",
                                      text=_serialize_tool_use(block)))
            elif btype == "tool_result":
                parts.append(TextPart(type="text",
                                      text=_serialize_tool_result(block)))
            elif btype == "desktop_action":
                parts.append(TextPart(type="text",
                                      text=_serialize_desktop_action(block)))
            # else: ignore unknown block types

        if parts:
            out.append(Message(role=role, content=parts))

    return _ConvResult(messages=out, n_placeholders=n_placeholders,
                       n_images_attached=img_cursor)


# ---- dataset -----------------------------------------------------------------


class _EmuVisionDataset(SupervisedDataset):
    """Holds pre-built tinker.Datum objects, batched at __init__ time."""

    def __init__(self, datums: list[tinker.Datum], batch_size: int):
        self._datums = datums
        self._batch_size = batch_size

    def __len__(self) -> int:
        return max(1, len(self._datums) // self._batch_size)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self._batch_size
        return self._datums[start:start + self._batch_size]

    def set_epoch(self, seed: int = 0) -> None:
        import random
        random.Random(seed).shuffle(self._datums)


@chz.chz
class EmuVisionDatasetBuilder(SupervisedDatasetBuilder):
    """Build train/eval datasets from synth_trajectories.json + real_trajs/."""

    model_name: str
    renderer_name: str
    batch_size: int
    max_length: int
    eval_fraction: float
    max_trajectories: int = 0

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        image_processor = get_image_processor(self.model_name)
        renderer = renderers.get_renderer(
            self.renderer_name, tokenizer, image_processor=image_processor)

        trajectories = json.loads(SYNTH_PATH.read_text())
        datums: list[tinker.Datum] = []
        n_skipped_no_imgs = n_skipped_arena = 0

        for item in trajectories:
            if (self.max_trajectories
                    and len(datums) >= self.max_trajectories):
                break
            meta = item.get("_meta", {})
            source_zip = (meta.get("source_zip")
                          or meta.get("osworld", {}).get("source_zip", ""))
            task_id = item.get("task_id", "")
            if source_zip == ARENA_STEM:
                # Arena has no screenshots on disk; skip for VLM SFT.
                n_skipped_arena += 1
                continue

            zip_stem = _normalize_zip_stem(source_zip)
            screenshots = _list_screenshots(zip_stem, task_id)
            if not screenshots:
                n_skipped_no_imgs += 1
                continue

            conv = _convert_messages(item, screenshots)
            if conv.n_images_attached == 0:
                n_skipped_no_imgs += 1
                continue

            # qwen3_vl_instruct doesn't satisfy the extension property, so
            # we cannot train on all assistant messages of a multi-turn
            # conversation in one pass. Per the Tinker docs, emit one
            # Datum per assistant turn with the prefix up through it,
            # training only on that last assistant message.
            assistant_idxs = [
                i for i, m in enumerate(conv.messages)
                if m.get("role") == "assistant"
            ]
            traj_datums = 0
            for ai in assistant_idxs:
                sub = conv.messages[: ai + 1]
                try:
                    model_input, weights = renderer.build_supervised_example(
                        sub,
                        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[builder] {task_id}@a{ai} render failed: {e}",
                          file=sys.stderr)
                    continue
                datum = datum_from_model_input_weights(
                    model_input, weights,
                    max_length=self.max_length,
                    reduction="mean",
                )
                datums.append(datum)
                traj_datums += 1
                if (self.max_trajectories
                        and len(datums) >= self.max_trajectories):
                    break

        print(f"[builder] kept {len(datums)} datums "
              f"(arena_skipped={n_skipped_arena}, "
              f"no_imgs={n_skipped_no_imgs})", file=sys.stderr)

        if not datums:
            raise RuntimeError(
                "No training examples — did you run prefetch_screenshots.py?")

        # Stable train/eval split.
        n_eval = max(1, int(len(datums) * self.eval_fraction))
        eval_datums = datums[:n_eval]
        train_datums = datums[n_eval:]

        train_ds = _EmuVisionDataset(train_datums, self.batch_size)
        eval_ds = (_EmuVisionDataset(eval_datums, self.batch_size)
                   if eval_datums else None)
        return train_ds, eval_ds


# ---- entry point -------------------------------------------------------------


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME, help="Model name override.")
    ap.add_argument("--renderer", default=RENDERER_NAME)
    ap.add_argument("--log-path", default=LOG_PATH)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    ap.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--lora-rank", type=int, default=LORA_RANK)
    ap.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    ap.add_argument("--max-trajectories", type=int, default=0,
                    help="Cap on number of trajectories used (0 = no cap). "
                         "Useful for smoke tests.")
    ap.add_argument("--wandb-project", default="emu-vlm-sft")
    args = ap.parse_args()

    config = train.Config(
        log_path=args.log_path,
        model_name=args.model,
        renderer_name=args.renderer,
        dataset_builder=EmuVisionDatasetBuilder(
            model_name=args.model,
            renderer_name=args.renderer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            eval_fraction=EVAL_FRACTION,
            max_trajectories=args.max_trajectories,
        ),
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        num_epochs=args.num_epochs,
        save_every=SAVE_EVERY,
        eval_every=EVAL_EVERY,
        wandb_project=args.wandb_project,
    )
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
