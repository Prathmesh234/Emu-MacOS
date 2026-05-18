"""Sample from the fine-tuned Qwen3.5-397B-A17B VLM checkpoint.

Loads the step-50 LoRA checkpoint from Tinker, builds a single-turn prompt
with a real screenshot, and prints the model's response.

Usage:
    export TINKER_API_KEY=...   # or rely on training/.env
    uv run python sample_vlm.py [--checkpoint tinker://.../sampler_weights/000050]
                                [--image /path/to/screenshot.png]
                                [--prompt "Your instruction here"]
                                [--base]   # also sample from the base model
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import tinker
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers import ImagePart, Message, TextPart

MODEL_NAME = "Qwen/Qwen3.5-397B-A17B"
RENDERER_NAME = "qwen3_5_disable_thinking"
DEFAULT_CHECKPOINT = (
    "tinker://2c65dd6d-78ae-53e5-a7c1-00b03ea21097:train:0/"
    "sampler_weights/000050"
)
DEFAULT_IMAGE = (
    Path(__file__).parent
    / "data" / "real_trajs"
    / "results_gemini_50_steps_aws"
    / "00fa164e-2612-4439-992e-157d019a8436"
    / "step_1_20250725@235857.png"
)
DEFAULT_PROMPT = (
    "You are a desktop computer-use agent. Look at the screenshot and describe "
    "what application is open and the most reasonable next action to take. "
    "Respond as a single JSON object with keys 'observation' and 'action'."
)


def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


async def _sample(
    sc: tinker.ServiceClient,
    label: str,
    *,
    model_path: str | None,
    base_model: str | None,
    renderer,
    prompt_messages: list[Message],
) -> None:
    print(f"\n=== {label} ===")
    if model_path:
        client = await sc.create_sampling_client_async(model_path=model_path)
    else:
        client = await sc.create_sampling_client_async(base_model=base_model)

    prompt = renderer.build_generation_prompt(prompt_messages)
    stops = renderer.get_stop_sequences()

    out = await client.sample_async(
        prompt=prompt,
        sampling_params=tinker.SamplingParams(
            max_tokens=512,
            temperature=0.2,
            top_p=0.95,
            stop=stops,
        ),
        num_samples=1,
    )
    seq = out.sequences[0]
    message, _ok = renderer.parse_response(seq.tokens)
    content = message.get("content")
    if isinstance(content, list):
        text = "".join(
            p.get("text", "") for p in content if isinstance(p, dict)
        )
    else:
        text = str(content)
    print(text.strip())
    print(f"[stop_reason={seq.stop_reason}, tokens={len(seq.tokens)}]")


async def main_async(args: argparse.Namespace) -> None:
    _load_dotenv()
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY not set (looked in env and .env)")

    img = Path(args.image).resolve()
    if not img.exists():
        raise SystemExit(f"Image not found: {img}")

    tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
    image_processor = get_image_processor(MODEL_NAME)
    renderer = renderers.get_renderer(
        RENDERER_NAME, tokenizer, image_processor=image_processor)

    prompt_messages = [
        Message(
            role="user",
            content=[
                ImagePart(type="image", image=img.as_uri()),
                TextPart(type="text", text=args.prompt),
            ],
        )
    ]

    sc = tinker.ServiceClient()
    print(f"Image: {img}")
    print(f"Prompt: {args.prompt}")

    await _sample(
        sc, f"Fine-tuned checkpoint ({args.checkpoint.rsplit('/', 1)[-1]})",
        model_path=args.checkpoint,
        base_model=None,
        renderer=renderer,
        prompt_messages=prompt_messages,
    )

    if args.base:
        await _sample(
            sc, f"Base model ({MODEL_NAME})",
            model_path=None,
            base_model=MODEL_NAME,
            renderer=renderer,
            prompt_messages=prompt_messages,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--base", action="store_true",
                    help="Also sample from the base (untuned) model.")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
