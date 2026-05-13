"""ask_checkpoint.py — single-turn prompt against a tinker checkpoint.

Loads a Tinker sampling client off a sampler-weights path, renders a single
user turn, samples one response, and prints it.

Default points at the QRA-RL v2 SFT checkpoint
(`HISAB_SFT_FROM_PIPELINE_V2_RECIPE` output) — flip CHECKPOINT_PATH to test
any other tinker:// path.

Run with:
    uv run scripts/ask_checkpoint.py "your question here"
    uv run scripts/ask_checkpoint.py            # uses DEFAULT_PROMPT
"""

import asyncio
import sys

import tinker
from dotenv import load_dotenv
from oai_utils.tinker import setup_tinkermodel

# === Config ==========================================================

MODEL_NAME = "Qwen/Qwen3-32B"

# Sampler-weights path — feed the *_sampler_weights/_ side of a tinker
# checkpoint URI. setup_tinkermodel(...).sampling_client routes here.
CHECKPOINT_PATH = (
    "tinker://1237cd7d-e163-5ffb-9ef9-82c98c281079:train:0/sampler_weights/init_sft"
)

DEFAULT_PROMPT = (
    "Write a Rust program using the `hisab` library that constructs a 3x3 "
    "identity matrix and prints it."
)

SYSTEM_PROMPT: str | None = None  # e.g. "You are a Rust expert..."

MAX_TOKENS = 1024
TEMPERATURE = 0.7


# === Implementation ==================================================


async def ask(prompt: str) -> str:
    service_client = tinker.ServiceClient()
    tinker_model, _, renderer = setup_tinkermodel(
        MODEL_NAME, path=CHECKPOINT_PATH, service_client=service_client
    )

    messages: list[dict] = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})

    model_input = renderer.build_generation_prompt(messages)
    sample_result = await tinker_model.sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            stop=renderer.get_stop_sequences(),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        ),
    )
    tokens = sample_result.sequences[0].tokens
    message, _ = renderer.parse_response(tokens)
    content = message["content"]
    if isinstance(content, list):
        # Pull text parts out of multimodal content (thinking / text / image).
        text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
        return "".join(text_parts)
    return content or ""


async def main() -> None:
    load_dotenv()
    prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_PROMPT

    print("=" * 80)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Prompt: {prompt}")
    print("=" * 80)
    response = await ask(prompt)
    print(response)
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
