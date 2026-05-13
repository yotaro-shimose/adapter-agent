"""solve_with_checkpoint.py — drive `solve_verify` with a TinkerModel.

Wires up a single `solve_verify` session where the solver is the trained
checkpoint (sampler-weights path), the verifier is Gemini, and the runtime
pool is hisab's cloudrun image. The agent gets the standard XML-tag tool
surface (<grep>/<read>/<ls>/<write_and_run>/<submit>) and runs to completion
or `max_turns`.

After the session ends we dump the full trajectory to stdout so you can
see every turn (assistant message + tool result) the way graphvis renders
trial logs.

Run with:
    uv run scripts/solve_with_checkpoint.py "your task instruction"
    uv run scripts/solve_with_checkpoint.py            # uses DEFAULT_TASK
"""

import asyncio
import logging
import re
import sys

import tinker
from agents import set_tracing_disabled
from dotenv import load_dotenv
from oai_utils.litellm import litellm_concurrent_limit
from oai_utils.tinker import setup_tinkermodel

from adapter_agent.hierarchical.process.solve_verify import solve_verify
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.library_spec import LibrarySpec
from adapter_agent.model_helper import get_gemini_lite
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.session_result import (
    RewireSessionResultError,
    RewireSessionResultFailure,
    RewireSessionResultSuccess,
)
from adapter_agent.util.logger_util import setup_base_loglevel

set_tracing_disabled(True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
setup_base_loglevel()
# Stream the solver-env turn-by-turn DEBUG logs (assistant text, tool args,
# cargo output, verifier verdict). Other libraries stay on INFO.
logging.getLogger("adapter_agent.rl.env.source_solver").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


# === Config ==========================================================

MODEL_NAME = "Qwen/Qwen3-32B"

# Sampler-weights path of the checkpoint to drive. setup_tinkermodel(...)
# uses this to spin up the sampling client.
CHECKPOINT_PATH = (
    "tinker://1237cd7d-e163-5ffb-9ef9-82c98c281079:train:0/sampler_weights/init_sft"
)

LIBRARY_SPEC = LibrarySpec.hisab()

DEFAULT_TASK = (
    "Using the `hisab` library, generate a 10x10 complex matrix where each "
    "real and imaginary component is sampled from a uniform distribution "
    "over [-0.5, 0.5]. Then convert it into a Hermitian matrix by adding "
    "the matrix to its own conjugate transpose. Print the resulting matrix "
    "and verify it equals its own conjugate transpose. The program must "
    "compile and run via `cargo run`."
)

MAX_TURNS = 12
ENABLE_SEARCH_TOOLS = True

# Cloudrun runtime — hisab's per-library settings. Pool size of 1 is fine
# for a single solver session; bump only if you parallelize.
RUNTIME_POOL_MAX_SIZE = 1


# === Trajectory printing ============================================

ROLE_LABEL = {
    "system": "SYSTEM",
    "user": "USER / TOOL RESULT",
    "assistant": "ASSISTANT",
    "tool": "TOOL",
}


def _flatten(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t == "thinking":
                out.append(f"<think>{p.get('thinking', '')}</think>")
            else:
                out.append(p.get("text", ""))
        return "".join(out)
    return "" if content is None else str(content)


def _print_separator(char: str = "=", n: int = 80) -> None:
    print(char * n)


def _print_trial(i: int, msg) -> None:
    role = msg.get("role", "?") if isinstance(msg, dict) else getattr(msg, "role", "?")
    content_obj = (
        msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
    )
    label = ROLE_LABEL.get(str(role), str(role).upper())
    text = _flatten(content_obj)
    _print_separator("-")
    print(f"[turn {i}] {label}")
    _print_separator("-")
    if not text.strip():
        print("(empty)")
        return
    # Highlight the key tags so it's easy to spot what the solver is doing.
    tags_seen = []
    for tag in ("submit", "write_and_run", "grep", "read", "ls"):
        if re.search(rf"<{tag}>", text):
            tags_seen.append(tag)
    if tags_seen:
        print(f"(tags: {', '.join(tags_seen)})")
    print(text)


# === Main ============================================================


async def main() -> None:
    load_dotenv()

    task_instruction = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK

    try:
        library_summary = LIBRARY_SPEC.read_summary()
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    print()
    _print_separator()
    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print(f"Library    : {LIBRARY_SPEC.name}")
    print(f"Max turns  : {MAX_TURNS}")
    print(f"Search ON  : {ENABLE_SEARCH_TOOLS}")
    print("Task       :")
    print(task_instruction)
    _print_separator()
    print()

    service_client = tinker.ServiceClient()
    solver_model, _, _ = setup_tinkermodel(
        MODEL_NAME, path=CHECKPOINT_PATH, service_client=service_client
    )
    verifier_model = get_gemini_lite()

    runtime_pool: RuntimePool | None = None
    # The solver hits LiteLLM (verifier) and Tinker (sampler). Cap at a small
    # pool — a single session never exceeds a couple of concurrent calls.
    async with litellm_concurrent_limit(max_concurrent=8):
        try:
            runtime_pool = RuntimePool(
                settings=LIBRARY_SPEC.cloudrun_runtime(),
                max_size=RUNTIME_POOL_MAX_SIZE,
            )

            result = await solve_verify(
                solver_model=solver_model,
                verifier_model=verifier_model,
                task=Task(instruction=task_instruction),
                libdir=LIBRARY_SPEC.libdir,
                library_name=LIBRARY_SPEC.name,
                runtime_pool=runtime_pool,
                max_turns=MAX_TURNS,
                reference_knowledge=library_summary,
                enable_search_tools=ENABLE_SEARCH_TOOLS,
            )
        finally:
            if runtime_pool is not None:
                await runtime_pool.close_all()

    # --- Result ---
    print()
    _print_separator()
    if isinstance(result, RewireSessionResultSuccess):
        verdict = "✓ VERIFIED"
    elif isinstance(result, RewireSessionResultFailure):
        verdict = f"✗ FAILED ({result.conclusion})"
    elif isinstance(result, RewireSessionResultError):
        verdict = f"⚠ ERROR ({result.conclusion})"
    else:
        verdict = f"? UNKNOWN ({type(result).__name__})"
    print(f"Result: {verdict}")
    _print_separator()
    print()

    trials = getattr(result, "trials", None)
    if not trials:
        print("(no trials recorded)")
        return

    print(f"Trajectory ({len(trials)} turns):")
    print()
    for i, msg in enumerate(trials, start=1):
        _print_trial(i, msg)
    _print_separator()


if __name__ == "__main__":
    asyncio.run(main())
