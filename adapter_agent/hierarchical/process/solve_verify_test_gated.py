"""Test-gated single-turn evaluation.

Sibling of `solve_verify` for tasks that ship with their own assertion-based
test harness. The task is one Rust source file with a marked student
region; the student submits a replacement for that region; we re-compose
the file and run `cargo run`. Pass/fail is decided purely by the cargo
exit code (assert!() panics → exit ≠ 0). No LLM-judge, no multi-turn
search.

Why this exists:
  - LLM-as-judge verifiers are noisy ("facade-only" detection in particular
    misfires on tasks where hisab simply lacks the API the verifier expects
    to see). Mechanical asserts remove that noise.
  - Generation-time validation (does the task have a valid hisab solution
    at all?) becomes mechanical too.

This module is intentionally minimal — it's the runtime side of the
test-gated curriculum PoC. Generation-side helpers live in
`generate_test_gated.py` and `generate_test_gated_with_tools.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import tinker
from coder_mcp.runtime import CoderMCPRuntimeError, Runtime
from oai_utils.tinker import TinkerModel
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.util.parsing import extract_rust_code

from .student_region import StudentRegion, compose_with_student_code

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestGatedResult:
    """Outcome of one test-gated evaluation.

    `passed`: True iff cargo exit code == 0 (all asserts held).
    `student_code`: the rust block extracted from the model's response.
    `composed_code`: student_code + harness (what actually ran).
    `execution_output`: stdout/stderr from cargo.
    `parse_error`: True if no `<submit>` block was found.
    """

    passed: bool
    student_code: str
    composed_code: str
    execution_output: str
    parse_error: bool = False


# Composition is delegated to `student_region.compose_with_student_code`:
# the task ships as one full Rust file with marked student region, and the
# rollout-time / gen-time runner just substitutes the region body.


def _extract_submit(text: str) -> str | None:
    """Pull rust code out of a single-turn response.

    The trained policy was taught to emit ```rust ... ``` fenced blocks,
    so that's the primary signal. We also accept the legacy `<submit>`
    envelope and a raw-code fallback, in this priority:

      1. first ```rust fenced block anywhere (trained convention)
      2. `<submit>...</submit>` (strict)
      3. `<submit>` opened but never closed (eaten by a stop sequence)
      4. raw rust starting from the first `use ` / `fn ` / `pub fn ` line

    Tests gate the result either way, so a sloppy envelope shouldn't kill
    the rollout. Returning None means we genuinely couldn't find code."""
    import re
    fence = re.search(r"```rust\n(.*?)\n```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    m = re.search(r"<submit>(.*?)</submit>", text, re.DOTALL)
    if m:
        return extract_rust_code(m.group(1))
    open_only = re.search(r"<submit>(.*)$", text, re.DOTALL)
    if open_only:
        return extract_rust_code(open_only.group(1))
    # No envelope at all — model dumped raw Rust. Cut from the first plausible
    # rust top-level keyword.
    raw = re.search(
        r"^[ \t]*(?:use\s+\w|pub\s+fn\b|fn\b)",
        text,
        re.MULTILINE,
    )
    if raw:
        return text[raw.start():].strip()
    return None


async def _sample_one(
    solver_model: TinkerModel,
    instruction: str,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    """Render `(system, user=instruction)` → tokens → text. Single turn."""
    renderer = solver_model.renderer
    messages: list[TinkerMessage] = [
        TinkerMessage(role="user", content=instruction),
    ]
    prompt = renderer.build_generation_prompt(messages)
    sample = await solver_model.sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            stop=renderer.get_stop_sequences(),
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    )
    msg, _ = renderer.parse_response(sample.sequences[0].tokens)
    content = msg.get("content")
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return content or ""


async def _run_composed(runtime_pool: RuntimePool, code: str) -> tuple[str, bool]:
    """Write to src/main.rs, run cargo. Returns (output, success)."""
    async def _closure(runtime: Runtime) -> tuple[str, bool]:
        await runtime.set_content("src/main.rs", code)
        return await runtime.run_cargo()

    return await runtime_pool.execute_with_retry(_closure)


async def solve_verify_test_gated(
    *,
    solver_model: TinkerModel,
    instruction: str,
    region: StudentRegion,
    runtime_pool: RuntimePool,
    max_tokens: int = 6000,
    temperature: float = 1.0,
) -> TestGatedResult:
    """Single-turn test-gated evaluation.

    The instruction (typically including the program with the region body
    blanked out) tells the student what to implement. The student replies
    with rust code; we substitute it for `region.region_body` in the
    parent program and run `cargo run`. Pass/fail is the cargo exit code.
    """
    response_text = await _sample_one(
        solver_model,
        instruction,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    student_code = _extract_submit(response_text)
    if student_code is None:
        logger.debug("No code block found in response.")
        return TestGatedResult(
            passed=False,
            student_code="",
            composed_code="",
            execution_output=response_text[:2000],
            parse_error=True,
        )

    composed = compose_with_student_code(region, student_code)
    try:
        output, success = await _run_composed(runtime_pool, composed)
    except CoderMCPRuntimeError as e:
        logger.exception("Runtime error during test-gated execution")
        return TestGatedResult(
            passed=False,
            student_code=student_code,
            composed_code=composed,
            execution_output=f"[runtime error] {e}",
        )
    return TestGatedResult(
        passed=success,
        student_code=student_code,
        composed_code=composed,
        execution_output=output,
    )


async def run_composed_directly(
    runtime_pool: RuntimePool,
    *,
    region: StudentRegion,
    student_code: str,
) -> TestGatedResult:
    """Substitute `student_code` for the region body, run cargo, return
    the result. Generation-time validators use this to confirm the
    reference body inside the markers actually passes the asserts in the
    file's `fn main()`. Same compose/run path as the sampled flow so the
    gate is identical at gen-time and rollout-time."""
    composed = compose_with_student_code(region, student_code)
    try:
        output, success = await _run_composed(runtime_pool, composed)
    except CoderMCPRuntimeError as e:
        return TestGatedResult(
            passed=False,
            student_code=student_code,
            composed_code=composed,
            execution_output=f"[runtime error] {e}",
        )
    return TestGatedResult(
        passed=success,
        student_code=student_code,
        composed_code=composed,
        execution_output=output,
    )
