"""Single-shot generator for test-gated tasks.

Given a verified `pipeline_v2_inv` row (an InvestigationTarget that has a
known-good hisab solution), ask Gemini to author a self-contained
test-gated task:

  - `instruction`: what the student is asked to implement, in domain
    language. Mentions the function signature explicitly so the harness
    has something to call.
  - `reference_solution`: a hisab implementation of that function. Must
    pass the harness on its own — this is the ground-truth answer.
  - `test_harness`: a `fn main()` plus assertions that exercise the
    student function on a non-trivial fixed input and compare against
    expected values.

Returned as `TestGatedDraft`. Caller is expected to compile-check the
reference against the harness via
`solve_verify_test_gated.run_composed_directly` to confirm the task has
at least one valid solution before adding it to a curriculum cache.

Note: we deliberately do NOT generate a "facade" entry. Black-box tests
cannot distinguish a hisab implementation from a numerically-equivalent
std reimplementation — that distinction is delegated to a separate
reference-anchored LLM judge used purely for observability, never for
reward.

Public surface:
    generate_test_gated_draft(...) -> TestGatedDraft | None
"""

from __future__ import annotations

import json
import logging
import re

from agents.extensions.models.litellm_model import LitellmModel
from pydantic import BaseModel
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.rl.completer import LiteLLMMessageCompleter
from adapter_agent.util.exception import MaximumContextExceeded

from .student_region import (
    STUDENT_REGION_END_MARKER,
    STUDENT_REGION_START_MARKER,
    StudentRegionParseError,
    parse_student_region,
)

logger = logging.getLogger(__name__)


class TestGatedDraft(BaseModel):
    """Untrusted output of the generator.

    The single `full_program` field is the complete Rust source the agent
    confirmed runs end-to-end. It must contain exactly one
    `STUDENT_REGION_START_MARKER` / `STUDENT_REGION_END_MARKER` pair: the
    body between markers is what the student replaces at rollout time;
    everything outside is the fixed harness.

    Caller still validates by re-running the program in a fresh runtime
    (`run_composed_directly`) to confirm the reference body inside the
    markers actually passes the asserts in `fn main()`."""

    instruction: str
    full_program: str


def _build_system_prompt(library_name: str, library_summary: str) -> str:
    return f"""<Role>
You are authoring an atom-level training task for the `{library_name}` Rust
library. The task is shipped as ONE complete Rust source file that
compiles and runs end-to-end. The file contains the reference solution
inside marked region; at training time, the student's code replaces the
region body and the file is re-run.

You will be given a verified (instruction, answer) pair from an existing
investigation. Use it as inspiration: the new task should exercise the
SAME `{library_name}` API, in a fully self-contained, function-shaped
form.
</Role>

<OutputContract>
Respond with a single JSON object — no prose, no markdown fences around
the object itself — containing exactly these two fields:

  {{
    "instruction": "<what the student sees>",
    "full_program": "<one complete Rust source file, marker-delimited>"
  }}

`full_program` is raw Rust source code (no markdown fences inside the
JSON string either). It is what gets written to `src/main.rs` and run
with `cargo run`.
</OutputContract>

<StudentRegionMarkers>
`full_program` MUST contain exactly one pair of markers, each on its own
line, marking the section the student will replace:

    {STUDENT_REGION_START_MARKER}
    fn solve(...) -> ... {{
        // your reference implementation goes inside the markers
    }}
    {STUDENT_REGION_END_MARKER}

Rules for the markers:
- They are plain Rust comments and must appear EXACTLY as shown above
  (no extra punctuation, no translation).
- Exactly ONE pair per file. Both must be at the start of their own
  line (leading whitespace is fine).
- All `{library_name}`-using code that is the actual *answer* to the
  task goes BETWEEN the markers (this is what the student replaces).
- The harness — `fn main()`, asserts, fixed inputs, helpers — goes
  OUTSIDE the markers (above them, below them, or both).
- `use ...;` lines that the harness needs go OUTSIDE the markers.
  `use ...;` lines that only the student function needs may go INSIDE
  the markers — but if you put them outside they must not be duplicated
  inside.
</StudentRegionMarkers>

<RulesForInstruction>
- Tell the student EXACTLY what function signature(s) they must
  implement (the ones inside the markers), including full module paths
  for `{library_name}` types (e.g.
  `fn foo(m: &hisab::num::DenseMatrix) -> hisab::num::DenseMatrix`).
- Describe what the function should compute in domain language.
- State that the student must use `{library_name}` for the core
  operation (not std-only loops).
- Keep the instruction self-contained: no references to external files,
  no network, no I/O beyond stdout.
- The instruction is shown to the student ALONGSIDE the program with
  the region blanked out, so don't repeat what the program already
  shows. Just describe what the missing function should do.
</RulesForInstruction>

<RulesForReferenceImplementation>
(this is what you put INSIDE the student region)

- Uses `{library_name}` for the core operation. No std-only fallback.
- Compiles when the file is run end-to-end via `cargo run`.
- Does NOT define `fn main()` (main lives outside the markers).
</RulesForReferenceImplementation>

<RulesForHarness>
(this is what you put OUTSIDE the student region)

- Provides `fn main()` and any helper functions it needs.
- Calls the student function(s) on a hard-coded fixed input.
- Asserts numerical correctness with a small tolerance (e.g. 1e-6) for
  floating point. Use exact equality for integer / count outputs.
- Prints `"[TEST PASSED]"` followed by a brief description on success.
- The fixed input must be NON-TRIVIAL. Avoid all-zero matrices, identity
  matrices, length-one vectors, all-equal entries, or any input where
  obviously-wrong implementations (return zeros, return input unchanged,
  return a constant) would coincidentally satisfy the asserts. Use a
  small but irregular hand-picked input whose expected output is unique
  to the correct algorithm.
- Do NOT define the student's function outside the markers — that would
  be a duplicate definition.
</RulesForHarness>

<LibrarySummary>
{library_summary}
</LibrarySummary>
"""


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _flatten_content(msg: TinkerMessage) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return "" if content is None else str(content)


def _strip_rust_fence(s: str) -> str:
    """LLMs often wrap rust strings in ```rust fences inside the JSON
    despite being told not to. Peel them off if present."""
    s = s.strip()
    m = re.match(r"```(?:rust)?\n(.*?)\n```$", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return s


async def generate_test_gated_draft(
    *,
    seed_instruction: str,
    seed_answer: str | None,
    library_name: str,
    library_summary: str,
    solver_model: LitellmModel,
) -> TestGatedDraft | None:
    """Single-shot draft generator. Returns None on parse failure.

    The seed is inspiration — the LLM is free to pick a function shape of
    its choice, as long as the resulting task exercises a real hisab API.
    Pass `seed_answer=None` (e.g. for raw gh_archive tasks that don't ship
    a reference solution) — the generator will author one itself.

    Caller MUST validate the draft (reference must pass the harness)
    before trusting it.
    """
    system_prompt = _build_system_prompt(library_name, library_summary)
    if seed_answer:
        user_prompt = (
            f"<SeedInstruction>\n{seed_instruction}\n</SeedInstruction>\n"
            f"<SeedAnswer>\n```rust\n{seed_answer}\n```\n</SeedAnswer>"
        )
    else:
        user_prompt = (
            f"<SeedInstruction>\n{seed_instruction}\n</SeedInstruction>\n"
            f"<SeedAnswer>\n(no reference solution provided — author one "
            f"yourself as part of the draft)\n</SeedAnswer>"
        )
    history: list[TinkerMessage] = [
        TinkerMessage(role="system", content=system_prompt),
        TinkerMessage(role="user", content=user_prompt),
    ]
    completer = LiteLLMMessageCompleter.from_litellm_model(solver_model)

    try:
        action = await completer(history)
    except MaximumContextExceeded:
        logger.warning("test-gated generator: context length exceeded")
        return None

    msg = action.message if hasattr(action, "message") else action  # type: ignore[attr-defined]
    text = _flatten_content(msg)

    m = _JSON_OBJECT_RE.search(text)
    if m is None:
        logger.warning("test-gated generator: no JSON object in response")
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"test-gated generator: JSON decode error: {e}")
        return None

    required = ("instruction", "full_program")
    if not isinstance(obj, dict) or not all(
        isinstance(obj.get(k), str) for k in required
    ):
        logger.warning(
            f"test-gated generator: missing fields. keys={list(obj) if isinstance(obj, dict) else type(obj)}"
        )
        return None

    full_program = _strip_rust_fence(obj["full_program"])
    try:
        parse_student_region(full_program)
    except StudentRegionParseError as e:
        logger.warning(f"test-gated generator: bad student-region markers: {e}")
        return None

    return TestGatedDraft(
        instruction=obj["instruction"].strip(),
        full_program=full_program,
    )
