"""Multi-turn, source-aware test-gated task generator.

Drives a Gemini agent through `<grep>` / `<read>` / `<ls>` /
`<write_and_run>` to author a self-contained test-gated task. The agent
explores the actual library source instead of guessing API names from a
SUMMARY, and can prototype its draft in `cargo run` before finalizing.
This eliminates the two failure modes of the single-shot generator:

  1. fake / hallucinated APIs (e.g. `DenseMatrix.svd()` when none exists)
  2. library shadowing (`mod hisab { ... }` baked into the harness so
     "tests pass" without the real crate being involved)

The agent submits via:
    <submit>{
      "instruction": "...",
      "reference_solution": "...",
      "test_harness": "..."
    }</submit>

Public surface:
    generate_test_gated_draft_with_tools(...) -> TestGatedDraft | None
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from agents.extensions.models.litellm_model import LitellmModel
from coder_mcp.runtime import CoderMCPRuntimeError, Runtime
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.rl.completer import LiteLLMMessageCompleter
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.source_solver import do_grep, do_ls, do_read
from adapter_agent.util.exception import MaximumContextExceeded

from .generate_test_gated import TestGatedDraft
from .student_region import (
    STUDENT_REGION_END_MARKER,
    STUDENT_REGION_START_MARKER,
    StudentRegionParseError,
    parse_student_region,
)

logger = logging.getLogger(__name__)


def _short(text: str, limit: int = 400) -> str:
    """Single-line preview for log readability when an assistant or tool
    payload would otherwise dump hundreds of lines into the stream."""
    one_line = text.replace("\n", " ⏎ ")
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit] + f" ... ({len(one_line) - limit} more chars)"


def _build_system_prompt(library_name: str, library_summary: str, max_turns: int) -> str:
    one_tag_rule = (
        "Emit EXACTLY ONE tool tag per response. "
        "The system processes only the first tag it finds."
    )
    return f"""<Role>
You are authoring an atom-level training task for the `{library_name}` Rust
library. The task asks the student to implement ONE function. A test
harness will run the student's code and assert numerical correctness.
You will be SEEDED with an L3 problem statement (a real-world task that
the student is supposed to solve). Your job is to extract a single,
SOLVABLE, SELF-CONTAINED function-shaped task from that seed and ship it
along with a working reference implementation and a discriminative
harness.

You have access to the actual `{library_name}` source tree — USE IT. The
reason this multi-turn surface exists is to stop you from hallucinating
APIs or shadowing the crate. Every API you put in the reference and
harness must exist in the real source.
</Role>

<Tools>
{one_tag_rule}

- `<ls>relative/path</ls>` — list directory entries inside the library source.
- `<read>relative/path</read>` or `<read>relative/path:START-END</read>` — read a source file (optionally a 1-indexed inclusive line range).
- `<grep>pattern</grep>` or `<grep>{{"pattern": "...", "path": "subdir/"}}</grep>` — Python-regex search across `*.rs` files. Default path is the library root.
- `<write_and_run>FULL_RUST_CODE</write_and_run>` — write the contents to `src/main.rs` of a `{library_name}`-dependent workspace and run `cargo run`. Use this to test that the file compiles end-to-end and the asserts in `fn main()` pass. The agent receives stdout + stderr.
- `<submit>{{ "instruction": "...", "full_program": "..." }}</submit>` — final draft. JSON object. Ends the session. `full_program` is the full Rust file you just confirmed runs successfully — copied verbatim with student-region markers added. RAW Rust source, no markdown fences inside the JSON string.
</Tools>

<Budget>
You have at most {max_turns} turns total (1 tool call per turn). Each
tool result is prefixed with `[turn k/{max_turns}]`. If you reach the
final turn without `<submit>`, your work is lost. Pace yourself: a
typical session does a few `<ls>` / `<grep>` / `<read>` calls, one or
two `<write_and_run>` to verify, then `<submit>`.
</Budget>

<LibrarySummary>
{library_summary}
</LibrarySummary>

<StudentRegionMarkers>
The `full_program` you submit MUST contain exactly one pair of markers,
each on its own line, marking the section the student will replace at
training time:

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

When you `<write_and_run>` a candidate, write the file as-it-will-be-
shipped (markers included). The markers are comments so they do not
affect compilation. After it passes, copy that exact file verbatim into
the `full_program` field of `<submit>`.
</StudentRegionMarkers>

<RulesForInstruction>
- Tell the student EXACTLY what function signature to implement, with
  full `{library_name}` module paths for any library types
  (e.g. `fn foo(m: &hisab::num::DenseMatrix) -> hisab::num::DenseMatrix`).
- The function's job MUST be smaller than the seed's. The seed is a
  full L3 problem; you should pick ONE crisp sub-piece that reduces to
  a single function the harness can drive.
- Tell the student to use `{library_name}` for the core operation, NOT
  std-only loops.
- Self-contained: no external files, no network, no I/O beyond stdout.
- The student is shown the full file with the region body blanked out
  alongside this instruction, so you don't need to re-show signatures
  the file already declares — just describe what the missing region
  should do.
</RulesForInstruction>

<RulesForHarness>
(this is what goes OUTSIDE the markers)

- Provides `fn main()` and any helper functions it needs.
- Calls the student's function on a hard-coded NON-TRIVIAL fixed input.
  Avoid all-zero / identity / single-element / all-equal inputs that
  any wrong implementation would coincidentally satisfy.
- Asserts numerical correctness with a small tolerance for floating
  point. Use exact equality for integer / count outputs.
- Prints `"[TEST PASSED]"` followed by a brief description on success.
- DO NOT redefine the student function outside the markers — that would
  be a duplicate definition.
- DO NOT define `mod {library_name}` or `pub mod ...` blocks that
  shadow the real `{library_name}` crate. The harness must run against
  the genuine `{library_name}` dependency, NOT a fake module.
</RulesForHarness>

<HardConstraints>
- Every `{library_name}` API name in `full_program` must appear in the
  actual source. Verify with `<grep>` before submitting if you're unsure.
- Use `<write_and_run>` AT LEAST ONCE before `<submit>` with the file
  EXACTLY as you intend to ship it (markers included), and confirm the
  asserts pass.
- If `<write_and_run>` reports a compile or assert failure, fix it and
  retry — never `<submit>` a draft you haven't seen pass.
- The `full_program` field in `<submit>` must be the verbatim contents
  of the LAST `<write_and_run>` whose cargo result indicated success.
  Do NOT rewrite, refactor, or "clean up" the code at submit time.
</HardConstraints>
"""


_TOOL_TAGS_RE = re.compile(r"<(grep|read|ls|write_and_run|submit)\b", re.DOTALL)
_GREP_RE = re.compile(r"<grep>(.*?)</grep>", re.DOTALL)
_READ_RE = re.compile(r"<read>(.*?)</read>", re.DOTALL)
_LS_RE = re.compile(r"<ls>(.*?)</ls>", re.DOTALL)
_WRITE_AND_RUN_RE = re.compile(r"<write_and_run>(.*?)</write_and_run>", re.DOTALL)
_SUBMIT_RE = re.compile(r"<submit>(.*?)</submit>", re.DOTALL)

# Fewer than 5 stops so Gemini's stopSequences honors them all (we have
# the same 5-entry-silent-drop trap the planner has). We drop </submit>
# because it's the final tag and trailing junk doesn't matter.
_GEN_STOP_SEQUENCES = ["</grep>", "</read>", "</ls>", "</write_and_run>"]
_GEN_TAG_NAMES = ["submit", "grep", "read", "ls", "write_and_run"]


def _flatten_content(msg: TinkerMessage) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return "" if content is None else str(content)


def _restore_closing_tag(text: str, tag_names: list[str]) -> str:
    """If exactly one tool tag was opened but its closer was eaten by a
    stop sequence, append the matching `</tag>`."""
    for name in tag_names:
        opener = f"<{name}>"
        closer = f"</{name}>"
        if opener in text and closer not in text:
            return text + closer
    return text


def _strip_rust_fence(s: str) -> str:
    """Peel ```rust ... ``` if the model wrapped JSON-string Rust in fences
    despite the rules. Same heuristic as the single-shot generator."""
    s = s.strip()
    m = re.match(r"```(?:rust)?\n(.*?)\n```$", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return s


def _parse_submit(body: str) -> tuple[TestGatedDraft | None, str]:
    """Return (draft, error_msg). If parsing fails, draft is None and
    error_msg explains what to fix so the agent can retry."""
    try:
        obj = json.loads(body.strip())
    except json.JSONDecodeError as e:
        return None, f"submit body is not valid JSON: {e}"
    if not isinstance(obj, dict):
        return None, "submit JSON must be an object."
    required = ("instruction", "full_program")
    missing = [k for k in required if not isinstance(obj.get(k), str)]
    if missing:
        return None, (
            f"submit JSON missing or non-string field(s): {missing}. "
            f"Expected shape: {{\"instruction\": str, \"full_program\": str}}."
        )

    full_program = _strip_rust_fence(obj["full_program"])
    try:
        parse_student_region(full_program)
    except StudentRegionParseError as e:
        return None, (
            f"full_program does not have a well-formed student region: {e}. "
            f"It must contain exactly one '{STUDENT_REGION_START_MARKER}' "
            f"line and exactly one '{STUDENT_REGION_END_MARKER}' line, in "
            f"that order, each at the start of its own line."
        )
    return (
        TestGatedDraft(
            instruction=obj["instruction"].strip(),
            full_program=full_program,
        ),
        "",
    )


async def _run_in_runtime(runtime_pool: RuntimePool, code: str) -> str:
    async def _closure(runtime: Runtime) -> str:
        await runtime.set_content("src/main.rs", code)
        out, _ok = await runtime.run_cargo()
        return out

    try:
        return await runtime_pool.execute_with_retry(_closure)
    except CoderMCPRuntimeError as e:
        return f"[runtime error] {e}"


async def generate_test_gated_draft_with_tools(
    *,
    seed_instruction: str,
    seed_answer: str | None,
    library_name: str,
    libdir: Path,
    library_summary: str,
    runtime_pool: RuntimePool,
    solver_model: LitellmModel,
    max_turns: int = 16,
    seed_id: str = "anon",
) -> TestGatedDraft | None:
    """Multi-turn draft generator. Returns None if the agent never lands a
    valid `<submit>` within `max_turns` turns.

    Workflow per turn (one tag only):
      ls / read / grep   — explore the library source
      write_and_run      — try a candidate (reference + harness concat) in cargo
      submit             — finalize the draft as JSON
    """
    system_prompt = _build_system_prompt(library_name, library_summary, max_turns)
    if seed_answer:
        seed_block = (
            f"<SeedInstruction>\n{seed_instruction}\n</SeedInstruction>\n"
            f"<SeedAnswer>\n```rust\n{seed_answer}\n```\n</SeedAnswer>"
        )
    else:
        seed_block = (
            f"<SeedInstruction>\n{seed_instruction}\n</SeedInstruction>\n"
            f"<SeedAnswer>\n(none — author your own reference)\n</SeedAnswer>"
        )
    history: list[TinkerMessage] = [
        TinkerMessage(role="system", content=system_prompt),
        TinkerMessage(role="user", content=seed_block),
    ]
    completer = LiteLLMMessageCompleter.from_litellm_model(
        solver_model,
        stop_condition=_GEN_STOP_SEQUENCES,
    )

    def _user(turn_idx: int, body: str) -> TinkerMessage:
        return TinkerMessage(
            role="user",
            content=f"[turn {turn_idx + 1}/{max_turns}]\n{body}",
        )

    for turn in range(max_turns):
        try:
            action = await completer(history)
        except MaximumContextExceeded:
            logger.warning(f"[{seed_id}] context length exceeded at turn {turn + 1}")
            return None

        msg = action.message if hasattr(action, "message") else action  # type: ignore[attr-defined]
        raw_text = _flatten_content(msg)
        text = _restore_closing_tag(raw_text, _GEN_TAG_NAMES)
        if text != raw_text:
            msg = TinkerMessage(role=msg.get("role", "assistant"), content=text)
        history.append(msg)

        logger.debug(
            "[%s] turn %d/%d assistant: %s",
            seed_id, turn + 1, max_turns, _short(text),
        )

        # Multi-tag rejection.
        tags = _TOOL_TAGS_RE.findall(text)
        if len(tags) > 1:
            logger.debug(
                "[%s] turn %d/%d rejected: multiple tags %s",
                seed_id, turn + 1, max_turns, sorted(set(tags)),
            )
            history.append(_user(turn, (
                "[SYSTEM ERROR] MULTIPLE_TOOL_TAGS_DETECTED — "
                f"saw {', '.join(set(tags))}. Emit only ONE tool tag per response."
            )))
            continue

        sub = _SUBMIT_RE.search(text)
        if sub:
            draft, err = _parse_submit(sub.group(1))
            if draft is None:
                logger.debug(
                    "[%s] turn %d/%d submit parse error: %s",
                    seed_id, turn + 1, max_turns, err,
                )
                history.append(_user(turn, f"[SYSTEM ERROR] {err}"))
                continue
            logger.debug(
                "[%s] turn %d/%d submit accepted (instruction=%d chars, "
                "full_program=%d chars)",
                seed_id, turn + 1, max_turns,
                len(draft.instruction),
                len(draft.full_program),
            )
            return draft

        wr = _WRITE_AND_RUN_RE.search(text)
        if wr:
            code = wr.group(1)
            logger.debug(
                "[%s] turn %d/%d write_and_run (%d chars): %s",
                seed_id, turn + 1, max_turns, len(code), _short(code, 200),
            )
            output = await _run_in_runtime(runtime_pool, code)
            logger.debug(
                "[%s] turn %d/%d cargo result: %s",
                seed_id, turn + 1, max_turns, _short(output),
            )
            history.append(_user(turn, f"<CargoRunResult>\n{output}\n</CargoRunResult>"))
            continue

        gr = _GREP_RE.search(text)
        if gr:
            arg = gr.group(1).strip()
            output = do_grep(libdir, arg)
            logger.debug(
                "[%s] turn %d/%d grep %s -> %s",
                seed_id, turn + 1, max_turns, _short(arg, 100), _short(output),
            )
            history.append(_user(turn, output))
            continue

        rd = _READ_RE.search(text)
        if rd:
            arg = rd.group(1).strip()
            output = do_read(libdir, arg)
            logger.debug(
                "[%s] turn %d/%d read %r -> %s",
                seed_id, turn + 1, max_turns, arg, _short(output),
            )
            history.append(_user(turn, output))
            continue

        ls_m = _LS_RE.search(text)
        if ls_m:
            arg = ls_m.group(1).strip()
            output = do_ls(libdir, arg)
            logger.debug(
                "[%s] turn %d/%d ls %r -> %s",
                seed_id, turn + 1, max_turns, arg, _short(output),
            )
            history.append(_user(turn, output))
            continue

        logger.debug(
            "[%s] turn %d/%d no recognized tag", seed_id, turn + 1, max_turns,
        )
        history.append(_user(turn, (
            "[SYSTEM ERROR] NO_TOOL_TAG — your response must contain one of "
            "<grep>, <read>, <ls>, <write_and_run>, or <submit>."
        )))

    logger.warning(
        f"[{seed_id}] max_turns={max_turns} exhausted without a valid submit"
    )
    return None
