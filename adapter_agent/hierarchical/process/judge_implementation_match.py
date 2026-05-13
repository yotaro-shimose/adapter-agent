"""Reference-anchored implementation-match judge.

Given a known-good `reference` solution and a `student` solution that BOTH
pass the same numerical asserts, decide whether the student is using the
same target-library API the reference uses, or whether they bypassed the
library and reimplemented the math with std loops.

This is *observability-only* — it never feeds reward. Use it to monitor
the curriculum / RL loop for facade drift, not to gate rollouts.

Why anchor on reference: a generic "are they using <library>?" judge is
noisy and biased toward whatever the model's prior expects. By showing
the judge the reference's idiomatic library usage and asking "does the
student do the equivalent?", the judgment becomes a structural diff
("they call the same kind of API" vs "they replaced it with hand-rolled
loops") which LLMs are reliably good at.

Public surface:
    judge_implementation_match(...) -> JudgeResult | None
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

logger = logging.getLogger(__name__)


class JudgeResult(BaseModel):
    """`uses_library` is the headline signal:
      - True  → student exercises the same library API the reference uses
      - False → student bypassed it (std loops, hand-rolled math, facade)
    `reasoning` carries the judge's justification for audit / debugging."""

    uses_library: bool
    reasoning: str


def _build_system_prompt(library_name: str) -> str:
    return f"""<Role>
You are auditing whether a STUDENT implementation actually uses the
`{library_name}` Rust library, by comparing it side-by-side with a
REFERENCE implementation that is known to use it correctly.

Both the reference and the student already pass the same numerical
assertions on a fixed input. The question is structural, not numerical:
does the student's code exercise the same kind of `{library_name}` API
the reference exercises, or did they bypass `{library_name}` and
hand-roll the math with std loops, Vec arithmetic, etc.?
</Role>

<HowToDecide>
Look at WHERE the actual computation happens in each:

- REFERENCE: identify the line(s) where `{library_name}` does the heavy
  lifting (e.g. `m.transpose()`, `a.mul_mat(&b)`, `svd(&m)`, etc.).
- STUDENT: do the same line(s) of computation go through `{library_name}`
  APIs, or are they replaced by hand-written for-loops over Vec / arrays
  that just happen to produce the same numbers?

Examples:

  ✓ uses_library = TRUE
    Reference calls `m.transpose().mul_mat(&m)`.
    Student writes `let t = m.transpose(); t.mul_mat(&m)?` — same APIs.
    Student may rename, restructure, error-handle differently — that's fine.

  ✗ uses_library = FALSE
    Reference calls `m.transpose().mul_mat(&m)`.
    Student writes a triple-nested for loop that computes Mᵀ·M element by
    element using `Vec<Vec<f64>>`, then optionally wraps the result with
    `DenseMatrix::from_rows(...)` at the end. The library is touched only
    cosmetically; the math is hand-rolled. This is a "facade" pattern.

  ✗ uses_library = FALSE
    Student avoids `{library_name}` types entirely and returns the result
    via std types. (Should not actually pass the harness, but if it
    somehow does, still false.)

A line like `use {library_name}::...;` or wrapping a final result in a
`{library_name}` type does not count as "using the library" if the
algorithm itself is std-only.
</HowToDecide>

<OutputContract>
Respond with a single JSON object — no prose around it, no markdown
fences — with this exact shape:

  {{"uses_library": true|false, "reasoning": "<one or two sentences>"}}
</OutputContract>
"""


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _flatten_content(msg: TinkerMessage) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return "" if content is None else str(content)


async def judge_implementation_match(
    *,
    instruction: str,
    reference_solution: str,
    student_solution: str,
    library_name: str,
    model: LitellmModel,
) -> JudgeResult | None:
    """Reference-anchored LLM judge. Returns None on parse failure."""
    system_prompt = _build_system_prompt(library_name)
    user_prompt = (
        f"<Instruction>\n{instruction}\n</Instruction>\n"
        f"<Reference>\n```rust\n{reference_solution}\n```\n</Reference>\n"
        f"<Student>\n```rust\n{student_solution}\n```\n</Student>"
    )
    history: list[TinkerMessage] = [
        TinkerMessage(role="system", content=system_prompt),
        TinkerMessage(role="user", content=user_prompt),
    ]
    completer = LiteLLMMessageCompleter.from_litellm_model(model)

    try:
        action = await completer(history)
    except MaximumContextExceeded:
        logger.warning("impl-match judge: context length exceeded")
        return None

    msg = action.message if hasattr(action, "message") else action  # type: ignore[attr-defined]
    text = _flatten_content(msg)

    m = _JSON_OBJECT_RE.search(text)
    if m is None:
        logger.warning("impl-match judge: no JSON object in response")
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"impl-match judge: JSON decode error: {e}")
        return None
    if (
        not isinstance(obj, dict)
        or not isinstance(obj.get("uses_library"), bool)
        or not isinstance(obj.get("reasoning"), str)
    ):
        logger.warning(f"impl-match judge: bad schema: {obj!r}")
        return None
    return JudgeResult(
        uses_library=obj["uses_library"],
        reasoning=obj["reasoning"].strip(),
    )
