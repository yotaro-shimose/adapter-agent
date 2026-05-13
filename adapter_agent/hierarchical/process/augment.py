"""Single-shot variation generator.

Given a verified `(problem, answer)` pair, asks Gemini to propose N variant
task instructions that exercise the SAME `library_name` API as the original.
The (problem, answer) already pins down which API matters, so no source
exploration is needed — one LLM call, JSON in, JSON out.

Used by `study2_pipeline.py` / `study2_augment.py` to expand the
verified-knowledge SFT cache without overfitting to original phrasings.

Public surface:
    propose_variants(...) -> Variants | None

Response shape (JSON object only):
    {"variants": ["...", "...", "..."]}
"""

from __future__ import annotations

import json
import logging
import re

from agents.extensions.models.litellm_model import LitellmModel
from pydantic import BaseModel, Field
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.rl.completer import LiteLLMMessageCompleter
from adapter_agent.util.exception import MaximumContextExceeded

logger = logging.getLogger(__name__)


class Variants(BaseModel):
    variants: list[str] = Field(default_factory=list)


def _build_system_prompt(
    library_name: str,
    library_summary: str,
    n_variants: int,
) -> str:
    return f"""<Role>
You are augmenting a Rust SFT dataset for the `{library_name}` library. The
training objective is: given a problem description, the model must RECALL the
right `{library_name}` API and write code with it. So variants must describe
the PROBLEM, not the solution.

You will be given an ORIGINAL (problem, verified answer) pair. Read the
verified answer to learn which API the original problem is "really about",
then produce {n_variants} NEW task instructions that:

  1. Are solvable using the SAME `{library_name}` API.
  2. DESCRIBE A GOAL IN PROBLEM-DOMAIN LANGUAGE — what the program should
     compute / simulate / output — WITHOUT naming the library API.
  3. DIFFER from the original in scenario, input shape, parameter choice, or
     expected output — not just a paraphrase.
  4. Are solvable in a single `fn main()` with no external I/O / network /
     filesystem dependency.
  5. Are concrete and short (1–3 sentences each).
</Role>

<HardConstraint>
Variants must NOT name `{library_name}` types, modules, methods, or function
names. The model is being trained to recall these from the description alone.

  BAD  — leaks API: "Use `hisab::num::Pcg32::next_f32()` to generate 10
          random floats in [0,1) and print them."
  GOOD — pure problem: "Print 10 pseudo-random floats uniformly distributed
          in [0, 1) using a deterministic seed so the output is reproducible."

  BAD  — leaks API: "Build a `CsrMatrix` of shape 5×5 and call `nnz()`."
  GOOD — pure problem: "Build a sparse 5×5 matrix where entry (i,j) is set
          when i+j is even, then print the count of stored entries."

You may name plain Rust stdlib types (Vec, String, i32, etc.) since those
don't reveal the lesson.
</HardConstraint>

<OutputFormat>
Respond with a single JSON object — no prose, no markdown fences — with this shape:

  {{"variants": ["<variant 1>", "<variant 2>", ...]}}

Submit EXACTLY {n_variants} variants — no more, no less.
</OutputFormat>

<LibrarySummary>
{library_summary}
</LibrarySummary>

<Guidelines>
- The original answer code tells you which API the variant must implicitly
  require. Anchor every variant on that — but describe it from the user's
  problem perspective, not the implementation perspective.
- A good test: if you removed the `{library_name}` library entirely, would
  your variant still be a sensible task description? It should be.
- Don't repeat the original problem nearly verbatim. Genuinely different
  scenarios: different shapes / values / pipeline steps / output style.
</Guidelines>
"""


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _flatten_content(msg: TinkerMessage) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return "" if content is None else str(content)


async def propose_variants(
    *,
    original_instruction: str,
    original_answer: str,
    library_name: str,
    library_summary: str,
    n_variants: int,
    solver_model: LitellmModel,
) -> Variants | None:
    """Single-shot variant proposer. Returns None on parse failure."""
    system_prompt = _build_system_prompt(library_name, library_summary, n_variants)
    user_prompt = (
        f"<OriginalProblem>\n{original_instruction}\n</OriginalProblem>\n"
        f"<OriginalAnswer>\n```rust\n{original_answer}\n```\n</OriginalAnswer>"
    )
    history: list[TinkerMessage] = [
        TinkerMessage(role="system", content=system_prompt),
        TinkerMessage(role="user", content=user_prompt),
    ]
    completer = LiteLLMMessageCompleter.from_litellm_model(solver_model)

    try:
        action = await completer(history)
    except MaximumContextExceeded:
        logger.warning("augmenter: context length exceeded")
        return None

    msg = action.message if hasattr(action, "message") else action  # type: ignore[attr-defined]
    text = _flatten_content(msg)

    m = _JSON_OBJECT_RE.search(text)
    if m is None:
        logger.warning("augmenter: no JSON object in response")
        return None

    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"augmenter: JSON decode error: {e}")
        return None

    variants = obj.get("variants") if isinstance(obj, dict) else None
    if not isinstance(variants, list) or not all(isinstance(s, str) for s in variants):
        logger.warning('augmenter: response missing "variants": [str, ...]')
        return None

    return Variants(variants=variants)
