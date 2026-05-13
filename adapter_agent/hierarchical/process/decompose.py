"""Single-shot mid-level decomposer.

Given a long, multi-step gh_archive problem statement, asks Gemini to split
it into N intermediate sub-problems that each compose 2-3 distinct library
APIs — sitting between single-API drills (`pipeline_v2_qra`) and the full
gh_archive task. Used by `study2_decompose.py` to grow a "middle-level"
SFT cache for hisab.

Public surface:
    propose_decomposition(...) -> Decomposition | None

Response shape (JSON object only):
    {"sub_tasks": ["...", "...", ...]}
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


class Decomposition(BaseModel):
    sub_tasks: list[str] = Field(default_factory=list)


def _build_system_prompt(library_name: str, library_summary: str) -> str:
    return f"""<Role>
You are decomposing a complex Rust programming task that uses the `{library_name}`
library. Your output is training data: the model will later learn to RECALL the
right `{library_name}` APIs from a problem description and write code with them.

The original task is a multi-step pipeline that composes 4-7 `{library_name}`
APIs. We already have lots of training data at the OPPOSITE extreme — single-API
drills like "build a CSR matrix and print its nnz". What we lack is the MIDDLE.

Your job: split the original into INTERMEDIATE sub-tasks that each chain
2-3 `{library_name}` APIs together. Not single-API drills. Not the full
original. The middle.
</Role>

<HowToDecompose>
1. Read the original task and identify its sequential steps (often numbered).
2. Group adjacent steps so each group requires 2-3 distinct `{library_name}` APIs.
   - Number of sub-tasks ≈ ceil(number_of_steps / 2). Do NOT over-decompose.
   - For a 4-step original, output 2 sub-tasks. For 6 steps, 3 sub-tasks. Etc.
3. Each sub-task must STAND ALONE: it has its own data initialization (carry
   over the relevant numbers from the original) and its own observable output.
   A reader who never saw the original should be able to solve it.
4. Carry the original's DOMAIN VOCABULARY into the sub-tasks (e.g. "Hermitian
   matrix", "satellite trajectory", "cosine kNN"). Do not strip context to a
   generic linear-algebra exercise.
</HowToDecompose>

<HardConstraints>
- Each sub-task MUST require AT LEAST 2 distinct `{library_name}` APIs in
  composition. Single-API sub-tasks defeat the purpose — those are drills.
- Sub-tasks must NOT name `{library_name}` types, modules, methods, or function
  names. The model is being trained to recall these from the description alone.

  BAD  — leaks API: "Use `hisab::num::CsrMatrix::new` and `Cholesky::solve` to
          factor and solve Ax=b."
  GOOD — pure problem: "Build a 100x100 sparse symmetric positive-definite matrix
          A from a tridiagonal pattern, then solve Ax=b for a given dense b
          using a numerically stable factorization, and report the residual norm."

  You may name plain Rust stdlib types (Vec, String, i32, etc.).

- Each sub-task is solvable in a single `fn main()` with NO external I/O,
  network, or filesystem access.
- Length per sub-task: 2-5 sentences, roughly 300-700 characters. Shorter
  than the original, longer than a one-line drill.
- Specify enough numerics (sizes, seeds, key constants) that the verifier
  has a deterministic target — but do NOT copy the original's full data
  spec verbatim if a smaller equivalent will do.
</HardConstraints>

<OutputFormat>
Respond with a single JSON object — no prose, no markdown fences — with this shape:

  {{"sub_tasks": ["<sub-task 1>", "<sub-task 2>", ...]}}

Output BETWEEN 2 AND 5 sub-tasks. Choose the count based on the original's
step count (≈ ceil(steps/2)).
</OutputFormat>

<LibrarySummary>
{library_summary}
</LibrarySummary>

<SelfCheck>
Before emitting, verify each sub-task:
  (a) requires 2-3 distinct `{library_name}` APIs (not 1, not 5+);
  (b) names no `{library_name}` symbols;
  (c) preserves a slice of the original's domain language;
  (d) is self-contained — initializes its own data, prints its own output.
If any sub-task fails (a)-(d), revise it. If you can't make it pass while
staying under 5 sub-tasks total, drop it.
</SelfCheck>
"""


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _flatten_content(msg: TinkerMessage) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return "" if content is None else str(content)


async def propose_decomposition(
    *,
    original_instruction: str,
    library_name: str,
    library_summary: str,
    solver_model: LitellmModel,
) -> Decomposition | None:
    """Single-shot decomposer. Returns None on parse failure."""
    system_prompt = _build_system_prompt(library_name, library_summary)
    user_prompt = f"<OriginalProblem>\n{original_instruction}\n</OriginalProblem>"
    history: list[TinkerMessage] = [
        TinkerMessage(role="system", content=system_prompt),
        TinkerMessage(role="user", content=user_prompt),
    ]
    completer = LiteLLMMessageCompleter.from_litellm_model(solver_model)

    try:
        action = await completer(history)
    except MaximumContextExceeded:
        logger.warning("decomposer: context length exceeded")
        return None

    msg = action.message if hasattr(action, "message") else action  # type: ignore[attr-defined]
    text = _flatten_content(msg)

    m = _JSON_OBJECT_RE.search(text)
    if m is None:
        logger.warning("decomposer: no JSON object in response")
        return None

    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"decomposer: JSON decode error: {e}")
        return None

    sub_tasks = obj.get("sub_tasks") if isinstance(obj, dict) else None
    if not isinstance(sub_tasks, list) or not all(isinstance(s, str) for s in sub_tasks):
        logger.warning('decomposer: response missing "sub_tasks": [str, ...]')
        return None

    return Decomposition(sub_tasks=sub_tasks)
