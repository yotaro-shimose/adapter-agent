"""Source-aware variation generator.

Given a verified `(problem, answer)` pair, drives a Gemini agent through
`<grep>` / `<read>` / `<ls>` exploration and asks it to submit N variant
task instructions that exercise the SAME `library_name` API as the original.

Used by `study2_augment.py` to expand the verified-knowledge SFT cache
without overfitting to original phrasings.

Public surface:
    propose_variants(...) -> Variants | None

Submit format:
    <submit>{"variants": ["...", "...", "..."]}</submit>
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from agents.extensions.models.litellm_model import LitellmModel
from pydantic import BaseModel, Field
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.rl.completer import LiteLLMMessageCompleter
from adapter_agent.rl.env.source_solver import do_grep, do_ls, do_read
from adapter_agent.util.exception import MaximumContextExceeded

logger = logging.getLogger(__name__)


class Variants(BaseModel):
    """Re-exported so callers don't depend on the JSON envelope shape."""

    variants: list[str] = Field(default_factory=list)


def _build_system_prompt(
    library_name: str,
    library_summary: str,
    n_variants: int,
    max_turns: int,
) -> str:
    one_tag_rule = (
        "Emit EXACTLY ONE tool tag per response. "
        "The system processes only the first tag it finds."
    )
    return f"""<Role>
You are augmenting a Rust SFT dataset for the `{library_name}` library. The
training objective is: given a problem description, the model must RECALL the
right `{library_name}` API and write code with it. So variants must describe
the PROBLEM, not the solution.

You will be given an ORIGINAL (problem, verified answer) pair. Read the
verified answer to learn which API the original problem is "really about",
then produce {n_variants} NEW task instructions that:

  1. Are solvable using the SAME `{library_name}` API (or a close sibling in
     the same module — confirm with `<grep>` if unsure).
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

<Tools>
{one_tag_rule}

- `<ls>relative/path</ls>` — list immediate children of a directory inside the library source.
- `<read>relative/path</read>` or `<read>relative/path:START-END</read>` — read a source file (1-indexed inclusive line range).
- `<grep>pattern</grep>` or `<grep>{{"pattern": "...", "path": "subdir/"}}</grep>` — Python-regex search across `*.rs`. Default path is the library root.
- `<submit>{{"variants": ["...", "...", "..."]}}</submit>` — final list of {n_variants} variant task instructions, as JSON. ENDS the augmentation step.
</Tools>

<Budget>
You have at most {max_turns} turns total (1 tool call per turn). Each tool
result is prefixed with `[turn k/{max_turns}]`. If you reach the final turn
without `<submit>`, the variants are LOST. Use early turns to confirm sibling
APIs / parameter shapes if needed, then submit while you still have budget.
</Budget>

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
- Submit EXACTLY {n_variants} variants — no more, no less.
</Guidelines>
"""


_TOOL_TAGS_RE = re.compile(r"<(grep|read|ls|submit)\b", re.DOTALL)
_GREP_RE = re.compile(r"<grep>(.*?)</grep>", re.DOTALL)
_READ_RE = re.compile(r"<read>(.*?)</read>", re.DOTALL)
_LS_RE = re.compile(r"<ls>(.*?)</ls>", re.DOTALL)
_SUBMIT_RE = re.compile(r"<submit>(.*?)</submit>", re.DOTALL)

_AUGMENTER_STOP_SEQUENCES = ["</submit>", "</grep>", "</read>", "</ls>"]
_AUGMENTER_TAG_NAMES = ["submit", "grep", "read", "ls"]


def _restore_closing_tag(text: str, tag_names: list[str]) -> str:
    """If exactly one tool tag was opened but its closer was eaten by the
    stop sequence, append the matching `</tag>`. No-op otherwise."""
    for name in tag_names:
        opener = f"<{name}>"
        closer = f"</{name}>"
        if opener in text and closer not in text:
            return text + closer
    return text


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
    libdir: Path,
    library_summary: str,
    n_variants: int,
    solver_model: LitellmModel,
    max_turns: int = 16,
) -> Variants | None:
    """Run the source-aware augmenter and return its proposed variants.

    Returns None if the agent never submits a parseable JSON within
    `max_turns`, or the submission shape is wrong. Caller decides whether
    to retry / skip.
    """
    system_prompt = _build_system_prompt(
        library_name, library_summary, n_variants, max_turns,
    )
    user_prompt = (
        f"<OriginalProblem>\n{original_instruction}\n</OriginalProblem>\n"
        f"<OriginalAnswer>\n```rust\n{original_answer}\n```\n</OriginalAnswer>"
    )
    history: list[TinkerMessage] = [
        TinkerMessage(role="system", content=system_prompt),
        TinkerMessage(role="user", content=user_prompt),
    ]
    completer = LiteLLMMessageCompleter.from_litellm_model(
        solver_model,
        stop_condition=_AUGMENTER_STOP_SEQUENCES,
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
            logger.warning("augmenter: context length exceeded")
            return None

        msg = action.message if hasattr(action, "message") else action  # type: ignore[attr-defined]
        raw_text = _flatten_content(msg)
        text = _restore_closing_tag(raw_text, _AUGMENTER_TAG_NAMES)
        if text != raw_text:
            msg = TinkerMessage(role=msg.get("role", "assistant"), content=text)
        history.append(msg)

        tags = _TOOL_TAGS_RE.findall(text)
        if len(tags) > 1:
            history.append(_user(turn, (
                "[SYSTEM ERROR] MULTIPLE_TOOL_TAGS_DETECTED — "
                f"saw {', '.join(set(tags))}. Emit only ONE tool tag per response."
            )))
            continue

        sub = _SUBMIT_RE.search(text)
        if sub:
            try:
                obj = json.loads(sub.group(1).strip())
            except json.JSONDecodeError as e:
                history.append(_user(turn, (
                    f"[SYSTEM ERROR] SUBMIT_NOT_JSON: {e}. "
                    f'Submit must be a JSON object with shape {{"variants": ["...", ...]}}.'
                )))
                continue
            variants = obj.get("variants") if isinstance(obj, dict) else None
            if not isinstance(variants, list) or not all(isinstance(s, str) for s in variants):
                history.append(_user(turn, (
                    '[SYSTEM ERROR] Submit JSON missing "variants": [str, ...]. '
                    "Re-submit with the correct shape."
                )))
                continue
            return Variants(variants=variants)

        gr = _GREP_RE.search(text)
        if gr:
            history.append(_user(turn, do_grep(libdir, gr.group(1))))
            continue

        rd = _READ_RE.search(text)
        if rd:
            history.append(_user(turn, do_read(libdir, rd.group(1))))
            continue

        ls_m = _LS_RE.search(text)
        if ls_m:
            history.append(_user(turn, do_ls(libdir, ls_m.group(1))))
            continue

        history.append(_user(turn, (
            "[SYSTEM ERROR] NO_TOOL_TAG — your response must contain one of "
            "<grep>, <read>, <ls>, or <submit>."
        )))

    logger.warning(f"augmenter: max_turns={max_turns} exhausted without a valid submit")
    return None
