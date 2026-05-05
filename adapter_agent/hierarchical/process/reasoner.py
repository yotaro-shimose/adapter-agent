"""QRA reasoner — fills in the chain-of-thought between a verified Q and A.

The augmentation pipeline produces (problem, answer-code) pairs anchored on a
specific `library_name` API. SFT training wants (question, reasoning, answer)
triples; this module produces the missing R.

The model sees BOTH the problem and the verified answer. It writes reasoning
that is consistent with the answer — post-hoc, but stylistically authentic
("a senior engineer thinking before typing"). This is a deliberate trade:
without seeing the answer the reasoner would frequently propose a different
API path than the one the answer actually uses, producing misaligned R/A.

Public surface:
    fill_reasoning(...) -> str | None
"""

from __future__ import annotations

import logging

from agents.extensions.models.litellm_model import LitellmModel
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.rl.completer import LiteLLMMessageCompleter
from adapter_agent.util.exception import MaximumContextExceeded

logger = logging.getLogger(__name__)


def _build_system_prompt(library_name: str, library_summary: str) -> str:
    return f"""<Role>
You are writing the chain-of-thought (reasoning) that connects a Rust coding
problem to its verified solution code. The reasoning must read like authentic
problem-solving — what a senior Rust engineer who knows `{library_name}` would
think BEFORE typing the solution.

You will be given the PROBLEM and the verified ANSWER. Produce the reasoning
that bridges them. The reasoning should:
  1. Restate / parse the problem in your own words.
  2. Identify which `{library_name}` API or types are relevant, and why.
  3. Sketch the program shape — data flow, key calls, ordering.
  4. Note any subtle requirement (mutability, types, edge values).
</Role>

<HardConstraints>
- DO NOT include any Rust code in the reasoning. The code comes after.
- DO NOT say "the answer is...", "the solution shows...", or otherwise refer
  to having seen the answer. Write as if you are about to derive the code.
- DO NOT use markdown headers or bullet lists. Write in flowing prose.
- Keep it 80–250 words. Concise but complete.
- Output the reasoning text directly. No surrounding tags, JSON, or preamble
  ("Here's my reasoning:"). The first character of your response is the
  first word of the reasoning.
</HardConstraints>

<LibrarySummary>
{library_summary}
</LibrarySummary>
"""


def _build_user_prompt(question: str, answer: str) -> str:
    return f"""<Problem>
{question}
</Problem>

<Solution>
```rust
{answer}
```
</Solution>"""


async def fill_reasoning(
    *,
    question: str,
    answer: str,
    library_name: str,
    library_summary: str,
    model: LitellmModel,
) -> str | None:
    """Generate the chain-of-thought that bridges (question, answer).

    Returns the reasoning string, or None if the LLM call fails (context
    exceeded, empty completion, etc.). Caller decides whether to retry.
    """
    system_prompt = _build_system_prompt(library_name, library_summary)
    user_prompt = _build_user_prompt(question, answer)
    history: list[TinkerMessage] = [
        TinkerMessage(role="system", content=system_prompt),
        TinkerMessage(role="user", content=user_prompt),
    ]
    completer = LiteLLMMessageCompleter.from_litellm_model(model)

    try:
        action = await completer(history)
    except MaximumContextExceeded:
        logger.warning("reasoner: context length exceeded")
        return None

    msg = action.message if hasattr(action, "message") else action  # type: ignore[attr-defined]
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(p.get("text", "") for p in content if p.get("type") == "text")
    else:
        text = ""
    text = text.strip()
    if not text:
        return None
    return text
