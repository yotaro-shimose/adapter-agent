"""Source-aware investigation planner.

Drives a Gemini agent through `<grep>` / `<read>` / `<ls>` tool calls (the
same set `solve_verify` exposes) and asks it to finalize an investigation
plan as JSON. This replaces the structured-output planner whose output was
based purely on SUMMARY.md and tended to hallucinate API names.

Public surface:
    plan_with_tools(...) -> InvestigationPlan | None

The planner submits via:
    <submit>{"items": ["...", "..."]}</submit>
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


class InvestigationPlan(BaseModel):
    """Re-exported so callers don't have to depend on the JSON envelope."""

    items: list[str] = Field(default_factory=list)


def _build_system_prompt(library_name: str, library_summary: str, max_turns: int) -> str:
    one_tag_rule = (
        "Emit EXACTLY ONE tool tag per response. "
        "The system processes only the first tag it finds."
    )
    return f"""<Role>
You are a Rust engineer planning what `{library_name}` API surface to investigate
before solving a problem. Use the read-only source-tree tools to discover what
actually exists in `{library_name}` — do NOT guess based on the summary alone.
</Role>

<Tools>
{one_tag_rule}

- `<ls>relative/path</ls>` — list immediate children of a directory inside the library source.
- `<read>relative/path</read>` or `<read>relative/path:START-END</read>` — read a source file (optionally a 1-indexed inclusive line range).
- `<grep>pattern</grep>` or `<grep>{{"pattern": "...", "path": "subdir/"}}</grep>` — Python-regex search across `*.rs` files. Default path is the library root.
- `<submit>{{"items": ["...", "..."]}}</submit>` — final investigation plan, as a JSON object with an `items` array of strings. ENDS the planning step.
</Tools>

<Budget>
You have at most {max_turns} turns total (1 tool call per turn). Each tool result
will be prefixed with `[turn k/{max_turns}]` so you always know where you are.
If you reach the final turn without `<submit>`, the plan is LOST. Pace yourself:
spend the first ~70% exploring, then submit while you still have budget.
</Budget>

<LibrarySummary>
{library_summary}
</LibrarySummary>

<Guidelines>
- Each item in your final plan MUST name a real type / function / module that you confirmed exists in the source (e.g. via grep / read).
- DO NOT include items about API that you couldn't find — uncertainty becomes hallucination downstream.
- Each item should be a concrete lookup target, e.g. "signature of hisab::num::CsrMatrix::new", "behavior of CsrMatrix::spmv".
- Skip generic Rust knowledge (ownership, iterators, Vec, etc.) — only `{library_name}`-specific items.
- Aim for 3–8 items. Submit only after you have grep'd / read enough source to back each one up.
</Guidelines>
"""


_TOOL_TAGS_RE = re.compile(r"<(grep|read|ls|submit)\b", re.DOTALL)
_GREP_RE = re.compile(r"<grep>(.*?)</grep>", re.DOTALL)
_READ_RE = re.compile(r"<read>(.*?)</read>", re.DOTALL)
_LS_RE = re.compile(r"<ls>(.*?)</ls>", re.DOTALL)
_SUBMIT_RE = re.compile(r"<submit>(.*?)</submit>", re.DOTALL)

# LiteLLM consumes the stop sequence and DOES NOT include it in the response.
# We pass these stops so the agent can't keep writing past the first tool call
# (the autoregressive-hallucination failure mode), then restore the closing
# tag manually so downstream regex parsing still works.
_PLANNER_STOP_SEQUENCES = ["</submit>", "</grep>", "</read>", "</ls>"]
_PLANNER_TAG_NAMES = ["submit", "grep", "read", "ls"]


def _restore_closing_tag(text: str, tag_names: list[str]) -> str:
    """If exactly one tool tag was opened but its closer was eaten by the
    stop sequence, append the matching `</tag>`. No-op if no opens or if a
    matching closer is already present."""
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


async def plan_with_tools(
    *,
    task_instruction: str,
    library_name: str,
    libdir: Path,
    library_summary: str,
    solver_model: LitellmModel,
    max_turns: int = 16,
) -> InvestigationPlan | None:
    """Run a tool-using planner against `libdir` and return the submitted plan.

    Returns None if the agent never submits a parseable JSON plan within
    `max_turns`. Caller is responsible for downstream handling (retry,
    skip-with-warning, etc.).
    """
    system_prompt = _build_system_prompt(library_name, library_summary, max_turns)
    user_prompt = f"<Problem>\n{task_instruction}\n</Problem>"
    history: list[TinkerMessage] = [
        TinkerMessage(role="system", content=system_prompt),
        TinkerMessage(role="user", content=user_prompt),
    ]
    completer = LiteLLMMessageCompleter.from_litellm_model(
        solver_model,
        stop_condition=_PLANNER_STOP_SEQUENCES,
    )

    def _user(turn_idx: int, body: str) -> TinkerMessage:
        # Prefix with the turn the agent JUST used, so it knows the budget left.
        return TinkerMessage(
            role="user",
            content=f"[turn {turn_idx + 1}/{max_turns}]\n{body}",
        )

    for turn in range(max_turns):
        try:
            action = await completer(history)
        except MaximumContextExceeded:
            logger.warning("planner: context length exceeded")
            return None

        msg = action.message if hasattr(action, "message") else action  # type: ignore[attr-defined]
        # Restore the closing tag the stop sequence ate, so downstream regex
        # finds the full `<tag>...</tag>` pair.
        raw_text = _flatten_content(msg)
        text = _restore_closing_tag(raw_text, _PLANNER_TAG_NAMES)
        if text != raw_text:
            msg = TinkerMessage(role=msg.get("role", "assistant"), content=text)
        history.append(msg)

        # Multi-tag rejection (mirrors source_solver behavior).
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
                    f'Submit must be a JSON object with shape {{"items": ["...", ...]}}.'
                )))
                continue
            items = obj.get("items") if isinstance(obj, dict) else None
            if not isinstance(items, list) or not all(isinstance(s, str) for s in items):
                history.append(_user(turn, (
                    '[SYSTEM ERROR] Submit JSON missing "items": [str, ...]. '
                    "Re-submit with the correct shape."
                )))
                continue
            return InvestigationPlan(items=items)

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

        # No recognized tag.
        history.append(_user(turn, (
            "[SYSTEM ERROR] NO_TOOL_TAG — your response must contain one of "
            "<grep>, <read>, <ls>, or <submit>."
        )))

    logger.warning(f"planner: max_turns={max_turns} exhausted without a valid submit")
    return None
