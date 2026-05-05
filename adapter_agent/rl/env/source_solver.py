"""Source-grep-based solver environment.

A leaner variant of `SimplifiedSolverEnv`: instead of a rustdoc-JSON keyword
search + wiki, the solver is given direct read-only access to the library
source via three tools — `<grep>`, `<read>`, `<ls>`. Plus the existing
`<write_and_run>` and `<submit>` for code testing.

All filesystem access is scoped to `library_spec.libdir`. The env never
shells out — `grep` is a pure-Python regex over `*.rs` files; `read` and
`ls` are direct filesystem reads. No `bash` rights leak to the agent.

This module is parallel to `simplified_solver.py` and does NOT reuse its
env class; that lets us iterate on prompt/tool surface here without risking
the existing `ss_solve_verify` callers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.renderers.base import ToolSpec
from tinker_cookbook.rl.message_env import MessageEnv

from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.env.conclusion import SSConclusion
from adapter_agent.rl.env.injection import _inject_tools_into_prompt
from adapter_agent.rl.env.reward import LLMAsAJudgeSingleTurn
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.search_tool import SimplifiedSolverMutableState
from adapter_agent.rl.env.simplified_solver import SSStepResult
from adapter_agent.rl.solved_subtask import SolvedSubtask

logger = logging.getLogger(__name__)


CARGO_INIT_MAIN_RS = """\
fn main() {
    println!("Hello, world!");
}
"""

# --- Tool output caps. Keep responses bounded so the solver's context
#     doesn't blow up from a single ill-targeted grep. ---
GREP_MAX_RESULTS = 200
READ_MAX_LINES_DEFAULT = 500
LS_MAX_ENTRIES = 200
EXCLUDE_DIRS = {"target", ".git", "node_modules", ".cache"}


@dataclass
class SourceSolverEnvState:
    """Inputs for one solver session.

    `libdir` is the root the agent is allowed to grep/read/ls under. All
    paths in tool calls are resolved relative to this and rejected if they
    escape it.
    """

    task: Task
    library_name: str
    libdir: Path
    qwen_no_think: bool = False
    solved_subtasks: list[SolvedSubtask] = field(default_factory=list)
    reference_knowledge: str | None = None

    @classmethod
    def for_library(
        cls,
        task: Task,
        library_name: str,
        libdir: Path,
        qwen_no_think: bool = False,
        solved_subtasks: list[SolvedSubtask] | None = None,
        reference_knowledge: str | None = None,
    ) -> Self:
        return cls(
            task=task,
            library_name=library_name,
            libdir=libdir,
            qwen_no_think=qwen_no_think,
            solved_subtasks=solved_subtasks or [],
            reference_knowledge=reference_knowledge,
        )


# Reuse the same step-result shape as SimplifiedSolverEnv so callers (the
# loop in solve_verify.py) handle conclusions identically.
__all__ = [
    "SourceSolverEnvState",
    "SourceSolverEnv",
    "build_source_solver_msg_env",
    "get_source_solver_initial_messages",
    "resolve_under_libdir",
    "do_ls",
    "do_read",
    "do_grep",
]


# --- Pure tool implementations (libdir-scoped, side-effect-free). ---
# These are exposed as module-level functions so non-env callers (e.g. the
# planner-with-tools loop) can use the same grep/read/ls behavior without
# building a SimplifiedSolverEnv-shaped object.


def resolve_under_libdir(libdir: Path, rel: str) -> Path | None:
    """Resolve `rel` strictly under `libdir`. Returns None for paths that
    escape via `..` or are absolute. The returned path is `.resolve()`-ed."""
    rel = rel.strip().strip("/")
    if not rel:
        rel = "."
    try:
        base = libdir.resolve()
        target = (libdir / rel).resolve()
        target.relative_to(base)
        return target
    except (ValueError, OSError):
        return None


def do_ls(libdir: Path, rel: str) -> str:
    p = resolve_under_libdir(libdir, rel)
    if p is None:
        return f"[error] path '{rel}' is invalid or escapes the library root."
    if not p.exists():
        return f"[error] path '{rel}' does not exist."
    if p.is_file():
        return f"[error] '{rel}' is a file, use <read>{rel}</read>."
    entries: list[str] = []
    for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
        if item.name in EXCLUDE_DIRS:
            continue
        entries.append(f"{item.name}/" if item.is_dir() else item.name)
        if len(entries) >= LS_MAX_ENTRIES:
            entries.append(f"... (truncated at {LS_MAX_ENTRIES} entries)")
            break
    if not entries:
        return "(empty)"
    rel_disp = "." if rel.strip().strip("/") == "" else rel
    return f"# {rel_disp}\n" + "\n".join(entries)


def do_read(libdir: Path, spec: str) -> str:
    """`spec` is `path` or `path:START-END` (1-indexed, inclusive)."""
    spec = spec.strip()
    m = re.match(r"^(.*?)(?::(\d+)-(\d+))?$", spec)
    if not m:
        return f"[error] could not parse '{spec}', expected `path` or `path:start-end`."
    path_part, start_s, end_s = m.group(1), m.group(2), m.group(3)
    p = resolve_under_libdir(libdir, path_part)
    if p is None:
        return f"[error] path '{path_part}' is invalid or escapes the library root."
    if not p.exists() or not p.is_file():
        return f"[error] '{path_part}' is not a readable file."
    try:
        text = p.read_text(errors="replace")
    except OSError as e:
        return f"[error] failed to read '{path_part}': {e}"
    lines = text.splitlines()
    if start_s and end_s:
        start, end = int(start_s), int(end_s)
        if start < 1 or end < start:
            return f"[error] invalid range {start}-{end} (need 1 ≤ start ≤ end)."
        end = min(end, len(lines))
        sliced = lines[start - 1 : end]
        width = len(str(end))
        body = "\n".join(f"{i:>{width}}: {ln}" for i, ln in enumerate(sliced, start))
        header = f"# {path_part} (lines {start}-{end} of {len(lines)})"
        return f"{header}\n{body}"
    if len(lines) > READ_MAX_LINES_DEFAULT:
        return (
            f"[truncated] '{path_part}' has {len(lines)} lines (>{READ_MAX_LINES_DEFAULT}). "
            f"Re-read with a slice, e.g. <read>{path_part}:1-{READ_MAX_LINES_DEFAULT}</read>."
        )
    width = len(str(len(lines)))
    body = "\n".join(f"{i:>{width}}: {ln}" for i, ln in enumerate(lines, 1))
    header = f"# {path_part} ({len(lines)} lines)"
    return f"{header}\n{body}"


def do_grep(libdir: Path, body: str) -> str:
    """`body` is JSON `{"pattern": "...", "path": "..."}` or a bare pattern."""
    body = body.strip()
    pattern: str
    path: str = "."
    if body.startswith("{"):
        try:
            obj = json.loads(body)
            pattern = obj.get("pattern", "")
            path = obj.get("path", ".") or "."
        except json.JSONDecodeError as e:
            return f"[error] grep body is not valid JSON: {e}"
    else:
        pattern = body
    if not pattern:
        return "[error] empty pattern."
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"[error] invalid regex: {e}"
    target = resolve_under_libdir(libdir, path)
    if target is None:
        return f"[error] path '{path}' is invalid or escapes the library root."
    if not target.exists():
        return f"[error] path '{path}' does not exist."
    files: list[Path] = []
    if target.is_file():
        files = [target]
    else:
        for f in sorted(target.rglob("*.rs")):
            if any(part in EXCLUDE_DIRS for part in f.parts):
                continue
            files.append(f)
    results: list[str] = []
    truncated = False
    libdir_resolved = libdir.resolve()
    for f in files:
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        try:
            rel = f.relative_to(libdir_resolved)
        except ValueError:
            rel = f
        for i, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                results.append(f"{rel}:{i}: {line}")
                if len(results) >= GREP_MAX_RESULTS:
                    truncated = True
                    break
        if truncated:
            break
    if not results:
        return "(no matches)"
    out = "\n".join(results)
    if truncated:
        out += f"\n... (truncated at {GREP_MAX_RESULTS} matches — narrow the pattern or path)"
    return out


@dataclass
class SourceSolverEnv(MessageEnv):
    initial_state: SourceSolverEnvState
    runtime_pool: RuntimePool
    initial_messages: list[TinkerMessage]
    reward_fn: LLMAsAJudgeSingleTurn
    mutable_state: SimplifiedSolverMutableState
    history: list[TinkerMessage] = field(default_factory=list)
    # When False, `<grep>` / `<read>` / `<ls>` are rejected with an error.
    # Used by the augmentation pipeline where the SolvedSubtask hint already
    # tells the solver which API to use — re-discovery via search would defeat
    # the purpose. Also keeps stop-sequence count low enough for Gemini.
    enable_search_tools: bool = True

    async def initial_observation(self) -> list[TinkerMessage]:
        if not self.history:
            self.history = list(self.initial_messages)
        return self.history

    @property
    def libdir(self) -> Path:
        return self.initial_state.libdir

    # --- Tool implementations (delegate to module-level pure functions
    # so the planner can reuse them without instantiating an env). ---

    def _do_ls(self, rel: str) -> str:
        return do_ls(self.libdir, rel)

    def _do_read(self, spec: str) -> str:
        return do_read(self.libdir, spec)

    def _do_grep(self, body: str) -> str:
        return do_grep(self.libdir, body)

    def _user_msg(self, body: str) -> TinkerMessage:
        # Prefix each tool-result/error so the agent always sees its budget.
        used = self.mutable_state.total_turns - self.mutable_state.remaining_turns
        total = self.mutable_state.total_turns
        return TinkerMessage(role="user", content=f"[turn {used}/{total}]\n{body}")

    # --- Step loop ---

    async def step(self, message: TinkerMessage) -> SSStepResult:
        from coder_mcp.runtime import CoderMCPRuntimeError
        from adapter_agent.util.exception import CodingEnvironmentError

        try:
            self.history.append(message)
            self.mutable_state.remaining_turns -= 1

            if isinstance(message["content"], str):
                text_content = message["content"]
            else:
                text_content = "".join(
                    part["text"]
                    for part in message["content"]
                    if part["type"] == "text"
                )

            # Multi-tag rejection (one tool tag per response).
            all_tags = re.findall(
                r"<(grep|read|ls|write_and_run|submit)\b",
                text_content,
                re.DOTALL,
            )
            if len(all_tags) > 1:
                error_content = (
                    f"[SYSTEM ERROR] MULTIPLE_TOOL_TAGS_DETECTED\n"
                    f"You attempted to use multiple tool actions in a single response: {', '.join(set(all_tags))}.\n"
                    f"You MUST only use ONE tool tag per turn. Re-think and emit only a single action."
                )
                self.history.append(self._user_msg(error_content))
                if self.mutable_state.remaining_turns <= 0:
                    return SSStepResult(
                        reward=0.0,
                        episode_done=True,
                        next_messages=self.history,
                        conclusion="max_turns_exceeded",
                    )
                return SSStepResult(
                    reward=0.0,
                    episode_done=False,
                    next_messages=self.history,
                    conclusion="multiple_tool_tags",
                )

            # write_and_run — overwrite src/main.rs and run cargo.
            wr = re.search(r"<write_and_run>(.*?)</write_and_run>", text_content, re.DOTALL)
            if wr:
                code = wr.group(1).strip()
                async with self.runtime_pool.acquire() as runtime:
                    await runtime.set_content("src/main.rs", code)
                    run_ret, _ = await runtime.run_cargo()
                content = f"<CargoRunResult>\n{run_ret}\n</CargoRunResult>"
                self.history.append(self._user_msg(content))
                if self.mutable_state.remaining_turns <= 0:
                    return SSStepResult(
                        reward=0.0,
                        episode_done=True,
                        next_messages=self.history,
                        conclusion="max_turns_exceeded",
                    )
                return SSStepResult(
                    reward=0.0,
                    episode_done=False,
                    next_messages=self.history,
                    conclusion="not_finished",
                )

            # submit — final answer; runs verifier and ends the episode.
            sub = re.search(r"<submit>(.*?)</submit>", text_content, re.DOTALL)
            if sub:
                code = sub.group(1).strip()
                if not code:
                    self.history.append(self._user_msg(
                        "[SYSTEM ERROR] EMPTY_SUBMIT — your <submit> block was empty. Provide a complete `fn main()` program.",
                    ))
                    if self.mutable_state.remaining_turns <= 0:
                        return SSStepResult(
                            reward=0.0,
                            episode_done=True,
                            next_messages=self.history,
                            conclusion="max_turns_exceeded",
                        )
                    return SSStepResult(
                        reward=0.0,
                        episode_done=False,
                        next_messages=self.history,
                        conclusion="no_code_found",
                    )
                async with self.runtime_pool.acquire() as runtime:
                    await runtime.set_content("src/main.rs", code)
                    reward, conclusion, observation = await self.reward_fn(
                        self.history, runtime=runtime
                    )
                self.history.append(TinkerMessage(role="user", content=observation))
                return SSStepResult(
                    reward=reward,
                    episode_done=True,
                    next_messages=self.history,
                    conclusion=conclusion,
                )

            # ls
            ls_match = re.search(r"<ls>(.*?)</ls>", text_content, re.DOTALL)
            if ls_match:
                if not self.enable_search_tools:
                    output = (
                        "[SYSTEM ERROR] SEARCH_DISABLED — `<ls>` is not "
                        "available in this session. Use `<write_and_run>` "
                        "to test code or `<submit>` to finalize."
                    )
                else:
                    output = self._do_ls(ls_match.group(1))
                self.history.append(self._user_msg(output))
                if self.mutable_state.remaining_turns <= 0:
                    return SSStepResult(
                        reward=0.0,
                        episode_done=True,
                        next_messages=self.history,
                        conclusion="max_turns_exceeded",
                    )
                return SSStepResult(
                    reward=0.0,
                    episode_done=False,
                    next_messages=self.history,
                    conclusion="not_finished",
                )

            # read
            rd = re.search(r"<read>(.*?)</read>", text_content, re.DOTALL)
            if rd:
                if not self.enable_search_tools:
                    output = (
                        "[SYSTEM ERROR] SEARCH_DISABLED — `<read>` is not "
                        "available in this session. Use `<write_and_run>` "
                        "to test code or `<submit>` to finalize."
                    )
                else:
                    output = self._do_read(rd.group(1))
                self.history.append(self._user_msg(output))
                if self.mutable_state.remaining_turns <= 0:
                    return SSStepResult(
                        reward=0.0,
                        episode_done=True,
                        next_messages=self.history,
                        conclusion="max_turns_exceeded",
                    )
                return SSStepResult(
                    reward=0.0,
                    episode_done=False,
                    next_messages=self.history,
                    conclusion="not_finished",
                )

            # grep
            gr = re.search(r"<grep>(.*?)</grep>", text_content, re.DOTALL)
            if gr:
                if not self.enable_search_tools:
                    output = (
                        "[SYSTEM ERROR] SEARCH_DISABLED — `<grep>` is not "
                        "available in this session. Use `<write_and_run>` "
                        "to test code or `<submit>` to finalize."
                    )
                else:
                    output = self._do_grep(gr.group(1))
                self.history.append(self._user_msg(output))
                if self.mutable_state.remaining_turns <= 0:
                    return SSStepResult(
                        reward=0.0,
                        episode_done=True,
                        next_messages=self.history,
                        conclusion="max_turns_exceeded",
                    )
                return SSStepResult(
                    reward=0.0,
                    episode_done=False,
                    next_messages=self.history,
                    conclusion="not_finished",
                )

            # No recognized tag — nudge the agent.
            available = (
                "<grep>, <read>, <ls>, <write_and_run>, or <submit>"
                if self.enable_search_tools
                else "<write_and_run> or <submit>"
            )
            error_content = (
                "[SYSTEM ERROR] NO_TOOL_TAG\n"
                "Your response contained no recognized tool tag. "
                f"Emit one of {available}."
            )
            self.history.append(self._user_msg(error_content))
            if self.mutable_state.remaining_turns <= 0:
                return SSStepResult(
                    reward=0.0,
                    episode_done=True,
                    next_messages=self.history,
                    conclusion="max_turns_exceeded",
                )
            return SSStepResult(
                reward=0.0,
                episode_done=False,
                next_messages=self.history,
                conclusion="not_finished",
            )
        except CoderMCPRuntimeError as e:
            raise CodingEnvironmentError(f"Environment error during step: {e}") from e


def get_source_solver_initial_messages(
    env_state: SourceSolverEnvState,
    tools: list[ToolSpec],
    renderer: Renderer | None,
    max_turns: int,
    enable_search_tools: bool = True,
) -> list[TinkerMessage]:
    one_tag_rule = (
        "Emit EXACTLY ONE tool tag per response. "
        "The system processes only the first tag it finds."
    )

    role_blurb = (
        f"You are a Rust engineer iteratively solving a coding task in a cargo "
        f"project. The `{env_state.library_name}` library is preinstalled as a "
        f"dependency."
    )
    if enable_search_tools:
        role_blurb += (
            " The library's source tree is available read-only — explore it to "
            "discover what types and methods exist before writing code."
        )

    if enable_search_tools:
        tools_block = """- `<ls>relative/path</ls>` — list immediate children of a directory inside the library source.
- `<read>relative/path</read>` or `<read>relative/path:START-END</read>` — read a source file (optionally a 1-indexed inclusive line range).
- `<grep>pattern</grep>` or `<grep>{"pattern": "...", "path": "subdir/"}</grep>` — Python-regex search across `*.rs` files. Default path is the library root. JSON form is optional.
- `<write_and_run>...rust source...</write_and_run>` — overwrite `src/main.rs` with the contents and run `cargo run`; output is returned to you.
- `<submit>...rust source...</submit>` — your final `src/main.rs`. ENDS the task and triggers verification."""
    else:
        tools_block = """- `<write_and_run>...rust source...</write_and_run>` — overwrite `src/main.rs` with the contents and run `cargo run`; output is returned to you.
- `<submit>...rust source...</submit>` — your final `src/main.rs`. ENDS the task and triggers verification."""

    PROMPT = f"""<Role>
{role_blurb}
</Role>

<Tools>
{one_tag_rule}

{tools_block}
</Tools>

<Budget>
You have at most {max_turns} turns total (1 tool call per turn). Each tool result
is prefixed with `[turn k/{max_turns}]` so you always know where you are.
If you reach the final turn without `<submit>`, the task is LOST. Pace yourself:
{"explore early, then verify with `<write_and_run>` and `<submit>` while you still have budget."
if enable_search_tools else
"iterate with `<write_and_run>` to debug, then `<submit>` while you still have budget."}
</Budget>
"""

    if enable_search_tools:
        guideline_bullets = [
            f"- Start by exploring the `{env_state.library_name}` source layout (`<ls>src/</ls>`) before writing code.",
            "- Use `<grep>` to locate types/functions (e.g. `pub struct CsrMatrix`, `pub fn .*row`). Use `<read>` for surrounding context.",
            "- Always verify with `<write_and_run>` before `<submit>`.",
        ]
    else:
        guideline_bullets = [
            "- You do NOT have library-source search this session. Rely on the "
            "task description and any `<ReferenceKnowledge>` / `<SolvedSubtasks>` "
            "below to know which API to call.",
            "- Always verify with `<write_and_run>` before `<submit>`.",
        ]
    if env_state.reference_knowledge:
        guideline_bullets.insert(
            0,
            "- `<ReferenceKnowledge>` (in the user message) is authoritative for the library API. Prefer it over guessing.",
        )
    if env_state.solved_subtasks:
        guideline_bullets.append(
            "- `<SolvedSubtasks>` shows verified solutions to related problems — reuse the patterns."
        )
    PROMPT += f"\n<Guidelines>\n{chr(10).join(guideline_bullets)}\n</Guidelines>\n"

    if env_state.solved_subtasks:
        subtask_blocks = []
        for idx, sub in enumerate(env_state.solved_subtasks, start=1):
            subtask_blocks.append(
                f'<SolvedSubtask index="{idx}">\n'
                f"<Problem>\n{sub.instruction}\n</Problem>\n"
                f"<Solution>\n{sub.submit_code}\n</Solution>\n"
                f"</SolvedSubtask>"
            )
        PROMPT += (
            "\n<SolvedSubtasks>\n"
            + "\n\n".join(subtask_blocks)
            + "\n</SolvedSubtasks>\n"
        )

    initial_message = f"""<Task>
{env_state.task.instruction}
</Task>
"""
    if env_state.reference_knowledge:
        initial_message += (
            f"\n<ReferenceKnowledge>\n{env_state.reference_knowledge}\n</ReferenceKnowledge>\n"
        )
    if env_state.qwen_no_think:
        initial_message = "/no_think " + initial_message

    if renderer is not None:
        system_prompt_with_tools = _inject_tools_into_prompt(renderer, tools, PROMPT)
    else:
        system_prompt_with_tools = [TinkerMessage(role="system", content=PROMPT)]

    return system_prompt_with_tools + [
        TinkerMessage(role="user", content=initial_message),
    ]


async def build_source_solver_msg_env(
    env_state: SourceSolverEnvState,
    verifier: Verifier,
    runtime_pool: RuntimePool,
    max_turns: int = 12,
    renderer: Renderer | None = None,
    enable_search_tools: bool = True,
) -> SourceSolverEnv:
    exclude = ["target", ".git"]
    async with runtime_pool.acquire() as runtime:
        tree_structure = await runtime.tree(".", exclude=exclude, truncate=20)

    mutable_state = SimplifiedSolverMutableState(
        remaining_turns=max_turns, total_turns=max_turns
    )

    return SourceSolverEnv(
        initial_state=env_state,
        runtime_pool=runtime_pool,
        enable_search_tools=enable_search_tools,
        initial_messages=get_source_solver_initial_messages(
            env_state=env_state,
            tools=[],
            renderer=renderer,
            max_turns=max_turns,
            enable_search_tools=enable_search_tools,
        ),
        reward_fn=LLMAsAJudgeSingleTurn(
            task=env_state.task,
            verifier=verifier,
            tree_structure=tree_structure,
        ),
        mutable_state=mutable_state,
    )
