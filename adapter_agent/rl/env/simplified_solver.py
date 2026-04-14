import asyncio
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Self, cast

import tinker
from coder_mcp.runtime import (
    CoderMCPRuntimeError,
    Runtime,
)
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.renderers.base import TextPart, ToolCall, ToolSpec
from tinker_cookbook.rl import types
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult
from tinker_cookbook.tool_use.tools import handle_tool_call
from typing_extensions import AsyncGenerator

from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.wiki_manager import WikiManager
from adapter_agent.rl.env.conclusion import SSConclusion
from adapter_agent.rl.env.injection import _inject_tools_into_prompt
from adapter_agent.rl.env.reward import LLMAsAJudgeSingleTurn
from adapter_agent.rl.env.search_tool import SearchTool, SimplifiedSolverMutableState
from adapter_agent.util.exception import CodingEnvironmentError

CARGO_INIT_MAIN_RS = """\
fn main() {
    println!("Hello, world!");
}
"""


@dataclass
class SimplifiedSolverEnvState:
    task: Task
    library_name: str
    internalized_knowledge: str | None
    messages: list[TinkerMessage]
    blocked_knowledge_ids: set[str] = field(default_factory=set)
    qwen_no_think: bool = False

    @classmethod
    def numrs2(
        cls,
        task: Task,
        internalized_knowledge: str | None = None,
        blocked_knowledge_ids: set[str] | None = None,
        qwen_no_think: bool = False,
    ) -> Self:
        return cls(
            task=task,
            library_name="numrs2",
            internalized_knowledge=internalized_knowledge,
            messages=[],
            blocked_knowledge_ids=blocked_knowledge_ids or set(),
            qwen_no_think=qwen_no_think,
        )

    def with_messages(self, messages: list[TinkerMessage]) -> Self:
        return self.__class__(
            task=self.task,
            library_name=self.library_name,
            internalized_knowledge=self.internalized_knowledge,
            messages=messages,
            blocked_knowledge_ids=self.blocked_knowledge_ids,
            qwen_no_think=self.qwen_no_think,
        )


@dataclass
class SSStepResult(MessageStepResult):
    conclusion: SSConclusion = field(default="not_finished")


@dataclass(kw_only=True)
class SSTokenEnvResult(types.StepResult):
    conclusion: SSConclusion


@dataclass
class SimplifiedSolverEnv(MessageEnv):
    initial_state: SimplifiedSolverEnvState
    rust_env: Runtime
    search_model: Any
    wiki_manager: WikiManager
    rust_doc_analyzer: AsyncRustDocAnalyzer
    initial_messages: list[TinkerMessage]
    reward_fn: LLMAsAJudgeSingleTurn
    mutable_state: SimplifiedSolverMutableState
    history: list[TinkerMessage] = field(default_factory=list)

    def __post_init__(self):
        self.search_tool = SearchTool(
            self.rust_doc_analyzer,
            self.wiki_manager,
            self.mutable_state,
        )
        self.tools = {}

    async def initial_observation(self) -> list[TinkerMessage]:
        if not self.history:
            self.history = list(self.initial_messages)
        return self.history

    async def step(self, message: TinkerMessage) -> SSStepResult:
        try:
            self.history.append(message)
            self.mutable_state.remaining_turns -= 1

            tool_calls: list[ToolCall] = list(message.get("tool_calls", []))
            if tool_calls:
                tool_results = await asyncio.gather(
                    *[handle_tool_call(self.tools, tc) for tc in tool_calls]  # type: ignore
                )
                for tool_result in tool_results:
                    for msg in tool_result.messages:
                        self.history.append(msg)

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

            else:
                if isinstance(message["content"], str):
                    text_content = message["content"]
                else:
                    text_content = "".join(
                        part["text"]
                        for part in message["content"]
                        if part["type"] == "text"
                    )

                # Detect if the agent tried to use multiple tool tags at once
                all_tags = re.findall(
                    r"<(wiki_ls|wiki_read|search_library_doc|write_and_run|submit)>",
                    text_content,
                    re.DOTALL,
                )
                if len(all_tags) > 1:
                    new_message = TinkerMessage(
                        role="user",
                        content=(
                            f"[SYSTEM ERROR] MULTIPLE_TOOL_TAGS_DETECTED\n"
                            f"You attempted to use multiple tool actions in a single response: {', '.join(set(all_tags))}.\n"
                            f"You MUST only use ONE tool tag per turn. Please re-think and emit only a single action."
                        ),
                    )
                    self.history.append(new_message)

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

                # Check for XML code test submission
                write_match = re.search(
                    r"<write_and_run>(.*?)</write_and_run>", text_content, re.DOTALL
                )
                if write_match:
                    code_to_test = write_match.group(1).strip()
                    await self.rust_env.set_content("src/main.rs", code_to_test)

                    run_ret, run_success = await self.rust_env.run_cargo()
                    
                    search_limit = self.mutable_state.total_turns // 2
                    remaining_search_quota = max(0, search_limit - self.mutable_state.search_count)
                    
                    content = (
                        f"<FileWritten>\nsrc/main.rs has been updated.\n</FileWritten>\n"
                        f"<CargoRunResult>\n{run_ret}\n</CargoRunResult>\n\n"
                        f"[STATUS]\n"
                        f"RemainingSearchQuota: {remaining_search_quota}\n"
                        f"RemainingTurns: {self.mutable_state.remaining_turns}"
                    )

                    new_message = TinkerMessage(role="user", content=content)
                    self.history.append(new_message)

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

                # Check for Wiki Exploration Tools
                ls_match = re.search(r"<wiki_ls\s*/>", text_content)
                if ls_match:
                    titles = await self.wiki_manager.ls()
                    title_list = "\n".join([f"- [[{t}]]" for t in titles]) if titles else "No articles found."
                    content = (
                        f"## Wiki Article List (Version: {self.wiki_manager.version})\n"
                        f"{title_list}\n\n"
                        f"[STATUS]\n"
                        f"RemainingTurns: {self.mutable_state.remaining_turns}"
                    )
                    new_message = TinkerMessage(role="user", content=content)
                    self.history.append(new_message)
                    return SSStepResult(reward=0.0, episode_done=False, next_messages=self.history, conclusion="not_finished")

                read_match = re.search(r"<wiki_read>(.*?)</wiki_read>", text_content, re.DOTALL)
                if read_match:
                    title = read_match.group(1).strip()
                    article_content = await self.wiki_manager.read(title)
                    if article_content:
                        res_content = f"## Wiki Article: {title}\n\n{article_content}"
                    else:
                        res_content = f"Error: Article '{title}' not found."
                    
                    content = (
                        f"{res_content}\n\n"
                        f"[STATUS]\n"
                        f"RemainingTurns: {self.mutable_state.remaining_turns}"
                    )
                    new_message = TinkerMessage(role="user", content=content)
                    self.history.append(new_message)
                    return SSStepResult(reward=0.0, episode_done=False, next_messages=self.history, conclusion="not_finished")

                # Check for XML search tool
                search_match = re.search(
                    r"<search_library_doc>(.*?)</search_library_doc>",
                    text_content,
                    re.DOTALL,
                )
                if search_match:
                    query = search_match.group(1).strip()
                    final_ans, ctx = await self.search_tool.search(query)

                    search_limit = self.mutable_state.total_turns // 2
                    remaining_search_quota = max(
                        0, search_limit - self.mutable_state.search_count
                    )

                    content = (
                        f"{final_ans}\n\n"
                        f"[STATUS]\n"
                        f"RemainingSearchQuota: {remaining_search_quota}\n"
                        f"RemainingTurns: {self.mutable_state.remaining_turns}"
                    )

                    new_message = TinkerMessage(role="user", content=content)
                    self.history.append(new_message)

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

                # Check for Final Answer Submission
                submit_match = re.search(
                    r"<submit>(.*?)</submit>", text_content, re.DOTALL
                )
                if not submit_match:
                    new_message = TinkerMessage(
                        role="user",
                        content="No code found in the message. Make sure you use `<write_and_run>...</write_and_run>` to test, or wrap the final valid runnable code in `<submit>...</submit>` to submit.",
                    )
                    self.history.append(new_message)

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

                code = submit_match.group(1).strip()
                await self.rust_env.set_content("src/main.rs", code)

                reward, conclusion, final_obs = await self.reward_fn(self.history)
                new_message = TinkerMessage(role="user", content=final_obs)
                self.history.append(new_message)

                return SSStepResult(
                    reward=reward,
                    episode_done=True,
                    next_messages=self.history,
                    conclusion=conclusion,
                )
        except CoderMCPRuntimeError as e:
            raise CodingEnvironmentError(f"Environment error during step: {e}") from e

    async def get_state(self) -> SimplifiedSolverEnvState:
        return self.initial_state.with_messages(self.history)


def get_simplified_solver_initial_messages(
    env_state: SimplifiedSolverEnvState,
    tree_structure: str,
    tools: list[ToolSpec],
    renderer: Renderer,
) -> list[TinkerMessage]:
    PROMPT = f"""<Role>
You are a Rust engineer.
Your goal is to create a solution to achieve the user's request through the iterative process.
You create a solution using simple playground to find the correct code.
</Role>

<Context>
You are working in a cargo-initialized project.
You need to solve the problem using `{env_state.library_name}` library which is already installed as a dependency (e.g. via `cargo add`).
Note `{env_state.library_name}` is a new library you should be unfamilier with.
</Context>

<HowTo>
You have a simplified coding environment with the following tools:
- `wiki_ls`: Use `<wiki_ls />` to list all available Wiki articles.
- `wiki_read`: Use `<wiki_read>Article_Title</wiki_read>` to read specific knowledge.
- `search_library_doc`: Use `<search_library_doc>query</search_library_doc>` to search official Rust documentation for specific API signatures or error details.

IMPORTANT: You MUST ONLY use ONE tool tag (`<wiki_ls />`, `<wiki_read>`, `<search_library_doc>`, `<write_and_run>`, or `<submit>`) in a single response. Do not combine multiple actions or mix searching with testing. 
The system will only process the FIRST tag it detects if the response is valid, but using multiple tags is discouraged and may lead to errors.

To TEST your code implementation without ending the task, simply output your Rust code inside plain XML tags like this (do NOT use JSON!):
<write_and_run>
fn main() {{
    // your test code here
}}
</write_and_run>
The system will automatically extract the code, write it to `src/main.rs`, run `cargo run`, and give you the output to help you iterate.

Once you confirmed that the solution works, you must output your FINAL answer as a complete, fully functioning source code enclosed in a `<submit> ... </submit>` block. You can also provide any necessary explanation.
Note: Outputting a `<submit> ... </submit>` block will IMMEDIATELY SUBMIT your answer and run the final verification, which will END the task.
</HowTo>

<ResourceManagement>
At the end of every tool response, you will see a `[STATUS]` section:
- `RemainingSearchQuota`: Number of `search_library_doc` calls you can still make. Wiki exploration tools (`wiki_ls`, `wiki_read`) do NOT consume this quota.
- `RemainingTurns`: Total number of actions (turns) remaining before the task is forced to end.
</ResourceManagement>
"""

    guidelines = """\
Verification: Once again, you MUST verify your answer. You should make your best efforts to avoid hallucination and make sure your answer is correct.
Self-contained: Note your solution has been fully self-contained including both fully functioning source code and explanation.
Testing Code: Before submitting the final answer, use the `<write_and_run>...</write_and_run>` tags to test code and see outputs. Avoid JSON syntax errors.
One Action at a Time: You MUST only use one tool tag per turn. Response containing multiple tags (e.g. two searches or a search and a test) will be rejected.
Code block inclusion: Your final answer MUST include exactly one `<submit>\\n<your_code_here>\\n</submit>` block. Its content will be pasted to main.rs and executed for final verification to END the task.
Wiki Exploration: Always check the internal Wiki (`wiki_ls`, `wiki_read`) first for high-quality, project-specific knowledge.
Documentation Fallback: Use `search_library_doc` only if the Wiki does not contain the necessary information. Note that this tool provides raw technical details from the crate's official documentation.
Error Reflection: If `<write_and_run>` test fails, analyze the compiler error carefully. Check the Wiki or documentation for specific causes of the error code (e.g., E0308).
"""
    PROMPT += f"\n<Guidelines>\n{guidelines}\n</Guidelines>"

    initial_message = f"""<Task>
{env_state.task.instruction}
</Task>

<Current Directory Structure>
{tree_structure}
</Current Directory Structure>

<Current src/main.rs>
{CARGO_INIT_MAIN_RS}
</Current src/main.rs>
"""

    if env_state.qwen_no_think:
        initial_message = "/no_think " + initial_message

    system_prompt_with_tools = _inject_tools_into_prompt(renderer, tools, PROMPT)
    return system_prompt_with_tools + [
        TinkerMessage(role="user", content=initial_message),
    ]


class SimplifiedSolverTokenEnv(types.Env):
    def __init__(
        self,
        renderer: Renderer,
        message_env: SimplifiedSolverEnv,
        internalized_knowledge: str | None = None,
        failed_parse_reward: float = -1.0,
        terminate_on_parse_error: bool = True,
        max_trajectory_tokens: int | None = None,
    ):
        self.renderer = renderer
        self.message_env = message_env
        self.failed_parse_reward = failed_parse_reward
        self.terminate_on_parse_error = terminate_on_parse_error
        self.max_trajectory_tokens = max_trajectory_tokens
        self.internalized_knowledge = internalized_knowledge
        self._base_stop_condition = renderer.get_stop_sequences()

    async def get_state(self) -> SimplifiedSolverEnvState:
        return await self.message_env.get_state()  # type: ignore

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        messages = await self.message_env.initial_observation()
        # No tags, just raw technical fact statement to start the response
        prefill = self.internalized_knowledge
        return self.renderer.build_generation_prompt(
            messages, prefill=prefill
        ), self._base_stop_condition

    async def step(
        self, action: types.Action, *, extra: types.ActionExtra | None = None
    ) -> SSTokenEnvResult:
        assistant_message, parse_success = self.renderer.parse_response(action)

        if not parse_success:
            return SSTokenEnvResult(
                reward=self.failed_parse_reward,
                episode_done=self.terminate_on_parse_error,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics={"parse_error": 1.0},
                conclusion="parse_failed",
            )

        if self.internalized_knowledge:
            if isinstance(assistant_message["content"], str):
                assistant_message["content"] = [
                    TextPart(type="text", text=assistant_message["content"])
                ]

            # Prepend the internalized knowledge (plain text context) to the first TextPart
            # Or add it as the very first part if preferred.
            assistant_content = [
                TextPart(
                    type="text",
                    text=self.internalized_knowledge,
                ),
                *assistant_message["content"],
            ]
            assistant_message["content"] = assistant_content
            self.internalized_knowledge = None

        msg_step = await self.message_env.step(assistant_message)
        next_observation = self.renderer.build_generation_prompt(msg_step.next_messages)
        next_stop_condition = msg_step.next_stop_condition or self._base_stop_condition

        if (
            self.max_trajectory_tokens is not None
            and next_observation.length > self.max_trajectory_tokens
        ):
            return SSTokenEnvResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics={**msg_step.metrics, "context_overflow": 1.0},
                conclusion="context_length_exceeded",
            )

        return SSTokenEnvResult(
            reward=msg_step.reward,
            episode_done=msg_step.episode_done,
            next_observation=next_observation,
            next_stop_condition=next_stop_condition,
            metrics=msg_step.metrics,
            conclusion=msg_step.conclusion,
        )


@asynccontextmanager
async def build_simplified_solver_env(
    env_state: SimplifiedSolverEnvState,
    renderer: Renderer,
    verifier: Verifier,
    rust_doc_analyzer: AsyncRustDocAnalyzer,
    runtime: Runtime,
    search_model: Any,
    wiki_manager: WikiManager,
    max_trajectory_tokens: int = 32 * 1024,
    max_turns: int = 10,
) -> AsyncGenerator[SimplifiedSolverTokenEnv, None]:
    exclude = ["target", ".git"]
    tree_structure = await runtime.tree(".", exclude=exclude, truncate=20)

    mutable_state = SimplifiedSolverMutableState(
        remaining_turns=max_turns, total_turns=max_turns
    )

    msg_env = SimplifiedSolverEnv(
        initial_state=env_state,
        rust_env=runtime,
        search_model=search_model,
        wiki_manager=wiki_manager,
        rust_doc_analyzer=rust_doc_analyzer,
        initial_messages=get_simplified_solver_initial_messages(
            env_state=env_state,
            tree_structure=tree_structure,
            tools=[],
            renderer=renderer,
        ),
        reward_fn=LLMAsAJudgeSingleTurn(
            task=env_state.task,
            rust_env=runtime,
            verifier=verifier,
            tree_structure=tree_structure,
        ),
        mutable_state=mutable_state,
    )

    yield SimplifiedSolverTokenEnv(
        renderer=renderer,
        message_env=msg_env,
        internalized_knowledge=env_state.internalized_knowledge,
        failed_parse_reward=-1.0,
        max_trajectory_tokens=max_trajectory_tokens,
    )
