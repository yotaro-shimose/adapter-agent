import asyncio
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Self

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
from tinker_cookbook.rl.message_env import (
    EnvFromMessageEnv,
    MessageEnv,
    MessageStepResult,
)
from tinker_cookbook.tool_use.tools import handle_tool_call
from typing_extensions import AsyncGenerator

from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.knowledge_db import KnowledgeDB
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


@dataclass
class SimplifiedSolverEnv(MessageEnv):
    initial_state: SimplifiedSolverEnvState
    rust_env: Runtime
    search_model: Any
    knowledge_db: KnowledgeDB
    rust_doc_analyzer: AsyncRustDocAnalyzer
    initial_messages: list[TinkerMessage]
    reward_fn: LLMAsAJudgeSingleTurn
    mutable_state: SimplifiedSolverMutableState
    history: list[TinkerMessage] = field(default_factory=list)

    def __post_init__(self):
        search_tool = SearchTool(
            self.search_model,
            self.rust_doc_analyzer,
            self.knowledge_db,
            self.mutable_state,
            blocked_knowledge_ids=self.initial_state.blocked_knowledge_ids,
        )
        self.tools = {
            search_tool.name: search_tool,
        }

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

                # Check for XML code test submission
                write_match = re.search(
                    r"<write_and_run>(.*?)</write_and_run>", text_content, re.DOTALL
                )
                if write_match:
                    code_to_test = write_match.group(1).strip()
                    await self.rust_env.set_content("src/main.rs", code_to_test)

                    run_ret, run_success = await self.rust_env.run_cargo()
                    content = f"<FileWritten>\\nsrc/main.rs has been updated.\\n</FileWritten>\\n<CargoRunResult>\\n{run_ret}\\n</CargoRunResult>\\n<RemainingTurns>\\n{self.mutable_state.remaining_turns}\\n</RemainingTurns>"

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
You are an **expert** in the `{env_state.library_name}` library. 
You possess **deep internalized technical knowledge** of this library. 
Before using the `search` tool, you MUST consult your own internal reasoning (thoughts) to retrieve API signatures, patterns, and rules. 
Only search for highly specific details or error resolutions that are not already present in your memory.
</Context>

<HowTo>
You have a simplified coding environment with the following tools:
- `search`: A JSON tool. Use this to search both symbol names and documentation for functionality, concepts, or how-to guides.

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
"""

    guidelines = """\
Verification: Once again, you MUST verify your answer. You should make your best efforts to avoid hallucination and make sure your answer is correct.
Self-contained: Note your solution has been fully self-contained including both fully functioning source code and explanation.
Testing Code: Before submitting the final answer, use the `<write_and_run>...</write_and_run>` tags to test code and see outputs. Avoid JSON syntax errors.
Code block inclusion: Your final answer MUST include exactly one `<submit>\\n<your_code_here>\\n</submit>` block. Its content will be pasted to main.rs and executed for final verification to END the task.
Trust Internal Knowledge: Favor your internalized recollections (summarized in your initial thoughts) over repetitive searching. Use the `search` tool only for details you don't already possess.
Avoid Redundant Search: The number of `search` calls is strictly limited. Avoid repeated or redundant queries for information already provided or recalled in your thoughts.
Error Reflection: If `<write_and_run>` test fails, analyze the compiler error carefully. If you find your understanding is flawed, use the `search` tool sparingly to fill specific gaps.
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


class SimplifiedSolverTokenEnv(EnvFromMessageEnv):
    def __init__(
        self,
        renderer: Renderer,
        message_env: SimplifiedSolverEnv,
        internalized_knowledge: str | None = None,
        failed_parse_reward: float = -1.0,
        terminate_on_parse_error: bool = True,
        max_trajectory_tokens: int | None = None,
    ):
        super().__init__(
            renderer=renderer,
            message_env=message_env,
            failed_parse_reward=failed_parse_reward,
            terminate_on_parse_error=terminate_on_parse_error,
            max_trajectory_tokens=max_trajectory_tokens,
        )
        self.internalized_knowledge = internalized_knowledge

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
    ) -> types.StepResult:
        assistant_message, parse_success = self.renderer.parse_response(action)

        if not parse_success:
            return types.StepResult(
                reward=self.failed_parse_reward,
                episode_done=self.terminate_on_parse_error,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics={"parse_error": 1.0},
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
            return types.StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics={**msg_step.metrics, "context_overflow": 1.0},
            )

        return types.StepResult(
            reward=msg_step.reward,
            episode_done=msg_step.episode_done,
            next_observation=next_observation,
            next_stop_condition=next_stop_condition,
            metrics=msg_step.metrics,
        )


@asynccontextmanager
async def build_simplified_solver_env(
    env_state: SimplifiedSolverEnvState,
    renderer: Renderer,
    verifier: Verifier,
    rust_doc_analyzer: AsyncRustDocAnalyzer,
    runtime: Runtime,
    search_model: Any,
    knowledge_db: KnowledgeDB,
    max_trajectory_tokens: int = 32 * 1024,
    max_turns: int = 10,
) -> AsyncGenerator[SimplifiedSolverTokenEnv, None]:
    exclude = ["target", ".git"]
    tree_structure = await runtime.tree(".", exclude=exclude, truncate=20)

    mutable_state = SimplifiedSolverMutableState(
        remaining_turns=max_turns, total_turns=max_turns
    )
    tools_list = [
        SearchTool(search_model, rust_doc_analyzer, knowledge_db, mutable_state),
    ]

    msg_env = SimplifiedSolverEnv(
        initial_state=env_state,
        rust_env=runtime,
        search_model=search_model,
        knowledge_db=knowledge_db,
        rust_doc_analyzer=rust_doc_analyzer,
        initial_messages=get_simplified_solver_initial_messages(
            env_state=env_state,
            tree_structure=tree_structure,
            tools=[t.to_spec() for t in tools_list],
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
