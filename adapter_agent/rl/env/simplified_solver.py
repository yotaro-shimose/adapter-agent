import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Self, TypedDict, cast

import tinker
from coder_mcp.runtime import RustCodingEnvironment
from coder_mcp.runtime.rust_env import RustEnvError
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.renderers.base import TextPart, ThinkingPart, ToolCall, ToolSpec
from tinker_cookbook.rl import types
from tinker_cookbook.rl.message_env import (
    EnvFromMessageEnv,
    MessageEnv,
    MessageStepResult,
)
from tinker_cookbook.tool_use.tools import handle_tool_call
from tinker_cookbook.tool_use.types import Tool, ToolInput, ToolResult
from typing_extensions import AsyncGenerator

from adapter_agent.hierarchical.agent.simplified_solver import CARGO_INIT_MAIN_RS
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.rl.env.injection import _inject_tools_into_prompt
from adapter_agent.rl.env.reward import LLMAsAJudgeSingleTurn
from adapter_agent.rl.env.standard import EnvironmentError
from adapter_agent.util.parsing import extract_rust_code


@dataclass
class SimplifiedSolverEnvState:
    task: Task
    image_name: str
    library_name: str
    prethink: str | None
    messages: list[TinkerMessage]
    qwen_no_think: bool = False

    @classmethod
    def numrs2(
        cls, task: Task, prethink: str | None = None, qwen_no_think: bool = False
    ) -> Self:
        return cls(
            task=task,
            library_name="numrs2",
            image_name="coder-mcp-numrs2:latest",
            prethink=prethink,
            messages=[],
            qwen_no_think=qwen_no_think,
        )

    def with_messages(self, messages: list[TinkerMessage]) -> Self:
        return self.__class__(
            task=self.task,
            image_name=self.image_name,
            library_name=self.library_name,
            prethink=self.prethink,
            messages=messages,
        )


@dataclass
class SimplifiedSolverMutableState:
    remaining_turns: int


class SearchTool(Tool):
    def __init__(self, analyzer: RustDocAnalyzer):
        self.analyzer = analyzer

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search the Rust documentation for both symbols and concepts. It searches BOTH the name of the item and its explanation text."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }

    async def run(self, input: ToolInput) -> ToolResult:
        query = input.arguments.get("query", "")
        results = self.analyzer.search(query, limit=10)
        output = json.dumps([r.model_dump() for r in results])
        assert input.call_id is not None
        msg = TinkerMessage(
            role="tool",
            content=output if output != "[]" else "No results found.",
            tool_call_id=input.call_id,
        )
        return ToolResult(messages=[msg])


class ReplaceAndRunTool(Tool):
    def __init__(
        self,
        runtime: RustCodingEnvironment,
        mutable_state: SimplifiedSolverMutableState,
    ):
        self.runtime = runtime
        self.mutable_state = mutable_state

    @property
    def name(self) -> str:
        return "replace_and_run"

    @property
    def description(self) -> str:
        return "Edit src/main.rs by finding and replacing an exact string. If successful, runs cargo run."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
            },
            "required": ["old_str", "new_str"],
        }

    async def run(self, input: ToolInput) -> ToolResult:
        old_str = input.arguments.get("old_str", "")
        new_str = input.arguments.get("new_str", "")

        replace_ret, replace_success = await self.runtime.str_replace(old_str, new_str)
        if not replace_success:
            content = replace_ret
        else:
            self.mutable_state.remaining_turns -= 1
            run_ret, run_success = await self.runtime.run_cargo()
            content = f"<TextReplacementPerformed>\n{replace_ret}\n</TextReplacementPerformed>\n<CargoRunResult>\n{run_ret}\n</CargoRunResult>\n<RemainingTurns>\n{self.mutable_state.remaining_turns}\n</RemainingTurns>"

        assert input.call_id is not None
        msg = TinkerMessage(
            role="tool",
            content=content,
            tool_call_id=input.call_id,
        )
        return ToolResult(messages=[msg])


type SSConclusion = Literal[
    "success",
    "context_length_exceeded",
    "no_code_found",
    "no_text_content",
    "code_did_not_compile",
    "verification_failed",
    "verification_error",
    "environment_error",
    "rewire_failed",
    "max_turns_exceeded",
    "not_finished",
    "parse_failed",
]


class SSMetrics(TypedDict):
    success: float
    context_length_exceeded: float
    no_code_found: float
    no_text_content: float
    code_did_not_compile: float
    verification_failed: float
    verification_error: float
    environment_error: float
    rewire_failed: float
    max_turns_exceeded: float
    not_finished: float
    parse_failed: float


def conclusion_to_metrics(conclusion: SSConclusion) -> dict[str, float]:
    return cast(
        dict[str, float],
        SSMetrics(
            success=1.0 if conclusion == "success" else 0.0,
            context_length_exceeded=1.0
            if conclusion == "context_length_exceeded"
            else 0.0,
            no_code_found=1.0 if conclusion == "no_code_found" else 0.0,
            no_text_content=1.0 if conclusion == "no_text_content" else 0.0,
            code_did_not_compile=1.0 if conclusion == "code_did_not_compile" else 0.0,
            verification_failed=1.0 if conclusion == "verification_failed" else 0.0,
            verification_error=1.0 if conclusion == "verification_error" else 0.0,
            environment_error=1.0 if conclusion == "environment_error" else 0.0,
            rewire_failed=1.0 if conclusion == "rewire_failed" else 0.0,
            max_turns_exceeded=1.0 if conclusion == "max_turns_exceeded" else 0.0,
            not_finished=1.0 if conclusion == "not_finished" else 0.0,
            parse_failed=1.0 if conclusion == "parse_failed" else 0.0,
        ),
    )


@dataclass
class SSStepResult(MessageStepResult):
    conclusion: SSConclusion = field(default="not_finished")


@dataclass
class SimplifiedSolverEnv(MessageEnv):
    initial_state: SimplifiedSolverEnvState
    rust_env: RustCodingEnvironment
    rust_doc_analyzer: RustDocAnalyzer
    initial_messages: list[TinkerMessage]
    reward_fn: LLMAsAJudgeSingleTurn
    mutable_state: SimplifiedSolverMutableState
    history: list[TinkerMessage] = field(default_factory=list)

    def __post_init__(self):
        search_tool = SearchTool(self.rust_doc_analyzer)
        replace_tool = ReplaceAndRunTool(self.rust_env, self.mutable_state)
        self.tools = {
            search_tool.name: search_tool,
            replace_tool.name: replace_tool,
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

                code = extract_rust_code(text_content)
                if not code:
                    new_message = TinkerMessage(
                        role="user",
                        content="No code found in the message. Make sure you wrap the final valid runnable code in ```rust ... ```.",
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

                await self.rust_env.set_content("src/main.rs", code)

                reward, reward_metrics = await self.reward_fn(self.history)

                return SSStepResult(
                    reward=reward,
                    episode_done=True,
                    next_messages=self.history,
                    conclusion="success",
                )
        except RustEnvError as e:
            raise EnvironmentError(f"Environment error during step: {e}") from e

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
- `search`: Use this to search both symbol names and documentation for functionality, concepts, or how-to guides.
- `replace_and_run`: Edit src/main.rs by finding and replacing an exact string. The target file is always src/main.rs. If replacement is successful, runs `cargo run` and returns the output.

First search the necessary information using search tool, and then use replace_and_run tool to replace the content and run the code.
Unverified answer gets automatically rejected. You should confirm your work by running the code and verifying the output.
Once you confirmed that the solution works, you must output your final answer as a complete, fully functioning source code enclosed in a ```rust ... ``` block. You can also provide any necessary explanation.
Note you MUST submit your answer after running the code and verifying the output. Code which does not work will be automatically rejected.
</HowTo>

<Guidelines>
Verification: Once again, you MUST verify your answer. You should make your best efforts to avoid hallucination and make sure your answer is correct.
Self-contained: Note your solution has to be fully self-contained including both fully functioning source code and explanation.
Code block inclusion: Your final answer must include exactly one ```rust\n<your_code_here>\n``` block. It's content will be pasted to main.rs and executed for verification.
Simple Search Keyword: When using the `search` tool, it is highly recommended to use only one keyword such as "array" or "conv2d" otherwise the search tool does not return anything.
Error Reflection: If replace_and_run fails, analyze the error carefully. WHen you find your understanding about the library is wrong, use search tool again.
</Guidelines>"""

    initial_message = f"""<Task>
{env_state.task.instruction}
</Task>

<Current Directory Structure>
{tree_structure}
</Current Directory Structure>

<Current src/main.rs>
{CARGO_INIT_MAIN_RS}
</Current src/main.rs>"""
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
        prethink: str | None = None,
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
        self.prethink = prethink

    async def get_state(self) -> SimplifiedSolverEnvState:
        return await self.message_env.get_state()  # type: ignore

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        messages = await self.message_env.initial_observation()
        prefill = f"<think>{self.prethink}</think>" if self.prethink else None
        return self.renderer.build_generation_prompt(
            messages, prefill=prefill
        ), self._base_stop_condition

    async def step(self, action: types.Action) -> types.StepResult:
        assistant_message, parse_success = self.renderer.parse_response(action)

        if not parse_success:
            return types.StepResult(
                reward=self.failed_parse_reward,
                episode_done=self.terminate_on_parse_error,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics={"parse_error": 1.0},
            )

        if self.prethink:
            if isinstance(assistant_message["content"], str):
                assistant_message["content"] = [
                    TextPart(type="text", text=assistant_message["content"])
                ]
            assistant_content = [
                ThinkingPart(
                    type="thinking",
                    thinking=self.prethink,
                ),
                *assistant_message["content"],
            ]
            assistant_message["content"] = assistant_content
            self.prethink = None

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
    rust_doc_analyzer: RustDocAnalyzer,
    max_trajectory_tokens: int = 32 * 1024,
    max_turns: int = 10,
) -> AsyncGenerator[SimplifiedSolverTokenEnv, None]:
    async with RustCodingEnvironment(image_name=env_state.image_name) as rust_env:
        exclude = ["target", ".git"]
        tree_structure = await rust_env.tree(".", exclude=exclude, truncate=20)

        mutable_state = SimplifiedSolverMutableState(remaining_turns=max_turns)
        tools_list = [
            SearchTool(rust_doc_analyzer),
            ReplaceAndRunTool(rust_env, mutable_state),
        ]

        msg_env = SimplifiedSolverEnv(
            initial_state=env_state,
            rust_env=rust_env,
            rust_doc_analyzer=rust_doc_analyzer,
            initial_messages=get_simplified_solver_initial_messages(
                env_state=env_state,
                tree_structure=tree_structure,
                tools=[t.to_spec() for t in tools_list],
                renderer=renderer,
            ),
            reward_fn=LLMAsAJudgeSingleTurn(
                task=env_state.task,
                rust_env=rust_env,
                verifier=verifier,
                tree_structure=tree_structure,
            ),
            mutable_state=mutable_state,
        )

        yield SimplifiedSolverTokenEnv(
            renderer=renderer,
            message_env=msg_env,
            prethink=env_state.prethink,
            failed_parse_reward=-1.0,
            max_trajectory_tokens=max_trajectory_tokens,
        )
