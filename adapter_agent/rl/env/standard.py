import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Self

from coder_mcp.runtime import RustCodingEnvironment
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import Message, ToolSpec
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.rl.types import Action, Env, Observation, StepResult
from tinker_cookbook.tool_use import AgentToolMessageEnv, tool
from tinker_cookbook.tool_use.agent_tool_message_env import RewardFn
from tinker_cookbook.tool_use.tools import FunctionTool
from tinker_cookbook.tool_use.types import Tool, ToolResult

from adapter_agent.hierarchical.agent.solver import (
    CONTEXT,
    EFFICIENCY_GUIDELINES,
)
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.env.injection import _inject_tools_into_prompt
from adapter_agent.rl.env.prefillable_env import PrefillableMessageEnv
from adapter_agent.rl.env.reward import LLMAsAJudge
from adapter_agent.util.exception import CodingEnvironmentError

logger = logging.getLogger(__name__)

CARGO_INIT_MAIN_RS = """\
fn main() {
    println!("Hello, world!");
}
"""


@dataclass
class CoderEnvTool:
    remaining_turns: int
    rust_env: RustCodingEnvironment

    @tool
    async def run(self) -> ToolResult:
        """
        Run `cargo run` in the workspace. Returns the combined stdout/stderr output and exit code.
        """
        output, success = await self.rust_env.run_cargo()
        self.remaining_turns -= 1
        ret_str = self.with_remaining_turns(output)
        return ToolResult(messages=[TinkerMessage(role="tool", content=ret_str)])

    @tool
    async def str_replace(self, old_str: str, new_str: str) -> ToolResult:
        """
        Find and replace an exact string in src/main.rs. The file target is always src/main.rs. Returns error if string not found or multiple matches. Shows context snippet after edit.
        """
        output, success = await self.rust_env.str_replace(old_str, new_str)
        self.remaining_turns -= 1
        return ToolResult(
            messages=[
                TinkerMessage(role="tool", content=self.with_remaining_turns(output))
            ]
        )

    @tool
    async def str_replace_and_run(self, old_str: str, new_str: str) -> ToolResult:
        """
        Find and replace an exact string in src/main.rs. The file target is always src/main.rs. Returns error if string not found or multiple matches. Shows context snippet after edit.
        """
        replace_ret, replace_success = await self.rust_env.str_replace(old_str, new_str)
        if not replace_success:
            return ToolResult(
                messages=[
                    TinkerMessage(
                        role="tool", content=self.with_remaining_turns(replace_ret)
                    )
                ]
            )

        run_ret, run_success = await self.rust_env.run_cargo()
        self.remaining_turns -= 1
        combined_output = f"""\
<TextReplacementPerformed>
{replace_ret}
</TextReplacementPerformed>
<CargoRunResult>
{run_ret}
</CargoRunResult>
"""

        ret_str = self.with_remaining_turns(combined_output)
        return ToolResult(messages=[TinkerMessage(role="tool", content=ret_str)])

    @tool
    async def report_success(self, answer: str) -> ToolResult:
        """
        Report success to the verifier.
        """
        self.remaining_turns -= 1
        return ToolResult(
            messages=[TinkerMessage(role="tool", content="")], should_stop=True
        )

    def with_remaining_turns(self, text: str) -> str:
        return f"{text}\n<RemainingTurns>{self.remaining_turns}</RemainingTurns>"


@dataclass
class EnvStateBase:
    task: Task
    code_history: list[
        str
    ]  # The history of code for every turn (we add new entry even if no changes are made)
    max_turns: int
    remaining_turns: int  # If no action is taken, the turn is 0
    image_name: str
    library_name: str


@dataclass
class InitEnvState(EnvStateBase):
    messages: None
    prethink: None

    @classmethod
    def numrs2(
        cls,
        task: Task,
        max_turns: int,
        library_name: str = "numrs2",
        image_name: str = "coder-mcp-numrs2:latest",
    ) -> Self:
        return cls(
            task=task,
            code_history=[CARGO_INIT_MAIN_RS],
            max_turns=max_turns,
            remaining_turns=max_turns,
            image_name=image_name,
            library_name=library_name,
            messages=None,
            prethink=None,
        )


@dataclass
class ResumedEnvState(EnvStateBase):
    messages: list[TinkerMessage]
    prethink: str | None


type EnvState = InitEnvState | ResumedEnvState


@dataclass
class RustCoderEnv(Env):
    initial_state: EnvStateBase
    rust_env: RustCodingEnvironment
    internal: PrefillableMessageEnv
    code_history: list[str]

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        try:
            return await self.internal.initial_observation()
        except Exception as e:
            raise CodingEnvironmentError(f"Failed to get initial observation: {e}")

    async def step(self, action: Action) -> StepResult:
        try:
            result = await self.internal.step(action)
            code = await self.rust_env.view_file("src/main.rs")
            self.code_history.append(code)
            return result
        except Exception as e:
            raise CodingEnvironmentError(f"Failed to step: {e}")

    async def get_state(self) -> ResumedEnvState:
        message_env = self.internal.message_env
        return ResumedEnvState(
            task=self.initial_state.task,
            code_history=self.code_history,
            max_turns=self.initial_state.max_turns,
            remaining_turns=message_env._turn_count,
            image_name=self.initial_state.image_name,
            library_name=self.initial_state.library_name,
            messages=message_env.history,
            prethink=None,
        )

    @classmethod
    async def from_env_state(
        cls,
        rust_env: RustCodingEnvironment,
        env_state: EnvState,
        renderer: Renderer,
        verifier: Verifier,
        max_trajectory_tokens: int = 32 * 1024,
    ) -> Self:
        tool_set = CoderEnvTool(
            remaining_turns=env_state.remaining_turns, rust_env=rust_env
        )
        exclude = ["target", ".git"]
        tree_structure = await rust_env.tree(".", exclude=exclude, truncate=20)
        await rust_env.set_content("src/main.rs", env_state.code_history[-1])

        tools: list[Tool] = [
            # tool_set.run,
            # tool_set.str_replace,
            tool_set.str_replace_and_run,
            tool_set.report_success,
        ]
        internal = build_agent_tool_env(
            renderer=renderer,
            tools=tools,
            initial_messages=_get_initial_messages(
                env_state=env_state,
                tree_structure=tree_structure,
                tools=[t.to_spec() for t in tools],
                renderer=renderer,
            ),
            reward_fn=LLMAsAJudge(
                task=env_state.task,
                rust_env=rust_env,
                verifier=verifier,
                tree_structure=tree_structure,
            ),
            prethink=env_state.prethink,
            max_trajectory_tokens=max_trajectory_tokens,
            max_turns=env_state.remaining_turns,
            turn_count=env_state.max_turns - env_state.remaining_turns,
        )

        return cls(
            initial_state=env_state,
            rust_env=rust_env,
            internal=internal,
            code_history=env_state.code_history,
        )


@asynccontextmanager
async def build_coder_env(
    env_state: EnvState,
    renderer: Renderer,
    verifier: Verifier,
    max_trajectory_tokens: int = 32 * 1024,
) -> AsyncGenerator[RustCoderEnv, None]:
    async with RustCodingEnvironment(image_name=env_state.image_name) as rust_env:
        yield await RustCoderEnv.from_env_state(
            rust_env=rust_env,
            env_state=env_state,
            renderer=renderer,
            verifier=verifier,
            max_trajectory_tokens=max_trajectory_tokens,
        )


def _get_initial_messages(
    env_state: EnvState,
    tree_structure: str,
    tools: list[ToolSpec],
    renderer: Renderer,
) -> list[TinkerMessage]:
    if env_state.messages is not None:
        return env_state.messages
    system_prompt = f"""
<Role>
You are an expert Rust software engineer.
Your task is to solve user provided problem.
</Role>

{CONTEXT.format(library_name=env_state.library_name)}

<HowTo>
You have a simplified coding environment with one tool:
- `str_replace_and_run`: Edit src/main.rs by finding and replacing an exact string. The target file is always src/main.rs. The code is automatically compiled and run, returning the combined stdout/stderr output and exit code.
You DO NOT have access to library documents or source codes. You are expected to solve the problem based on your knowledge / recall.
You must end with a call to `report_success` to create your final answer to the question.
</HowTo>

{EFFICIENCY_GUIDELINES}

<Guidelines>
You can only edit src/main.rs. The current content of src/main.rs is provided below.
You must not read documents or source codes of the library.
You have turn limit of {env_state.max_turns}.
If you hit the turn limit without reporting success, it will be considered as a failure.
Note your solution has to be fully self-contained including both full source code and a detailed explanation so that we can verify the solution later.
Verification will solely based on your final solution and your source code in the coding environment will be discarded in verification.
</Guidelines>
"""
    system_prompt_with_tools = _inject_tools_into_prompt(
        renderer, [tool for tool in tools], system_prompt
    )
    initial_message = f"""
<Task>
{env_state.task.instruction}
</Task>

<Current Directory Structure>
{tree_structure}
</Current Directory Structure>

<Current src/main.rs>
{CARGO_INIT_MAIN_RS}
</Current src/main.rs>

"""
    return system_prompt_with_tools + [
        TinkerMessage(role="user", content=initial_message),
    ]


def build_agent_tool_env(
    renderer: Renderer,
    tools: list[Tool | FunctionTool],
    initial_messages: list[Message],
    reward_fn: RewardFn,
    *,
    prethink: str | None,
    max_turns: int = 5,
    failed_parse_reward: float = -0.1,
    max_trajectory_tokens: int | None = None,
    turn_count: int = 0,
) -> PrefillableMessageEnv:
    """Convenience method to build an EnvFromMessageEnv for tool-using agents.

    Args:
        renderer: The renderer for tokenizing messages.
        tools: List of tools the agent can call (must implement Tool protocol).
        initial_messages: Initial conversation history (system prompt, user message, etc.).
        reward_fn: Function that grades a completed episode. Takes the full message
            history and returns (reward, metrics). Called once at episode end.
        max_turns: Maximum turns before episode ends.
        failed_parse_reward: Reward when model output fails to parse.
        max_trajectory_tokens: Maximum tokens in trajectory before terminating episode.

    Returns:
        An EnvFromMessageEnv ready for RL training.
    """
    msg_env = AgentToolMessageEnv(
        tools=tools,
        initial_messages=initial_messages,
        max_turns=max_turns,
        reward_fn=reward_fn,
        _turn_count=turn_count,
    )
    return PrefillableMessageEnv(
        renderer=renderer,
        message_env=msg_env,
        failed_parse_reward=failed_parse_reward,
        max_trajectory_tokens=max_trajectory_tokens,
        prethink=prethink,
    )
