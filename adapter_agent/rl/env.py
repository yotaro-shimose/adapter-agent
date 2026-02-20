import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Self, cast

from coder_mcp.runtime import RustCodingEnvironment
from pydantic import ValidationError
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.types import Action, Env, Observation, StepResult
from tinker_cookbook.tool_use import tool
from tinker_cookbook.tool_use.agent_tool_message_env import (
    AgentToolMessageEnv,
    RewardFn,
)
from tinker_cookbook.tool_use.types import Tool

from adapter_agent.hierarchical.agent.solver import (
    CONTEXT,
    EFFICIENCY_GUIDELINES,
    report_success,
)
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.qra import QA

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
    async def run(self) -> str:
        """
        Run `cargo run` in the workspace. Returns the combined stdout/stderr output and exit code.
        """
        output, success = await self.rust_env.run_cargo()
        self.remaining_turns -= 1
        return self.with_remaining_turns(output)

    @tool
    async def str_replace(self, old_str: str, new_str: str) -> str:
        """
        Find and replace an exact string in src/main.rs. The file target is always src/main.rs. Returns error if string not found or multiple matches. Shows context snippet after edit.
        """
        output = await self.rust_env.str_replace(old_str, new_str)
        self.remaining_turns -= 1
        return self.with_remaining_turns(output)

    @tool
    async def report_success(self, question: str, answer: str) -> str:
        """
        Report success to the verifier.
        """
        self.remaining_turns -= 1
        return self.with_remaining_turns("")

    def with_remaining_turns(self, text: str) -> str:
        return f"{text}\n<RemainingTurns>{self.remaining_turns}</RemainingTurns>"


@dataclass
class EnvState:
    task: Task
    code: str
    max_turns: int
    remaining_turns: int  # If no action is taken, the turn is 0
    image_name: str
    library_name: str
    messages: list[TinkerMessage] | None

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
            code=CARGO_INIT_MAIN_RS,
            max_turns=max_turns,
            remaining_turns=max_turns,
            image_name=image_name,
            library_name=library_name,
            messages=None,
        )


@dataclass
class RustCoderEnv(Env):
    initial_state: EnvState
    rust_env: RustCodingEnvironment
    internal: EnvFromMessageEnv

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return await self.internal.initial_observation()

    async def step(self, action: Action) -> StepResult:
        return await self.internal.step(action)

    async def get_state(self) -> EnvState:
        message_env: AgentToolMessageEnv = cast(
            AgentToolMessageEnv, self.internal.message_env
        )
        code = await self.rust_env.view_file("src/main.rs")
        return EnvState(
            task=self.initial_state.task,
            code=code,
            max_turns=self.initial_state.max_turns,
            remaining_turns=message_env._turn_count,
            image_name=self.initial_state.image_name,
            library_name=self.initial_state.library_name,
            messages=message_env.history,
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
        tools = [tool_set.run, tool_set.str_replace, tool_set.report_success]
        internal = build_agent_tool_env(
            renderer=renderer,
            tools=tools,  # type: ignore
            initial_messages=_get_initial_messages(
                env_state=env_state,
                tree_structure=tree_structure,
            ),
            reward_fn=LLMAsAJudge(
                rust_env=rust_env,
                verifier=verifier,
                tree_structure=tree_structure,
            ),
            max_trajectory_tokens=max_trajectory_tokens,
            max_turns=env_state.remaining_turns,
            turn_count=env_state.remaining_turns,
        )

        return cls(
            initial_state=env_state,
            rust_env=rust_env,
            internal=internal,
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
) -> list[TinkerMessage]:
    if env_state.messages is not None:
        return env_state.messages
    system_prompt = f"""
<Role>
You are an expert Rust software engineer.
Your task is to answer user provided question with verified solution.
</Role>

{CONTEXT.format(library_name=env_state.library_name)}

<HowTo>
You have a simplified coding environment with the following tools:
- `str_replace`: Edit src/main.rs by finding and replacing an exact string. The target file is always src/main.rs.
- `run`: Run `cargo run` to compile and execute the code.
- `search_docs`: Use this to find functionality keywords, concepts, or how-to guides in the documentation.
- `search_symbol`: Use this to find specific types, functions, or traits by name.

You can access library documents via search tools.
Once you have created a solution, you must run the code and confirm it works (to the best of your ability).
You must end with a call to `report_success` to create your final answer to the question.
</HowTo>

{EFFICIENCY_GUIDELINES}

<Guidelines>
You can edit src/main.rs using str_replace tool. The current content of src/main.rs is provided below.
You have turn limit of {env_state.max_turns}.
If you hit the turn limit without reporting success, it will be considered as a failure.
Note your solution has to be fully self-contained including both source code and explanation so that we can verify the solution later.
Verification will solely based on your final solution and your source code in the coding environment will be discarded in verification.
You should not use release build for faster debugging.
</Guidelines>
"""

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
    return [
        TinkerMessage(role="user", content=system_prompt),
        TinkerMessage(role="assistant", content=initial_message),
    ]


@dataclass
class LLMAsAJudge:
    rust_env: RustCodingEnvironment
    verifier: Verifier
    tree_structure: str

    async def __call__(
        self, history: list[TinkerMessage]
    ) -> tuple[float, dict[str, float]]:
        execution_output, success = await self.rust_env.run_cargo()
        if not success:
            return 0.0, {}
        if len(history) == 0:
            return 0.0, {}
        final_message = history[-1]
        if "tool_calls" not in final_message:
            return 0.0, {}
        final_tool_calls = final_message["tool_calls"]
        if len(final_tool_calls) == 0:
            return 0.0, {}
        report_success_function_call = None
        for tool_call in final_tool_calls:
            if report_success.name == tool_call.function.name:
                report_success_function_call = tool_call
                break

        if report_success_function_call is None:
            return 0.0, {}

        try:
            report_success_args = QA.model_validate_json(
                report_success_function_call.function.arguments
            )
            content = await self.rust_env.view_file("src/main.rs")
            verification_result = await self.verifier.verify(
                qa=report_success_args,
                tree_structure=self.tree_structure,
                execution_output=execution_output,
                main_rs_content=content,
            )
            if verification_result.success:
                return 1.0, {}
            else:
                return 0.0, {}

        except ValidationError as e:
            logger.debug(f"Failed to parse report_success arguments: {e}")
            return 0.0, {}

    @classmethod
    def is_successful_reward(cls, reward: float) -> bool:
        return reward > 0.0


def build_agent_tool_env(
    renderer: Renderer,
    tools: list[Tool],
    initial_messages: list[Message],
    reward_fn: RewardFn,
    *,
    max_turns: int = 5,
    failed_parse_reward: float = -0.1,
    max_trajectory_tokens: int | None = None,
    turn_count: int = 0,
) -> EnvFromMessageEnv:
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
    return EnvFromMessageEnv(
        renderer=renderer,
        message_env=msg_env,
        failed_parse_reward=failed_parse_reward,
        max_trajectory_tokens=max_trajectory_tokens,
    )
