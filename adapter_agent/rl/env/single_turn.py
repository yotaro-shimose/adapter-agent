import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Self

import tinker
from coder_mcp.runtime import RustCodingEnvironment
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.renderers.base import TextPart, ThinkingPart
from tinker_cookbook.rl import types
from tinker_cookbook.rl.types import Env
from tinker_cookbook.tool_use.agent_tool_message_env import RewardFn
from typing_extensions import AsyncGenerator

from adapter_agent.hierarchical.agent.solver import CONTEXT
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.env.reward import LLMAsAJudgeSingleTurn
from adapter_agent.util.parsing import extract_rust_code

logger = logging.getLogger(__name__)


@dataclass
class SingleTurnEnvState:
    task: Task
    image_name: str
    library_name: str
    prethink: str | None
    messages: list[TinkerMessage]

    @classmethod
    def numrs2(cls, task: Task, prethink: str | None = None) -> Self:
        return cls(
            task=task,
            library_name="numrs2",
            image_name="coder-mcp-numrs2:latest",
            prethink=prethink,
            messages=[],
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
class SingleTurnEnv(Env):
    initial_state: SingleTurnEnvState
    rust_env: RustCodingEnvironment
    renderer: Renderer
    initial_messages: list[TinkerMessage]
    reward_fn: RewardFn
    history: list[TinkerMessage] = field(default_factory=list)
    prethink: str | None = None

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        if not self.history:
            self.history = self.initial_messages

        prefill = f"<think>{self.prethink}</think>" if self.prethink else None
        return (
            self.renderer.build_generation_prompt(
                self.initial_messages, prefill=prefill
            ),
            self.renderer.get_stop_sequences(),
        )

    async def step(self, action: types.Action) -> types.StepResult:
        """Parse tokens to a message, delegate to MessageEnv, and render response."""
        assistant_message, parse_success = self.renderer.parse_response(action)
        logger.debug(f"Assistant message: {assistant_message}")

        if not parse_success:
            return types.StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.renderer.get_stop_sequences(),
                metrics={"parse_error": 1.0},
            )

        if self.prethink:
            if isinstance(assistant_message["content"], str):
                assistant_message["content"] = [
                    TextPart(type="text", text=assistant_message["content"])
                ]
            assistant_cotent = [
                ThinkingPart(
                    type="thinking",
                    thinking=self.prethink,
                ),
                *assistant_message["content"],
            ]
            assistant_message["content"] = assistant_cotent
            self.prethink = None
        self.history.append(assistant_message)
        return await self.process_message(assistant_message)

    async def process_message(self, message: TinkerMessage) -> types.StepResult:
        if isinstance(message["content"], str):
            text_content = message["content"]
        else:
            text_content = ""
            for part in message["content"]:
                if part["type"] == "text":
                    text_content += part["text"]
        if not text_content:
            error_message = "No text content in the message."
            new_message = TinkerMessage(role="user", content=error_message)
            self.history.append(new_message)
            return types.StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=self.renderer.build_generation_prompt(self.history),
                next_stop_condition=self.renderer.get_stop_sequences(),
                metrics=dict(
                    no_text_content=1.0,
                    no_code_found=0.0,
                    code_did_not_compile=0.0,
                    verifier_failed=0.0,
                    verifier_error=0.0,
                ),
            )
        code = extract_rust_code(text_content)
        if not code:
            error_message = "No code found in the message. Make sure you wrap the code in ```rust ... ```."
            new_message = TinkerMessage(role="user", content=error_message)
            self.history.append(new_message)
            return types.StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=self.renderer.build_generation_prompt(self.history),
                next_stop_condition=self.renderer.get_stop_sequences(),
                metrics=dict(
                    no_text_content=0.0,
                    no_code_found=1.0,
                    code_did_not_compile=0.0,
                    verifier_failed=0.0,
                    verifier_error=0.0,
                ),
            )

        await self.rust_env.set_content("src/main.rs", code)
        output, success = await self.rust_env.run_cargo()
        new_message = TinkerMessage(role="user", content=output)
        self.history.append(new_message)
        if not success:
            return types.StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=self.renderer.build_generation_prompt(self.history),
                next_stop_condition=self.renderer.get_stop_sequences(),
                metrics=dict(
                    no_text_content=0.0,
                    no_code_found=0.0,
                    code_did_not_compile=1.0,
                    verifier_failed=0.0,
                    verifier_error=0.0,
                ),
            )
        reward, reward_metrics = await self.reward_fn(self.history)
        return types.StepResult(
            reward=reward,
            episode_done=True,
            next_observation=self.renderer.build_generation_prompt(self.history),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics=dict(
                no_text_content=0.0,
                no_code_found=0.0,
                **reward_metrics,
            ),
        )

    async def get_state(self) -> SingleTurnEnvState:
        return self.initial_state.with_messages(self.history)

    @classmethod
    async def from_env_state(
        cls,
        rust_env: RustCodingEnvironment,
        env_state: SingleTurnEnvState,
        renderer: Renderer,
        verifier: Verifier,
        max_trajectory_tokens: int = 32 * 1024,
    ) -> Self:
        exclude = ["target", ".git"]
        tree_structure = await rust_env.tree(".", exclude=exclude, truncate=20)

        return cls(
            initial_state=env_state,
            rust_env=rust_env,
            renderer=renderer,
            initial_messages=get_single_turn_initial_messages(
                env_state=env_state,
            ),
            reward_fn=LLMAsAJudgeSingleTurn(
                task=env_state.task,
                rust_env=rust_env,
                verifier=verifier,
                tree_structure=tree_structure,
            ),
            prethink=env_state.prethink,
        )


def get_single_turn_initial_messages(
    env_state: SingleTurnEnvState,
) -> list[TinkerMessage]:
    if env_state.messages:
        return env_state.messages
    system_prompt = f"""
<Role>
You are an expert Rust software engineer.
Your task is to solve user provided problem.
</Role>

{CONTEXT.format(library_name=env_state.library_name)}

<HowTo>
You DO NOT have access to library documents or source codes. You are expected to solve the problem based on your knowledge / recall.
Respond with a message including ```rust
<code here>
```
The code will be automatically pasted into src/main.rs, compiled and run. So that the code should be a complete program.
</HowTo>
"""

    initial_message = f"""
<Task>
{env_state.task.instruction}
</Task>
"""
    return [
        TinkerMessage(role="system", content=system_prompt),
        TinkerMessage(role="user", content=initial_message),
    ]


@asynccontextmanager
async def build_single_turn_env(
    env_state: SingleTurnEnvState,
    renderer: Renderer,
    verifier: Verifier,
    max_trajectory_tokens: int = 32 * 1024,
) -> AsyncGenerator[SingleTurnEnv, None]:
    async with RustCodingEnvironment(image_name=env_state.image_name) as rust_env:
        yield await SingleTurnEnv.from_env_state(
            rust_env=rust_env,
            env_state=env_state,
            renderer=renderer,
            verifier=verifier,
            max_trajectory_tokens=max_trajectory_tokens,
        )
