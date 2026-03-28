import tinker
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import TextPart, ThinkingPart
from tinker_cookbook.rl import ActionExtra, types
from tinker_cookbook.tool_use import AgentToolMessageEnv


class PrefillableMessageEnv(types.Env):
    """Adapter that wraps a MessageEnv to implement the token-level Env interface.

    This bridges the message-level abstraction to the token-level interface
    expected by the RL training loop.

    We additionally enable prefilling, which allows us to prefill the model with
    some text before the first token is generated.
    The prefill might better serve as the field of AgentToolMessageEnv, but we keep it here for ease of implementation.
    """

    def __init__(
        self,
        renderer: Renderer,
        message_env: AgentToolMessageEnv,
        failed_parse_reward: float = -1.0,
        terminate_on_parse_error: bool = True,
        max_trajectory_tokens: int | None = None,
        prethink: str | None = None,
    ):
        self.renderer = renderer
        self.message_env = message_env
        self.failed_parse_reward = failed_parse_reward
        self.terminate_on_parse_error = terminate_on_parse_error
        self.max_trajectory_tokens = max_trajectory_tokens
        self._base_stop_condition = renderer.get_stop_sequences()
        self.prethink = prethink

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        messages = await self.message_env.initial_observation()
        prefill = f"<think>{self.prethink}</think>" if self.prethink else None
        return (
            self.renderer.build_generation_prompt(messages, prefill=prefill),
            self._base_stop_condition,
        )

    async def step(
        self, action: types.Action, *, extra: ActionExtra | None = None
    ) -> types.StepResult:
        """Parse tokens to a message, delegate to MessageEnv, and render response."""
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
            assistant_cotent = [
                ThinkingPart(
                    type="thinking",
                    thinking=self.prethink,
                ),
                *assistant_message["content"],
            ]
            assistant_message["content"] = assistant_cotent
            self.prethink = None

        msg_step = await self.message_env.step(assistant_message)
        next_observation = self.renderer.build_generation_prompt(msg_step.next_messages)
        next_stop_condition = msg_step.next_stop_condition or self._base_stop_condition

        # Check if trajectory exceeds max token limit
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
