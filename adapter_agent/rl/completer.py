from dataclasses import dataclass
from typing import Self, cast

import tinker
from agents.extensions.models.litellm_model import LitellmModel
from litellm import acompletion
from tinker_cookbook import renderers
from tinker_cookbook.completers import (
    MessageCompleter,
    StopCondition,
    TokenCompleter,
    TokensWithLogprobs,
)

from adapter_agent.data import TinkerMessage
from adapter_agent.util.exception import MaximumContextExceeded


class MessageWithLogprobs(renderers.Message):
    tokens_with_logprobs: TokensWithLogprobs


@dataclass
class TinkerTokenCompleter(TokenCompleter):
    """
    The most standard TokenCompleter, which uses a tinker.SamplingClient to sample actions.
    """

    sampling_client: tinker.SamplingClient
    max_tokens: int | None
    temperature: float = 1.0

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        """Sample an action from the policy given an observation."""
        # Sample from the model
        sample_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )

        # Extract tokens and logprobs from the first (and only) sample
        sampled_tokens = sample_result.sequences[0].tokens
        sampled_logprobs = sample_result.sequences[0].logprobs
        assert sampled_logprobs is not None

        return TokensWithLogprobs(
            tokens=sampled_tokens, maybe_logprobs=sampled_logprobs
        )


class TinkerMessageCompleter(MessageCompleter):
    """A custom completer that uses the actual model to generate responses and retains tools."""

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        max_tokens: int | None,
        stop_condition: StopCondition | None = None,
        temperature: float = 1.0,
    ):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.max_tokens = max_tokens
        self.temperature = temperature
        if stop_condition is None:
            self.stop_condition = self.renderer.get_stop_sequences()
        else:
            self.stop_condition = stop_condition

    async def __call__(self, messages: list[renderers.Message]) -> MessageWithLogprobs:
        # Render the conversation for the model
        model_input = self.renderer.build_generation_prompt(messages)

        # Sample from the model
        try:
            response = await self.sampling_client.sample_async(
                model_input,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=self.stop_condition,
                ),
            )
        except tinker.APIStatusError as e:
            if (
                e.status_code == 400
                and "Prompt length plus max_tokens exceeds the model's context window"
                in e.message
            ):
                raise MaximumContextExceeded(e.message) from e
            else:
                raise
        except Exception:
            raise
        # Decode the response
        parsed_message, _success = self.renderer.parse_response(
            response.sequences[0].tokens
        )

        parsed_message_with_logprobs = cast(MessageWithLogprobs, parsed_message)
        parsed_message_with_logprobs["tokens_with_logprobs"] = TokensWithLogprobs(
            tokens=response.sequences[0].tokens,
            maybe_logprobs=response.sequences[0].logprobs,
        )

        return parsed_message_with_logprobs


class LiteLLMMessageCompleter(MessageCompleter):
    """A custom completer that uses Litellm to generate responses, maintaining a similar interface."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None,
        max_tokens: int | None,
        temperature: float = 1.0,
        stop_condition: StopCondition | None = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop_condition = stop_condition

    @classmethod
    def from_litellm_model(
        cls,
        model: LitellmModel,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        stop_condition: StopCondition | None = None,
    ) -> Self:
        return cls(
            model_name=model.model,
            api_key=model.api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_condition=stop_condition,
        )

    async def __call__(self, messages: list[TinkerMessage]) -> MessageWithLogprobs:
        litellm_messages = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                content = "".join(
                    [part["text"] for part in content if part["type"] == "text"]
                )
            litellm_messages.append(
                {"role": msg.get("role", "user"), "content": content}
            )

        try:
            response = await acompletion(  # type: ignore
                model=self.model_name,
                api_key=self.api_key,
                messages=litellm_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=list(self.stop_condition) if self.stop_condition else None,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if (
                "context length" in err_msg
                or "context window" in err_msg
                or (
                    hasattr(e, "status_code")
                    and getattr(e, "status_code") == 400
                    and "exceed" in err_msg
                )
            ):
                raise MaximumContextExceeded(str(e)) from e
            raise

        text_response = response.choices[0].message.content or ""

        parsed_message_with_logprobs = cast(
            MessageWithLogprobs, {"role": "assistant", "content": text_response}
        )
        parsed_message_with_logprobs["tokens_with_logprobs"] = TokensWithLogprobs(
            tokens=[],
            maybe_logprobs=None,
        )

        return parsed_message_with_logprobs
