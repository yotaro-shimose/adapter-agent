from dataclasses import dataclass

import tinker
from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs


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
