import logging

import tinker
from agents.extensions.models.litellm_model import LitellmModel
from coder_mcp.runtime import CoderMCPRuntimeError
from oai_utils.tinker import TinkerModel
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.renderers.base import TextPart
from tinker_cookbook.rl.types import TokensWithLogprobs, Trajectory, Transition
from adapter_agent.rl.env.simplified_solver import SSStepResult

from adapter_agent.hierarchical.agent.knowledge_normalizer import (
    AgentsSDKModel,
)
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.wiki_manager import WikiManager
from adapter_agent.rl.completer import LiteLLMMessageCompleter
from adapter_agent.rl.env.reward import LLMAsAJudge
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.env.session_result import (
    RewireSessionResult,
    RewireSessionResultError,
    RewireSessionResultFailure,
    RewireSessionResultSuccess,
)
from adapter_agent.rl.env.simplified_solver import (
    SimplifiedSolverEnvState,
    build_simplified_solver_msg_env,
)
from adapter_agent.util.exception import CodingEnvironmentError, MaximumContextExceeded

CONTEXT_EXCEEDED_MESSAGE = TinkerMessage(
    role="tool",
    content="The result of your action resulted in context length overflow.",
)
logger = logging.getLogger(__name__)


async def ss_solve_verify(
    solver_model: TinkerModel | LitellmModel,
    verifier_model: AgentsSDKModel,
    rust_doc_analyzer: AsyncRustDocAnalyzer,
    task: Task,
    max_turns: int,
    runtime_settings: RuntimeSettings,
    wiki_manager: WikiManager,
    qwen_no_think: bool = False,
    internalized_knowledge: str | None = None,
    blocked_knowledge_ids: set[str] | None = None,
) -> RewireSessionResult:
    verifier = Verifier(model=verifier_model)
    ss_state = SimplifiedSolverEnvState.numrs2(
        task=task,
        qwen_no_think=qwen_no_think,
        internalized_knowledge=internalized_knowledge,
        blocked_knowledge_ids=blocked_knowledge_ids,
    )
    try:
        renderer = (
            solver_model.renderer if isinstance(solver_model, TinkerModel) else None
        )

        msg_env = await build_simplified_solver_msg_env(
            env_state=ss_state,
            verifier=verifier,
            rust_doc_analyzer=rust_doc_analyzer,
            search_model=verifier_model,
            wiki_manager=wiki_manager,
            max_turns=max_turns,
            runtime_settings=runtime_settings,
            renderer=renderer,
        )

        ob_messages = await msg_env.initial_observation()
        internalized_knowledge = ss_state.internalized_knowledge

        if isinstance(solver_model, TinkerModel):
            assert renderer is not None, "TinkerModel must have a renderer"
            transitions = []
            stop_condition = renderer.get_stop_sequences()
            while True:
                prefill = internalized_knowledge
                internalized_knowledge = None

                model_input = renderer.build_generation_prompt(
                    ob_messages, prefill=prefill
                )
                try:
                    sample_result = await solver_model.sampling_client.sample_async(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=tinker.SamplingParams(
                            stop=stop_condition,
                            temperature=1.0,
                        ),
                    )
                    sampled_tokens = sample_result.sequences[0].tokens
                    maybe_logprobs = sample_result.sequences[0].logprobs
                except tinker.APIStatusError as e:
                    if (
                        e.status_code == 400
                        and "Prompt length plus max_tokens exceeds the model's context window"
                        in e.message
                    ):
                        logger.debug(
                            "Maximum context length exceeded during the rewiring loop"
                        )
                        msg_env.history.append(CONTEXT_EXCEEDED_MESSAGE)
                        trajectory = Trajectory(
                            transitions=transitions,
                            final_ob=renderer.build_generation_prompt(msg_env.history),
                        )
                        return RewireSessionResultFailure(
                            task=task,
                            trials=msg_env.history,
                            conclusion="context_length_exceeded",
                            trajectory=trajectory,
                        )
                    else:
                        raise

                assistant_message, parse_success = renderer.parse_response(
                    sampled_tokens
                )

                if not parse_success:
                    reward = -1.0
                    episode_done = True
                    metrics = {"parse_error": 1.0}
                    transition = Transition(
                        ob=model_input,
                        ac=TokensWithLogprobs(
                            tokens=sampled_tokens, maybe_logprobs=maybe_logprobs
                        ),
                        reward=reward,
                        episode_done=episode_done,
                        metrics=metrics,
                        logs={},
                    )
                    transitions.append(transition)
                    # Create a mock step result to pass to _process_solve_result

                    step_result = SSStepResult(
                        reward=reward,
                        episode_done=episode_done,
                        next_messages=msg_env.history,
                        conclusion="parse_failed",
                    )
                    break

                if prefill:
                    if isinstance(assistant_message["content"], str):
                        assistant_message["content"] = [
                            TextPart(type="text", text=assistant_message["content"])
                        ]
                    assistant_message["content"] = [
                        TextPart(type="text", text=prefill),
                        *assistant_message["content"],
                    ]

                step_result = await msg_env.step(assistant_message)
                ob_messages = step_result.next_messages

                transition = Transition(
                    ob=model_input,
                    ac=TokensWithLogprobs(
                        tokens=sampled_tokens, maybe_logprobs=maybe_logprobs
                    ),
                    reward=step_result.reward,
                    episode_done=step_result.episode_done,
                    metrics=step_result.metrics,
                    logs={},
                )
                transitions.append(transition)

                if step_result.episode_done:
                    break

            trajectory = Trajectory(
                transitions=transitions,
                final_ob=renderer.build_generation_prompt(msg_env.history),
            )
            trials = msg_env.history

        else:
            litellm_completer = LiteLLMMessageCompleter.from_litellm_model(solver_model)
            while True:
                try:
                    action = await litellm_completer(ob_messages)
                except MaximumContextExceeded:
                    logger.debug(
                        "Maximum context length exceeded during the rewiring loop"
                    )
                    msg_env.history.append(CONTEXT_EXCEEDED_MESSAGE)
                    return RewireSessionResultFailure(
                        task=task,
                        trials=msg_env.history,
                        conclusion="context_length_exceeded",
                        trajectory=None,
                    )

                step_result = await msg_env.step(action)
                ob_messages = step_result.next_messages

                if step_result.episode_done:
                    break

            trials = msg_env.history
            trajectory = None

        if not LLMAsAJudge.is_successful_reward(step_result.reward):
            return RewireSessionResultFailure(
                task=task,
                trials=trials,
                conclusion=step_result.conclusion,
                trajectory=trajectory,
            )

        return RewireSessionResultSuccess(
            task=task,
            trials=trials,
            trajectory=trajectory,
        )

    except (CodingEnvironmentError, CoderMCPRuntimeError) as e:
        logger.error(f"Environment error: {e}")
        return RewireSessionResultError(
            task=task,
            conclusion="environment_error",
        )


