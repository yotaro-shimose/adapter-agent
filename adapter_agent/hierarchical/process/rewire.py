import logging
from typing import Any, cast

import tinker
from coder_mcp.runtime import CoderMCPRuntimeError
from oai_utils.tinker import TinkerModel
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.rl.types import TokensWithLogprobs, Trajectory, Transition

from adapter_agent.hierarchical.agent.reflector import Reflector
from adapter_agent.hierarchical.agent.knowledge_normalizer import (
    AgentsSDKModel,
)
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.wiki_manager import WikiManager
from adapter_agent.rl.env.reward import LLMAsAJudge
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.env.session_result import (
    Citation,
    RewireSessionResult,
    RewireSessionResultError,
    RewireSessionResultFailure,
    RewireSessionResultSuccess,
)
from adapter_agent.rl.env.simplified_solver import (
    SimplifiedSolverEnvState,
    build_simplified_solver_env,
)
from adapter_agent.util.exception import CodingEnvironmentError, MaximumContextExceeded

CONTEXT_EXCEEDED_MESSAGE = TinkerMessage(
    role="tool",
    content="The result of your action resulted in context length overflow.",
)
logger = logging.getLogger(__name__)


async def ss_solve_verify(
    solver_model: TinkerModel,
    verifier_model: AgentsSDKModel,
    rust_doc_analyzer: AsyncRustDocAnalyzer,
    task: Task,
    max_turns: int,
    runtime_settings: RuntimeSettings,
    wiki_manager: WikiManager,
    qwen_no_think: bool = False,
    internalized_knowledge: str | None = None,
    blocked_knowledge_ids: set[str] | None = None,
    include_wiki: bool = True,
) -> RewireSessionResult:
    verifier = Verifier(model=verifier_model, rust_doc_analyzer=rust_doc_analyzer)
    ss_state = SimplifiedSolverEnvState.numrs2(
        task=task,
        qwen_no_think=qwen_no_think,
        internalized_knowledge=internalized_knowledge,
        blocked_knowledge_ids=blocked_knowledge_ids,
    )
    try:
        async with runtime_settings.build_runtime() as runtime:
            async with build_simplified_solver_env(
                env_state=ss_state,
                renderer=solver_model.renderer,
                verifier=verifier,
                rust_doc_analyzer=verifier.rust_doc_analyzer,
                search_model=verifier_model,
                wiki_manager=wiki_manager,
                max_turns=max_turns,
                runtime=runtime,
                include_wiki=include_wiki,
            ) as env:
                # Use TokenEnv directly to handle prefilling
                ob, stop = await env.initial_observation()
                transitions = []
                while True:
                    try:
                        # Sample from the model
                        # We use sampling_client directly because ob is ModelInput (which supports prefill)
                        sample_result = await solver_model.sampling_client.sample_async(
                            prompt=ob,
                            num_samples=1,
                            sampling_params=tinker.SamplingParams(
                                stop=stop,
                                temperature=1.0,
                            ),
                        )
                        sampled_tokens = sample_result.sequences[0].tokens
                        ac_with_logprobs = TokensWithLogprobs(
                            tokens=sampled_tokens,
                            maybe_logprobs=sample_result.sequences[0].logprobs,
                        )
                    except MaximumContextExceeded:
                        logger.debug(
                            "Maximum context length exceeded during the rewiring loop"
                        )
                        # Fallback to message env for recording the failure
                        msg_env = env.message_env
                        msg_env.history.append(CONTEXT_EXCEEDED_MESSAGE)
                        trajectory = Trajectory(
                            transitions=transitions,
                            final_ob=solver_model.renderer.build_generation_prompt(
                                msg_env.history
                            ),
                        )
                        return RewireSessionResultFailure(
                            task=task,
                            trials=msg_env.history,
                            conclusion="context_length_exceeded",
                            trajectory=trajectory,
                        )

                    model_input_before = ob
                    step_result = await env.step(sampled_tokens)
                    transition = Transition(
                        ob=model_input_before,
                        ac=ac_with_logprobs,
                        reward=step_result.reward,
                        episode_done=step_result.episode_done,
                        metrics=getattr(step_result, "metrics", {}),
                        logs=getattr(step_result, "logs", {}),
                    )
                    transitions.append(transition)
                    ob = step_result.next_observation
                    stop = step_result.next_stop_condition

                    if step_result.episode_done:
                        break

                msg_env = env.message_env
                trajectory = Trajectory(
                    transitions=transitions,
                    final_ob=solver_model.renderer.build_generation_prompt(
                        msg_env.history
                    ),
                )

                return await _process_solve_result(
                    task=task,
                    history=msg_env.history,
                    trajectory=trajectory,
                    step_result=step_result,
                    verifier_model=verifier_model,
                    wiki_manager=wiki_manager,
                    runtime=runtime,
                )

    except (CodingEnvironmentError, CoderMCPRuntimeError) as e:
        logger.error(f"Environment error: {e}")
        return RewireSessionResultError(
            task=task,
            conclusion="environment_error",
        )


async def _process_solve_result(
    task: Task,
    history: list[TinkerMessage],
    trajectory: Trajectory,
    step_result: Any, # SimplifiedSolverEnvStepResult
    verifier_model: AgentsSDKModel,
    wiki_manager: WikiManager,
    runtime: Any, # CoderMCPRuntime
) -> RewireSessionResult:
    """
    Handles the post-solve logic: citation extraction, knowledge normalization,
    verification, and uniqueness checking.
    """
    # Extract citations from the trials
    citations = []
    for turn_idx, msg in enumerate(history):
        msg_dict = cast(dict[str, Any], msg)
        knowledge_id = msg_dict.get("knowledge_id")
        if knowledge_id:
            citations.append(
                Citation(
                    knowledge_id=cast(str, knowledge_id),
                    turn_index=turn_idx,
                    content=msg_dict.get("knowledge_content"),
                    title=msg_dict.get("knowledge_title"),
                )
            )

    if not LLMAsAJudge.is_successful_reward(step_result.reward):
        return RewireSessionResultFailure(
            task=task,
            trials=history,
            conclusion=step_result.conclusion,
            trajectory=trajectory,
            citations=citations,
        )

    # 1. Extract Reflections
    logger.info("Extracting reflections from trajectory...")
    reflector = Reflector(model=verifier_model)
    reflections = await reflector.reflect(history)

    if not reflections:
        logger.warning("No reflections extracted.")
        # We can still return success if the goal was reached, but knowledge extraction failed.
        # However, for consistency with the previous logic:
        return RewireSessionResultFailure(
            task=task,
            trials=history,
            conclusion="knowledge_normalization_failed",
            trajectory=trajectory,
            citations=citations,
        )

    logger.info(f"Extracted {len(reflections)} reflections for async distillation.")
    return RewireSessionResultSuccess(
        task=task,
        trials=history,
        knowledges=[], # Knowledges will be populated by Studier
        reflections=reflections,
        trajectory=trajectory,
        reasoning="Delegated to async KnowledgeStudier",
        citations=citations,
    )
