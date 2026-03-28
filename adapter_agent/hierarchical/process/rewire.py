import logging
from typing import cast

from coder_mcp.runtime import CoderMCPRuntimeError
from oai_utils.tinker import TinkerModel
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.rl.types import Trajectory, Transition

from adapter_agent.hierarchical.agent.knowledge_normalizer import (
    AgentsSDKModel,
    KnowledgeNormalizer,
    verify_normalized_knowledge,
)
from adapter_agent.hierarchical.agent.oc_rewriter import OCRewriter
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.rl.env.session_result import (
    RewireSessionResult,
    RewireSessionResultError,
    RewireSessionResultFailure,
    RewireSessionResultSuccess,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.knowledge_db import KnowledgeDB
from adapter_agent.rl.completer import TinkerMessageCompleter
from adapter_agent.rl.env.reward import LLMAsAJudge
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.env.simplified_solver import (
    SimplifiedSolverEnv,
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
    knowledge_db: KnowledgeDB,
    qwen_no_think: bool = False,
) -> RewireSessionResult:
    verifier = Verifier(model=verifier_model, rust_doc_analyzer=rust_doc_analyzer)
    ss_state = SimplifiedSolverEnvState.numrs2(task=task, qwen_no_think=qwen_no_think)
    try:
        async with runtime_settings.build_runtime() as runtime:
            async with build_simplified_solver_env(
                env_state=ss_state,
                renderer=solver_model.renderer,
                verifier=verifier,
                rust_doc_analyzer=verifier.rust_doc_analyzer,
                search_model=verifier_model,
                knowledge_db=knowledge_db,
                max_turns=max_turns,
                runtime=runtime,
            ) as env:
                msg_env = cast(SimplifiedSolverEnv, env.message_env)

                completer = TinkerMessageCompleter(
                    solver_model.sampling_client,
                    renderer=solver_model.renderer,
                    max_tokens=None,  # type: ignore
                )

                ob = await msg_env.initial_observation()
                transitions = []
                while True:
                    try:
                        ac_message = await completer(ob)
                        ac_with_logprobs = ac_message["tokens_with_logprobs"]
                    except MaximumContextExceeded:
                        logger.debug(
                            "Maximum context length exceeded during the rewiring loop"
                        )
                        ob.append(CONTEXT_EXCEEDED_MESSAGE)
                        trajectory = Trajectory(
                            transitions=transitions,
                            final_ob=solver_model.renderer.build_generation_prompt(ob),
                        )
                        return RewireSessionResultFailure(
                            task=task,
                            trials=ob,
                            conclusion="context_length_exceeded",
                            trajectory=trajectory,
                        )

                    model_input = solver_model.renderer.build_generation_prompt(ob)
                    step_result = await msg_env.step(ac_message)
                    transition = Transition(
                        ob=model_input,
                        ac=ac_with_logprobs,
                        reward=step_result.reward,
                        episode_done=step_result.episode_done,
                        metrics=getattr(step_result, "metrics", {}),
                        logs=getattr(step_result, "logs", {}),
                    )
                    transitions.append(transition)

                    ob = step_result.next_messages
                    if step_result.episode_done:
                        break

                trajectory = Trajectory(
                    transitions=transitions,
                    final_ob=solver_model.renderer.build_generation_prompt(ob),
                )

                # Always attempt to extract normalized knowledge from the trials
                logger.info("Extracting normalized knowledge from trajectory...")
                normalizer = KnowledgeNormalizer(model=verifier_model)
                normalized_knowledge = await normalizer.normalize(ob)

                if LLMAsAJudge.is_successful_reward(step_result.reward):
                    verification_result = await verify_normalized_knowledge(
                        runtime=runtime,
                        normalized_knowledge=normalized_knowledge,
                        model=verifier_model,
                    )

                    if not verification_result.success:
                        logger.warning(
                            f"Knowledge verification failed: {verification_result.reasoning}"
                        )
                        return RewireSessionResultFailure(
                            task=task,
                            trials=ob,
                            conclusion="verification_failed",
                            trajectory=trajectory,
                            reasoning=verification_result.reasoning,
                            knowledge=normalized_knowledge,
                        )

                    logger.info(
                        "Problem solved, recording verified knowledge to KnowledgeDB"
                    )
                    await knowledge_db.add_knowledge(
                        task.instruction, normalized_knowledge
                    )

                    # Perform OC Conversion
                    logger.info(
                        "Performing Open-to-Close (OC) Conversion on successful trajectory..."
                    )
                    try:
                        oc_trials = await oc_convert_trajectory(
                            ob, model=verifier_model, knowledge_db=knowledge_db
                        )
                    except Exception as e:
                        logger.error(f"Failed OC Conversion: {e}")
                        oc_trials = None

                    return RewireSessionResultSuccess(
                        task=task,
                        trials=ob,
                        oc_trials=oc_trials,
                        knowledge=normalized_knowledge,
                        trajectory=trajectory,
                        reasoning=verification_result.reasoning,
                    )
                else:
                    return RewireSessionResultFailure(
                        task=task,
                        trials=ob,
                        conclusion=step_result.conclusion,
                        trajectory=trajectory,
                        knowledge=normalized_knowledge,
                    )
    except (CodingEnvironmentError, CoderMCPRuntimeError) as e:
        logger.error(f"Environment error: {e}")
        return RewireSessionResultError(
            task=task,
            conclusion="environment_error",
        )


async def oc_convert_trajectory(
    messages: list[TinkerMessage],
    model: AgentsSDKModel,
    knowledge_db: KnowledgeDB,
) -> list[TinkerMessage]:
    """
    Scans a trajectory for knowledge retrieval turns and rewrites them using OCRewriter.
    This effectively converts an "Open Book" (search-based) trajectory into a
    "Closed Book" (recall-based) trajectory for SFT.
    """
    rewriter = OCRewriter(model=model)
    new_trajectory = list(messages)

    # 1. Identify roles and citation turns
    # We find tool results that have a knowledge_id.
    # The citation_turn_idx is the index of the assistant turn that called that tool (generally i-1).
    retrieval_indices = []
    for i, msg in enumerate(messages):
        # A TinkerMessage is a dict, and we added knowledge_id in SimplifiedSolverEnv.step (via ToolResult)
        if msg.get("role") == "tool" and msg.get("knowledge_id"):
            # O_i is msg, so A_i is i-1
            # We assume the turn before the tool result is the one that triggered the search.
            retrieval_indices.append(i - 1)

    # 2. Rewrite each identified triplet (Ai, Oi, Ai+1)
    # We iterate BACKWARDS to keep citation indices stable since we are shrinking the list
    # (replacing 3 messages with 1).
    for citation_turn_idx in sorted(retrieval_indices, reverse=True):
        # Get the knowledge_id from Oi
        Oi = new_trajectory[citation_turn_idx + 1]
        knowledge_id = cast(str, Oi.get("knowledge_id"))

        # Retrieve the actual knowledge content from DB
        knowledge_doc = await knowledge_db.get_knowledge_by_id(knowledge_id)
        if not knowledge_doc:
            logger.warning(
                f"Knowledge content not found for ID: {knowledge_id}. Skipping rewrite."
            )
            continue

        knowledge_content = knowledge_doc["content"]

        # Perform the rewrite which merges [Ai, Oi, Ai+1] into [Ai']
        new_trajectory = await rewriter.rewrite_trajectory(
            trajectory=new_trajectory,
            knowledge_id=knowledge_id,
            knowledge_content=knowledge_content,
            citation_turn_idx=citation_turn_idx,
        )

    return new_trajectory
