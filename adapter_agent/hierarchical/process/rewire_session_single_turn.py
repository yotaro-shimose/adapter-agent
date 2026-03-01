import logging
from dataclasses import dataclass

from oai_utils.tinker import TinkerModel
from tinker_cookbook.renderers import TextPart, ThinkingPart
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.rl.types import (
    Trajectory,
    Transition,
)

from adapter_agent.data import QRA
from adapter_agent.hierarchical.agent.rewirer import SingleTurnRewirer
from adapter_agent.hierarchical.agent.simplified_solver import SimplifiedSolver
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.process.rewire_session import (
    FloatMetrics,
    RewireSessionResult,
    get_total_reward,
    log_trajectory_if_debug,
    metrics_with_prefix,
)
from adapter_agent.hierarchical.process.solve_verify import solve_verify
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.completer import TinkerTokenCompleter
from adapter_agent.rl.env.reward import LLMAsAJudge
from adapter_agent.rl.env.single_turn import (
    SingleTurnEnvState,
    build_single_turn_env,
    get_single_turn_initial_messages,
)
from adapter_agent.rl.env.standard import EnvironmentError

logger = logging.getLogger(__name__)


@dataclass
class SolveVerifyTinkerSingleTurnResult:
    trajectory: Trajectory
    env_state: SingleTurnEnvState
    reward: float
    metrics: FloatMetrics

    def is_success(self) -> bool:
        return LLMAsAJudge.is_successful_reward(self.reward)


async def rewire_session_single_turn(
    solver_model: TinkerModel,
    verifier: Verifier,
    rewirer: SingleTurnRewirer,
    task: Task,
    max_rewire: int,
) -> RewireSessionResult:
    state = SingleTurnEnvState.numrs2(
        task=task,
    )
    try:
        ret = await solve_verify_tinker_single_turn(
            solver_model=solver_model,
            verifier=verifier,
            env_state=state,
        )
    except EnvironmentError as e:
        logger.error(f"Failed to solve: {e}")
        return RewireSessionResult.enviroment_error(task=task, error_message=str(e))
    log_trajectory_if_debug(ret.env_state.messages)
    init_metrics = metrics_with_prefix(ret.metrics, "first")

    if ret.is_success():
        logger.info("Successfully solved in first attempt")
        return RewireSessionResult.success(
            task=task,
            messages=ret.env_state.messages,
            num_rewire=0,
            metrics=init_metrics,
        )
    # Rewire
    for i in range(max_rewire):
        solver = SimplifiedSolver(
            model=verifier.model, rust_doc_analyzer=verifier.rust_doc_analyzer
        )
        ret = await solve_verify(
            solver=solver,
            verifier=verifier,
            task=task,
            image_name=state.image_name,
            library_name=state.library_name,
            collect_trajectory=False,
            use_search=True,
            max_turns=10,
        )
        if ret.is_success() and ret.qa is not None:
            message = get_single_turn_initial_messages(state)
            if isinstance(ret.qa, QRA):
                new_message = TinkerMessage(
                    role="assistant",
                    content=[
                        ThinkingPart(
                            type="thinking",
                            thinking=ret.qa.reasoning,
                        ),
                        TextPart(
                            type="text",
                            text=ret.qa.answer,
                        ),
                    ],
                )
            else:
                new_message = TinkerMessage(role="assistant", content=ret.qa.answer)

            whole_traj = message + [new_message]
            log_trajectory_if_debug(whole_traj)
            metrics = dict(
                no_text_content=0.0,
                no_code_found=0.0,
                code_did_not_compile=0.0,
                verifier_failed=0.0,
                verifier_error=0.0,
            )
            logger.info(f"Successfully solved after {i + 1} rewirings")
            return RewireSessionResult.success(
                task=task,
                messages=whole_traj,
                num_rewire=i + 1,
                metrics=init_metrics | metrics_with_prefix(metrics, f"rewire{i + 1}"),
            )
        else:
            logger.info(f"Failed to solve after {i + 1} rewirings: {ret.cause}")
    logger.info("Failed to solve after max_rewire")
    return RewireSessionResult.error(
        task=task, error_message="Failed to solve after max_rewire"
    )


async def solve_verify_tinker_single_turn(
    solver_model: TinkerModel,
    env_state: SingleTurnEnvState,
    verifier: Verifier,
) -> SolveVerifyTinkerSingleTurnResult:
    async with build_single_turn_env(
        env_state=env_state,
        renderer=solver_model.renderer,
        verifier=verifier,
    ) as env:
        solver_sampler = solver_model.sampling_client
        policy = TinkerTokenCompleter(solver_sampler, max_tokens=None)
        transitions = []
        ob, stop_condition = await env.initial_observation()
        while True:
            ac_with_logprobs = await policy(ob, stop_condition)
            step_result = await env.step(ac_with_logprobs.tokens)
            transition = Transition(
                ob=ob,
                ac=ac_with_logprobs,
                reward=step_result.reward,
                episode_done=step_result.episode_done,
                metrics=step_result.metrics,
                logs=step_result.logs,
            )
            transitions.append(transition)
            ob = step_result.next_observation
            if step_result.episode_done:
                break
        env_state = await env.get_state()
        metrics = step_result.metrics
        trajectory = Trajectory(transitions=transitions, final_ob=ob)
        return SolveVerifyTinkerSingleTurnResult(
            trajectory=trajectory,
            env_state=env_state,
            reward=get_total_reward(trajectory),
            metrics=metrics,
        )
