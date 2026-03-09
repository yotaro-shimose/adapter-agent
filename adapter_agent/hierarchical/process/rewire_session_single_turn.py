import logging
from dataclasses import dataclass

from agents.extensions.models.litellm_model import LitellmModel
from oai_utils.tinker import TinkerModel
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.rl.types import (
    Trajectory,
    Transition,
)

from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.process.rewire import run_rewiring_loop
from adapter_agent.hierarchical.process.rewire_session import (
    FloatMetrics,
    RewireSessionResult,
    get_total_reward,
    log_trajectory_if_debug,
    metrics_with_prefix,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.completer import TinkerTokenCompleter
from adapter_agent.rl.env.reward import LLMAsAJudge
from adapter_agent.rl.env.single_turn import (
    SingleTurnEnvState,
    build_single_turn_env,
)

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
    rewirer_model: LitellmModel | TinkerModel,
    task: Task,
    max_rewire: int,
) -> RewireSessionResult:
    state = SingleTurnEnvState.numrs2(task=task)
    trials: list[list[TinkerMessage]] = []
    try:
        ret = await solve_verify_tinker_single_turn(
            solver_model=solver_model,
            verifier=verifier,
            env_state=state,
        )
    except EnvironmentError as e:
        logger.error(f"Failed to solve: {e}")
        return RewireSessionResult.error(task=task, error_message=str(e), trials=trials)
    log_trajectory_if_debug(ret.env_state.messages)
    init_metrics = metrics_with_prefix(ret.metrics, "first")
    trials.append(ret.env_state.messages)

    if ret.is_success():
        logger.info("Successfully solved in first attempt")
        return RewireSessionResult.success(
            task=task,
            messages=ret.env_state.messages,
            num_rewire=0,
            metrics=init_metrics,
            trials=trials,
        )
    # Rewire
    return await run_rewiring_loop(
        solver_model=solver_model,
        verifier=verifier,
        rewirer_model=rewirer_model,
        task=task,
        max_rewire=max_rewire,
        init_metrics=init_metrics,
        init_trials=trials,
        max_turns=5,
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
