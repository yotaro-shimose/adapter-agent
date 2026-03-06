import logging
from dataclasses import dataclass
from typing import cast

from agents.extensions.models.litellm_model import LitellmModel
from oai_utils.tinker import TinkerModel
from tinker_cookbook.rl.types import (
    Trajectory,
    Transition,
)

from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.process.rewire_session import (
    FloatMetrics,
    RewireSessionResult,
    get_total_reward,
    log_trajectory_if_debug,
    metrics_with_prefix,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.completer import TinkerMessageCompleter, TinkerTokenCompleter
from adapter_agent.rl.env.reward import LLMAsAJudge
from adapter_agent.rl.env.simplified_solver import (
    SimplifiedSolverEnv,
    SimplifiedSolverEnvState,
    build_simplified_solver_env,
)
from adapter_agent.rl.env.single_turn import (
    SingleTurnEnvState,
    build_single_turn_env,
)
from adapter_agent.rl.litellm_completer import LitellmMessageCompleter

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
    return await run_rewiring_loop(
        solver_model=solver_model,
        verifier=verifier,
        rewirer_model=rewirer_model,
        task=task,
        max_rewire=max_rewire,
        init_metrics={},
        # init_metrics=init_metrics,
    )


async def run_rewiring_loop(
    solver_model: TinkerModel,
    verifier: Verifier,
    rewirer_model: LitellmModel | TinkerModel,
    task: Task,
    max_rewire: int,
    init_metrics: dict,
) -> RewireSessionResult:
    for i in range(max_rewire):
        ss_state = SimplifiedSolverEnvState.numrs2(task=task)
        async with build_simplified_solver_env(
            env_state=ss_state,
            renderer=solver_model.renderer,
            verifier=verifier,
            rust_doc_analyzer=verifier.rust_doc_analyzer,
            max_turns=10,
        ) as env:
            msg_env = cast(SimplifiedSolverEnv, env.message_env)
            if isinstance(rewirer_model, LitellmModel):
                completer = LitellmMessageCompleter(
                    model=rewirer_model.model,
                    renderer=solver_model.renderer,
                    tools=list(msg_env.tools.values()),
                )
            else:
                completer = TinkerMessageCompleter(
                    rewirer_model.sampling_client,
                    renderer=rewirer_model.renderer,
                    max_tokens=None,  # type: ignore
                )

            ob = await msg_env.initial_observation()
            while True:
                ac_message = await completer(ob)
                step_result = await msg_env.step(ac_message)
                ob = step_result.next_messages
                if step_result.episode_done:
                    break

            if LLMAsAJudge.is_successful_reward(step_result.reward):
                # Retrieve the final valid Rust code from the environment history
                # Assuming the last assistant message before reward contains the code
                last_assistant_msg = next(
                    (m for m in reversed(ob) if m["role"] == "assistant"), None
                )
                if last_assistant_msg:
                    whole_traj = ob

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
                        metrics=init_metrics
                        | metrics_with_prefix(metrics, f"rewire{i + 1}"),
                    )
            else:
                log_trajectory_if_debug(ob)
                logger.info(f"Failed to solve after {i + 1} rewirings")
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
