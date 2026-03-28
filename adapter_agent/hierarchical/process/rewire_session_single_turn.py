import logging
from dataclasses import dataclass

from oai_utils.tinker import TinkerModel
from tinker_cookbook.rl.types import (
    Trajectory,
    Transition,
)

from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.rl.completer import TinkerTokenCompleter
from adapter_agent.rl.env.conclusion import SSConclusion
from adapter_agent.rl.env.reward import LLMAsAJudge
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.env.session_result import get_total_reward
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
    conclusion: SSConclusion

    def is_success(self) -> bool:
        return LLMAsAJudge.is_successful_reward(self.reward)


async def solve_verify_tinker_single_turn(
    solver_model: TinkerModel,
    env_state: SingleTurnEnvState,
    verifier: Verifier,
    runtime_settings: RuntimeSettings,
) -> SolveVerifyTinkerSingleTurnResult:
    async with build_single_turn_env(
        env_state=env_state,
        renderer=solver_model.renderer,
        verifier=verifier,
        runtime_settings=runtime_settings,
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
        trajectory = Trajectory(transitions=transitions, final_ob=ob)
        return SolveVerifyTinkerSingleTurnResult(
            trajectory=trajectory,
            env_state=env_state,
            reward=get_total_reward(trajectory),
            conclusion=step_result.conclusion,
        )
