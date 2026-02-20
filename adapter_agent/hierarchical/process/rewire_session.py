from dataclasses import dataclass

from oai_utils.tinker import TinkerModel
from tinker_cookbook.rl.types import (
    Trajectory,
    Transition,
)
from tinker_cookbook.utils import logtree

from adapter_agent.hierarchical.agent.rewirer import Rewirer
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.completer import TinkerTokenCompleter
from adapter_agent.rl.env import EnvState, LLMAsAJudge, build_coder_env


@dataclass
class SolveVerifyTinkerResult:
    trajectory: Trajectory
    env_state: EnvState
    reward: float

    def is_success(self) -> bool:
        return LLMAsAJudge.is_successful_reward(self.reward)

    def total_reward(self) -> float:
        return sum(transition.reward for transition in self.trajectory.transitions)


async def rewire_session(
    solver_model: TinkerModel,
    verifier: Verifier,
    rewirer: Rewirer,
    task: Task,
    max_turns: int = 5,
    max_rewire: int = 1,
) -> SolveVerifyTinkerResult:
    state = EnvState.numrs2(
        task=task,
        max_turns=max_turns,
    )
    ret = await solve_verify_tinker(
        solver_model=solver_model,
        verifier=verifier,
        env_state=state,
    )
    if ret.is_success():
        return ret

    # Rewire
    for _ in range(max_rewire):
        branching_state = await rewirer.rewire(ret.env_state)
        ret = await solve_verify_tinker(
            solver_model=solver_model,
            verifier=verifier,
            env_state=branching_state,
        )

        if ret.is_success():
            return ret
    return ret


@logtree.scope_header_decorator
async def solve_verify_tinker(
    solver_model: TinkerModel,
    env_state: EnvState,
    verifier: Verifier,
) -> SolveVerifyTinkerResult:
    async with build_coder_env(
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
        trajectory = Trajectory(transitions=transitions, final_ob=ob)
        return SolveVerifyTinkerResult(
            trajectory=trajectory,
            env_state=env_state,
            reward=get_total_reward(trajectory),
        )


def get_total_reward(trajectory: Trajectory) -> float:
    return sum([t.reward for t in trajectory.transitions])
