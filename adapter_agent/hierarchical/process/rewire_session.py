import logging
from dataclasses import dataclass
from typing import Self

from oai_utils.agent import AgentRunFailure
from oai_utils.tinker import TinkerModel
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.rl.types import (
    Trajectory,
    Transition,
)
from tinker_cookbook.utils import logtree

from adapter_agent.hierarchical.agent.rewirer import (
    Rewirer,
    format_trajectory_transcript,
)
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.completer import TinkerTokenCompleter
from adapter_agent.rl.env.env import (
    EnvState,
    InitEnvState,
    LLMAsAJudge,
    ResumedEnvState,
    build_coder_env,
)

logger = logging.getLogger(__name__)


type FloatMetrics = dict[str, float]


def _log_trajectory_debug(transcript: str) -> None:
    """Helper function to log the trajectory using rich if debugging is enabled."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    console = Console()
    console.print(
        Panel(
            Markdown(transcript),
            title="Acquired Trajectory",
            border_style="blue",
        )
    )


@dataclass
class Rewired:
    messages: list[TinkerMessage]
    n_rewire: int


@dataclass
class RewireSessionResult:
    task: Task
    rewired: Rewired | None
    error_message: str | None
    metrics: FloatMetrics

    @classmethod
    def error(cls, task: Task, error_message: str) -> Self:
        return cls(task=task, rewired=None, error_message=error_message, metrics={})

    @classmethod
    def success(
        cls,
        task: Task,
        messages: list[TinkerMessage],
        num_rewire: int,
        metrics: FloatMetrics,
    ) -> Self:
        return cls(
            task=task,
            rewired=Rewired(messages=messages, n_rewire=num_rewire),
            error_message=None,
            metrics=metrics,
        )

    def is_successful(self) -> bool:
        return self.error_message is None


@dataclass
class SolveVerifyTinkerResult:
    trajectory: Trajectory
    env_state: ResumedEnvState
    reward: float
    metrics: FloatMetrics

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
) -> RewireSessionResult:
    state = InitEnvState.numrs2(
        task=task,
        max_turns=max_turns,
    )
    ret = await solve_verify_tinker(
        solver_model=solver_model,
        verifier=verifier,
        env_state=state,
    )
    log_trajectory_if_debug(ret)
    init_metrics = metrics_with_prefix(ret.metrics, "first")

    if ret.is_success():
        return RewireSessionResult.success(
            task=task,
            messages=ret.env_state.messages,
            num_rewire=0,
            metrics=init_metrics,
        )
    # Rewire
    for i in range(max_rewire):
        try:
            branching_state = await rewirer.rewire(ret.env_state)
        except AgentRunFailure as e:
            logger.error(f"Failed to rewire: {e}")
            return RewireSessionResult.error(task=task, error_message=str(e))
        ret = await solve_verify_tinker(
            solver_model=solver_model,
            verifier=verifier,
            env_state=branching_state,
        )

        if ret.is_success():
            log_trajectory_if_debug(ret)
            return RewireSessionResult.success(
                task=task,
                messages=ret.env_state.messages,
                num_rewire=i + 1,
                metrics=init_metrics
                | metrics_with_prefix(ret.metrics, f"rewire{i + 1}"),
            )

    return RewireSessionResult.error(
        task=task, error_message="Failed to solve after max_rewire"
    )


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
        metrics = step_result.metrics
        trajectory = Trajectory(transitions=transitions, final_ob=ob)
        return SolveVerifyTinkerResult(
            trajectory=trajectory,
            env_state=env_state,
            reward=get_total_reward(trajectory),
            metrics=metrics,
        )


def get_total_reward(trajectory: Trajectory) -> float:
    return sum([t.reward for t in trajectory.transitions])


def metrics_with_prefix(
    metrics: FloatMetrics, prefix: str, sep: str = "_"
) -> FloatMetrics:
    return {f"{prefix}{sep}{k}": v for k, v in metrics.items()}


def log_trajectory_if_debug(ret: SolveVerifyTinkerResult) -> None:
    if ret.is_success():
        if logger.isEnabledFor(logging.DEBUG):
            transcript = format_trajectory_transcript(
                ret.env_state.messages, use_thinking=True
            )
            _log_trajectory_debug(transcript)
