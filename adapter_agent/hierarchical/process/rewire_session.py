import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from oai_utils.tinker import TinkerModel
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.rl.types import (
    Trajectory,
    Transition,
)
from tinker_cookbook.utils import logtree

from adapter_agent.hierarchical.agent.rewirer import (
    format_trajectory_transcript,
)
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.completer import TinkerTokenCompleter
from adapter_agent.rl.env.conclusion import SSConclusion
from adapter_agent.rl.env.standard import (
    EnvState,
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


@dataclass(kw_only=True)
class RewireSessionResultNormal:
    task: Task
    trials: list[TinkerMessage]
    conclusion: SSConclusion
    trajectory: Trajectory
    reward: float


@dataclass(kw_only=True)
class RewireSessionResultSuccess(RewireSessionResultNormal):
    conclusion: SSConclusion = field(default="success", init=False)
    knowledge: str
    reward: float = 1.0

    def is_successful(self) -> bool:
        return True


@dataclass(kw_only=True)
class RewireSessionResultFailure(RewireSessionResultNormal):
    reward: float = 0.0

    def is_successful(self) -> bool:
        return False


@dataclass(kw_only=True)
class RewireSessionResultError:
    task: Task
    conclusion: SSConclusion

    def is_successful(self) -> bool:
        return False


type RewireSessionResult = (
    RewireSessionResultSuccess | RewireSessionResultFailure | RewireSessionResultError
)


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


def log_trajectory_if_debug(messages: list[TinkerMessage]) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        log_trajectory(messages)


def log_trajectory(messages: list[TinkerMessage], flip_tag: bool = False) -> None:
    transcript = format_trajectory_transcript(
        messages, use_thinking=True, flip_tag=flip_tag
    )
    _log_trajectory_debug(transcript)


def mean_metrics(trajectories: list[FloatMetrics]) -> FloatMetrics:
    raw_metrics = defaultdict(list)
    metrics = {}
    for ret in trajectories:
        for key, value in metrics.items():
            raw_metrics[key].append(value)
    for key, value in raw_metrics.items():
        metrics[key] = np.mean(value).item()
    return metrics
