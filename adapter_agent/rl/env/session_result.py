from dataclasses import dataclass, field

from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.rl.types import Trajectory

from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.env.conclusion import SSConclusion


@dataclass(kw_only=True)
class RewireSessionResultNormal:
    task: Task
    trials: list[TinkerMessage]
    conclusion: SSConclusion
    trajectory: Trajectory
    reward: float
    knowledge: str | None = None
    reasoning: str | None = None


@dataclass(kw_only=True)
class RewireSessionResultSuccess(RewireSessionResultNormal):
    conclusion: SSConclusion = field(default="success", init=False)
    oc_trials: list[TinkerMessage] | None = None
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


def get_total_reward(trajectory: Trajectory) -> float:
    return sum([t.reward for t in trajectory.transitions])
