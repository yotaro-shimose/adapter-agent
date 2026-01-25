from dataclasses import dataclass

from oai_utils.agent import AgentsSDKModel

from adapter_agent.hierarchical.analyzer import Analyzer
from adapter_agent.hierarchical.decomposer import Decomposer, DecomposerInput
from adapter_agent.hierarchical.solver import Solver, SolverResult
from adapter_agent.hierarchical.state import TaskList
from adapter_agent.hierarchical.types import Memory, Task, Trajectory
from adapter_agent.hierarchical.verifier import QA, VerificationResult, Verifier


@dataclass
class Agents:
    solver: Solver
    verifier: Verifier
    analyzer: Analyzer
    decomposer: Decomposer

    @classmethod
    def from_model(cls, model: AgentsSDKModel):
        return cls(
            solver=Solver(
                model=model,
                memory=Memory[Task, SolverResult](),
            ),
            verifier=Verifier(
                model=model,
                memory=Memory[QA, VerificationResult](),
            ),
            analyzer=Analyzer(
                model=model,
                memory=Memory[Trajectory, Task](),
            ),
            decomposer=Decomposer(
                model=model,
                memory=Memory[DecomposerInput, TaskList](),
            ),
        )
