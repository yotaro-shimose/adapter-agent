from dataclasses import dataclass

from oai_utils.agent import AgentsSDKModel

from adapter_agent.hierarchical.agent.analyzer import Analyzer
from adapter_agent.hierarchical.agent.decomposer import Decomposer, DecomposerInput
from adapter_agent.hierarchical.agent.solver import Solver, SolverResult
from adapter_agent.hierarchical.agent.verifier import QA, VerificationResult, Verifier
from adapter_agent.hierarchical.state import TaskList
from adapter_agent.hierarchical.types import Memory, Task, Trajectory
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer


@dataclass
class Agents[T: AgentsSDKModel]:
    solver: Solver[T]
    verifier: Verifier[T]
    analyzer: Analyzer[T]
    decomposer: Decomposer[T]

    @classmethod
    def from_model(cls, model: T, rust_doc_analyzer: RustDocAnalyzer):
        return cls(
            solver=Solver(
                model=model,
                memory=Memory[Task, SolverResult](),
                rust_doc_analyzer=rust_doc_analyzer,
            ),
            verifier=Verifier(
                model=model,
                memory=Memory[QA, VerificationResult](),
                rust_doc_analyzer=rust_doc_analyzer,
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
