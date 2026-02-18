from dataclasses import dataclass

from oai_utils.agent import AgentsSDKModel

from adapter_agent.hierarchical.agent.analyzer import Analyzer
from adapter_agent.hierarchical.agent.decomposer import Decomposer
from adapter_agent.hierarchical.agent.solver import Solver
from adapter_agent.hierarchical.agent.verifier import Verifier
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
                rust_doc_analyzer=rust_doc_analyzer,
            ),
            verifier=Verifier(
                model=model,
                rust_doc_analyzer=rust_doc_analyzer,
            ),
            analyzer=Analyzer(
                model=model,
            ),
            decomposer=Decomposer(
                model=model,
            ),
        )
