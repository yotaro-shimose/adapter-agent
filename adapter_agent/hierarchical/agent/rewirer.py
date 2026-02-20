from dataclasses import dataclass

from oai_utils import AgentsSDKModel

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.rl.env import EnvState


@dataclass
class Rewirer[T: AgentsSDKModel](BaseAgent[T]):
    rust_doc_analyzer: RustDocAnalyzer

    async def rewire(self, state: EnvState) -> EnvState:
        raise NotImplementedError
