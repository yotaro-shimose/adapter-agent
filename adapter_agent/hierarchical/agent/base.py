from dataclasses import dataclass
from pathlib import Path

from oai_utils import AgentsSDKModel
from pydantic import BaseModel

from adapter_agent.hierarchical.types import Memory


@dataclass(kw_only=True)
class BaseAgent[M: AgentsSDKModel, I: BaseModel, O: BaseModel]:
    model: M
    memory: Memory[I, O] | None

    def maybe_add_to_memory(self, input: I, output: O) -> None:
        if self.memory is not None:
            self.memory.add(input, output)

    def maybe_save(self, path: Path) -> None:
        if self.memory is not None:
            self.memory.save(path)
