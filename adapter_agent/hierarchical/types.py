import time
import uuid
from pathlib import Path
from typing import Self

from agents import TResponseInputItem
from pydantic import BaseModel, Json


class Task(BaseModel):
    id: str
    instruction: str

    @classmethod
    def from_instruction(cls, instruction: str) -> Self:
        return cls(id=str(uuid.uuid4()), instruction=instruction)


class Trajectory(BaseModel):
    input_list: list[TResponseInputItem]

    def as_str(self) -> str:
        # Convert input list to a readable string representation
        buffer = []
        for item in self.input_list:
            if isinstance(item, dict):
                role = item.get("role", "unknown")
                content = item.get("content", "")
                buffer.append(f"[{role}]: {content}")
            else:
                buffer.append(str(item))
        return "\n\n".join(buffer)

    def add_item(self, item: TResponseInputItem) -> None:
        self.input_list.append(item)


class MemoryItem[InputT: BaseModel | Json, OutputT: BaseModel | Json](BaseModel):
    input: InputT
    output: OutputT
    timestamp: float


class Memory[InputT: BaseModel | Json, OutputT: BaseModel | Json](BaseModel):
    items: list[MemoryItem[InputT, OutputT]] = []

    def add(self, input: InputT, output: OutputT) -> None:
        self.items.append(MemoryItem(input=input, output=output, timestamp=time.time()))

    def save(self, path: Path) -> None:
        with path.open("w") as f:
            f.write(self.model_dump_json(indent=2))

    def load(self, path: Path) -> None:
        if path.exists():
            with path.open("r") as f:
                data = f.read()
                loaded = self.model_validate_json(data)
                self.items = loaded.items
