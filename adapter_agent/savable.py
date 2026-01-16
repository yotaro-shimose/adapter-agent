from pathlib import Path
from typing import Self

from pydantic import BaseModel


class Savable(BaseModel):
    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json())

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text())
