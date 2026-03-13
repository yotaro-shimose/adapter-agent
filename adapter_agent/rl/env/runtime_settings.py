from typing import Literal

from coder_mcp.runtime import CloudRunRuntime, DockerRuntime, Runtime
from pydantic import BaseModel


class RuntimeSettings(BaseModel):
    type: Literal["cloudrun", "docker"]
    image_uri: str

    def build_runtime(self) -> Runtime:
        if self.type == "cloudrun":
            return CloudRunRuntime(image_uri=self.image_uri)
        elif self.type == "docker":
            return DockerRuntime(image_name=self.image_uri)
        else:
            raise ValueError(f"Unknown runtime type: {self.type}")
