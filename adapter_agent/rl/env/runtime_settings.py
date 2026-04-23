from typing import Literal, Self

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

    @classmethod
    def cloudrun_numrs2(cls) -> Self:
        return cls(
            type="cloudrun",
            image_uri="europe-north1-docker.pkg.dev/dsat2-405406/shimose-repo/coder-mcp-numrs2:latest",
        )

    @classmethod
    def docker_numrs2(cls) -> Self:
        return cls(type="docker", image_uri="coder-mcp-numrs2:latest")
