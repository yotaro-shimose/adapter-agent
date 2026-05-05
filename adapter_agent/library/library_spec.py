from pathlib import Path
from typing import Self

from pydantic import BaseModel

from adapter_agent.rl.env.runtime_settings import RuntimeSettings


class LibrarySpec(BaseModel):
    """Per-library configuration bundle for a study/eval run.

    Aggregates the library-specific paths and runtime images so a single
    instance can drive `study.py` end-to-end. Switching libraries is meant
    to be a one-line change at the call site.
    """

    name: str
    libdir: Path
    benchmark_csv: Path
    docker_image: str
    cloudrun_image_uri: str
    # Difficulty filter the benchmark loader should apply by default. numrs2's
    # CSV has plenty of Easy rows so "Easy"-only is the long-standing choice;
    # hisab's CSV is mostly Medium (298 vs 21 Easy), so None (= no filter)
    # is needed for slices like [0:150] to be reachable.
    default_difficulty: str | None = "Easy"

    def docker_runtime(self) -> RuntimeSettings:
        return RuntimeSettings(type="docker", image_uri=self.docker_image)

    def cloudrun_runtime(self) -> RuntimeSettings:
        return RuntimeSettings(type="cloudrun", image_uri=self.cloudrun_image_uri)

    @property
    def summary_path(self) -> Path:
        return self.libdir / "SUMMARY.md"

    def read_summary(self) -> str:
        """Return SUMMARY.md contents. Raises FileNotFoundError if missing."""
        path = self.summary_path
        if not path.exists():
            raise FileNotFoundError(f"Library summary missing: {path}")
        return path.read_text()

    @classmethod
    def numrs2(cls) -> Self:
        return cls(
            name="numrs2",
            libdir=Path("repositories/numrs"),
            benchmark_csv=Path("data/benchmarks/numrs2_2026-04-29/diverse_enhanced.csv"),
            docker_image="coder-mcp-numrs2:latest",
            cloudrun_image_uri="europe-north1-docker.pkg.dev/dsat2-405406/shimose-repo/coder-mcp-numrs2:latest",
        )

    @classmethod
    def hisab(cls) -> Self:
        return cls(
            name="hisab",
            libdir=Path("repositories/hisab"),
            benchmark_csv=Path("data/benchmarks/hisab_2026-05-04/diverse_enhanced.csv"),
            docker_image="coder-mcp-hisab:latest",
            cloudrun_image_uri="europe-north1-docker.pkg.dev/dsat2-405406/shimose-repo/coder-mcp-hisab:latest",
            default_difficulty=None,
        )
