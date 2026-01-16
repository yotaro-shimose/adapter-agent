import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from agents.extensions.models.litellm_model import LitellmModel
from agents.tracing import add_trace_processor
from dotenv.main import load_dotenv
from oai_utils.tracing import AgentContentPrinter
from pydantic import BaseModel, Field

from adapter_agent.exam.exam import CodingExam
from adapter_agent.exam.pipeline import generate_exams, generate_exercises
from adapter_agent.exam.repository import GitRepository
from adapter_agent.exam.syllabus import LearningTopic


class BenchmarkConfig(BaseModel):
    benchmark_id: str = Field(
        default="test_numrs_syllabus2", description="Unique ID for this benchmark"
    )
    model_name: str = Field(
        default="gemini/gemini-3-flash-preview", description="LLM model name"
    )
    project_dir: Path = Field(
        default=Path("projects/test_numrs_syllabus2"),
        description="Working directory for projects",
    )
    template_dir: Path = Field(
        default=Path("templates/rust_exam_template"),
        description="Local directory of the project-specific exam template",
    )
    master_boilerplate_dir: Path = Field(
        default=Path("templates/rust_boilerplate"),
        description="Source boilerplate for new project templates",
    )
    repository_path: Path = Field(
        default=Path("repositories/numrs"),
        description="Local path to the source repository",
    )
    file_limit: int = Field(
        default=1, description="Maximum number of files to generate"
    )
    exam_limit: int = Field(
        default=6, description="Maximum number of exams to generate"
    )
    image_name: str = Field(
        default=os.getenv("OPENHANDS_IMAGE_NAME", "coder-mcp"),
        description="Docker image to use for the MCP server",
    )
    max_concurrent_exams: int = Field(
        default=3, description="Maximum number of concurrent exam generation tasks"
    )

    def get_project_dir(self) -> Path:
        return self.project_dir.resolve()

    def get_template_dir(self) -> Path:
        return self.template_dir.resolve()

    def get_repository_path(self) -> Path:
        return self.repository_path.resolve()

    def save(self):
        """Save the benchmark config to a JSON file in the benchmarks/ directory."""
        output_dir = Path("benchmarks")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.benchmark_id}.json"
        output_path.write_text(self.model_dump_json(indent=2))
        print(f"✅ Benchmark config saved to {output_path}")

    @classmethod
    def load(cls, benchmark_id: str) -> "BenchmarkConfig":
        """Load a BenchmarkConfig from the benchmarks/ directory by its ID."""
        input_path = Path("benchmarks") / f"{benchmark_id}.json"
        if not input_path.exists():
            raise FileNotFoundError(f"Benchmark config not found at {input_path}")
        return cls.model_validate_json(input_path.read_text())


def ensure_project_template(config: BenchmarkConfig) -> GitRepository:
    """Ensure the project-specific template exists and is initialized."""
    template_dir = config.get_template_dir()
    if not template_dir.exists():
        boilerplate_dir = config.master_boilerplate_dir.resolve()
        print(f"Initializing project template from {boilerplate_dir}...")
        if not boilerplate_dir.exists():
            raise FileNotFoundError(
                f"Master boilerplate not found at {boilerplate_dir}"
            )

        shutil.copytree(boilerplate_dir, template_dir)

        # Initialize as a fresh git repository
        try:
            subprocess.run(
                ["git", "init"], cwd=str(template_dir), check=True, capture_output=True
            )
            subprocess.run(
                ["git", "add", "."],
                cwd=str(template_dir),
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "Initial commit from boilerplate"],
                cwd=str(template_dir),
                check=True,
                capture_output=True,
            )
            print("✅ Project template initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize project template: {e}")

    return GitRepository(local_dir=template_dir)


async def get_or_generate_topics(
    config: BenchmarkConfig, model: LitellmModel, library: GitRepository
) -> list[LearningTopic]:
    """Load existing topics or generate new ones."""
    print("=== Phase 1: Generating Exercises ===")
    project_dir = config.get_project_dir()
    topics_path = project_dir / "curriculum" / "topics.json"
    topics = []

    if topics_path.exists():
        print(f"Found existing topics in {topics_path}. Loading...")
        try:
            data = json.loads(topics_path.read_text())
            topics = [LearningTopic.model_validate(item) for item in data]
            print(f"Loaded {len(topics)} topics.")
        except Exception as e:
            print(f"Failed to load existing topics: {e}")

    if not topics:
        topics = await generate_exercises(
            model, library, project_dir, limit=config.file_limit
        )
        if topics:
            print(f"Saved {len(topics)} topics to {topics_path}")

    return topics


async def generate_benchmark_exams(
    config: BenchmarkConfig,
    model: LitellmModel,
    library: GitRepository,
    exam_template: GitRepository,
    topics: list[LearningTopic],
) -> list[CodingExam]:
    """Generate exams for the given topics."""
    print("\n=== Phase 2: Generating Exams ===")
    project_dir = config.get_project_dir()

    target_exercises = topics[: config.exam_limit]
    print(f"Targeting {len(target_exercises)} exercises for exam generation.")

    exams = await generate_exams(
        model=model,
        exercises=target_exercises,
        library_repository=library,
        exam_template=exam_template,
        work_dir=project_dir,
        image_name=config.image_name,
        max_concurrent=config.max_concurrent_exams,
    )

    if exams:
        exams_path = project_dir / "exams.json"
        exams_path.write_text(
            json.dumps([ex.model_dump(mode="json") for ex in exams], indent=2)
        )
        print(f"Saved {len(exams)} exams to {exams_path}")

    return exams


async def main():
    load_dotenv()
    # Enable tracing to see agent activity
    add_trace_processor(AgentContentPrinter())
    config = BenchmarkConfig()
    config.save()

    # Configure Model
    model = LitellmModel(model=config.model_name, api_key=os.environ["GEMINI_API_KEY"])

    # Work Dir / Output Project
    project_dir = config.get_project_dir()
    project_dir.mkdir(parents=True, exist_ok=True)

    # Source Library definition
    library_path = config.get_repository_path()
    if not library_path.exists():
        print(f"Error: {library_path} does not exist.")
        return

    library = GitRepository(local_dir=library_path)

    # Scaffolding & Preparation
    exam_template = ensure_project_template(config)

    # Phase 1: Topics
    topics = await get_or_generate_topics(config, model, library)
    if not topics:
        print("No topics available. Exiting.")
        return

    # Phase 2: Exams
    exams = await generate_benchmark_exams(
        config, model, library, exam_template, topics
    )

    print(f"\nPipeline Complete. Generated {len(exams)} exams.")
    if exams:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
