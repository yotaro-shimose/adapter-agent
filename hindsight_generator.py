from adapter_agent.savable import Savable
from oai_utils.agent import AgentRunFailure
import asyncio
import os
import uuid
from itertools import chain
from pathlib import Path

from agents.extensions.models.litellm_model import LitellmModel
from agents.tracing import add_trace_processor
from oai_utils.agent import AgentsSDKModel, AgentWrapper
from oai_utils.tracing import AgentContentPrinter
from pydantic import BaseModel, Field

from adapter_agent.async_util import gather_with_semaphore
from coder_mcp.runtime import DockerRuntime
from coder_mcp.runtime.rust_env import RustCodingEnvironment
from coder_mcp.runtime.temp_workspace import TempWorkspace

# --- PYDANTIC MODELS ---


class HindsightConfig(BaseModel):
    experiment_id: str = Field(
        default="experiment_generic_aligned",
        description="ID for the experiment/output folder",
    )
    library_name: str = Field(
        default="numrs", description="Name of the library (e.g., 'numrs')"
    )
    model_name: str = Field(
        default="gemini/gemini-3-flash-preview", description="LLM model name"
    )
    # Using relative paths assuming execution from project root
    library_path: Path = Field(
        default=Path("repositories/numrs"),
        description="Path to the source library to learn from",
    )
    curriculum_dir: Path = Field(
        default=Path("projects/workspace_curriculum_hierarchical/curriculum"),
        description="Path to the generated curriculum markdown files",
    )
    boilerplate_dir: Path = Field(
        default=Path("templates/rust_boilerplate"),
        description="Path to the rust boilerplate template",
    )
    output_base_dir: Path = Field(
        default=Path("data/qra"),
        description="Base directory for outputting generated QRA data",
    )
    image_name: str = Field(
        default=os.getenv("OPENHANDS_IMAGE_NAME", "coder-mcp"),
        description="Docker image to use",
    )
    max_concurrent_tasks: int = Field(
        default=10, description="Number of concurrent generation agents"
    )

    def get_library_path(self) -> Path:
        return self.library_path.resolve()

    def get_curriculum_dir(self) -> Path:
        return self.curriculum_dir.resolve()

    def get_boilerplate_dir(self) -> Path:
        return self.boilerplate_dir.resolve()

    def get_output_dir(self) -> Path:
        return (self.output_base_dir / self.experiment_id).resolve()


def load_curriculum_mdfiles(curriculum_path: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in curriculum_path.glob("**/*.md")
            if p.name not in set(["comprehensive_summary.md", "instruction.md"])
        ]
    )


class TeachableItem(BaseModel):
    slug: str = Field(
        description="A short, url-friendly identifier (e.g., 'broadcast_ops')."
    )
    description: str = Field(
        description="A sentence describing what the user should learn."
    )


class TeachablesList(BaseModel):
    items: list[TeachableItem] = Field(description="List of extracted teachables.")


class Teachable(TeachableItem):
    chapter: str = Field(description="The chapter this teachable belongs to.")


class QRAContent(BaseModel):
    question: str = Field(description="The Markdown question text.")
    reasoning: str = Field(
        description="Internal monologue explaining the verification and thought process."
    )
    answer: str = Field(description="The final natural language answer for the user.")


class HindsightOutput(Savable):
    id: str
    slug: str
    concept: str
    question: str
    reasoning: str
    answer: str


# --- PROMPTS ---

TOPIC_EXTRACTOR_PROMPT = """You are an expert Technical Curriculum Architect.
Your goal is to extract "Teachables" from a given curriculum chapter for the `{library_name}` Rust library.

<INSTRUCTIONS>
1. Read the provided chapter content.
2. Identify ALL distinctive concepts or practical skills taught in this chapter.
3. **Mix Types**: Include both:
    - **Coding Challenges**: "How to reshape an array?"
    - **Conceptual Questions**: "How does broadcasting work with different shapes?" or "What is the difference between a View and a Copy?"
4. **Comprehensive List**: Do not limit yourself to a small number. Extract a comprehensive list of tasks that covers the chapter thoroughly.
</INSTRUCTIONS>
"""

QRA_GENERATOR_PROMPT = """You are an expert Rust Developer and Technical Writer who has internalized the `{library_name}` library.
Your task is to create a verified QRA (Question, Reasoning, Answer) triplet for a specific concept.

<CONTEXT>
Library: `{library_name}`
Detailed context about the library and the specific concept is provided in the input message.
</CONTEXT>

<TASK_PROGRESSION>
1. **Content Design**: Create a practical coding challenge or a conceptual question based on the provided concept and chapter context. To ensure the question is self-contained for future training, it must explicitly mention the `{library_name}` library by name (e.g., "In `{library_name}`, how do I..." or "Explain the benefit of X in `{library_name}`").

2. **Technical Verification**:
    - If your answer includes any Rust code snippets, you must verify them using an agentic loop.
    - Write a verification script (e.g., `src/bin/verify_x.rs`) using `write_to_file`.
    - Execute it with `run_command` (e.g., `cargo run --bin verify_x`).
    - Resolve any compilation or runtime errors until the solution is perfectly verified.
    - For purely conceptual text answers, verification is mental.

3. **Expert Drafting**:
    - **Reasoning**: Explain your internal technical logic and best practices from the perspective of an expert who already knows the library well.
    - **Answer**: Provide the final, naturally phrased technical answer.
</TASK_PROGRESSION>

<GUIDELINE>
- Internalization: Act as if the knowledge is internalized; never mention searching, documentation, 
    discovery, or the verification process (e.g., do NOT say "I found that..." 
    or "The code compiled successfully") for both reasoning and answer.
- The reasoning should start from user intent analysis
</GUIDELINE>

<ENVIRONMENT>
Commands should be executed in the current working directory, which is always `/workspace`.
</ENVIRONMENT>
"""


async def generate_qra_task(
    teachable: Teachable,
    output_dir: Path,
    model: AgentsSDKModel,
    boilerplate_dir: Path,
    library_path: Path,
    curriculum_src: Path,
    image_name: str,
    library_name: str,
):
    # Deterministic UUID for the task
    task_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"{teachable.slug}")
    output_file = output_dir / f"{task_uuid}.json"

    if output_file.exists():
        print(f"Skipping {teachable.slug} (already exists at {output_file.name})")
        return

    print(f"Starting generation for: {teachable.slug} ({task_uuid})")

    injections = {
        library_path: "repos/library",
        curriculum_src: "curriculum",
    }

    try:
        # Use TempWorkspace context manager
        with TempWorkspace(
            template_dir=boilerplate_dir,
            injections=injections,
            prefix=f"hindsight_{teachable.slug}_",
        ) as sandbox_dir:
            async with RustCodingEnvironment(
                workspace_dir=sandbox_dir, image_name=image_name
            ) as runtime:
                # The workspace inside the container is /workspace
                async with runtime.coder_mcp() as coder_mcp:
                    generator_agent = AgentWrapper[QRAContent].create(
                        name=f"QRA_Architect_{teachable.slug}",
                        instructions=QRA_GENERATOR_PROMPT.format(
                            library_name=library_name
                        ),
                        model=model,
                        mcp_servers=[coder_mcp],
                        output_type=QRAContent,
                    )

                    prompt = f"""Create a verified QRA for the following concept.

<CONCEPT>
{teachable.description}
</CONCEPT>

<CHAPTER_CONTEXT>
{teachable.chapter}
</CHAPTER_CONTEXT>

Please use tools to verify your code solution if it contains any Rust snippets.
"""
                    try:
                        result = await generator_agent.run(
                            prompt,
                            max_turns=30,
                        )
                    except AgentRunFailure as e:
                        print(f"AgentRunFailure: {e}")
                        return

                final_obj = result.final_output()

                # Save as structured JSON
                output_model = HindsightOutput(
                    id=str(task_uuid),
                    slug=teachable.slug,
                    concept=teachable.description,
                    question=final_obj.question,
                    reasoning=final_obj.reasoning,
                    answer=final_obj.answer,
                )

                output_file.write_text(output_model.model_dump_json(indent=2))
                print(f"Completed {teachable.slug} -> {output_file.name}")

    except Exception as e:
        print(f"Error generating {teachable.slug}: {e}")
        # Optional: write error log


async def extract_from_chapter(
    chapter_file: Path,
    model: AgentsSDKModel,
    runtime: DockerRuntime,
    library_name: str,
) -> list[Teachable]:
    if chapter_file.name.startswith("00_"):
        return []

    print(f"Processing Chapter: {chapter_file.name}")
    content = chapter_file.read_text()

    # Extract Teachables
    async with runtime.coder_mcp() as coder_mcp:
        extractor_agent = AgentWrapper[TeachablesList].create(
            name="TopicExtractor",
            instructions=TOPIC_EXTRACTOR_PROMPT.format(library_name=library_name),
            model=model,
            mcp_servers=[coder_mcp],
            output_type=TeachablesList,
        )

        try:
            extract_result = await extractor_agent.run(
                f"Extract teachables from this chapter:\n\n{content}", max_turns=5
            )
            teachables = [
                Teachable(slug=item.slug, description=item.description, chapter=content)
                for item in extract_result.final_output().items
            ]

            print(f"  Found {len(teachables)} teachables in {chapter_file.name}.")

            return teachables

        except Exception as e:
            print(f"Failed to extract info from {chapter_file.name}: {e}")
            return []


async def extract_teachables(
    chapter_files: list[Path],
    library_path: Path,
    model: AgentsSDKModel,
    max_concurrent: int,
    library_name: str,
):
    injections = {
        library_path: "repos/library",
    }

    # Create extraction tasks
    with TempWorkspace(injections=injections) as sandbox_dir:
        async with DockerRuntime(workspace_dir=sandbox_dir) as runtime:
            extraction_tasks = [
                extract_from_chapter(chapter_file, model, runtime, library_name)
                for chapter_file in chapter_files
            ]
            results = await gather_with_semaphore(
                extraction_tasks, max_concurrent=max_concurrent
            )

    return results


async def main():
    add_trace_processor(AgentContentPrinter())
    config = HindsightConfig()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return

    # Resolve paths
    library_path = config.get_library_path()
    curriculum_src = config.get_curriculum_dir()
    boilerplate_dir = config.get_boilerplate_dir()
    output_dir = config.get_output_dir()

    # Path validation
    if not library_path.exists():
        print(f"Error: Library not found at {library_path}")
        return

    if not curriculum_src.exists():
        print(f"Error: Curriculum not found at {curriculum_src}")
        return

    if not boilerplate_dir.exists():
        print(f"Error: Boilerplate not found at {boilerplate_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Model for Topic Extraction
    model = LitellmModel(model=config.model_name, api_key=api_key)

    print(f"Initializing Hindsight Generator (Config: {config.experiment_id})...")

    # List curriculum files
    chapters = load_curriculum_mdfiles(curriculum_src)

    print(f"Extracting teachables from {len(chapters)} chapters concurrently...")
    results = await extract_teachables(
        chapter_files=chapters,
        library_path=library_path,
        model=model,
        max_concurrent=config.max_concurrent_tasks,
        library_name=config.library_name,
    )

    teachables = list(chain.from_iterable(results))

    tasks = [
        generate_qra_task(
            teachable=t,
            output_dir=output_dir,
            model=model,
            boilerplate_dir=boilerplate_dir,
            library_path=library_path,
            curriculum_src=curriculum_src,
            image_name=config.image_name,
            library_name=config.library_name,
        )
        for t in teachables
    ]

    # Run in parallel with semaphore
    await gather_with_semaphore(tasks, max_concurrent=config.max_concurrent_tasks)
    print("All tasks completed.")


if __name__ == "__main__":
    asyncio.run(main())
