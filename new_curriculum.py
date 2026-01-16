import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from agents import RunContextWrapper, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from agents.tracing import add_trace_processor
from coder_mcp.runtime import LocalRuntime
from dotenv.main import load_dotenv
from oai_utils.agent import AgentWrapper
from oai_utils.tracing import AgentContentPrinter
from pydantic import BaseModel

from topic_db import Topic, TopicDatabase


class LibrarySummary(BaseModel):
    summary: str


@dataclass
class ContextType:
    db: TopicDatabase
    target_file_rel: str


@function_tool
async def register_topic(
    wrapper: RunContextWrapper[ContextType],
    id: str,
    title: str,
    description: str,
    related_apis: list[str],
) -> str:
    """Register a new topic in the database."""
    ctx = wrapper.context
    topic = Topic(
        id=id,
        title=title,
        description=description,
        related_apis=related_apis,
        source_file=ctx.target_file_rel,
    )
    print(f"  -> Registering topic: {title} ({id})")
    ctx.db.add_topic(topic)
    return f"Topic '{id}' registered successfully."


# --- PROMPTS ---

EXPLORER_PROMPT = """You are an expert Rust Researcher.
Your goal is to explore the provided Rust library at `repos/library` and generate a **Comprehensive Library Summary** in `library_summary.md`.

The summary should cover:
- High-level purpose of the library.
- Main modules and their responsibilities.
- Key structs and traits.
- Common usage patterns (look at examples).

You have access to tools to explore the file system and read files.
Once you have created the summary, you will return it as your final output.
CRITICAL: Your output must be ONLY a valid raw JSON object matching the schema `{"summary": "..."}`.
Do NOT use "thought:" prefix.
Do NOT output internal thoughts or reasoning.
Do NOT use markdown code blocks or backticks.
Just the raw JSON string starting with `{`.
"""

DETAILED_TOPIC_GENERATOR_PROMPT_TEMPLATE = """You are a specialized Topic Extraction Agent.
Your goal is to extract learning topics from the file: `{file_path}`.

CONTEXT:
**Library Summary**:
{library_summary}

**Current File**: `{file_path}`.

INSTRUCTIONS:
1. Read the current file.
2. Identify "Topics" that a user of this library should learn.
   - Distinguish between **Public API** (user-facing) and **Internal Implementation**.
   - ONLY extract public/user-facing topics.
   - **IGNORE** Trait Implementations (e.g. `impl MyTrait for MyType`) unless they add new public methods that are NOT part of the trait. Usage of standard traits should be covered under the Trait's topic or the Struct's main topic, not as a separate topic for the implementation file.
   - **IGNORE** purely internal helper functions.
3. For each topic found, use the `register_topic` tool to save it.
   - **ID**: A unique snake_case identifier (e.g., `tensor_creation`, `matrix_multiplication`).
   - **Title**: Human readable title.
   - **Related APIs**: List of functions/structs involved. IMPORTANT: Use the **FULL PATH** (e.g., `crate::module::struct::method` or `module::struct::method`). Do not use short names.
   - **Description**: Brief explanation of what this concept is.
4. If no user-facing topics are found in this file (e.g., internal utility), that is fine.
5. Say "finished" when done.
"""

# --- MAIN ---


async def prepare_workspace(workspace_dir: Path, repository_path: Path) -> Path:
    """Prepare the workspace and copy the library."""
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    lib_dest = workspace_dir / "repos" / "library"
    lib_dest.parent.mkdir(parents=True, exist_ok=True)
    if not lib_dest.exists():
        if repository_path.exists():
            shutil.copytree(repository_path, lib_dest, dirs_exist_ok=True)
        else:
            print(
                f"Warning: Repository path {repository_path} not found. Ensure it exists."
            )
    return lib_dest


async def run_explorer_phase(model, runtime: LocalRuntime, workspace_dir: Path) -> str:
    """Phase 1: Explorer Agent."""
    print("\n=== Phase 1: Explorer Agent ===")
    async with runtime.coder_mcp_readonly() as coder_mcp:
        explorer = AgentWrapper[LibrarySummary].create(
            name="Explorer",
            instructions=EXPLORER_PROMPT,
            model=model,
            mcp_servers=[coder_mcp],
            output_type=LibrarySummary,
        )

        result = await explorer.run(
            "Please explore the library and generate the summary. The library is located in 'repos/library'.",
            max_turns=60,
        )

    library_summary = result.final_output().summary
    print("✅ Library Summary Generated.")

    # Save for reference
    summary_path = workspace_dir / "library_summary.md"
    summary_path.write_text(library_summary)
    return library_summary


async def run_topic_generation_phase(
    model,
    runtime: LocalRuntime,
    lib_dest: Path,
    workspace_dir: Path,
    library_summary: str,
):
    """Phase 2: Detailed Topic Generation."""
    print("\n=== Phase 2: Detailed Topic Generation ===")

    # Initialize Topic DB
    db = TopicDatabase(db_path=str(workspace_dir / "topics.json"))

    # List files to process
    src_dir = lib_dest / "src"
    files_to_process = list(src_dir.rglob("*.rs"))
    print(f"Found {len(files_to_process)} files to process.")

    if not files_to_process:
        return

    # Select interesting files
    interesting_files = [
        "lib.rs",
        "array.rs",
        "linalg/mod.rs",
        "stats.rs",
        "traits/implementations.rs",
    ]
    selected_files = []
    for name in interesting_files:
        found = next((f for f in files_to_process if str(f).endswith(name)), None)
        if found:
            selected_files.append(found)

    for f in files_to_process:
        if len(selected_files) >= 5:
            break
        if f not in selected_files:
            selected_files.append(f)

    print(f"Testing Topic Extraction on {len(selected_files)} files:")
    for f in selected_files:
        print(f" - {f.relative_to(lib_dest)}")

    # Loop over selected files
    async with runtime.coder_mcp_readonly() as coder_mcp:
        for target_file in selected_files:
            relative_path = target_file.relative_to(lib_dest)
            agent_file_path = Path("repos/library") / relative_path
            print(f"\nProcessing: {agent_file_path}")

            prompt = DETAILED_TOPIC_GENERATOR_PROMPT_TEMPLATE.format(
                file_path=str(agent_file_path), library_summary=library_summary
            )

            topic_agent = AgentWrapper[str].create(
                name=f"TopicGen-{relative_path.name}",
                instructions=prompt,
                model=model,
                mcp_servers=[coder_mcp],
                tools=[register_topic],
            )

            ctx = ContextType(db=db, target_file_rel=str(relative_path))
            try:
                await topic_agent.run(
                    f"Analyze {agent_file_path} and register topics.",
                    context=ctx,
                    max_turns=30,
                )
            except Exception as e:
                print(f"Error processing {relative_path}: {e}")

    print("\n=== Topics in DB ===")
    for t in db.topics:
        print(f"- [{t.id}] {t.title}: {t.related_apis}")


async def main():
    load_dotenv()
    add_trace_processor(AgentContentPrinter())

    # Config
    model_name = "gemini/gemini-3-flash-preview"
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    workspace_dir = Path("workspace_new_curriculum").resolve()
    repository_path = Path("repositories/numrs").resolve()

    if not repository_path.exists():
        repository_path = Path(__file__).parent / "repositories/numrs"

    print(f"Workspace: {workspace_dir}")
    print(f"Library: {repository_path}")

    # Prepare Workspace
    lib_dest = await prepare_workspace(workspace_dir, repository_path)

    # Model
    model = LitellmModel(model=model_name, api_key=api_key)

    # Initialize Environment and Run Phases
    async with LocalRuntime(workdir=str(workspace_dir)) as runtime:
        library_summary = await run_explorer_phase(model, runtime, workspace_dir)
        await run_topic_generation_phase(
            model, runtime, lib_dest, workspace_dir, library_summary
        )


if __name__ == "__main__":
    asyncio.run(main())
