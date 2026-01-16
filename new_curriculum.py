import asyncio
import os
import shutil
from pathlib import Path

from agents.extensions.models.litellm_model import LitellmModel
from agents.tracing import add_trace_processor
from dotenv.main import load_dotenv
from oai_utils.agent import AgentWrapper
from oai_utils.tracing import AgentContentPrinter
from pydantic import BaseModel

from adapter_agent.exam.repository import chmod_recursive
from coder_mcp.runtime.rust_env import RustCodingEnvironment
from topic_db import Topic, TopicDatabase


class LibrarySummary(BaseModel):
    summary: str


# --- PROMPTS ---

EXPLORER_PROMPT = """You are an expert Rust Researcher.
Your goal is to explore the provided Rust library at `/workspace/repos/library` and generate a **Comprehensive Library Summary** in `/workspace/library_summary.md`.

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


async def main():
    load_dotenv()
    add_trace_processor(AgentContentPrinter())

    # Config
    model_name = "gemini/gemini-3-flash-preview"
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return

    workspace_dir = Path("workspace_new_curriculum").resolve()
    repository_path = Path(
        "repositories/numrs"
    ).resolve()  # Default, might need adjusting

    if not repository_path.exists():
        repository_path = Path(__file__).parent / "repositories/numrs"

    print(f"Workspace: {workspace_dir}")
    print(f"Library: {repository_path}")

    # Prepare Workspace
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    chmod_recursive(workspace_dir)

    lib_dest = workspace_dir / "repos" / "library"
    lib_dest.parent.mkdir(parents=True, exist_ok=True)
    if not lib_dest.exists():
        if repository_path.exists():
            shutil.copytree(repository_path, lib_dest, dirs_exist_ok=True)
            chmod_recursive(lib_dest)
        else:
            print(
                f"Warning: Repository path {repository_path} not found. Ensure it exists."
            )

    # Model
    model = LitellmModel(model=model_name, api_key=api_key)

    # Initialize Environment
    async with RustCodingEnvironment(
        workspace_dir=workspace_dir, image_name="coder-mcp"
    ) as runtime:
        # --- Phase 1: Explorer Agent ---
        print("\n=== Phase 1: Explorer Agent ===")
        # Check if summary already exists (skip if debugging/re-running?)
        # For now, always run.

        async with runtime.coder_mcp() as coder_mcp:
            explorer = AgentWrapper[LibrarySummary].create(
                name="Explorer",
                instructions=EXPLORER_PROMPT,
                model=model,
                mcp_servers=[coder_mcp],
                output_type=LibrarySummary,
            )

            result = await explorer.run(
                "Please explore the library and generate the summary.", max_turns=40
            )

        library_summary = result.final_output().summary
        print("✅ Library Summary Generated.")

        # Save for reference
        summary_path = workspace_dir / "library_summary.md"
        summary_path.write_text(library_summary)

        # --- Phase 2: Detailed Topic Generation ---
        print("\n=== Phase 2: Detailed Topic Generation ===")

        # Initialize Topic DB
        db = TopicDatabase(db_path=str(workspace_dir / "topics.json"))

        # List files to process
        # We want to process src/*.rs and maybe examples in the container
        # But we can list them from host since it's mounted
        src_dir = lib_dest / "src"
        files_to_process = list(src_dir.rglob("*.rs"))

        print(f"Found {len(files_to_process)} files to process.")

        # Test with just 1 file for now as per Todo
        if files_to_process:
            # Sort files to be deterministic or just pick 5 interesting ones
            # For now, let's pick 5 files, preferring some we know are interesting if present
            # Include 'traits/implementations.rs' as a NEGATIVE TEST (should yield no topics)
            interesting_files = [
                "lib.rs",
                "array.rs",
                "linalg/mod.rs",
                "stats.rs",
                "traits/implementations.rs",
            ]
            selected_files = []

            # First try to find interesting files
            for name in interesting_files:
                found = next(
                    (f for f in files_to_process if str(f).endswith(name)), None
                )
                if found:
                    selected_files.append(found)

            # Fill up with others if needed
            for f in files_to_process:
                if len(selected_files) >= 5:
                    break
                if f not in selected_files:
                    selected_files.append(f)

            print(f"Testing Topic Extraction on {len(selected_files)} files:")
            for f in selected_files:
                print(f" - {f.relative_to(lib_dest)}")

            # Context class for function tool
            from dataclasses import dataclass

            from agents import RunContextWrapper, function_tool

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

            # Loop over selected files
            async with runtime.coder_mcp() as coder_mcp:
                for target_file in selected_files:
                    relative_path = target_file.relative_to(lib_dest)
                    container_path = Path("/workspace/repos/library") / relative_path
                    print(f"\nProcessing: {container_path}")

                    prompt = DETAILED_TOPIC_GENERATOR_PROMPT_TEMPLATE.format(
                        file_path=str(container_path), library_summary=library_summary
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
                            f"Analyze {container_path} and register topics.",
                            context=ctx,
                            max_turns=30,  # Increased from 10
                        )
                    except Exception as e:
                        print(f"Error processing {relative_path}: {e}")

            print("\n=== Topics in DB ===")
            for t in db.topics:
                print(f"- [{t.id}] {t.title}: {t.related_apis}")


if __name__ == "__main__":
    asyncio.run(main())
