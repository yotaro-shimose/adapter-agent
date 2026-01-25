import shutil
from pathlib import Path

from agents import ModelSettings
from coder_mcp.runtime import LocalRuntime
from oai_utils.agent import AgentWrapper
from pydantic import BaseModel


class LibrarySummary(BaseModel):
    summary: str


EXPLORER_PROMPT = """You are an expert Rust Researcher.
Your goal is to explore the provided Rust library at `repos/library` and generate a **Comprehensive Library Summary** in `library_summary.md`.

STRATEGY (Breadth-First):
1. **Priority 1: Documentation**: Start by reading `README.md`, any files in `docs/`, and `examples/`. These provide the best high-level overview.
2. **Priority 2: Entry Points**: Read `src/lib.rs` and top-level module files (e.g., `mod.rs` or high-level structs) to understand the architecture.
3. **Priority 3: Key Source Code**: Read specific source files only if documentation and entry points are insufficient to describe the abstract.

CRITICAL:
- **AVOID Deep Dives**: Do NOT read hundreds of lines of implementation logic or private functions. Focus on public APIs, struct definitions, and trait declarations.
- **Maintain High-Level Perspective**: Your goal is a summary, not a code audit. If a file looks like complex internal logic, skip it.
- **Efficiency**: You have a limited turn budget. Do not waste turns reading large files line-by-line unless they are core to the library's identity.

Your summary should cover:
- High-level purpose of the library.
- Main modules and their responsibilities.
- Key structs and traits.
- Common usage patterns (look at examples).
"""


async def prepare_workspace(workspace_dir: Path, repository_path: Path) -> Path:
    """Prepare the workspace and copy the library."""
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    lib_dest = workspace_dir / "repos" / repository_path.name
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
            model_settings=ModelSettings(parallel_tool_calls=True),
        )

        result = await explorer.run(
            "Please explore the library and generate the summary. The library is located in 'repos/library'. You can read up to 15 files.",
            max_turns=60,
        )

    library_summary = result.final_output().summary
    print("✅ Library Summary Generated.")

    # Save for reference
    summary_path = workspace_dir.parent / "library_summary.md"
    summary_path.write_text(library_summary)
    return library_summary
