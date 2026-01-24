import asyncio
import warnings
from pathlib import Path

from agents import add_trace_processor
from coder_mcp.runtime import LocalRuntime
from dotenv.main import load_dotenv
from oai_utils.tracing import AgentContentPrinter

from adapter_agent.model_helper import get_gemini
from curriculum.database import TopicDatabase
from curriculum.library import prepare_workspace, run_explorer_phase
from curriculum.topic import run_topic_generation_for_file

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


async def main():
    load_dotenv()
    add_trace_processor(AgentContentPrinter())

    # --- CONFIGURATION ---
    experiment_name = "exp_debug260123"  # Temporal experiment name
    target_file_rel = "src/lib.rs"  # CHANGE THIS to debug different files
    # ---------------------

    experiment_dir = Path("experiments").resolve() / experiment_name
    workspace_dir = experiment_dir / "workspace"

    # Check if we messed up the experiment dir path in previous manual edits or not.
    # The snippet showed 'experiments/exp_1768614688'.

    repository_path = Path("repositories/numrs").resolve()

    # Prepare library (copy if needed) - reusing shared logic
    library_path = await prepare_workspace(workspace_dir, repository_path)

    # Model Setup (Same as gen_topics.py)
    model = get_gemini()

    # Database
    library_name = "numrs"
    db_path = experiment_dir / f"{library_name}_topics.json"
    db = TopicDatabase(db_path=str(db_path))

    # Library Summary - Logic to generate if missing
    summary_path = experiment_dir / "library_summary.md"

    async with LocalRuntime(workdir=str(workspace_dir)) as runtime:
        if not summary_path.exists():
            print(f"Library summary not found at {summary_path}. Generating...")
            library_summary = await run_explorer_phase(model, runtime, workspace_dir)
        else:
            library_summary = summary_path.read_text()

        # Target File
        target_file = library_path / target_file_rel
        if not target_file.exists():
            print(f"Error: Target file {target_file} not found.")
            return

        print(f"DEBUG: Processing {target_file_rel}")
        print(f"       Experiment: {experiment_name}")

        async with runtime.coder_mcp_readonly() as coder_mcp:
            await run_topic_generation_for_file(
                model=model,
                coder_mcp=coder_mcp,
                target_file=target_file,
                lib_dest=library_path,
                db=db,
                library_summary=library_summary,
            )


if __name__ == "__main__":
    asyncio.run(main())
