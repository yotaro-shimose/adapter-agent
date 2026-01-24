import asyncio
import time
import warnings
from pathlib import Path

from agents.tracing import add_trace_processor
from coder_mcp.runtime import LocalRuntime
from dotenv.main import load_dotenv
from oai_utils.agent import AgentsSDKModel
from oai_utils.tracing import AgentContentPrinter
from oai_utils.vllm import RopeScaling, VLLMSetup

from adapter_agent.async_util import gather_with_semaphore
from curriculum.database import TopicDatabase
from curriculum.library import prepare_workspace, run_explorer_phase
from curriculum.topic import run_topic_generation_for_file

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


async def topic_generation_recursive(
    model: AgentsSDKModel,
    runtime: LocalRuntime,
    lib_dest: Path,
    workspace_dir: Path,
    library_summary: str,
    lib_name: str,
    max_concurrent: int = 5,
):
    # Initialize Topic DB
    data_dir = workspace_dir.parent
    db_path = data_dir / f"{lib_name}_topics.json"
    db = TopicDatabase(db_path=str(db_path))


async def run_topic_generation_phase(
    model: AgentsSDKModel,
    runtime: LocalRuntime,
    lib_dest: Path,
    workspace_dir: Path,
    library_summary: str,
    lib_name: str,
    max_concurrent: int = 5,
):
    """Phase 2: Detailed Topic Generation."""
    print("\n=== Phase 2: Detailed Topic Generation ===")

    # Initialize Topic DB
    data_dir = workspace_dir.parent
    db_path = data_dir / f"{lib_name}_topics.json"
    db = TopicDatabase(db_path=str(db_path))

    # List files to process
    src_dir = lib_dest / "src"
    files_to_process = list(src_dir.rglob("*.rs"))
    print(f"Found {len(files_to_process)} files to process.")

    if not files_to_process:
        return

    print(f"Topic Extraction on {len(files_to_process)} files:")
    for f in files_to_process:
        print(f" - {f.relative_to(lib_dest)}")

    # Process files in parallel
    async with runtime.coder_mcp_readonly() as coder_mcp:
        await gather_with_semaphore(
            [
                run_topic_generation_for_file(
                    model=model,
                    coder_mcp=coder_mcp,
                    target_file=f,
                    lib_dest=lib_dest,
                    db=db,
                    library_summary=library_summary,
                )
                for f in files_to_process
            ],
            max_concurrent=max_concurrent,
        )

    print("\n=== Topics in DB ===")
    for t in db.topics:
        print(f"- [{t.id}] {t.title}: {t.related_apis}")


async def main():
    load_dotenv()
    add_trace_processor(AgentContentPrinter())

    # Config
    max_concurrent = 100
    experiment_id = f"exp_{int(time.time())}"
    base_dir = Path("experiments") / experiment_id
    workspace_dir = base_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    repository_path = Path("repositories/numrs").resolve()

    if not repository_path.exists():
        repository_path = Path(__file__).parent / "repositories/numrs"

    print(f"Experiment: {experiment_id}")
    print(f"Base Directory: {base_dir}")
    print(f"Workspace: {workspace_dir}")
    print(f"Library: {repository_path}")

    # Prepare Workspace
    lib_dest = await prepare_workspace(workspace_dir, repository_path)

    # Model
    data_parallel_size = 1
    yarn = 4
    base_max_len = 32768
    vllm_setup = VLLMSetup(
        model="Qwen/Qwen3-8B",
        reasoning_parser="deepseek_r1",
        data_parallel_size=data_parallel_size,
        quantization="fp8",
        rope_scaling=RopeScaling(
            rope_type="yarn",
            factor=yarn,
            original_max_position_embeddings=base_max_len,
        ),
    )
    model = vllm_setup.as_litellm_model()

    # Initialize Environment and Run Phases
    async with LocalRuntime(workdir=str(workspace_dir)) as runtime:
        library_summary = await run_explorer_phase(model, runtime, workspace_dir)
        await run_topic_generation_phase(
            model,
            runtime,
            lib_dest,
            workspace_dir,
            library_summary,
            lib_name=repository_path.name,
            max_concurrent=max_concurrent,
        )


if __name__ == "__main__":
    asyncio.run(main())
