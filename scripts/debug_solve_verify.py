import asyncio
from pathlib import Path
import tinker
from agents.tracing import add_trace_processor
from oai_utils.tracing import AgentContentPrinter

from adapter_agent.hierarchical.agent.solver import Solver
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task

try:
    from adapter_agent.hierarchical.process.solve_verify import solve_verify
except ImportError:
    from adapter_agent.hierarchical.solve_verify import solve_verify

from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.util.logger_util import setup_base_loglevel
from adapter_agent.model_helper import get_gemini
from oai_utils.tinker import setup_tinkermodel


# Copy of helper from sft_tinker.py
def setup_rust_doc_analyzer(host_lib_dir: Path) -> RustDocAnalyzer:
    doc_path = host_lib_dir / "target" / "doc"
    pubapi_path = host_lib_dir / "pubapi.txt"
    json_path = None
    if doc_path.exists():
        if (doc_path / "numrs2.json").exists():
            json_path = doc_path / "numrs2.json"

    if json_path and json_path.exists():
        return RustDocAnalyzer.from_json(json_path, pubapi_path=pubapi_path)
    else:
        # Fallback or error
        # For debug script, we might want to try finding it or just fail
        raise FileNotFoundError(
            f"Could not find rustdoc json in {doc_path}. Ensure you are in the workspace root or path is correct."
        )


async def main():
    # setup_base_loglevel()
    # Add trace processor for printing behavior
    add_trace_processor(AgentContentPrinter())

    service_client = tinker.ServiceClient()
    model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    path = (
        "tinker://c25c1802-91fa-5e5d-badf-d1ef12a28c32:train:0/sampler_weights/000030"
    )

    print(f"Setting up model {model_name} from {path}...")
    model, _tokenizer, _renderer = setup_tinkermodel(service_client, model_name, path)

    # Setup dependencies for Solver/Verifier
    # We assume the script is run from project root, so repositories/numrs should be accessible
    host_lib_dir = Path("repositories/numrs")
    if not host_lib_dir.exists():
        # Try finding it relative to workspace root if we are in scripts dir
        # But usually we run from root. Let's start with check.
        if Path("../repositories/numrs").exists():
            host_lib_dir = Path("../repositories/numrs")
        else:
            print(f"Warning: repositories/numrs not found at {host_lib_dir.absolute()}")

    try:
        rust_doc_analyzer = setup_rust_doc_analyzer(host_lib_dir)
        print("RustDocAnalyzer setup successful.")
    except Exception as e:
        print(f"Failed to setup RustDocAnalyzer: {e}")
        return

    # Solver
    print("Initializing Solver and Verifier...")
    solver = Solver(model=model, rust_doc_analyzer=rust_doc_analyzer, memory=None)

    # Verifier (using Gemini as in sft_tinker.py)
    verifier_model = get_gemini()
    verifier = Verifier(
        model=verifier_model, rust_doc_analyzer=rust_doc_analyzer, memory=None
    )

    # Task
    user_input = "I am calculating the sum of a 2D array using the `numrs` library. The result is a 1D array, but I want to keep it as a 2D array with size 1 for the summed dimension. What arguments should I change to achieve this?"
    task = Task.from_instruction(user_input)

    workspace_template = Path("templates/rust_template")
    if not workspace_template.exists():
        if Path("../templates/rust_template").exists():
            workspace_template = Path("../templates/rust_template")

    print("\n--- Starting Solve Verify ---")
    result = await solve_verify(
        solver=solver,
        verifier=verifier,
        task=task,
        workspace_template=workspace_template,
        library_name="numrs2",
        max_turns=15,
        collect_trajectory=False,
        use_search=False,
    )

    print("\n--- Result ---")
    if result.qa:
        print(f"Question: {result.qa.question}")
        print(f"Answer: {result.qa.answer}")
    else:
        print("No QA produced.")

    if result.verification_result:
        print(f"Verification Success: {result.verification_result.success}")
        print(f"Reasoning: {result.verification_result.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
