import asyncio
import time
from pathlib import Path

from agents import add_trace_processor
from dotenv import load_dotenv
from oai_utils.tracing import AgentContentPrinter

from adapter_agent.hierarchical.h_agent import Agents
from adapter_agent.hierarchical.runner import process_task
from adapter_agent.hierarchical.state import SFTDataset, TaskPool
from adapter_agent.hierarchical.types import Task
from adapter_agent.model_helper import get_gemini


async def main():
    load_dotenv()
    add_trace_processor(AgentContentPrinter())

    model = get_gemini()

    # Setup Experiment Directory
    experiment_id = f"hh_exp_{int(time.time())}"
    base_dir = Path("experiments") / experiment_id
    workspace_template_location = Path("templates") / "rust_template"
    lib_path = Path("repositories") / "numrs"

    # Ensure directories exist
    base_dir.mkdir(parents=True, exist_ok=True)
    assert workspace_template_location.exists()
    assert lib_path.exists()

    print(f"Experiment ID: {experiment_id}")
    print(f"Base Directory: {base_dir}")

    agents = Agents.from_model(model)

    task_pool = TaskPool(tasks={})
    sft_dataset = SFTDataset(items=[])

    # Initial Task
    initial_task = Task.from_instruction(
        instruction="Using the Rust programming language and the `numrs` library, implement a 2D Convolution (Conv2D) layer. The implementation should include a struct for the layer and a forward pass method.",
    )
    task_pool.register(initial_task)

    # Process loop (simple version)
    # Just run until pool empty or max steps
    max_steps = 3
    step = 0

    while step < max_steps:
        current_task = task_pool.pop_random()
        if not current_task:
            print("Task pool empty.")
            break

        await process_task(
            agents=agents,
            task=current_task,
            task_pool=task_pool,
            sft_dataset=sft_dataset,
            host_lib_dir=lib_path,
            workspace_template_location=workspace_template_location,
            experiment_dir=base_dir,
        )
        step += 1


if __name__ == "__main__":
    asyncio.run(main())
