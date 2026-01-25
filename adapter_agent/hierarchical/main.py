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

    # Task 1: Conversational request (Time Series Analysis)
    task_pool.register(
        Task.from_instruction(
            instruction="I'm analyzing stock prices and need to smooth the data. Could you implement a simple moving average function using `numrs`? It should take a 1D array of prices and a window size, returning the smoothed series.",
        )
    )

    # Task 2: Formal specification (Machine Learning - Gradient Descent)
    task_pool.register(
        Task.from_instruction(
            instruction="Implement a function `gradient_descent_step` using `numrs`. Input: Feature matrix X (2D), Target vector y (1D), Current weights w (1D), Learning rate alpha (f64). Output: Updated weights vector w_new. Formula: w_new = w - alpha * (X^T * (X * w - y)) / n_samples."
        )
    )

    # Task 3: Direct functional instruction (Clustering / Geometry)
    task_pool.register(
        Task.from_instruction(
            instruction="Write a Rust function using `numrs` that computes the pairwise Euclidean distance between two sets of row vectors, A (NxD) and B (MxD). The result should be an NxM matrix where entry (i, j) is the distance between A[i] and B[j]."
        )
    )

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
