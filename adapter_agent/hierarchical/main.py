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
    await task_pool.register(
        Task.from_instruction(
            instruction="I'm analyzing stock prices and need to smooth the data. Could you implement a simple moving average function using `numrs2`? It should take a 1D array of prices and a window size, returning the smoothed series.",
        )
    )

    # Task 2: Formal specification (Machine Learning - Gradient Descent)
    await task_pool.register(
        Task.from_instruction(
            instruction="Implement a function `gradient_descent_step` using `numrs2`. Input: Feature matrix X (2D), Target vector y (1D), Current weights w (1D), Learning rate alpha (f64). Output: Updated weights vector w_new. Formula: w_new = w - alpha * (X^T * (X * w - y)) / n_samples."
        )
    )

    # Task 3: Direct functional instruction (Clustering / Geometry)
    await task_pool.register(
        Task.from_instruction(
            instruction="Write a Rust function using `numrs2` that computes the pairwise Euclidean distance between two sets of row vectors, A (NxD) and B (MxD). The result should be an NxM matrix where entry (i, j) is the distance between A[i] and B[j]."
        )
    )

    async def worker(worker_id: int):
        print(f"Worker {worker_id} started.")
        while True:
            task = await task_pool.pop_task()
            if task is None:
                print(f"Worker {worker_id} stopping (shutdown signal received).")
                break

            try:
                await process_task(
                    agents=agents,
                    task=task,
                    task_pool=task_pool,
                    sft_dataset=sft_dataset,
                    host_lib_dir=lib_path,
                    workspace_template_location=workspace_template_location,
                    experiment_dir=base_dir,
                )
            except Exception as e:
                print(
                    f"Worker {worker_id} encountered an error processing task {task.id}: {e}"
                )
            finally:
                # Mark task as finished regardless of success or failure
                await task_pool.finish_task(task)

    # Spawn workers
    num_workers = 5
    workers = [asyncio.create_task(worker(i)) for i in range(num_workers)]
    await asyncio.gather(*workers)


if __name__ == "__main__":
    asyncio.run(main())
