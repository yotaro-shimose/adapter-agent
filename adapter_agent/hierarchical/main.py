from adapter_agent.hierarchical.state import SFTPool
import asyncio
import time
from pathlib import Path

import polars as pl
from agents import add_trace_processor
from dotenv import load_dotenv
from oai_utils.litellm import litellm_concurrent_limit
from oai_utils.tracing import AgentContentPrinter

from adapter_agent.hierarchical.h_agent import Agents
from adapter_agent.hierarchical.runner import process_task
from adapter_agent.hierarchical.state import TaskPool
from adapter_agent.hierarchical.types import Task
from adapter_agent.model_helper import get_gemini


from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer


async def main():
    load_dotenv()
    add_trace_processor(AgentContentPrinter())

    model = get_gemini()
    # model = get_qwen32b().as_litellm_model()

    # Setup Experiment Directory
    experiment_id = f"hh_exp_{int(time.time())}"
    base_dir = Path("experiments") / "hierarchical" / experiment_id
    workspace_template_location = Path("templates") / "rust_template"
    lib_path = Path("repositories") / "numrs"

    # Ensure directories exist
    base_dir.mkdir(parents=True, exist_ok=True)
    assert workspace_template_location.exists()
    assert lib_path.exists()

    print(f"Experiment ID: {experiment_id}")
    print(f"Base Directory: {base_dir}")

    # Initialize RustDocAnalyzer
    doc_path = lib_path / "target" / "doc"
    pubapi_path = lib_path / "pubapi.txt"
    json_path = None
    if doc_path.exists():
        if (doc_path / "numrs2.json").exists():
            json_path = doc_path / "numrs2.json"
        else:
            jsons = list(doc_path.glob("*.json"))
            if jsons:
                json_path = jsons[0]

    if json_path and json_path.exists():
        print(f"Loading RustDocAnalyzer from {json_path}")
        rust_doc_analyzer = RustDocAnalyzer.from_json(
            json_path, pubapi_path=pubapi_path
        )
    else:
        raise FileNotFoundError(f"Could not find rustdoc json in {doc_path}")

    agents = Agents.from_model(model, rust_doc_analyzer)

    task_pool = TaskPool(tasks={})
    sft_pool = SFTPool.new()

    benchmark_df = pl.read_csv("experiments/gh/benchmark_dataset.csv").filter(
        pl.col("appropriate")
    )
    for row in benchmark_df.iter_rows(named=True):
        await task_pool.register(Task.from_instruction(row["problem_statement"]))

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
                    sft_pool=sft_pool,
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
    num_workers = 1
    async with litellm_concurrent_limit(num_workers):
        workers = [asyncio.create_task(worker(i)) for i in range(num_workers)]
        await asyncio.gather(*workers)


if __name__ == "__main__":
    asyncio.run(main())
