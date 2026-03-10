import asyncio
from pathlib import Path

import tinker
from oai_utils.tinker import TinkerModel, setup_tinkermodel

from adapter_agent.hierarchical.agent.analyzer import Analyzer
from adapter_agent.hierarchical.agent.task_verifier import TaskVerifier
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.process.rewire import ss_solve
from adapter_agent.hierarchical.process.rewire_session import (
    RewireSessionResultNormal,
    RewireSessionResultSuccess,
    log_trajectory,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.model_helper import get_gemini


async def study_rollout(
    task: Task,
    solver_model: TinkerModel,
    verifier: Verifier,
):

    ret = await ss_solve(
        solver_model=solver_model,
        verifier=verifier,
        rewirer_model=solver_model,
        task=task,
        max_turns=5,
        qwen_no_think=True,
    )
    if isinstance(ret, RewireSessionResultNormal):
        log_trajectory(ret.trials)
    else:
        print("Session failed")
        return
    if isinstance(ret, RewireSessionResultSuccess):
        print("Agent generated successful trajectory")
        return
    analyzer = Analyzer(model=solver_model)
    subtask = await analyzer.analyze_trajectory(ret.trials)
    print(f"New Task: {subtask.instruction}")

    task_verifier = TaskVerifier(model=get_gemini())
    verification_result = await task_verifier.verify_task(
        task=subtask, library_name="numrs2"
    )
    print(f"Verification Result: {verification_result.output_type}")
    pass


async def main():
    # task = Task.from_instruction(
    #     "Please implement a three layer neural network with ReLU activation function and 100 units in each layer. Only the forward path should be implemented."
    # )
    # task = Task.from_instruction(
    #     instruction="Create a function which does 3d rotate operation of vectors around given point. The input should be N x 3 Array object and your function should return the array of the same shape but with rotated coordinates."
    # )

    # task = Task.from_instruction(
    #     "Task 1: Write a function that applies a simple 3D rotation matrix to a single vector using `numrs2` library."
    # )
    task = Task.from_instruction(
        "Implement a function that multiplies two 3x3 matrices using the `numrs2` library. The function should take two 3x3 matrices as input and return their product."
    )
    service_client = tinker.ServiceClient()
    model, _tokenizer, _renderer = setup_tinkermodel(
        service_client=service_client, model_name="Qwen/Qwen3-8B"
    )

    rust_doc_analyzer = RustDocAnalyzer.from_libdir(Path("repositories/numrs"))

    verifier = Verifier(
        model=get_gemini(),
        rust_doc_analyzer=rust_doc_analyzer,
    )
    await study_rollout(task, model, verifier)


if __name__ == "__main__":
    asyncio.run(main())
