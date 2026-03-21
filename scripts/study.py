import asyncio
from pathlib import Path

import tinker
from oai_utils import AgentsSDKModel
from oai_utils.tinker import TinkerModel, setup_tinkermodel

from adapter_agent.hierarchical.agent.analyzer import Analyzer
from adapter_agent.hierarchical.agent.knowledge_slicer import KnowledgeSlicer
from adapter_agent.hierarchical.agent.knowledge_summarizer import KnowledgeSummarizer
from adapter_agent.hierarchical.process.rewire import ss_solve_verify
from adapter_agent.hierarchical.process.rewire_session import (
    RewireSessionResultFailure,
    RewireSessionResultNormal,
    RewireSessionResultSuccess,
    log_trajectory,
)
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.task_net import StudyTaskContext, TaskNetwork
from scripts.uniray import load_gh_archive


async def study_rollout(
    task_network: TaskNetwork,
    solver_model: TinkerModel,
    verifier_model: AgentsSDKModel,
    rust_doc_analyzer: RustDocAnalyzer,
    vis_path: Path,
):
    tasks_queue = []
    knowledge_summarizer = KnowledgeSummarizer(model=get_gemini())
    while True:
        async with StudyTaskContext.next_task_from_network(task_network) as current:
            task_network.save_visualization(vis_path)

            task = current.task
            knowledges_str = None
            if task.knowledges:
                print(
                    f"Summarizing {len(task.knowledges)} knowledges for task: {task.task.instruction}"
                )
                knowledges_str = await knowledge_summarizer.summarize(
                    task.knowledges, task_instruction=task.task.instruction
                )
                print(f"Summary generated: {knowledges_str}")

            ret = await ss_solve_verify(
                solver_model=solver_model,
                verifier_model=verifier_model,
                rust_doc_analyzer=rust_doc_analyzer,
                task=task.task,
                max_turns=10,
                qwen_no_think=True,
                runtime_settings=RuntimeSettings(
                    type="docker",
                    image_uri="coder-mcp-numrs2:latest",
                ),
                knowledges=knowledges_str,
            )

            if isinstance(ret, RewireSessionResultNormal):
                print("Session completed with conclusion:", ret.conclusion)
                log_trajectory(ret.trials, flip_tag=True)

            if isinstance(ret, RewireSessionResultSuccess):
                slicer = KnowledgeSlicer(model=get_gemini())
                _qras = await slicer.slice(ret.knowledge)
                print(f"Generated {len(_qras)} QRAs from normalized knowledge.")

            if not task.is_generation or not isinstance(
                ret, RewireSessionResultFailure
            ):
                current.register_result(ret, new_task=None)
                continue

            try:
                analyzer = Analyzer(model=get_gemini())
                subtask = await analyzer.analyze_trajectory(ret.trials)
                print(f"New Task: {subtask.instruction}")

                # task_verifier = TaskVerifier(model=get_gemini())
                # verification_result = await task_verifier.verify_task(
                #     task=subtask, library_name="numrs2"
                # )
                # print(f"Verification Result: {verification_result.output_type}")

                # if verification_result.output_type == "valid":
                current.register_result(ret, new_task=subtask)

                task_network.save_visualization(vis_path)
                print(f"TaskNetwork Visualized at {vis_path}")
                continue
            except Exception as e:
                current.register_result(ret, new_task=None)
                print(f"Subtask generation failed: {e}")
                continue


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
    # task = Task.from_instruction(
    #     "Implement a function that multiplies two 3x3 matrices using the `numrs2` library. The function should take two 3x3 matrices as input and return their product."
    # )

    service_client = tinker.ServiceClient()
    model, _tokenizer, _renderer = setup_tinkermodel(
        service_client=service_client, model_name="Qwen/Qwen3-8B"
    )

    rust_doc_analyzer = RustDocAnalyzer.from_libdir(Path("repositories/numrs"))

    verifier_model = get_gemini()
    tasks = load_gh_archive()

    task_network = TaskNetwork(tasks_pool=tasks[:1])

    vis_path = Path("data/graphviz/task_net.html")
    launch_interval = 5
    num_workers = 50

    tasks = []
    for i in range(num_workers):
        print(f"Launching worker {i + 1}/{num_workers}...")
        tasks.append(
            asyncio.create_task(
                study_rollout(
                    task_network, model, verifier_model, rust_doc_analyzer, vis_path
                )
            )
        )
        if i < num_workers - 1:
            await asyncio.sleep(launch_interval)

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
