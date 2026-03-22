import asyncio
from dataclasses import dataclass
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
    log_trajectory,
)
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.task_net import (
    SlicingTask,
    SlicingTaskCompleted,
    StudyTask,
    StudyTaskCompleted,
    TaskContext,
    TaskNetwork,
    TaskResultContext,
    is_slice,
    is_study,
)
from scripts.uniray import load_gh_archive


@dataclass
class StudyActor:
    task_network: TaskNetwork
    solver_model: TinkerModel
    verifier_model: AgentsSDKModel
    rust_doc_analyzer: RustDocAnalyzer
    summarizer_model: AgentsSDKModel
    vis_path: Path
    json_path: Path

    async def run(self):
        while True:
            async with TaskContext.next_task_from_network(self.task_network) as current:
                if is_study(current):
                    await self.study(current)
                elif is_slice(current):
                    await self.slice(current)

    async def slice(
        self, current: TaskResultContext[SlicingTask, SlicingTaskCompleted]
    ):
        slicer = KnowledgeSlicer(model=self.summarizer_model)
        try:
            qra = await slicer.slice(current.task.knowledge.knowledge)
            current.register_result(current.task.complete(qra))
            self.task_network.save_json(self.json_path)
        except Exception as e:
            print(f"Slicing failed: {e}")
            current.register_result(current.task.complete(None))
            self.task_network.save_json(self.json_path)

    async def study(self, current: TaskResultContext[StudyTask, StudyTaskCompleted]):
        knowledge_summarizer = KnowledgeSummarizer(model=self.summarizer_model)
        self.task_network.save_visualization(self.vis_path)
        self.task_network.save_json(self.json_path)

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
            solver_model=self.solver_model,
            verifier_model=self.verifier_model,
            rust_doc_analyzer=self.rust_doc_analyzer,
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

        if not task.is_generation or not isinstance(ret, RewireSessionResultFailure):
            current.register_result(task.complete(ret))
            self.task_network.save_json(self.json_path)
            return

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
            current.register_result(task.complete(ret, new_task=subtask))

            self.task_network.save_visualization(self.vis_path)
            self.task_network.save_json(self.json_path)
            print(f"TaskNetwork Visualized at {self.vis_path} and {self.json_path}")
            return
        except Exception as e:
            current.register_result(task.complete(ret, new_task=None))
            print(f"Subtask generation failed: {e}")
            return


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
    json_path = Path("graphvis/public/data.json")
    launch_interval = 5
    num_workers = 10

    tasks = []
    for i in range(num_workers):
        print(f"Launching worker {i + 1}/{num_workers}...")
        study_actor = StudyActor(
            task_network=task_network,
            solver_model=model,
            verifier_model=verifier_model,
            rust_doc_analyzer=rust_doc_analyzer,
            summarizer_model=verifier_model,
            vis_path=vis_path,
            json_path=json_path,
        )
        tasks.append(asyncio.create_task(study_actor.run()))
        if i < num_workers - 1:
            await asyncio.sleep(launch_interval)

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
