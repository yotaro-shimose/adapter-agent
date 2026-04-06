import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tinker
from oai_utils import AgentsSDKModel
from oai_utils.tinker import TinkerModel, setup_tinkermodel

from adapter_agent.hierarchical.agent.analyzer import Analyzer
from adapter_agent.hierarchical.agent.rewirer import log_trajectory
from adapter_agent.hierarchical.gh import load_gh_archive
from adapter_agent.hierarchical.process.rewire import ss_solve_verify
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.knowledge_db import KnowledgeDB
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.env.session_result import (
    RewireSessionResultFailure,
    RewireSessionResultNormal,
)
from adapter_agent.rl.rl_database import RLDatabase
from adapter_agent.rl.task_net import (
    StudyTask,
    StudyTaskCompleted,
    TaskContext,
    TaskNetwork,
    TaskResultContext,
    is_study,
)
from adapter_agent.util.exception import AllTasksCompleted
from adapter_agent.util.logger_util import setup_base_loglevel


@dataclass
class StudyActor:
    task_network: TaskNetwork
    solver_model: TinkerModel
    verifier_model: AgentsSDKModel
    rust_doc_analyzer: AsyncRustDocAnalyzer
    knowledge_db: KnowledgeDB
    rl_db: RLDatabase
    vis_path: Path
    json_path: Path

    async def run(self):
        while True:
            try:
                async with TaskContext.next_task_from_network(
                    self.task_network
                ) as current:
                    if is_study(current):
                        await self.study(current)
                    else:
                        # SlicingTask is now disabled in TaskNetwork,
                        # but we keep this as a no-op just in case.
                        pass
            except AllTasksCompleted:
                print("All tasks completed. Worker exiting.")
                break

    async def study(self, current: TaskResultContext[StudyTask, StudyTaskCompleted]):
        await self.task_network.save_json(self.json_path)
        await self.rl_db.update_graph_json(self.task_network.to_dict())
        task = current.task

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
            knowledge_db=self.knowledge_db,
        )

        if isinstance(ret, RewireSessionResultNormal):
            print("Session completed with conclusion:", ret.conclusion)
            log_trajectory(ret.trials, flip_tag=True)

            # 1. Handle newly discovered knowledge (SSOT)
            knowledge_id = None
            if (
                isinstance(ret, RewireSessionResultNormal)
                and ret.reward > 0
                and ret.knowledge
            ):
                knowledge_id = await self.rl_db.create_knowledge(
                    knowledge_id=ret.knowledge.id,
                    task_id=task.id,
                    instruction=task.task.instruction,
                    title=ret.knowledge.title,
                    content=ret.knowledge.content,
                )
                # Update in-memory TaskNetwork knowledge ID for visualization mapping
                task_meta = self.task_network.nodes[task.id]
                for k in task_meta.knowledges.values():
                    # Link to the generated knowledge
                    if k.id == ret.knowledge.id:
                        k.knowledge_id = knowledge_id

            # 2. Extract citations
            citations_data = [
                {
                    "knowledge_id": c.knowledge_id,
                    "turn_index": c.turn_index,
                    "content": c.content,
                    "title": c.title,
                }
                for c in ret.citations
            ]

            # 3. Save trajectory
            await self.rl_db.add_trajectory(
                task_id=task.id,
                instruction=task.task.instruction,
                conclusion=ret.conclusion,
                reward=ret.reward,
                trajectory=ret.trials,
                knowledge_id=knowledge_id,
                final_knowledge=ret.knowledge.content if ret.knowledge else None,
                final_knowledge_title=ret.knowledge.title if ret.knowledge else None,
                citations=citations_data,
            )

        if not task.is_generation or not isinstance(ret, RewireSessionResultFailure):
            current.register_result(task.complete(ret))
            await self.task_network.save_json(self.json_path)
            await self.rl_db.update_graph_json(self.task_network.to_dict())
            return

        try:
            analyzer = Analyzer(model=get_gemini())
            subtask = await analyzer.analyze_trajectory(ret.trials)
            print(f"New Task: {subtask.instruction}")

            current.register_result(task.complete(ret, new_task=subtask))

            await self.task_network.save_json(self.json_path)
            await self.rl_db.update_graph_json(self.task_network.to_dict())
            print(f"TaskNetwork Visualized at {self.json_path}")
            return
        except Exception as e:
            current.register_result(task.complete(ret, new_task=None))
            await self.rl_db.update_graph_json(self.task_network.to_dict())
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
    setup_base_loglevel()

    # Initialize RLDatabase for visualization
    experiment_name = f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rl_db = RLDatabase()
    await rl_db.connect()
    await rl_db.register_experiment(experiment_name)
    print(f"Experiment started: {experiment_name}")

    service_client = tinker.ServiceClient()
    model, _tokenizer, _renderer = setup_tinkermodel(
        service_client=service_client,
        model_name="Qwen/Qwen3-8B",
        # path="tinker://3e320229-9d95-5b1a-b989-702de6f3fa88:train:0/sampler_weights/step_550",
    )

    rust_doc_analyzer = await AsyncRustDocAnalyzer.create_from_libdir(
        Path("repositories/numrs")
    )

    # Initialize KnowledgeDB once for this experiment (Postgres as SSOT)
    knowledge_db = KnowledgeDB.for_experiment(experiment_name)
    await knowledge_db.initialize()

    verifier_model = get_gemini()
    tasks = load_gh_archive()

    task_network = TaskNetwork(tasks_pool=tasks[:1])

    # Initial graph sync to database
    await rl_db.update_graph_json(task_network.to_dict())

    vis_path = Path("data/graphviz/task_net.html")
    json_path = Path("graphvis/public/data.json")
    launch_interval = 5
    num_workers = 20

    tasks = []
    for i in range(num_workers):
        print(f"Launching worker {i + 1}/{num_workers}...")
        study_actor = StudyActor(
            task_network=task_network,
            solver_model=model,
            verifier_model=verifier_model,
            rust_doc_analyzer=rust_doc_analyzer,
            knowledge_db=knowledge_db,
            rl_db=rl_db,
            vis_path=vis_path,
            json_path=json_path,
        )
        tasks.append(asyncio.create_task(study_actor.run()))
        if i < num_workers - 1:
            await asyncio.sleep(launch_interval)

    try:
        await asyncio.gather(*tasks)
    finally:
        await rl_db.close()


if __name__ == "__main__":
    asyncio.run(main())
