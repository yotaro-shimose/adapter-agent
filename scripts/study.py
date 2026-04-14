import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tinker
from oai_utils import AgentsSDKModel
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from prisma import Prisma

from adapter_agent.hierarchical.agent.analyzer import Analyzer
from adapter_agent.hierarchical.agent.rewirer import log_trajectory
from adapter_agent.hierarchical.gh import load_gh_archive
from adapter_agent.hierarchical.process.rewire import ss_solve_verify
from adapter_agent.internalize.studier import KnowledgeStudier
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.knowledge_db import KnowledgeDB
from adapter_agent.library.wiki_manager import WikiManager
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
    wiki_manager: WikiManager
    rl_db: RLDatabase
    studier: KnowledgeStudier
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
            wiki_manager=self.wiki_manager,
        )

        if isinstance(ret, RewireSessionResultNormal):
            print("Session completed with conclusion:", ret.conclusion)
            log_trajectory(ret.trials, flip_tag=True)

            # Save trajectory immediately
            # Note: knowledge_ids is currently empty, will be updated by Studier later
            await self.rl_db.add_trajectory(
                task_id=task.id,
                instruction=task.task.instruction,
                conclusion=ret.conclusion,
                reward=ret.reward,
                trajectory=ret.trials,
                knowledge_ids=[],
                final_knowledge=None,
                final_knowledge_title=None,
                citations=[
                    {
                        "knowledge_id": c.knowledge_id,
                        "turn_index": c.turn_index,
                        "content": c.content,
                        "title": c.title,
                    }
                    for c in ret.citations
                ],
            )

            # Enqueue for distillation if successful (and let Studier handle uniqueness & formalization)
            if ret.reward > 0:
                await self.studier.enqueue_trajectory(
                    task_id=task.id,
                    instruction=task.task.instruction,
                    reflections=ret.reflections,  # Use reflections instead of knowledges
                    trajectory=ret.trials,
                )

        # Mark task as completed IMMEDIATELY in TaskNetwork
        if not task.is_generation or not isinstance(ret, RewireSessionResultFailure):
            current.register_result(task.complete(ret))
        else:
            try:
                analyzer = Analyzer(model=get_gemini())
                subtask = await analyzer.analyze_trajectory(ret.trials)
                print(f"New Task: {subtask.instruction}")
                current.register_result(task.complete(ret, new_task=subtask))
            except Exception as e:
                current.register_result(task.complete(ret, new_task=None))
                print(f"Subtask generation failed: {e}")

        await self.task_network.save_json(self.json_path)
        await self.rl_db.update_graph_json(self.task_network.to_dict())
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

    parser = argparse.ArgumentParser(description="Wiki-Integrated Learning Pipeline")
    parser.add_argument("--version", type=str, default=None, help="Wiki version to use (defaults to experiment name)")
    parser.add_argument(
        "--reset", action="store_true", help="Reset the Wiki version before starting"
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of concurrent workers"
    )
    args = parser.parse_args()

    setup_base_loglevel()

    # Initialize RLDatabase for visualization
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"study_{timestamp}"
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

    # Initialize Prisma and WikiManager
    db = Prisma()
    await db.connect()
    wiki_version = args.version or experiment_name
    wiki_manager = WikiManager(db, version=wiki_version)

    if args.reset:
        await wiki_manager.reset()

    # Note: KnowledgeDB is kept for metadata/experiment tracking if needed
    knowledge_db = KnowledgeDB.for_experiment(experiment_name)
    await knowledge_db.initialize()

    verifier_model = get_gemini()
    tasks = load_gh_archive()

    task_network = TaskNetwork(tasks_pool=tasks[:5])

    # Initial graph sync to database
    await rl_db.update_graph_json(task_network.to_dict())

    json_path = Path("graphvis/public/data.json")
    launch_interval = 2
    num_workers = args.workers

    # Initialize KnowledgeStudier
    studier = KnowledgeStudier(
        verifier_model=verifier_model,
        wiki_manager=wiki_manager,
        rl_db=rl_db,
        runtime_settings=RuntimeSettings(
            type="docker",
            image_uri="coder-mcp-numrs2:latest",
        ),
        task_network=task_network,
    )
    await studier.start()

    tasks = []
    for i in range(num_workers):
        print(f"Launching worker {i + 1}/{num_workers}...")
        study_actor = StudyActor(
            task_network=task_network,
            solver_model=model,
            verifier_model=verifier_model,
            rust_doc_analyzer=rust_doc_analyzer,
            wiki_manager=wiki_manager,
            rl_db=rl_db,
            studier=studier,
            json_path=json_path,
        )
        tasks.append(asyncio.create_task(study_actor.run()))
        if i < num_workers - 1:
            await asyncio.sleep(launch_interval)

    try:
        await asyncio.gather(*tasks)
    finally:
        await studier.stop()
        await rl_db.close()


if __name__ == "__main__":
    asyncio.run(main())
