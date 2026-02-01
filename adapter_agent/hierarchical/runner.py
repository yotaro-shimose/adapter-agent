import logging
from pathlib import Path

from coder_mcp.runtime.rust_env import RustCodingEnvironment
from coder_mcp.runtime.temp_workspace import TempWorkspace

from adapter_agent.hierarchical.h_agent import Agents
from adapter_agent.hierarchical.state import SFTPool, TaskPool
from adapter_agent.hierarchical.types import Task
from adapter_agent.hierarchical.verifier import QA

logger = logging.getLogger(__name__)


async def process_task(
    agents: Agents,
    task: Task,
    task_pool: TaskPool,
    sft_pool: SFTPool,
    workspace_template_location: Path,
    host_lib_dir: Path,
    experiment_dir: Path | None = None,
):
    """
    1. リファレンスも使いながらエージェントがとく。
    2. Taskを解くのに成功したとエージェントが判断した場合
        ...
    """
    logger.info(f"Processing Task: {task.instruction}")

    # Inject the already prepared library
    injections = {host_lib_dir: f"repositories/{host_lib_dir.name}"}

    async def decompose_and_register(task: Task):
        new_tasks = await agents.decomposer.decompose(
            task.instruction, host_lib_dir.name
        )
        for new_task in new_tasks:
            logger.info(f"Generated practice task: {new_task.instruction}")
            await task_pool.register(new_task)

    async with TempWorkspace(
        workspace_template_location, injections=injections
    ) as temp_workspace:
        async with RustCodingEnvironment(workspace_dir=temp_workspace) as rust_env:
            solver_result = await agents.solver.try_solve(
                task, rust_env, host_lib_dir.name
            )

            if solver_result.is_max_turns_exceeded:
                logger.info("Solver max turns exceeded. Decomposing task...")
                await decompose_and_register(task)

            elif isinstance(solver_result.qa, QA):
                logger.info("Solver produced a QA. Verifying...")
                verification_result = await agents.verifier.verify(
                    solver_result.qa, rust_env
                )
                if verification_result.success:
                    logger.info("Verification SUCCESS.")
                    await sft_pool.register(solver_result.qa)
                    # Task is effectively done (popped from pool by caller or here?)
                    pass
                else:
                    # logger.info("Verification FAILED.")
                    # logger.info(verification_result.reasoning)
                    # solver_result.trajectory.add_item(
                    #     {
                    #         "role": "user",
                    #         "content": f"We ran verification process with another agent, but verification failed: {verification_result.reasoning}",
                    #     }
                    # )

                    # analysis = await agents.analyzer.analyze_trajectory(
                    #     solver_result.trajectory, rust_env
                    # )
                    # logger.info(f"Generated subtask: {analysis.instruction}")
                    # await task_pool.register(analysis)
                    logger.info("Verification FAILED, Decomposing task...")
                    await decompose_and_register(task)
            else:
                # Normal failure (report_failure called or other implicit failure without timeout)
                # logger.info("Solver failed to produce QA. Analyzing trajectory...")
                # try:
                #     trajectory_analysis = await agents.analyzer.analyze_trajectory(
                #         solver_result.trajectory, rust_env
                #     )
                # except AgentRunFailure as e:
                #     logger.error(f"Analyzer failed: {e}, skipping task.")
                #     return
                # logger.info(f"Generated subtask: {trajectory_analysis.instruction}")
                # await task_pool.register(trajectory_analysis)
                logger.info("Normal failure, Decomposing task...")
                await decompose_and_register(task)

    # Save state at the end of the task
    if experiment_dir is not None:
        task_pool.save(experiment_dir / "task_pool.json")
        sft_pool.dataset.save(experiment_dir / "sft_dataset.json")
        agents.solver.memory.save(experiment_dir / "memory_solver.json")
        agents.verifier.memory.save(experiment_dir / "memory_verifier.json")
        agents.analyzer.memory.save(experiment_dir / "memory_analyzer.json")
        agents.decomposer.memory.save(experiment_dir / "memory_decomposer.json")
