from pathlib import Path

from coder_mcp.runtime.rust_env import RustCodingEnvironment
from coder_mcp.runtime.temp_workspace import TempWorkspace

from adapter_agent.hierarchical.h_agent import Agents
from adapter_agent.hierarchical.state import SFTDataset, TaskPool
from adapter_agent.hierarchical.types import Task
from adapter_agent.hierarchical.verifier import QA


async def process_task(
    agents: Agents,
    task: Task,
    task_pool: TaskPool,
    sft_dataset: SFTDataset,
    workspace_template_location: Path,
    host_lib_dir: Path,
    experiment_dir: Path,
):
    """
    1. リファレンスも使いながらエージェントがとく。
    2. Taskを解くのに成功したとエージェントが判断した場合
        ...
    """
    print(f"Processing Task: {task.instruction}")

    # Inject the already prepared library
    injections = {host_lib_dir: f"repositories/{host_lib_dir.name}"}

    async with TempWorkspace(
        workspace_template_location, injections=injections
    ) as temp_workspace:
        async with RustCodingEnvironment(workspace_dir=temp_workspace) as rust_env:
            solver_result = await agents.solver.try_solve(
                task, rust_env, host_lib_dir.name
            )

            if solver_result.is_max_turns_exceeded:
                print("Solver timed out. Decomposing task...")
                new_tasks = await agents.decomposer.decompose(
                    solver_result.trajectory, task.instruction, host_lib_dir.name
                )
                for new_task in new_tasks:
                    print(f"Generated practice task: {new_task.instruction}")
                    await task_pool.register(new_task)

            elif isinstance(solver_result.qa, QA):
                print("Solver produced a QA. Verifying...")
                verification_result = await agents.verifier.verify(
                    solver_result.qa, rust_env
                )
                if verification_result.success:
                    print("Verification SUCCESS.")
                    sft_dataset.register(solver_result.qa)
                    # Task is effectively done (popped from pool by caller or here?)
                    pass
                else:
                    print("Verification FAILED.")
                    print(verification_result.reasoning)
                    solver_result.trajectory.add_item(
                        {
                            "role": "user",
                            "content": f"We ran verification process with another agent, but verification failed: {verification_result.reasoning}",
                        }
                    )

                    analysis = await agents.analyzer.analyze_trajectory(
                        solver_result.trajectory, rust_env
                    )
                    print(f"Generated subtask: {analysis.instruction}")
                    await task_pool.register(analysis)

            else:
                # Normal failure (report_failure called or other implicit failure without timeout)
                print("Solver failed to produce QA. Analyzing trajectory...")
                trajectory_analysis = await agents.analyzer.analyze_trajectory(
                    solver_result.trajectory, rust_env
                )
                print(f"Generated subtask: {trajectory_analysis.instruction}")
                await task_pool.register(trajectory_analysis)

    # Save state at the end of the task
    task_pool.save(experiment_dir / "task_pool.json")
    sft_dataset.save(experiment_dir / "sft_dataset.json")
    agents.solver.memory.save(experiment_dir / "memory_solver.json")
    agents.verifier.memory.save(experiment_dir / "memory_verifier.json")
    agents.analyzer.memory.save(experiment_dir / "memory_analyzer.json")
    agents.decomposer.memory.save(experiment_dir / "memory_decomposer.json")
