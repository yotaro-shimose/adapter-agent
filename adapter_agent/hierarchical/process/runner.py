import asyncio
import logging
from pathlib import Path

from adapter_agent.hierarchical.agent.h_agent import Agents
from adapter_agent.hierarchical.state import RLGroup, RLPool, SFTPool, TaskPool
from adapter_agent.hierarchical.types import Task
from adapter_agent.hierarchical.process.solve_verify import solve_verify

logger = logging.getLogger(__name__)


async def process_task(
    agents: Agents,
    task: Task,
    task_pool: TaskPool,
    sft_pool: SFTPool,
    workspace_template_location: Path,
    host_lib_dir: Path,
    rl_pool: RLPool,
    experiment_dir: Path | None = None,
    group_size: int = 1,
):
    """
    1. リファレンスも使いながらエージェントがとく。
    2. Taskを解くのに成功したとエージェントが判断した場合
        ...
    """
    logger.info(f"Processing Task: {task.instruction}")

    async def decompose_and_register(task: Task):
        new_tasks = await agents.decomposer.decompose(
            task.instruction, host_lib_dir.name
        )
        for new_task in new_tasks:
            logger.info(f"Generated practice task: {new_task.instruction}")
            await task_pool.register(new_task)

    # Main Solve & Verify
    result = await solve_verify(
        solver=agents.solver,
        verifier=agents.verifier,
        task=task,
        workspace_template=workspace_template_location,
        library_name=host_lib_dir.name,
        collect_trajectory=True,  # runner.py seems to use trajectory for RLGroup
        use_search=True,
    )

    if result.is_max_turns_exceeded:
        logger.info("Solver max turns exceeded. Decomposing task...")
        await decompose_and_register(task)

    elif result.qa and result.verification_result:
        if result.verification_result.success:
            logger.info("Verification SUCCESS.")
            await sft_pool.register(result.qa)

            logger.info(
                f"Details: RL Pool active. Collecting group rollout (size {group_size})."
            )

            trajectories = [result.trajectory]
            rewards = [1.0]

            async def single_rollout():
                # helper handles workspace creation internally
                res = await solve_verify(
                    solver=agents.solver,
                    verifier=agents.verifier,
                    task=task,
                    workspace_template=workspace_template_location,
                    library_name=host_lib_dir.name,
                    collect_trajectory=True,
                    use_search=True,
                )

                traj = res.trajectory
                rew = 0.0
                if res.verification_result and res.verification_result.success:
                    rew = 1.0
                return traj, rew

            if group_size > 1:
                tasks_rl = [single_rollout() for _ in range(group_size - 1)]
                results = await asyncio.gather(*tasks_rl)
                for t, r in results:
                    if t is not None:
                        trajectories.append(t)
                        rewards.append(r)

            if trajectories:
                await rl_pool.register(
                    RLGroup(trajectories=trajectories, rewards=rewards)
                )
                logger.info(
                    f"Details: Registered RL Group with {len(trajectories)} items."
                )
        else:
            logger.info("Verification FAILED, Decomposing task...")
            await decompose_and_register(task)
    else:
        logger.info("Normal failure (no QA produced), Decomposing task...")
        await decompose_and_register(task)

    # Save state at the end of the task
    if experiment_dir is not None:
        task_pool.save(experiment_dir / "task_pool.json")
        sft_pool.dataset.save(experiment_dir / "sft_dataset.json")
        rl_pool.save(experiment_dir / "rl_pool.json")
        agents.solver.maybe_save(experiment_dir / "memory_solver.json")
        agents.verifier.maybe_save(experiment_dir / "memory_verifier.json")
        agents.analyzer.maybe_save(experiment_dir / "memory_analyzer.json")
        agents.decomposer.maybe_save(experiment_dir / "memory_decomposer.json")
