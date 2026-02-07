import logging
from dataclasses import dataclass
from pathlib import Path

from coder_mcp.runtime.rust_env import RustCodingEnvironment
from coder_mcp.runtime.temp_workspace import TempWorkspace

from adapter_agent.hierarchical.agent.solver import Solver
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.qra import QA
from adapter_agent.hierarchical.agent.verifier import VerificationResult
from tinker_cookbook.rl.types import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class SolveVerifyResult:
    qa: QA | None
    verification_result: VerificationResult | None
    trajectory: Trajectory | None
    reasoning: str | None = None
    is_max_turns_exceeded: bool = False


async def solve_verify(
    solver: Solver,
    verifier: Verifier,
    task: Task,
    workspace_template: Path,
    library_name: str,
    max_turns: int = 15,
    collect_trajectory: bool = False,
    use_search: bool = True,
    exclude: list[str] | None = None,
) -> SolveVerifyResult:
    """
    Encapsulates the pattern of:
    1. Creating a TempWorkspace
    2. Setting up a RustCodingEnvironment
    3. Generating a file tree
    4. Solving the task (with or without search)
    5. Verifying the solution if a QA is produced
    """
    if exclude is None:
        exclude = ["target", ".git"]

    async with TempWorkspace(workspace_template) as temp_workspace:
        async with RustCodingEnvironment(workspace_dir=temp_workspace) as rust_env:
            tree_structure = await rust_env.tree(".", exclude=exclude, truncate=20)

            if use_search:
                solver_result = await solver.try_solve(
                    task,
                    rust_env,
                    library_name,
                    tree_structure,
                    max_turns=max_turns,
                    collect_trajectory=collect_trajectory,
                )
            else:
                solver_result = await solver.try_solve_without_search(
                    task,
                    rust_env,
                    library_name,
                    tree_structure,
                    max_turns=max_turns,
                    collect_trajectory=collect_trajectory,
                )

            qa = solver_result.qa
            trajectory = solver_result.trajectory if collect_trajectory else None
            is_max_turns_exceeded = solver_result.is_max_turns_exceeded
            verification_result = None
            reasoning = None

            if qa:
                logger.info("Solver produced a QA. Verifying...")
                verification_result = await verifier.verify(
                    qa, rust_env, tree_structure
                )
                if not verification_result.success:
                    reasoning = verification_result.reasoning
            else:
                logger.info("Solver failed to produce a QA.")

            return SolveVerifyResult(
                qa=qa,
                verification_result=verification_result,
                trajectory=trajectory,
                reasoning=reasoning,
                is_max_turns_exceeded=is_max_turns_exceeded,
            )
