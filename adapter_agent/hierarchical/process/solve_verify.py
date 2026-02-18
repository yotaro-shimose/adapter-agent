import logging
from dataclasses import dataclass

from coder_mcp.runtime.rust_env import RustCodingEnvironment
from tinker_cookbook.rl.types import Trajectory

from adapter_agent.hierarchical.agent.solver import Solver
from adapter_agent.hierarchical.agent.verifier import VerificationResult, Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.qra import QA

logger = logging.getLogger(__name__)


@dataclass
class SolveVerifyResult:
    qa: QA | None
    verification_result: VerificationResult | None
    trajectory: Trajectory | None
    reasoning: str | None = None
    is_max_turns_exceeded: bool = False
    cause: str | None = None
    turns: int = 0


async def solve_verify(
    solver: Solver,
    verifier: Verifier,
    task: Task,
    image_name: str,
    library_name: str,
    max_turns: int = 15,
    collect_trajectory: bool = False,
    use_search: bool = True,
    exclude: list[str] | None = None,
) -> SolveVerifyResult:
    """
    Encapsulates the pattern of:
    1. Setting up a RustCodingEnvironment using a pre-built image
    2. Generating a file tree
    3. Solving the task (with or without search)
    4. Verifying the solution if a QA is produced
    """
    if exclude is None:
        exclude = ["target", ".git"]

    async with RustCodingEnvironment(
        image_name=image_name, workspace_dir=None
    ) as rust_env:
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
        cause = solver_result.cause
        verification_result = None
        reasoning = None

        # Count turns from trajectory transitions
        turns = len(trajectory.transitions) if trajectory else 0

        if qa:
            logger.info("Solver produced a QA. Running code and verifying...")

            # Pre-run execution and fetching content for the Verifier
            execution_output, success = await rust_env.run_cargo()
            if not success:
                return SolveVerifyResult(
                    qa=None,
                    verification_result=None,
                    trajectory=trajectory,
                    reasoning=None,
                    is_max_turns_exceeded=is_max_turns_exceeded,
                    cause="code_compilation_failed",
                    turns=turns,
                )
            main_rs_content = await rust_env.view_file("src/main.rs")

            verification_result = await verifier.verify(
                qa=qa,
                tree_structure=tree_structure,
                execution_output=execution_output,
                main_rs_content=main_rs_content,
            )

            if not verification_result.success:
                reasoning = verification_result.reasoning
                cause = "verification_failed"
        else:
            logger.info("Solver failed to produce a QA.")

        return SolveVerifyResult(
            qa=qa,
            verification_result=verification_result,
            trajectory=trajectory,
            reasoning=reasoning,
            is_max_turns_exceeded=is_max_turns_exceeded,
            cause=cause,
            turns=turns,
        )
