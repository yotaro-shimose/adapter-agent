import logging
from pathlib import Path

from oai_utils.async_utils import gather_with_semaphore

from adapter_agent.hierarchical.agent.solver import Solver
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.process.solve_verify import solve_verify
from adapter_agent.hierarchical.types import Task
from adapter_agent.qra import QA

logger = logging.getLogger(__name__)


async def evaluate_sample(
    i: int,
    qa: QA,
    solver: Solver,
    verifier: Verifier,
    workspace_template: Path,
) -> bool:
    logger.info(f"Evaluating Test Sample {i + 1}: {qa.question}")

    task = Task.from_instruction(qa.question)

    result = await solve_verify(
        solver=solver,
        verifier=verifier,
        task=task,
        workspace_template=workspace_template,
        library_name="numrs",
        max_turns=15,
        collect_trajectory=False,
        use_search=False,
        exclude=["target", ".git"],
    )

    if result.qa:
        if result.verification_result and result.verification_result.success:
            logger.info(f"Sample {i + 1}: SUCCESS")
            return True
        else:
            reasoning = (
                result.verification_result.reasoning
                if result.verification_result
                else "No verification result"
            )
            logger.info(f"Sample {i + 1}: FAILED (Verification Failed: {reasoning})")
    else:
        logger.info(f"Sample {i + 1}: FAILED (Solver produced no QA)")

    return False


async def run_evaluation(
    name: str,
    qas: list[QA],
    solver: Solver,
    verifier: Verifier,
    workspace_template: Path,
):
    logger.info(f"Starting Evaluation on {name} Set ({len(qas)} samples)...")

    evaluation_tasks = [
        evaluate_sample(i, qa, solver, verifier, workspace_template)
        for i, qa in enumerate(qas)
    ]

    results = await gather_with_semaphore(evaluation_tasks, max_concurrent=16)
    success_count = sum(results)

    logger.info(
        f"Evaluation on {name} Set Completed. Success Rate: {success_count}/{len(qas)}"
    )
