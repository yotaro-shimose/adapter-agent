import asyncio
import logging
from pathlib import Path

from agents import add_trace_processor
from dotenv import load_dotenv
from oai_utils.async_utils import gather_with_semaphore
from oai_utils.tracing import AgentContentPrinter

from adapter_agent.hierarchical.agent.augmentor import Augmentor
from adapter_agent.hierarchical.agent.solver import Solver
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.process.solve_verify import solve_verify
from adapter_agent.hierarchical.state import SFTDataset
from adapter_agent.hierarchical.types import Memory, Task
from adapter_agent.model_helper import get_gemini
from adapter_agent.qra import QA
from h_tinker import setup_rust_doc_analyzer

logger = logging.getLogger(__name__)


async def main():
    model = get_gemini()
    rust_doc_analyzer = setup_rust_doc_analyzer(Path("repositories/numrs"))

    solver = Solver(model=model, memory=Memory(), rust_doc_analyzer=rust_doc_analyzer)
    verifier = Verifier(
        model=model, memory=Memory(), rust_doc_analyzer=rust_doc_analyzer
    )
    NUM_GENERATED_QUESTIONS = 64
    augmentor = Augmentor(model=model, memory=None)

    task = Task.from_instruction(
        instruction="Please tell me how I can get the sum of a tensor on a specific dimension in numrs.",
    )

    verified_qas: list[QA] = []
    # Queue of tasks to process. Start with the initial task.
    tasks_to_process = [task]

    async def process_task(task: Task) -> list[Task]:
        logger.info(f"Processing Task: {task.instruction}")
        new_tasks = []
        result = await solve_verify(
            solver=solver,
            verifier=verifier,
            task=task,
            workspace_template=Path("templates/rust_template"),
            library_name="numrs",
            collect_trajectory=False,
            use_search=True,
        )

        logger.info(f"Solver Result: {result.qa}")

        if result.qa:
            if result.verification_result:
                logger.info(f"Verification Result: {result.verification_result}")
                if result.verification_result.success:
                    logger.info("Verification SUCCESS. Adding to dataset.")
                    verified_qas.append(result.qa)

                    if len(verified_qas) < NUM_GENERATED_QUESTIONS:
                        logger.info("Augmenting verified QA...")
                        augmented_tasks = await augmentor.augment(result.qa)
                        logger.info(f"Generated {len(augmented_tasks)} new tasks.")
                        new_tasks.extend(augmented_tasks)
                else:
                    logger.info(f"Verification FAILED: {result.reasoning}")
            else:
                # Should not happen if qa is present, unless helper logic changes
                logger.info("Solver produced QA but verification result is missing.")
        else:
            logger.info("Solver failed to produce a QA.")
        return new_tasks

    while len(verified_qas) < NUM_GENERATED_QUESTIONS and tasks_to_process:
        # Take a batch of tasks to process in parallel
        # We can process all currently available tasks, but maybe limit concurrency
        current_batch = tasks_to_process
        tasks_to_process = []

        logger.info(f"Starting batch of {len(current_batch)} tasks...")

        # Run tasks in parallel with semaphore
        results = await gather_with_semaphore(
            [process_task(t) for t in current_batch],
            max_concurrent=20,
        )

        # Collect new tasks from results
        for task_list in results:
            tasks_to_process.extend(task_list)

        # Shuffle or prioritize? For now, just extend.
        # If verified_qas reached 16 during execution, we stop at the start of next loop.

    logger.info(f"Collected {len(verified_qas)} verified QAs.")

    # Save to SFTDataset
    dataset_path = Path("generated_qas.json")
    dataset = SFTDataset(items=verified_qas)
    dataset.save(dataset_path)
    logger.info(f"Saved dataset to {dataset_path.absolute()}")

    for i, qa in enumerate(verified_qas):
        logger.info(f"QA {i + 1}: {qa.question}")


if __name__ == "__main__":
    load_dotenv()
    add_trace_processor(AgentContentPrinter())

    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
