import asyncio
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from oai_utils import gather_with_semaphore
from pydantic import BaseModel, Field

from adapter_agent.data import QA
from adapter_agent.hierarchical.agent.augmentor import Augmentor
from adapter_agent.hierarchical.agent.simplified_solver import SimplifiedSolver
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.process.solve_verify import solve_verify
from adapter_agent.hierarchical.state import QASFTDataset
from adapter_agent.hierarchical.types import Task
from adapter_agent.model_helper import get_gemini
from adapter_agent.util.logger_util import setup_base_loglevel
from h_tinker import setup_rust_doc_analyzer

logger = logging.getLogger(__name__)


async def augment(
    qa: QA | None,
    solver: SimplifiedSolver,
    verifier: Verifier,
    augmentor: Augmentor,
    num_questions: int,
) -> list[QA]:
    if qa is None:
        return []
    verified_qas: list[QA] = [qa]
    while len(verified_qas) < num_questions:
        augmented_tasks = await augmentor.augment(qa)
        results = await gather_with_semaphore(
            [
                solve_verify(
                    solver, verifier, task, "coder-mcp-numrs2:latest", "numrs2"
                )
                for task in augmented_tasks
            ],
            max_concurrent=10,
        )
        successful_qas = [
            result.qa
            for result in results
            if result.qa
            and result.verification_result
            and result.verification_result.success
        ]
        if len(successful_qas) == 0:
            logger.warning("No successful QAs generated in this iteration. Stop.")
            return verified_qas
        verified_qas.extend(successful_qas)
        logger.info(
            f"Augmented {len(successful_qas)} new QAs. Total: {len(verified_qas)}"
        )
    return verified_qas


async def gen_qa(
    task: Task, solver: SimplifiedSolver, verifier: Verifier, max_retry: int = 3
) -> QA | None:
    for _ in range(max_retry):
        result = await solve_verify(
            solver=solver,
            verifier=verifier,
            task=task,
            image_name="coder-mcp-numrs2:latest",
            library_name="numrs2",
            collect_trajectory=False,
            use_search=True,
        )
        if (
            result.qa
            and result.verification_result
            and result.verification_result.success
        ):
            return result.qa
    raise ValueError(f"Failed to generate QA for task: {task.instruction}")


class GenSFTConfig(BaseModel):
    experiment_dir: Path = Field(
        default_factory=lambda: Path(
            f"data/sft/gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )
    questions_per_task: int = 16
    max_retry: int = 5
    seeds: list[Task]


async def main():
    setup_base_loglevel()
    logging.getLogger("adapter_agent.hierarchical.agent.simplified_solver").setLevel(
        logging.DEBUG
    )
    config = GenSFTConfig(
        seeds=[
            Task.from_instruction(
                "How do I create a multi-dimensional array or tensor (equivalent to `numpy.ndarray` or `torch.Tensor`) in this library?"
            ),
            Task.from_instruction(
                "What is the API for initializing a tensor with specific values: `zeros`, `ones`, `constant`, and `identity`?"
            ),
            Task.from_instruction(
                "How do I perform random initialization (specifically `uniform`, `normal`, and `truncated_normal`)? Please provide the syntax for setting a manual seed."
            ),
            Task.from_instruction(
                "What is the exact API for Matrix Multiplication (GEMM)? Does it use an operator (like `@`) or a function (like `matmul` or `dot`)?"
            ),
            Task.from_instruction(
                "How does the library handle **broadcasting**? If I add a 1D bias vector to a 2D result matrix, what are the syntax requirements to ensure it 'stretches' correctly?"
            ),
            Task.from_instruction(
                "What are the APIs for matrix manipulation: `transpose`, `reshape`, `flatten`, `squeeze`, and `unsqueeze` (adding dimensions)?"
            ),
            Task.from_instruction(
                "How do I perform basic element-wise arithmetic (addition, subtraction, multiplication) between tensors of different shapes?"
            ),
        ],
        checkpoint_dir=Path("data/sft/gen_20260216_233815"),
    )

    model = get_gemini()
    rust_doc_analyzer = setup_rust_doc_analyzer(Path("repositories/numrs"))

    solver = SimplifiedSolver(model=model, rust_doc_analyzer=rust_doc_analyzer)
    verifier = Verifier(model=model, rust_doc_analyzer=rust_doc_analyzer)
    augmentor = Augmentor(model=model)

    qas = await gather_with_semaphore(
        [
            gen_qa(task, solver, verifier, max_retry=config.max_retry)
            for task in config.seeds
        ],
        max_concurrent=10,
    )
    verified_qas = await gather_with_semaphore(
        [
            augment(
                qa=qa,
                solver=solver,
                verifier=verifier,
                augmentor=augmentor,
                num_questions=config.questions_per_task,
            )
            for qa in qas
        ],
        max_concurrent=4,
    )
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    sft_dataset = QASFTDataset(items=[qa for qa_list in verified_qas for qa in qa_list])
    sft_dataset.save(config.experiment_dir / "sft_dataset.json")
    logger.info(f"Saved dataset to {config.experiment_dir.absolute()}")


if __name__ == "__main__":
    load_dotenv()
    # add_trace_processor(AgentContentPrinter())

    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
