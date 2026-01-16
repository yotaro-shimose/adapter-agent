import asyncio
import json
import os
import shutil

from agents.extensions.models.litellm_model import LitellmModel
from agents.tracing import add_trace_processor
from dotenv.main import load_dotenv
from loguru import logger
from oai_utils.tracing import AgentContentPrinter

from benchmark import BenchmarkConfig
from adapter_agent.exam.exam import CodingExam
from adapter_agent.exam.repository import GitRepository
from adapter_agent.exam.runner import evaluate_exam, solve_exam


async def main():
    load_dotenv()

    # Configure Model
    model_name = "gemini/gemini-3-flash-preview"
    model = LitellmModel(model=model_name, api_key=os.environ["GEMINI_API_KEY"])

    # Enable tracing
    add_trace_processor(AgentContentPrinter())

    # Load benchmark config
    try:
        config = BenchmarkConfig.load("test_numrs_syllabus2")
    except FileNotFoundError:
        logger.error("Benchmark config not found. Please run benchmark.py first.")
        return

    # Load exam from checkpoint
    project_dir = config.get_project_dir()
    exams_path = project_dir / "exams.json"

    exams_data = json.loads(exams_path.read_text())
    if not exams_data:
        logger.error("No exams found in exams.json")
        return

    # Take the first exam for evaluation
    exam_dict = exams_data[0]
    exam = CodingExam.model_validate(exam_dict)

    logger.info(f"Evaluating gold solution for exam: {exam.id}")

    # Create a random temporary directory for evaluation
    temp_path = exam.setup_tempdir()
    try:
        # Initialize GitRepository for the cloned workspace
        workspace_repo = GitRepository(local_dir=temp_path)

        # Config User for the workspace (needed for git operations during solve)
        workspace_repo.run_git(["config", "user.name", "OpenHands Exam Solver"])
        workspace_repo.run_git(["config", "user.email", "solver@openhands.ai"])

        # Checkout the problem commit to start solving
        logger.info(f"Checking out problem commit: {exam.problem_commit}")
        workspace_repo.run_git(["checkout", exam.problem_commit])

        # Solve the exam
        logger.info("--- Phase 1: Solving the Exam ---")

        await solve_exam(
            model=model,
            exam=exam,
            workspace_path=temp_path,
        )

        # Evaluate the exam
        logger.info("\n--- Phase 2: Evaluating the Solution ---")
        evaluation = await evaluate_exam(
            model=model,
            exam=exam,
            workspace_path=temp_path,
        )

        print("\n=== Evaluation Result ===")
        print(f"Score: {evaluation.score}/100")
        print(f"Description:\n{evaluation.description}")

        if evaluation.score == 100:
            print("\nSUCCESS: Observed perfect score for gold standard solution.")
        else:
            print(f"\nWARNING: Score is {evaluation.score}, expected 100.")
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


if __name__ == "__main__":
    asyncio.run(main())
