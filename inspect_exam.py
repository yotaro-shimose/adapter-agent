import asyncio
import json
import os

from dotenv import load_dotenv
from loguru import logger

from benchmark import BenchmarkConfig
from adapter_agent.exam.exam import CodingExam
from adapter_agent.exam.repository import GitRepository
from coder_mcp.runtime import DockerRuntime


async def main():
    load_dotenv()

    # 1. Load benchmark config to find project
    try:
        config = BenchmarkConfig.load("test_numrs_syllabus2")
    except FileNotFoundError:
        logger.error("Benchmark config 'test_numrs_syllabus2' not found.")
        return

    project_dir = config.get_project_dir()
    exams_path = project_dir / "exams.json"

    if not exams_path.exists():
        logger.error(f"No exams found at {exams_path}")
        return

    # 2. Load and select exam
    exams_data = json.loads(exams_path.read_text())
    print(f"\nAvailable Exams in {project_dir}:")
    for i, ex in enumerate(exams_data):
        print(f"[{i}] {ex.get('id', 'N/A')} - {ex.get('question', '')[:50]}...")

    try:
        choice = input("\nSelect exam index to inspect [0]: ").strip()
        idx = int(choice) if choice else 0
        exam_dict = exams_data[idx]
    except (ValueError, IndexError):
        logger.error("Invalid selection.")
        return

    exam = CodingExam.model_validate(exam_dict)

    # 3. Ask for commit selection (problem vs solution)
    commit_choice = (
        input("\nInspect problem (p) or solution (s) commit? [p]: ").strip().lower()
    )
    if commit_choice == "s":
        commit_hash = exam.solution_commit
        commit_label = "solution"
    else:
        commit_hash = exam.problem_commit
        commit_label = "problem"

    # 4. Setup temporary workspace
    print(f"\n--- Setting up environment for {exam.id} ({commit_label} commit) ---")
    temp_path = exam.setup_tempdir()

    try:
        # Checkout selected commit
        workspace_repo = GitRepository(local_dir=temp_path)
        print(f"Checking out {commit_label} commit: {commit_hash[:7]}")
        workspace_repo.run_git(["checkout", commit_hash])

        # 5. Global Image Override (if set)
        image_name = os.getenv("OPENHANDS_IMAGE_NAME", exam.image_name)

        # 6. Launch DockerRuntime
        print(f"\n🚀 Launching Docker container with image: {image_name}")
        async with DockerRuntime(
            workspace_dir=str(temp_path), image_name=image_name
        ) as runtime:
            container_name = runtime.container_name
            print("\n" + "=" * 50)
            print("EXAM INSPECTION READY")
            print("=" * 50)
            print(f"Container Name: {container_name}")
            print(f"Workspace Path: {temp_path}")
            print("\nTo attach to the container, run:")
            print(f"  docker exec -it {container_name} bash")
            print("\nPress Ctrl+C to stop the container and clean up.")
            print("=" * 50)

            # Keep alive
            try:
                while True:
                    await asyncio.sleep(3600)
            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print(f"Cleaning up temp dir: {temp_path}")
        import shutil

        if temp_path.exists():
            shutil.rmtree(temp_path)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
