from adapter_agent.exam.repository import GitRepository, GitRepositoryDict
from typing import TypedDict
from pydantic import BaseModel
from pathlib import Path
import tempfile
import subprocess
import shutil
from loguru import logger


class CodingExam(BaseModel):
    id: str
    image_name: str
    project: GitRepository
    library: GitRepository
    solution_commit: str
    problem_commit: str
    question: str
    eval_rubric: str

    def setup_tempdir(self) -> Path:
        """Set up a temporary environment for the exam.

        1. Creates a random temporary directory.
        2. Clones the project repository into it.
        3. Handles permission issues (chmod 777).

        Returns:
            Path to the temporary directory.
        """
        temp_dir = tempfile.mkdtemp(prefix="exam_")
        temp_path = Path(temp_dir)
        logger.info(f"Created temporary environment at {temp_path}")

        try:
            # Clone the project repository
            logger.info(f"Cloning {self.project.local_dir} to {temp_path}")
            subprocess.run(
                ["git", "clone", str(self.project.local_dir), str(temp_path)],
                check=True,
                capture_output=True,
            )

            # Fix permissions
            logger.debug(f"Applying chmod -R 777 to {temp_path}")
            subprocess.run(["chmod", "-R", "777", str(temp_path)], check=True)

            return temp_path
        except Exception as e:
            logger.error(f"Failed to setup environment: {e}")
            # Clean up on failure
            if temp_path.exists():
                shutil.rmtree(temp_path)
            raise e


class CodingExamDict(TypedDict):
    id: str
    image_name: str
    project: GitRepositoryDict
    library: GitRepositoryDict
    solution_commit: str
    problem_commit: str
    question: str
