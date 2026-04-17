import logging
from dataclasses import dataclass
from typing import Literal

from oai_utils.agent import AgentsSDKModel, AgentWrapper
from pydantic import BaseModel

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.hierarchical.types import Task

logger = logging.getLogger(__name__)


class TaskVerificationResult(BaseModel):
    output_type: Literal[
        "success",
        "self-contained",
        "library-learning",
        "solvable",
        "affordable",
    ]


@dataclass(kw_only=True)
class TaskVerifier[T: AgentsSDKModel](BaseAgent[T]):
    async def verify_task(
        self, task: Task, library_name: str
    ) -> TaskVerificationResult:
        """
        与えられたタスクが、指定されたライブラリを学習するためのタスクとして適切かを判別する。
        """
        PROMPT = """\
You are an expert task evaluator. Your job is to verify if a given Task is appropriate for learning the specified library.
The task must satisfy ALL of the following 4 criteria:

1. self-contained: The task description must not have external references like "the code", "this file", etc. It must be completely self-contained.
2. library-learning: It must be a task where the user heavily learns how to use the given library.
3. solvable: It must be solvable ONLY by editing and running `main.rs`. For example, a task requiring the installation of other non-specified software or libraries is not allowed.
4. affordable: It must not require significant computational resources, time, or large datasets (e.g., training a random forest model using scikit-learn is not affordable, whereas a simple code transformation is).

Evaluate the given Task against these criteria.
If it satisfies ALL criteria perfectly, output "success" for output_type.
If it fails to satisfy one or more criteria, output the name of the first criterion it fails to meet (one of "self-contained", "library-learning", "solvable", or "affordable") for output_type.

Example output if successful:
{
  "output_type": "success"
}

Example output if failed to be self-contained:
{
  "output_type": "self-contained"
}
"""
        agent = AgentWrapper[TaskVerificationResult].create(
            name="TaskVerifier",
            instructions=PROMPT,
            model=self.model,
            output_type=TaskVerificationResult,
        )

        result = await agent.run(f"""\
Library Name: {library_name}

Task Instruction:
{task.instruction}
""")
        return result.final_output()
