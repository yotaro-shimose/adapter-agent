import logging
from dataclasses import dataclass

from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from pydantic import BaseModel

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.hierarchical.state import TaskList
from adapter_agent.hierarchical.types import Task, TaskInstructionList

logger = logging.getLogger(__name__)


class DecomposerInput(BaseModel):
    original_instruction: str


@dataclass
class Decomposer[T: AgentsSDKModel](BaseAgent[T, DecomposerInput, TaskList]):
    async def decompose(
        self, original_instruction: str, library_name: str
    ) -> list[Task]:
        """
        Solverが時間切れ(MaxTurnsExceeded)で失敗した場合に呼ばれる。
        元のタスクが難しすぎたため、より簡単でself-containedな練習問題（サブタスク）を作成する。
        """

        PROMPT = f"""
<Role>
You are a Teacher mentoring a student learning new library.
The student failed to complete the task because it was too complex and too big.
</Role>

<Goal>
Your goal is to create 2-3 *conceptually simpler* practice tasks that isolate core concepts required for the original task.
It is often the case that the student needs to understand the library's basic API usage.
The goal is NOT to solve the original task, but to "practice" specific parts of it.
</Goal>

<Constraints>
1. **Completely Self-contained**: Each task must be a standalone assignment. It must NOT assume any prior files, state, or context from previous attempts.
2. **Fresh Start**: Assume the agent starts in a fresh, clean cargo project. The `{
            library_name
        }` library is already available. Do not ask the agent to create a sub-directory for the project.
3. **No Context Continuity**: Do NOT reference the failed task's context or imply continuity (e.g., AVOID phrases like "continue with...", "before implementing X...", "as part of the previous goal..."). The task must make sense to someone seeing it for the first time without any background knowledge.
4. **Simpler**: Must be significantly easier than the original task.
5. **Targeted**: Focus on specific skills or API usages that the agent struggled with.
6. **Immediate**: The agent should be able to start coding immediately.
</Constraints>

<Example>
If the original task was "Implement a complex Neural Network Layer", and they got stuck on matrix multiplication,
a good practice task would be:
"Write a function that performs simple matrix multiplication using the `{
            library_name
        }` library and prints the result."
</Example>

<OutputFormat>
You must return a `TaskInstructionList` object which contains a list of `TaskInstruction` objects.
Each `TaskInstruction` must have:
- `instruction`: The string instruction for the practice task.

IMPORTANT: Your output must be PURE JSON. Do NOT include any markdown formatting such as ```json ... ```.
Example Output:
{{
  "tasks": [
    {{
      "instruction": "Write a function that creates a 3x3 matrix of zeros using `ndarray`."
    }},
    {{
      "instruction": "Write a function that adds two 1D arrays using `ndarray`."
    }}
  ]
}}
</OutputFormat>
"""
        agent = AgentWrapper[TaskInstructionList].create(
            name="Decomposer",
            instructions=PROMPT,
            model=self.model,
            output_type=TaskInstructionList,
        )

        try:
            result = await agent.run(
                f"<Task>\n{original_instruction}\n</Task>", max_turns=10
            )
            decomposed_list = result.final_output()

            new_tasks = []
            for d_task in decomposed_list.tasks:
                new_tasks.append(Task.from_instruction(d_task.instruction))

            task_list = TaskList(tasks=new_tasks)

            self.maybe_add_to_memory(
                DecomposerInput(original_instruction=original_instruction),
                task_list,
            )
            return task_list.tasks
        except AgentRunFailure as e:
            logger.error(f"Decomposer failed: {e}, skipping.")
            return []
