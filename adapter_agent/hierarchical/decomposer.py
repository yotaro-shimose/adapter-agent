import uuid
from dataclasses import dataclass

from oai_utils.agent import AgentsSDKModel, AgentWrapper
from pydantic import BaseModel

from adapter_agent.hierarchical.state import TaskList
from adapter_agent.hierarchical.types import Memory, Task, Trajectory


class DecomposerInput(BaseModel):
    trajectory: Trajectory
    original_instruction: str


@dataclass
class Decomposer:
    model: AgentsSDKModel
    memory: Memory[DecomposerInput, TaskList]

    async def decompose(
        self, trajectory: Trajectory, original_instruction: str, library_name: str
    ) -> list[Task]:
        """
        Solverが時間切れ(MaxTurnsExceeded)で失敗した場合に呼ばれる。
        元のタスクが難しすぎたため、より簡単でself-containedな練習問題（サブタスク）を作成する。
        """
        print("Details: Decomposing task due to MaxTurnsExceeded...")

        PROMPT = f"""
You are a Teacher mentoring a student (Agent).
The student failed to complete the following task because it was too difficult (Code limit exceeded).
Task: "{original_instruction}"

Here is the partial progress (Trajectory) of the student:
{trajectory.as_str()}

Your goal is to create 2-3 *conceptually simpler* practice tasks that isolate core concepts required for the original task.
The goal is NOT to solve the original task, but to "practice" specific parts of it.

Constraints for the new tasks:
1. **Completely Self-contained**: Each task must be a standalone assignment. It must NOT assume any prior files, state, or context from previous attempts.
2. **Fresh Start**: Assume the agent starts in a fresh, clean cargo project. The `{library_name}` library is already available. Do not ask the agent to create a sub-directory for the project.
3. **No Context Continuity**: Do NOT reference the failed task's context or imply continuity (e.g., AVOID phrases like "continue with...", "before implementing X...", "as part of the previous goal..."). The task must make sense to someone seeing it for the first time without any background knowledge.
4. **Simpler**: Must be significantly easier than the original task.
5. **Targeted**: Focus on specific skills or API usages that the agent struggled with.
6. **Immediate**: The agent should be able to start coding immediately.

Example:
If the original task was "Implement a complex Neural Network Layer", and they got stuck on matrix multiplication,
a good practice task would be:
"Write a function that performs simple matrix multiplication using the `{library_name}` library and prints the result."

Return a list of Tasks, each with a clear, self-contained instruction.
"""
        agent = AgentWrapper[TaskList].create(
            name="Decomposer",
            instructions=PROMPT,
            model=self.model,
            output_type=TaskList,
        )

        try:
            result = await agent.run("Create simplified practice tasks.", max_turns=10)
            task_list = result.final_output()
            for task in task_list.tasks:
                task.id = str(uuid.uuid4())

            self.memory.add(
                DecomposerInput(
                    trajectory=trajectory, original_instruction=original_instruction
                ),
                task_list,
            )
            return task_list.tasks
        except Exception as e:
            print(f"Decomposer failed: {e}")
            return [
                Task(
                    id=str(uuid.uuid4()),
                    instruction=f"Research the basics required for: {original_instruction}",
                )
            ]
