from dataclasses import dataclass

from coder_mcp.runtime.runtime import Runtime
from oai_utils.agent import AgentsSDKModel, AgentWrapper
from pydantic import BaseModel

from adapter_agent.hierarchical.types import Memory, Task, Trajectory


class TaskResponse(BaseModel):
    instruction: str

    def to_task(self) -> Task:
        return Task.from_instruction(self.instruction)


@dataclass
class Analyzer:
    model: AgentsSDKModel
    memory: Memory[Trajectory, Task]

    async def analyze_trajectory(
        self, trajectory: Trajectory, runtime: Runtime
    ) -> Task:
        """
        Trajectoryを分析してなぜSolverが失敗したのかを理解する。
        Solverが解けるであろうより小さな問題を生成する。
        """
        print("Details: Analyzing failure trajectory...")

        PROMPT = f"""
You are a Senior Engineer analyzing a Junior Engineer's failure.
The Junior Engineer tried to solve a task but failed.
You can see the current state of the workspace where the Junior Engineer finished execution.
The workspace is a cargo-initialized project.

Here is the execution log (Trajectory):
{trajectory.as_str()}

Your goal is to create a *simpler* sub-task that helps bridge the gap.
The sub-task should be:
1. Self-contained.
2. Easier than the original failed task.
3. Related to the specific failure point.

Return a new Task with a clear instruction.
"""
        # Analyzer doesn't necessarily need runtime tools, but consistency helps.
        # We can run it without coder_mcp if it's just text processing, but let's give it just in case.
        async with runtime.coder_mcp() as coder_mcp:
            agent = AgentWrapper[TaskResponse].create(
                name="Analyzer",
                instructions=PROMPT,
                model=self.model,
                mcp_servers=[coder_mcp],
                output_type=TaskResponse,
            )

            try:
                result = await agent.run(
                    "Analyze the trajectory and generate a sub-task.", max_turns=30
                )
                task = result.final_output().to_task()
                self.memory.add(trajectory, task)
                return task
            except Exception as e:
                print(f"Analyzer failed: {e}")
                # Fallback task
                return Task.from_instruction(
                    instruction="Read the numrs documentation and summarize basic usage.",
                )
