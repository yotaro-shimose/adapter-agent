from dataclasses import dataclass

from agents import RunContextWrapper, StopAtTools, function_tool
from coder_mcp.runtime.runtime import Runtime
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from pydantic import BaseModel

from adapter_agent.hierarchical.types import Memory, Task, Trajectory
from adapter_agent.qra import QA


class SolverContext(BaseModel):
    qra: QA | None = None


@function_tool
def report_success(
    wrapper: RunContextWrapper[SolverContext],
    question: str,
    answer: str,
) -> None:
    """
    Report that the task has been successfully solved.
    Args:
        question: The original task instruction or a refined version of it.
        answer: The final solution (code and explanation).
    """
    wrapper.context.qra = QA(
        question=question,
        answer=answer,
    )


@function_tool
def report_failure() -> None:
    """
    Report that the task could not be solved.
    Args:
        reason: The reason for failure.
    """
    pass


class SolverResult(BaseModel):
    qa: QA | None = None
    trajectory: Trajectory
    is_max_turns_exceeded: bool = False


@dataclass
class Solver:
    model: AgentsSDKModel
    memory: Memory[Task, SolverResult]

    async def try_solve(
        self, task: Task, runtime: Runtime, library_name: str
    ) -> SolverResult:
        """
        タスクを解いてみる。
        もしタスクを解くことができたらSolutionを生成してReturnする。
        もしタスクを解くことができなければ、実行結果からTrajectoryを生成してReturnする。
        """
        print(f"Details: Solver attempting task: {task.instruction}")

        PROMPT = f"""
You are an expert Rust software engineer.
Your task is to solve the following problem:
{task.instruction}

You are working in a cargo-initialized project.
The `{library_name}` library source code is located at `workspace_dir/repos/{library_name}` in case you do not know its API usage.
The library is just for reference and is already installed in the workspace_dir, so you do not need to run `cargo add`.
You must not add the repository as path dependency. Stick with the version that is already installed.

You have access to a coding environment. You can write and run code to test your solution.
Once you have defined a solution and confirmed it works (to the best of your ability), you MUST call the `report_success` tool.
If you find that you cannot solve the problem, you MUST call the `report_failure` tool with a reason.

If you hit the turn limit without reporting success or failure, it will be considered a failure.
"""

        async with runtime.coder_mcp() as coder_mcp:
            agent = AgentWrapper.create(
                name="Solver",
                instructions=PROMPT,
                model=self.model,
                tools=[report_failure, report_success],
                mcp_servers=[coder_mcp],
                tool_use_behavior=StopAtTools(
                    stop_at_tool_names=[report_failure.name, report_success.name]
                ),
            )

            context = SolverContext()

            try:
                result = await agent.run(
                    "Please solve the task.", max_turns=30, context=context
                )
                trajectory = Trajectory(input_list=result.to_input_list())
                if context.qra is not None:
                    result = SolverResult(qa=context.qra, trajectory=trajectory)
                else:
                    result = SolverResult(trajectory=trajectory)

                self.memory.add(task, result)
                return result

            except AgentRunFailure as e:
                if e.cause == "MaxTurnsExceeded":
                    print("Details: Solver hit MaxTurnsExceeded.")
                    input_list = e.to_input_list()
                    if input_list:
                        trajectory = Trajectory(input_list=input_list)
                        result = SolverResult(
                            trajectory=trajectory, is_max_turns_exceeded=True
                        )
                        self.memory.add(task, result)
                        return result
                    else:
                        # Should not happen if to_input_list works, but fallback
                        trajectory = Trajectory(input_list=[])
                        result = SolverResult(
                            trajectory=trajectory, is_max_turns_exceeded=True
                        )
                        self.memory.add(task, result)
                        return result
                else:
                    raise
            except Exception as e:
                print(f"Solver error: {e}")
                raise NotImplementedError
