from agents import ModelSettings
import logging
from dataclasses import dataclass

from agents import RunContextWrapper, StopAtTools, function_tool
from coder_mcp.runtime.runtime import Runtime
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from pydantic import BaseModel

from adapter_agent.hierarchical.types import Memory, Task, Trajectory
from adapter_agent.qra import QA
from adapter_agent.library.rust_doc_tools import (
    WithRustDocAnalyzer,
    search_docs,
    search_symbol,
)
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer


class SolverContext(WithRustDocAnalyzer):
    qra: QA | None = None


logger = logging.getLogger(__name__)


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
        answer: The answer to the question (including both code and explanation).
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
    rust_doc_analyzer: RustDocAnalyzer

    async def try_solve(
        self, task: Task, runtime: Runtime, library_name: str
    ) -> SolverResult:
        """
        タスクを解いてみる。
        もしタスクを解くことができたらSolutionを生成してReturnする。
        もしタスクを解くことができなければ、実行結果からTrajectoryを生成してReturnする。
        """

        PROMPT = f"""
<Role>
You are an expert Rust software engineer.
Your task is to solve user provided problem.
</Role>

<Context>
You are working in a cargo-initialized project.
You need to solve the problem using `{library_name}` library which is already installed in the workspace.
</Context>

<HowTo>
You have access to a coding environment via tools.
You can write and run commands to test your solution.
You have access to a set of tools to understand the library usage:
- `search_docs`: Use this to find functionality keyworks, concepts, or how-to guides in the documentation.
- `search_symbol`: Use this to find specific types, functions, or traits by name.
Once you have defined a solution and confirmed it works (to the best of your ability), you MUST call the `report_success` tool.
If you find that you cannot solve the problem, you MUST call the `report_failure` tool with a reason.
</HowTo>

<EfficiencyGuidelines>
You are evaluated on your efficiency.
1. **Minimize Turns**: Combine tool calls whenever possible. Use parallel tool execution to perform multiple actions (e.g., reading multiple files, creating multiple files) in a single turn.
2. **Minimize Tokens**: Do not read large files unless necessary. Use `grep` or specific line ranges if you only need parts of a file.
</EfficiencyGuidelines>

<Guidelines>
You should not use release build for faster debugging.
If you hit the turn limit without reporting success or failure, it will be considered a failure.
Note your solution has to be fully self-contained including both source code and explanation so that we can verify the solution later.
Verification will solely based on your final solution and your source code in the coding environment will be discarded in verification.
</Guidelines>
"""
        async with runtime.coder_mcp() as coder_mcp:
            agent = AgentWrapper.create(
                name="Solver",
                instructions=PROMPT,
                model=self.model,
                tools=[
                    report_failure,
                    report_success,
                    search_docs,
                    search_symbol,
                ],
                mcp_servers=[coder_mcp],
                tool_use_behavior=StopAtTools(
                    stop_at_tool_names=[report_failure.name, report_success.name]
                ),
                model_settings=ModelSettings(parallel_tool_calls=True),
            )

            context = SolverContext(rust_doc_analyzer=self.rust_doc_analyzer)

            crate_overview = self.rust_doc_analyzer.get_overview()

            try:
                result = await agent.run(
                    f"""
<Task>
{task.instruction}
</Task>

<Crate Overview>
{crate_overview}
</Crate Overview>
""",
                    max_turns=20,
                    context=context,
                )
                trajectory = Trajectory(input_list=result.to_input_list())
                if context.qra is not None:
                    result = SolverResult(qa=context.qra, trajectory=trajectory)
                else:
                    result = SolverResult(trajectory=trajectory)

                self.memory.add(task, result)
                return result

            except AgentRunFailure as e:
                if e.cause in [
                    "ContextWindowExceededError",
                    "BadRequestError",
                    "MaxTurnsExceeded",
                ]:
                    logger.error(f"Details: Solver hit {e.cause}.")
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
                logger.error(f"Solver error: {e}")
                raise NotImplementedError
