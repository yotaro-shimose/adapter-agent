import logging
from dataclasses import dataclass

from agents import ModelSettings, RunContextWrapper, StopAtTools, function_tool
from coder_mcp.runtime.runtime import Runtime
from coder_mcp.types import CoderToolName
from oai_utils import RunResultWrapper
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from oai_utils.tinker.litellm_model import result_to_trajectory
from pydantic import BaseModel
from tinker_cookbook.rl.types import Trajectory

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.library.rust_doc_tools import (
    WithRustDocAnalyzer,
    search_docs,
    search_symbol,
)
from adapter_agent.qra import QA


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
    trajectory: Trajectory | None
    is_max_turns_exceeded: bool = False


SOLVER_CODER_TOOLS: list[CoderToolName] = [
    "bash",
    "view_file",
    "list_directory",
    "create_file",
    "str_replace",
    "delete_file",
]

CONTEXT = """\
<Context>
You are working in a cargo-initialized project.
You need to solve the problem using `{library_name}` library which is already installed as a dependency (e.g. via `cargo add`)
You have no access to its source code in the current directory.
</Context>
"""

EFFICIENCY_GUIDELINES = """\
<EfficiencyGuidelines>
You are evaluated on your efficiency.
**Minimize Turns**: Combine tool calls whenever possible. Use parallel tool execution to perform multiple actions (e.g., reading multiple files, creating multiple files) in a single turn.
</EfficiencyGuidelines>
"""


@dataclass(kw_only=True)
class Solver[T: AgentsSDKModel](BaseAgent[T, Task, SolverResult]):
    rust_doc_analyzer: RustDocAnalyzer

    async def try_solve(
        self,
        task: Task,
        runtime: Runtime,
        library_name: str,
        tree_structure: str = "",
        max_turns: int = 10,
        collect_trajectory: bool = True,
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

{CONTEXT.format(library_name=library_name)}

<HowTo>
You have access to a coding environment via tools.
You can write and run commands to test your solution.
You have access to a set of tools to understand the library usage:
- `search_docs`: Use this to find functionality keyworks, concepts, or how-to guides in the documentation.
- `search_symbol`: Use this to find specific types, functions, or traits by name.
Once you have created a solution, you have to confirm it works (to the best of your ability).
You must end with a call to `report_success` to create your final answer to the question.
</HowTo>

{EFFICIENCY_GUIDELINES}

<Guidelines>
Use main.rs as a playground to test your solution.
You have turn limit of {max_turns}.
If you hit the turn limit without reporting success, it will be considered as a failure.
Note your solution has to be fully self-contained including both source code and explanation so that we can verify the solution later.
Verification will solely based on your final solution and your source code in the coding environment will be discarded in verification.
You should not use release build for faster debugging.
</Guidelines>
"""
        async with runtime.coder_mcp(
            allowed_tool_names=SOLVER_CODER_TOOLS
        ) as coder_mcp:
            agent = AgentWrapper.create(
                name="Solver",
                instructions=PROMPT,
                model=self.model,
                tools=[
                    report_success,
                    search_docs,
                    search_symbol,
                ],
                mcp_servers=[coder_mcp],
                tool_use_behavior=StopAtTools(stop_at_tool_names=[report_success.name]),
                model_settings=ModelSettings(
                    parallel_tool_calls=True, tool_choice="required"
                ),
                reset_tool_choice=False,
            )

            context = SolverContext(rust_doc_analyzer=self.rust_doc_analyzer)

            crate_overview = self.rust_doc_analyzer.get_overview()

            try:
                ret: RunResultWrapper = await agent.run(
                    f"""
<Task>
{task.instruction}
</Task>

<Current Directory Structure>
{tree_structure}
</Current Directory Structure>

<Library Overview>
{crate_overview}
</Library Overview>
""",
                    max_turns=max_turns,
                    context=context,
                )
                if collect_trajectory:
                    trajectory = result_to_trajectory(ret)
                else:
                    trajectory = None
                if context.qra is not None:
                    result = SolverResult(qa=context.qra, trajectory=trajectory)
                else:
                    result = SolverResult(trajectory=trajectory)

                self.maybe_add_to_memory(task, result)
                return result

            except AgentRunFailure as e:
                if e.cause in [
                    "ContextWindowExceededError",
                    "BadRequestError",
                    "MaxTurnsExceeded",
                ]:
                    result = SolverResult(trajectory=None, is_max_turns_exceeded=True)
                    self.maybe_add_to_memory(task, result)
                    return result
                else:
                    raise
            except Exception as e:
                logger.error(f"Solver error: {e}")
                raise NotImplementedError

    async def try_solve_without_search(
        self,
        task: Task,
        runtime: Runtime,
        library_name: str,
        tree_structure: str = "",
        max_turns: int = 10,
        collect_trajectory: bool = True,
    ) -> SolverResult:
        """
        タスクを解いてみる。ドキュメント検索等の探索ツールは使用しない。
        エージェントが既に知識を持っている (memorized) かどうかをテストするために使用する。
        """

        PROMPT = f"""
<Role>
You are an expert Rust software engineer.
Your task is to solve user provided problem.
</Role>

{CONTEXT.format(library_name=library_name)}


<HowTo>
You have access to a coding environment via tools.
You can write and run commands to test your solution.
You DO NOT have access to library documents or source codes. You are expected to solve the problem based on your knowledge / recall.
Do not try to find library source code in the current directory.
Once you have created a solution, you must run the code and confirm it works (to the best of your ability).
You must end with a call to `report_success` to create your final answer to the question.
</HowTo>

{EFFICIENCY_GUIDELINES}

<Guidelines>
Use main.rs as a playground to test your solution.
There is no API documentation for you. You are trained on the dataset containing the usage of the library and needs to recall the usage of the library.
You have turn limit of {max_turns}.
If you hit the turn limit without reporting success, it will be considered as a failure.
Note your solution has to be fully self-contained including both source code and explanation so that we can verify the solution later.
Verification will solely based on your final solution and your source code in the coding environment will be discarded in verification.
You should not use release build for faster debugging.
</Guidelines>
"""
        async with runtime.coder_mcp(
            allowed_tool_names=SOLVER_CODER_TOOLS
        ) as coder_mcp:
            agent = AgentWrapper.create(
                name="Solver(NoSearch)",
                instructions=PROMPT,
                model=self.model,
                tools=[
                    report_success,
                ],
                mcp_servers=[coder_mcp],
                tool_use_behavior=StopAtTools(stop_at_tool_names=[report_success.name]),
                model_settings=ModelSettings(
                    parallel_tool_calls=True, tool_choice="required"
                ),
                reset_tool_choice=False,
            )

            context = SolverContext(rust_doc_analyzer=self.rust_doc_analyzer)

            crate_overview = self.rust_doc_analyzer.get_overview()

            try:
                ret: RunResultWrapper = await agent.run(
                    f"""
<Task>
{task.instruction}
</Task>

<Current Directory Structure>
{tree_structure}
</Current Directory Structure>

<Library Overview>
{crate_overview}
</Library Overview>
""",
                    max_turns=max_turns,
                    context=context,
                )
                if collect_trajectory:
                    trajectory = result_to_trajectory(ret)
                else:
                    trajectory = None
                if context.qra is not None:
                    result = SolverResult(qa=context.qra, trajectory=trajectory)
                else:
                    result = SolverResult(trajectory=trajectory)

                self.maybe_add_to_memory(task, result)
                return result

            except AgentRunFailure as e:
                if e.cause in [
                    "ContextWindowExceededError",
                    "BadRequestError",
                    "MaxTurnsExceeded",
                ]:
                    result = SolverResult(trajectory=None, is_max_turns_exceeded=True)
                    self.maybe_add_to_memory(task, result)
                    return result
                else:
                    raise
            except Exception as e:
                logger.error(f"Solver error: {e}")
                raise NotImplementedError
