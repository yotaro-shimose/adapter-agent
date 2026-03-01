import logging
from dataclasses import dataclass

from agents import (
    AgentsException,
    ModelSettings,
    RunContextWrapper,
    StopAtTools,
    function_tool,
)
from coder_mcp.runtime import RustCodingEnvironment
from coder_mcp.runtime.runtime import Runtime
from coder_mcp.types import CoderToolName
from oai_utils import RunResultWrapper
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from oai_utils.tinker.agent_sdk_model import raw_responses_to_trajectory
from pydantic import BaseModel
from tinker_cookbook.rl.types import Trajectory

from adapter_agent.data import QA
from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.library.rust_doc_tools import (
    WithRustDocAnalyzer,
    search_docs,
    search_symbol,
)


class SolverContext(WithRustDocAnalyzer):
    qa: QA | None = None


logger = logging.getLogger(__name__)


@function_tool
def report_success(
    wrapper: RunContextWrapper[SolverContext],
    question: str,
    answer: str,
) -> None:
    """
    Report that the task has been successfully solved. Always call this tool when you are done.
    Args:
        question: The original task instruction or a refined version of it.
        answer: The answer to the question (including both code and explanation).
    """
    wrapper.context.qa = QA(
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
    cause: str | None = None


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
class Solver[T: AgentsSDKModel](BaseAgent[T]):
    rust_doc_analyzer: RustDocAnalyzer

    async def try_solve(
        self,
        task: Task,
        runtime: RustCodingEnvironment,
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
Do not forget to call `report_success` at the end, otherwise your solution will be considered as a failure.
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

            try:
                ret: RunResultWrapper = await agent.run(
                    f"""
<Task>
{task.instruction}
</Task>

<Current Directory Structure>
{tree_structure}
</Current Directory Structure>


""",
                    max_turns=max_turns,
                    context=context,
                )
                if collect_trajectory:
                    trajectory = raw_responses_to_trajectory(ret.result.raw_responses)
                else:
                    trajectory = None
                if context.qa is not None:
                    result = SolverResult(qa=context.qa, trajectory=trajectory)
                else:
                    result = SolverResult(trajectory=trajectory, cause="no_qa_produced")

                return result

            except AgentRunFailure as e:
                cause_map = {
                    "ContextWindowExceededError": "context_window_exceeded",
                    "BadRequestError": "bad_request",
                    "MaxTurnsExceeded": "max_turns_exceeded",
                    "ModelBehaviourError": "model_behaviour_error",
                }
                if e.cause in [
                    "ContextWindowExceededError",
                    "BadRequestError",
                    "MaxTurnsExceeded",
                ]:
                    result = SolverResult(
                        trajectory=None,
                        is_max_turns_exceeded=True,
                        cause=cause_map[e.cause],
                    )
                    return result
                elif e.cause == "ModelBehaviourError":
                    result = SolverResult(
                        trajectory=None,
                        is_max_turns_exceeded=False,
                        cause="model_behaviour_error",
                    )
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
You must not read documents or source codes of the library.
Your actions are recorded. If you try to find library source code in the current directory, it will be considered as a failure.
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

            try:
                ret: RunResultWrapper = await agent.run(
                    f"""
<Task>
{task.instruction}
</Task>

<Current Directory Structure>
{tree_structure}
</Current Directory Structure>


""",
                    max_turns=max_turns,
                    context=context,
                )
                if collect_trajectory:
                    trajectory = raw_responses_to_trajectory(ret.result.raw_responses)
                else:
                    trajectory = None
                if context.qa is not None:
                    result = SolverResult(qa=context.qa, trajectory=trajectory)
                else:
                    logger.debug("Solver failed to produce a QA after successful run.")
                    result = SolverResult(trajectory=trajectory, cause="no_qa_produced")

                return result

            except AgentRunFailure as e:
                if (
                    isinstance(e.original, AgentsException)
                    and e.original.run_data is not None
                    and e.original.run_data.raw_responses
                ):
                    trajectory = raw_responses_to_trajectory(
                        e.original.run_data.raw_responses
                    )
                else:
                    trajectory = None
                cause_map = {
                    "ContextWindowExceededError": "context_window_exceeded",
                    "BadRequestError": "bad_request",
                    "MaxTurnsExceeded": "max_turns_exceeded",
                    "ModelBehaviourError": "model_behaviour_error",
                }
                if e.cause in [
                    "ContextWindowExceededError",
                    "BadRequestError",
                    "MaxTurnsExceeded",
                ]:
                    logger.debug(f"Solver failed to produce a QA due to {e.cause}")
                    result = SolverResult(
                        trajectory=trajectory,
                        is_max_turns_exceeded=True,
                        cause=cause_map[e.cause],
                    )
                    return result
                elif e.cause == "ModelBehaviourError":
                    logger.debug(f"Solver failed to produce a QA due to {e.cause}")
                    result = SolverResult(
                        trajectory=trajectory,
                        is_max_turns_exceeded=False,
                        cause="model_behaviour_error",
                    )
                    return result
                else:
                    raise
            except Exception as e:
                if "Request Entity Too Large" in str(e):
                    logger.warning(
                        "Somehow Request Entity Too Large. Returning empty trajectory with failure."
                    )
                    result = SolverResult(
                        trajectory=None,
                        is_max_turns_exceeded=False,
                        cause="request_entity_too_large",
                    )
                    return result
                logger.error(f"Solver error: {e}")
                raise NotImplementedError
