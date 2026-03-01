import logging
from dataclasses import dataclass

from agents import (
    AgentsException,
    ModelSettings,
    RunContextWrapper,
    StopAtTools,
    function_tool,
)
from coder_mcp.runtime.runtime import Runtime
from coder_mcp.runtime.rust_env import RustCodingEnvironment
from oai_utils import RunResultWrapper
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from oai_utils.tinker.agent_sdk_model import raw_responses_to_trajectory

from adapter_agent.data import QA, QRA
from adapter_agent.hierarchical.agent.solver import (
    CONTEXT,
    EFFICIENCY_GUIDELINES,
    Solver,
    SolverResult,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_tools import (
    WithRustDocAnalyzer,
    search,
)

logger = logging.getLogger(__name__)

CARGO_INIT_MAIN_RS = """\
fn main() {
    println!("Hello, world!");
}
"""


class SimplifiedSolverContext(WithRustDocAnalyzer):
    question: str
    runtime: RustCodingEnvironment
    remaining_turns: int
    qa: QA | None = None
    qra: QRA | None = None


@function_tool
async def run_tool(
    wrapper: RunContextWrapper[SimplifiedSolverContext],
) -> str:
    """
    Run `cargo run` in the workspace. Returns the combined stdout/stderr output and exit code.
    """
    output = await wrapper.context.runtime.run_cargo()
    wrapper.context.remaining_turns -= 1
    return (
        f"{output}\n<RemainingTurns>{wrapper.context.remaining_turns}</RemainingTurns>"
    )


@function_tool
async def str_replace_tool(
    wrapper: RunContextWrapper[SimplifiedSolverContext],
    old_str: str,
    new_str: str,
) -> str:
    """
    Find and replace an exact string in src/main.rs. The file target is always src/main.rs. Returns error if string not found or multiple matches. Shows context snippet after edit.
    """
    output, success = await wrapper.context.runtime.str_replace(old_str, new_str)
    wrapper.context.remaining_turns -= 1
    return (
        f"{output}\n<RemainingTurns>{wrapper.context.remaining_turns}</RemainingTurns>"
    )


@function_tool
async def replace_and_run(
    wrapper: RunContextWrapper[SimplifiedSolverContext],
    old_str: str,
    new_str: str,
) -> str:
    """
    Find and replace an exact string in src/main.rs. The file target is always src/main.rs. Returns error if string not found or multiple matches.
    If replacement is successful, runs `cargo run` and returns the output.
    """
    if wrapper.context.runtime is None:
        return "Runtime not initialized."

    replace_ret, replace_success = await wrapper.context.runtime.str_replace(
        old_str, new_str
    )
    if not replace_success:
        return replace_ret
    wrapper.context.remaining_turns -= 1

    run_ret, run_success = await wrapper.context.runtime.run_cargo()
    combined_output = f"""\\
<TextReplacementPerformed>
{replace_ret}
</TextReplacementPerformed>
<CargoRunResult>
{run_ret}
</CargoRunResult>

<RemainingTurns>
{wrapper.context.remaining_turns}
</RemainingTurns>
"""
    return combined_output


@function_tool
def internalize_solution(
    wrapper: RunContextWrapper[SimplifiedSolverContext],
    reasoning: str,
    answer: str,
) -> None:
    """
    Create a training data to internalize what you have learned.
    Since it is internalizing process, the reasoning should not contain any of your observations during the task solving process.

    Args:
        reasoning: The reasoning process as if you are recalling the knowledge from your memory. This must start from "Okay, let's see." followed by briefly explaining the user's request.
            Then you start recalling the knowledge.
        answer: The final answer to the question (including both code and explanation). This answer must be self-contained.
    """
    wrapper.context.qa = QRA(
        question=wrapper.context.question,
        reasoning=reasoning,
        answer=answer,
    )


@dataclass(kw_only=True)
class SimplifiedSolver[T: AgentsSDKModel](Solver[T]):
    """Solver with a simplified action space: str_replace (fixed to src/main.rs), run (cargo run), and report_success."""

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
        Simplified solver with search tools.
        """
        PROMPT = f"""
<Role>
You are an expert Rust software engineer.
Your goal is to create a solution and reasoning process to achieve the solution.
You first create a solution using simple playground and then you generate a training data to internalize what you have learned.
</Role>

{CONTEXT.format(library_name=library_name)}

<HowTo>
You have a simplified coding environment with the following tools:
- `{replace_and_run.name}`: Edit src/main.rs by finding and replacing an exact string. The target file is always src/main.rs. If replacement is successful, runs `cargo run` and returns the output.
- `{search.name}`: Use this to search BOTH symbol names and documentation for functionality, concepts, or how-to guides.

You can access library documents via search tools.
Once you confirmed that the solution works, you must end with a call to `{internalize_solution.name}` to create your final answer to the question.
</HowTo>

{EFFICIENCY_GUIDELINES}

<Guidelines>
You have turn limit of {max_turns}.
If you hit the turn limit without reporting success, it will be considered as a failure.
Note your solution has to be fully self-contained including both source code and explanation so that we can verify the solution later.
Typically you need to write the exact same code you used in the playground in the answer with ```rust ... ```.
Your reasoning process should be written as if you are recalling the knowledge from your memory.
- You must not mention about search tools like `I will search the documentation for X`.
- You should instead say `I recall that X is implemented as Y`.
</Guidelines>
"""

        agent = AgentWrapper.create(
            name="SimplifiedSolver",
            instructions=PROMPT,
            model=self.model,
            tools=[
                internalize_solution,
                replace_and_run,
                search,
            ],
            mcp_servers=[],
            tool_use_behavior=StopAtTools(
                stop_at_tool_names=[internalize_solution.name]
            ),
            model_settings=ModelSettings(
                parallel_tool_calls=True, tool_choice="required"
            ),
            reset_tool_choice=False,
        )

        context = SimplifiedSolverContext(
            rust_doc_analyzer=self.rust_doc_analyzer,
            runtime=runtime,
            question=task.instruction,
            remaining_turns=max_turns,
        )

        try:
            ret: RunResultWrapper = await agent.run(
                f"""
<Task>
{task.instruction}
</Task>

<Current Directory Structure>
{tree_structure}
</Current Directory Structure>

<Current src/main.rs>
{CARGO_INIT_MAIN_RS}
</Current src/main.rs>

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
                logger.debug(
                    "SimplifiedSolver failed to produce a QA after successful run."
                )
                result = SolverResult(trajectory=trajectory, cause="no_qa_produced")

            return result

        except AgentRunFailure as e:
            if (
                isinstance(e.original, AgentsException)
                and e.original.run_data is not None
                and e.original.run_data.raw_responses
                and collect_trajectory
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
                logger.debug(
                    f"SimplifiedSolver failed to produce a QA due to {e.cause}"
                )
                result = SolverResult(
                    trajectory=trajectory,
                    is_max_turns_exceeded=True,
                    cause=cause_map[e.cause],
                )
                return result
            elif e.cause == "ModelBehaviourError":
                logger.debug(
                    f"SimplifiedSolver failed to produce a QA due to {e.cause}"
                )
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
            logger.error(f"SimplifiedSolver error: {e}")
            raise

    def get_agent_without_tool(self, library_name: str, max_turns: int) -> AgentWrapper:
        PROMPT = f"""
<Role>
You are an expert Rust software engineer.
Your task is to solve user provided problem.
</Role>

{CONTEXT.format(library_name=library_name)}

<HowTo>
You have a simplified coding environment with only two tools:
- `str_replace`: Edit src/main.rs by finding and replacing an exact string. The target file is always src/main.rs.
- `run`: Run `cargo run` to compile and execute the code.
You DO NOT have access to library documents or source codes. You are expected to solve the problem based on your knowledge / recall.
Once you have created a solution, you must run the code and confirm it works (to the best of your ability).
You must end with a call to `report_success` to create your final answer to the question.
</HowTo>

{EFFICIENCY_GUIDELINES}

<Guidelines>
You can only edit src/main.rs. The current content of src/main.rs is provided below.
You must not read documents or source codes of the library.
You have turn limit of {max_turns}.
If you hit the turn limit without reporting success, it will be considered as a failure.
Note your solution has to be fully self-contained including both source code and explanation so that we can verify the solution later.
Verification will solely based on your final solution and your source code in the coding environment will be discarded in verification.
You should not use release build for faster debugging.
</Guidelines>
"""

        agent = AgentWrapper.create(
            name="SimplifiedSolver",
            instructions=PROMPT,
            model=self.model,
            tools=[
                internalize_solution,
                run_tool,
                str_replace_tool,
            ],
            mcp_servers=[],
            tool_use_behavior=StopAtTools(
                stop_at_tool_names=[internalize_solution.name]
            ),
            model_settings=ModelSettings(
                parallel_tool_calls=True, tool_choice="required"
            ),
            reset_tool_choice=False,
        )
        return agent

    async def try_solve_without_search(
        self,
        task: Task,
        runtime: Runtime,
        library_name: str,
        tree_structure: str = "",
        max_turns: int = 5,
        collect_trajectory: bool = True,
    ) -> SolverResult:
        """
        Simplified solver. Only edits src/main.rs and runs cargo run.
        The agent receives the current main.rs content upfront.
        """
        agent = self.get_agent_without_tool(library_name, max_turns)

        context = SimplifiedSolverContext(
            rust_doc_analyzer=self.rust_doc_analyzer,
            runtime=runtime,
            remaining_turns=max_turns,
            question=task.instruction,
        )

        try:
            ret: RunResultWrapper = await agent.run(
                f"""
<Task>
{task.instruction}
</Task>

<Current Directory Structure>
{tree_structure}
</Current Directory Structure>

<Current src/main.rs>
{CARGO_INIT_MAIN_RS}
</Current src/main.rs>

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
                logger.debug(
                    "SimplifiedSolver failed to produce a QA after successful run."
                )
                result = SolverResult(trajectory=trajectory, cause="no_qa_produced")

            return result

        except AgentRunFailure as e:
            if (
                isinstance(e.original, AgentsException)
                and e.original.run_data is not None
                and e.original.run_data.raw_responses
                and collect_trajectory
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
                logger.debug(
                    f"SimplifiedSolver failed to produce a QA due to {e.cause}"
                )
                result = SolverResult(
                    trajectory=trajectory,
                    is_max_turns_exceeded=True,
                    cause=cause_map[e.cause],
                )
                return result
            elif e.cause == "ModelBehaviourError":
                logger.debug(
                    f"SimplifiedSolver failed to produce a QA due to {e.cause}"
                )
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
            logger.error(f"SimplifiedSolver error: {e}")
            raise
