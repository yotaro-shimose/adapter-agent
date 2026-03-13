import logging
from dataclasses import dataclass

from agents import (
    AgentsException,
    ModelSettings,
    RunContextWrapper,
    function_tool,
)
from coder_mcp.runtime import Runtime
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
    runtime: Runtime
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


@dataclass(kw_only=True)
class SimplifiedSolver[T: AgentsSDKModel](Solver[T]):
    """Solver with a simplified action space: str_replace (fixed to src/main.rs), run (cargo run), and report_success."""

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
        Simplified solver with search tools.
        """
        PROMPT = f"""
<Role>
You are a Rust engineer.
Your goal is to create a solution to achieve the user's request through the iterative process.
You create a solution using simple playground to find the correct code.
</Role>

<Context>
You are working in a cargo-initialized project.
You need to solve the problem using `{library_name}` library which is already installed as a dependency (e.g. via `cargo add`).
Note `{library_name}` is a new library you should be unfamilier with.
</Context>

<HowTo>
You have a simplified coding environment with the following tools:
- `{search.name}`: Use this to search both symbol names and documentation for functionality, concepts, or how-to guides.
- `{replace_and_run.name}`: Edit src/main.rs by finding and replacing an exact string. The target file is always src/main.rs. If replacement is successful, runs `cargo run` and returns the output.

First search the necessary information using {search.name} tool, and then use {replace_and_run.name} tool to replace the content and run the code.
Unverified answer gets automatically rejected. You should confirm your work by running the code and verifying the output.
Once you confirmed that the solution works, you must output your final answer as a complete, fully functioning source code enclosed in a ```rust ... ``` block. You can also provide any necessary explanation.
Note you MUST submit your answer after running the code and verifying the output. Code which does not work will be automatically rejected.
</HowTo>

<Guidelines>
You have turn limit of {max_turns}.
Once again, you MUST verify your answer. You should make your best efforts to avoid hallucination and make sure your answer is correct.
If you hit the turn limit without outputting your final answer, it will be considered as a failure.
Note your solution has to be fully self-contained including both fully functioning source code and explanation.
Your final answer must include exactly one ```rust\n<your_code_here>\n``` block. It's content will be pasted to main.rs and executed for verification.
When using the `{search.name}` tool, it is highly recommended to use only one keyword such as "array" or "conv2d" otherwise the search tool does not return anything.
</Guidelines>
"""

        agent = AgentWrapper[str].create(
            name="SimplifiedSolver",
            instructions=PROMPT,
            model=self.model,
            tools=[
                replace_and_run,
                search,
            ],
            mcp_servers=[],
            model_settings=ModelSettings(parallel_tool_calls=True, tool_choice="auto"),
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

            final_text = ret.final_output()
            if isinstance(final_text, str):
                import re

                code_match_1 = re.search(
                    r"```rust(.*?)```", final_text, flags=re.DOTALL
                )
                if not code_match_1:
                    logger.debug("SimplifiedSolver failed to output rust code.")
                    return SolverResult(trajectory=trajectory, cause="no_code_produced")

                original_code = code_match_1.group(1).strip()

                PROMPT2 = """
<Role>
You are an expert Rust engineer generating a training dataset for other AI agents. You are tasked with writing a first-person reasoning and explanation for solving a user's instruction.
</Role>

<Context>
You are given a user's instruction and the known correct answer which contains a correct Rust code block achieving the user's request.
</Context>

<HowTo>
You must output your response using `<reasoning>` and `<answer>` XML tags to simulate how a knowledgeable human engineer would solve the problem without using any external tools or documentation.

- `<reasoning>`: The reasoning process. This must start from "Okay, let's see." followed by briefly explaining the user's request. Then you start recalling the necessary knowledge to solve the problem. The reasoning should be described in the first person as if you remembered the knowledge by yourself without any external search. Do NOT mention any search tools, documentation searches, or trial-and-error process. 
- `<answer>`: The final answer to the question (including both explanation and the code block). The Rust code enclosed in ```rust ... ``` must be EXACTLY the same as the one provided in the known answer. Do not change a single character of the provided source code.
</HowTo>

<OutputFormat>
You must output exactly in this format:
<reasoning>
Okay, let's see. [Your reasoning here...]
</reasoning>
<answer>
[Your explanation here]
```rust
// <the exact code from the provided answer>
```
</answer>
</OutputFormat>
"""
                agent2 = AgentWrapper[str].create(
                    name="Internalizer",
                    instructions=PROMPT2,
                    model=self.model,
                    tools=[],
                    mcp_servers=[],
                    reset_tool_choice=False,
                )

                ret2: RunResultWrapper = await agent2.run(
                    f"Instruction:\n{task.instruction}\n\nKnown Answer:\n{final_text}",
                    max_turns=1,
                    context=context,
                )
                final_text2 = ret2.final_output()

                if isinstance(final_text2, str):
                    reasoning_match = re.search(
                        r"<reasoning>(.*?)</reasoning>", final_text2, flags=re.DOTALL
                    )
                    answer_match = re.search(
                        r"<answer>(.*?)</answer>", final_text2, flags=re.DOTALL
                    )
                    if not answer_match:
                        answer_match = re.search(
                            r"<answer>(.*)", final_text2, flags=re.DOTALL
                        )
                    if reasoning_match and answer_match:
                        answer_str = answer_match.group(1)
                        code_match_2 = re.search(
                            r"```rust(.*?)```", answer_str, flags=re.DOTALL
                        )

                        if (
                            not code_match_2
                            or code_match_2.group(1).strip() != original_code
                        ):
                            logger.debug("Internalizer produced different rust code.")
                            return SolverResult(
                                trajectory=trajectory, cause="code_mismatch"
                            )

                        context.qa = QRA(
                            question=context.question,
                            reasoning=reasoning_match.group(1).strip(),
                            answer=answer_match.group(1).strip(),
                        )
                    else:
                        logger.debug(
                            "Internalizer failed to parse <reasoning> or <answer> from final_text."
                        )

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
You must output your final answer in normal text using `<reasoning>` and `<answer>` XML tags.
- `<reasoning>`: The reasoning process. This must start from "Okay, let's see." followed by briefly explaining the user's request.
- `<answer>`: The final answer to the question (including both code and explanation). This answer must be self-contained.
</HowTo>

<OutputFormat>
You must output exactly in this format when you report the final answer:
<reasoning>
Okay, let's see. [Your reasoning here...]
</reasoning>
<answer>
[Your explanation here]
```rust
// <your_code_here>
```
</answer>
</OutputFormat>

{EFFICIENCY_GUIDELINES}

<Guidelines>
You can only edit src/main.rs. The current content of src/main.rs is provided below.
You must not read documents or source codes of the library.
You have turn limit of {max_turns}.
If you hit the turn limit without outputting your final answer, it will be considered as a failure.
Note your solution has to be fully self-contained including both source code and explanation so that we can verify the solution later.
Your final answer must include exactly one ```rust\n<your_code_here>\n``` block inside the `<answer>` tag.
Verification will solely based on your final solution and your source code in the coding environment will be discarded in verification.
</Guidelines>
"""

        agent = AgentWrapper.create(
            name="SimplifiedSolver",
            instructions=PROMPT,
            model=self.model,
            tools=[
                run_tool,
                str_replace_tool,
            ],
            mcp_servers=[],
            model_settings=ModelSettings(parallel_tool_calls=True, tool_choice="auto"),
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

            final_text = ret.final_output()
            if isinstance(final_text, str):
                import re

                reasoning_match = re.search(
                    r"<reasoning>(.*?)</reasoning>", final_text, flags=re.DOTALL
                )
                answer_match = re.search(
                    r"<answer>(.*?)</answer>", final_text, flags=re.DOTALL
                )
                if not answer_match:
                    answer_match = re.search(
                        r"<answer>(.*)", final_text, flags=re.DOTALL
                    )
                if reasoning_match and answer_match:
                    context.qa = QRA(
                        question=context.question,
                        reasoning=reasoning_match.group(1).strip(),
                        answer=answer_match.group(1).strip(),
                    )
                else:
                    logger.debug(
                        "SimplifiedSolver failed to parse <reasoning> or <answer> from final_text."
                    )

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
