import logging
from dataclasses import dataclass
from openai.types.shared import Reasoning
from agents import ModelSettings, RunContextWrapper, StopAtTools, function_tool
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from oai_utils.tinker import TinkerModel
from pydantic import BaseModel

from adapter_agent.data import QA
from adapter_agent.hierarchical.agent.base import BaseAgent

logger = logging.getLogger(__name__)


class VerificationResult(BaseModel):
    success: bool
    reasoning: str


class VerifierContext(BaseModel):
    result: VerificationResult | None = None


@function_tool
def report_success(
    wrapper: RunContextWrapper[VerifierContext],
    reasoning: str,
) -> None:
    """
    Report that the solution is correct and satisfies the question.
    Args:
        reasoning: The detailed reasoning for why the solution is correct.
    """
    wrapper.context.result = VerificationResult(
        success=True,
        reasoning=reasoning,
    )


@function_tool
def report_failure(
    wrapper: RunContextWrapper[VerifierContext],
    reasoning: str,
) -> None:
    """
    Report that the solution is incorrect or fails to satisfy the question.
    Args:
        reasoning: The detailed reasoning for why the solution is incorrect.
    """
    wrapper.context.result = VerificationResult(
        success=False,
        reasoning=reasoning,
    )


@dataclass(kw_only=True)
class Verifier[T: AgentsSDKModel](BaseAgent[T]):
    library_name: str
    qwen_no_think: bool = False

    async def verify(
        self,
        qa: QA,
        tree_structure: str,
        execution_output: str,
        main_rs_content: str,
    ) -> VerificationResult:
        """
        Questionに対してAnswerが問題を解決できるものとなっているかどうかをコードの実行結果などを通じて検証して、QAが正しければTrueをリターンする。
        """
        PROMPT = f"""
<Role>
You are a Quality Assurance engineer for Rust code.
Your task is to verify the following Solution for the given Question.
</Role>

<Context>
You are presented with the result of a Rust program execution.
The solution logic is contained in `main.rs`, and the execution output is provided.
</Context>

<HowTo>
You must:
1. Check the execution output. If it contains errors or panic messages, or incorrect results, report failure.
2. Check the source code `main.rs`. Ensure it addresses the question directly and isn't "cheating" (e.g., hardcoding the answer without logic, unless appropriate).
3. Compare the Question, Answer, Code, and Output to determine validty.
4. Report your finding using the `report_success` or `report_failure` tool.
</HowTo>

<Guidelines>
- If the execution output indicates a compilation error or runtime panic, report failure.
- If the output is logically incorrect for the question, report failure.
- If the provided answer is not self-contained verifiable answer, report failure, even when the execution output and main.rs are correct.
    - Carefully check if the answer contains the same verified code as main.rs.
- The solution must actually use the `{self.library_name}` library, not just import it. Concretely: `main.rs` must contain at least one `use {self.library_name}::...` import AND make a call to something brought in by that import (a function from the library, or a method on a type that came from the library). A solution that imports `{self.library_name}` but performs the real work via stdlib or hand-rolled functions has bypassed the task — report failure.
</Guidelines>
"""
        model_settings = ModelSettings(
            tool_choice="required",
            parallel_tool_calls=True,
            # Tinker does not support OpenAI-style reasoning effort, so only
            # set it for non-Tinker models.
            reasoning=None if isinstance(self.model, TinkerModel) else Reasoning(effort="none"),
        )
        agent = AgentWrapper.create(
            name="Verifier",
            instructions=PROMPT,
            model=self.model,
            mcp_servers=[],
            tools=[
                report_success,
                report_failure,
            ],
            model_settings=model_settings,
            tool_use_behavior=StopAtTools(
                stop_at_tool_names=[report_success.name, report_failure.name]
            ),
            reset_tool_choice=False,
        )

        context = VerifierContext()

        input_prompt = f"""\
<Task>
{qa.question}
</Task>

<Answer to Verify>
{qa.answer}
</Answer to Verify>

<Current Directory Structure>
{tree_structure}
</Current Directory Structure>

<Current src/main.rs>
{main_rs_content}
</Current src/main.rs>

<Execution Output>
{execution_output}
</Execution Output>
"""
        if self.qwen_no_think:
            input_prompt = "/no_think " + input_prompt

        try:
            await agent.run(input_prompt, context=context)
            if context.result is None:
                return VerificationResult(
                    success=False,
                    reasoning="Verifier finished without reporting success or failure.",
                )

            final_output = context.result
            return final_output
        except AgentRunFailure as e:
            if e.cause == "MaxTurnsExceeded":
                return VerificationResult(
                    success=False,
                    reasoning="Verifier exceeded maximum number of turns.",
                )
            logger.error(f"Verification process failed: {e}")
            raise
