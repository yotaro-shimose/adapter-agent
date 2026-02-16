import logging
from dataclasses import dataclass

from agents import ModelSettings, RunContextWrapper, StopAtTools, function_tool
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from pydantic import BaseModel

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.qra import QA

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
class Verifier[T: AgentsSDKModel](BaseAgent[T, QA, VerificationResult]):
    rust_doc_analyzer: RustDocAnalyzer

    async def verify(
        self,
        qra: QA,
        tree_structure: str,
        execution_output: str,
        main_rs_content: str,
    ) -> VerificationResult:
        """
        Questionに対してAnswerが問題を解決できるものとなっているかどうかをコードの実行結果などを通じて検証して、QAが正しければTrueをリターンする。
        """
        PROMPT = """
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
- If providing the solution in `main.rs` but the answer text says otherwise, verify based on the code's effectiveness.
</Guidelines>
"""
        agent = AgentWrapper.create(
            name="Verifier",
            instructions=PROMPT,
            model=self.model,
            mcp_servers=[],
            tools=[
                report_success,
                report_failure,
            ],
            model_settings=ModelSettings(
                tool_choice="required", parallel_tool_calls=True
            ),
            tool_use_behavior=StopAtTools(
                stop_at_tool_names=[report_success.name, report_failure.name]
            ),
            reset_tool_choice=False,
        )

        context = VerifierContext()
        crate_overview = self.rust_doc_analyzer.get_overview()

        input_prompt = f"""\
<Task>
{qra.question}
</Task>

<Answer to Verify>
{qra.answer}
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

<Crate Overview>
{crate_overview}
</Crate Overview>
"""

        try:
            await agent.run(input_prompt, max_turns=5, context=context)
            if context.result is None:
                return VerificationResult(
                    success=False,
                    reasoning="Verifier finished without reporting success or failure.",
                )

            final_output = context.result
            self.maybe_add_to_memory(qra, final_output)
            return final_output
        except AgentRunFailure as e:
            if e.cause == "MaxTurnsExceeded":
                return VerificationResult(
                    success=False,
                    reasoning="Verifier exceeded maximum number of turns.",
                )
            logger.error(f"Verification process failed: {e}")
            raise
