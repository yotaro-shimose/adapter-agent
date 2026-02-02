from dataclasses import dataclass

from agents import ModelSettings, RunContextWrapper, StopAtTools, function_tool
from coder_mcp.runtime.runtime import Runtime
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from pydantic import BaseModel

from adapter_agent.hierarchical.types import Memory
from adapter_agent.qra import QA
from adapter_agent.library.rust_doc_tools import (
    WithRustDocAnalyzer,
    search_docs,
    search_symbol,
)
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer


class VerificationResult(BaseModel):
    success: bool
    reasoning: str


class VerifierContext(WithRustDocAnalyzer):
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


@dataclass
class Verifier:
    model: AgentsSDKModel
    memory: Memory[QA, VerificationResult]
    rust_doc_analyzer: RustDocAnalyzer

    async def verify(self, qra: QA, runtime: Runtime) -> VerificationResult:
        """
        Questionに対してAnswerが問題を解決できるものとなっているかどうかをコードの実行などを通じて検証して、QAが正しければTrueをリターンする。
        """
        PROMPT = """
<Role>
You are a Quality Assurance engineer for Rust code.
Your task is to verify the following Solution for the given Question.
</Role>

<Context>
The workspace is a cargo-initialized project.
You are starting from the state where the Solver agent finished its implementation.
</Context>

<HowTo>
You have access to documentation tools to understand the codebase:
- `search_docs`: Find functionality keywords, concepts in documentation.
- `search_symbol`: Find specific types/functions by name.

You must:
1. First, check if the provided code works without error.
2. After that, read the source code to check if the solution is not cheating and satisfies the question.
3. Then, report your finding using the `report_success` or `report_failure` tool.
</HowTo>

<Guidelines>
You should not use release build for faster debugging.
Include your deep analysis in the reasoning argument, but you do not have to edit the code unless you REALLY need to check its intermediate output.
While you have edit tools, you are not supposed to write dedicated test code.
Most of the time you just execute the provided code, or insert debug print statements at maximum.
</Guidelines>
"""
        async with runtime.coder_mcp() as coder_mcp:
            agent = AgentWrapper.create(
                name="Verifier",
                instructions=PROMPT,
                model=self.model,
                mcp_servers=[coder_mcp],
                tools=[
                    report_success,
                    report_failure,
                    search_docs,
                    search_symbol,
                ],
                model_settings=ModelSettings(
                    tool_choice="required", parallel_tool_calls=True
                ),
                tool_use_behavior=StopAtTools(
                    stop_at_tool_names=[report_success.name, report_failure.name]
                ),
                reset_tool_choice=False,
            )

            context = VerifierContext(rust_doc_analyzer=self.rust_doc_analyzer)
            crate_overview = self.rust_doc_analyzer.get_overview()

            input_prompt = f"""\
<Task>
{qra.question}
</Task>

<Answer to Verify>
{qra.answer}
</Answer to Verify>

<Crate Overview>
{crate_overview}
</Crate Overview>
"""

            try:
                await agent.run(input_prompt, max_turns=30, context=context)
                if context.result is None:
                    # Should not happen because of StopAtTools or MaxTurnsExceeded (which raises)
                    # But if the agent stops without calling a tool (e.g. natural stop), treating as failure
                    print("Verifier finished without calling report tool.")
                    return VerificationResult(
                        success=False,
                        reasoning="Verifier finished without reporting success or failure.",
                    )

                final_output = context.result
                self.memory.add(qra, final_output)
                return final_output
            except AgentRunFailure as e:
                print(f"Verification process failed: {e}")
                raise
