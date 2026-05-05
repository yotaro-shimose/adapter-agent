import logging
import re
from dataclasses import dataclass
from openai.types.shared import Reasoning
from agents import ModelSettings, RunContextWrapper, StopAtTools, function_tool
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from pydantic import BaseModel

from adapter_agent.data import QA
from adapter_agent.hierarchical.agent.base import BaseAgent

logger = logging.getLogger(__name__)


def _build_stub_pattern(library_name: str) -> re.Pattern[str]:
    """Match a top-of-line `mod <library_name> {` declaration.

    Such a local module shadows the real crate so any `use <library_name>::...`
    resolves to the hand-rolled stub. Caller must escape `library_name` if it
    can contain regex metacharacters (real crate names are word chars, so the
    direct interpolation is fine in practice).
    """
    return re.compile(
        rf"^\s*(?:pub\s+)?mod\s+{re.escape(library_name)}\s*\{{",
        re.MULTILINE,
    )


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
        if _build_stub_pattern(self.library_name).search(main_rs_content):
            return VerificationResult(
                success=False,
                reasoning=(
                    f"Solution shadows the `{self.library_name}` crate by declaring a local "
                    f"`mod {self.library_name} {{ ... }}` block. Any `use {self.library_name}::...` "
                    f"then resolves to that hand-rolled stub instead of the real library, "
                    f"bypassing the task requirement to use the actual `{self.library_name}` crate."
                ),
            )
        PROMPT = """
<Role>
You are a Quality Assurance engineer for Rust code.
Your task is to verify the following Solution for the given Question.
</Role>

<Context>
You are presented with the result of a Rust program execution.
The solution logic is contained in `main.rs`, and the execution output is provided.
The Task names a specific external library crate (read it from the Task text)
that is already installed as a Cargo dependency. A correct solution must call
into that real crate.
</Context>

<HowTo>
You must:
1. Identify the required library crate by reading the Task — typically phrased
   as "Using <library>, implement ..." or "Using the <library> library, ...".
   Treat that crate name as the canonical library identifier for the rest of
   verification.
2. Check the execution output. If it contains errors or panic messages, or incorrect results, report failure.
3. Check the source code `main.rs`. Ensure it addresses the question directly and isn't "cheating" (e.g., hardcoding the answer without logic, unless appropriate).
4. Detect library shadowing. If `main.rs` declares a local module whose name
   matches the required library identified in step 1 (e.g. `mod <library> { ... }`),
   the solution is shadowing the real crate — every `use <library>::...` then
   resolves to the local hand-rolled stub instead of the actual dependency.
   This bypasses the task requirement and MUST be reported as failure even if
   the numerical output happens to be correct, because the solver has not
   exercised the real library at all.
5. Detect facade-only library usage. The Task requires the solver to actually
   *exercise* the named library — its functions, methods, or types must carry
   non-trivial computational steps that produce the answer. The library is NOT
   being exercised when it appears only as cosmetic decoration at the boundary,
   for example:
     - `use <library>::Type;` is present, but the only call into the crate is
       a trivial wrap/unwrap such as `Type::new(x, 0.0)`, `Type::from(x)`,
       `x.into()` applied to a value already fully computed with `std` /
       primitive arithmetic.
     - The library's types appear only in the final `Vec<...>` collect at the
       end while the algorithm itself uses no library function.
     - The crate is imported but never referenced, or referenced only inside
       an unreachable / `#[allow(dead_code)]` path.
   Apply the deletion test: mentally remove every `<library>::...` reference
   (and replace any wrapped values with their inner primitive). If the remaining
   code still answers the Task by itself, the library is acting as a facade
   and the solver has not satisfied the requirement. Report failure even when
   compilation and numerical output are fine.
6. Compare the Question, Answer, Code, and Output to determine validity.
7. Report your finding using the `report_success` or `report_failure` tool.
</HowTo>

<Guidelines>
- If the execution output indicates a compilation error or runtime panic, report failure.
- If the output is logically incorrect for the question, report failure.
- If `main.rs` shadows the required library crate by declaring a local module
  whose name matches the library named in the Task (see HowTo step 4), report
  failure.
- If `main.rs` uses the required library only as a facade — e.g. importing a
  type only to wrap a value computed entirely with std/primitive arithmetic,
  or referencing the crate only at the boundary while the algorithm itself
  contains zero library calls (see HowTo step 5) — report failure. A solution
  that survives mentally deleting every `<library>::...` reference is not
  exercising the library.
- If the provided answer is not self-contained verifiable answer, report failure, even when the execution output and main.rs are correct.
    - Carefully check if the answer contains the same verified code as main.rs.
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
                tool_choice="required", parallel_tool_calls=True, reasoning=Reasoning(effort="none")
            ),
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
