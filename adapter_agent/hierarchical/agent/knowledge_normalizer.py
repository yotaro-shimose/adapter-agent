import logging
import re
from dataclasses import dataclass

from agents import ModelSettings, RunConfig, StopAtTools
from coder_mcp.runtime import Runtime
from oai_utils.agent import AgentsSDKModel, AgentWrapper
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.hierarchical.agent.rewirer import format_trajectory_transcript
from adapter_agent.hierarchical.agent.verifier import (
    VerificationResult,
    VerifierContext,
    report_failure,
    report_success,
)
from adapter_agent.util.parsing import extract_rust_code

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class KnowledgeNormalizer[T: AgentsSDKModel](BaseAgent[T]):
    async def normalize(self, trajectory: list[TinkerMessage]) -> str:
        """
        Extracts "normalized knowledge" from an agent's trajectory log.
        This knowledge is a compact, necessary, and sufficient technical summary of the findings,
        patterns, and solutions discovered during the trial.
        """
        if not trajectory:
            raise ValueError("Trajectory cannot be empty")

        PROMPT = """\
You are a Senior Technical Writer and Knowledge Engineer.
Your goal is to extract "Normalized Knowledge" from a trajectory log of an AI agent solving a complex Rust coding task.

Normalized Knowledge MUST be a comprehensive, article-style explanatory text (in Markdown format) that thoroughly documents the technical insights, patterns, and solutions discovered during the trial. It should NOT just be a source code snippet with comments.

It should serve as a complete reference that contains all the information needed to:
1. Reconstruct the final correct solution.
2. Understand the underlying logic and reasoning behind the solution.
3. Identify relevant library patterns, common pitfalls, and specific symbol usages.

Guidelines:
- You MUST preserve all valuable technical details, API signatures, parameter descriptions, trait constraints, and caveats that the agent discovered via search tools during the trajectory. Do NOT over-summarize or discard technical depth.
- Write a highly detailed, article-style technical documentation (using Markdown headings `###`, bullet points, etc.) that clearly explains the library usage, concepts, and key API signatures.
- For every key function or struct used, document its purpose, function signature, parameters, and trait constraints clearly, just like official documentation.
- If the agent encountered errors and solved them, detail the exact problem and the technical reasoning behind the solution.
- Ensure the normalized knowledge includes exactly one ```rust ... ``` code block containing a final, complete, and runnable solution that demonstrates the learned knowledge. This code block should serve as a concrete use-case example at the end of the article.
- Do NOT just return a Rust code block. The comprehensive text explanation is the most important part of the knowledge base.

OUTPUT FORMAT:
Provide your reasoning, and then output the normalized knowledge base inside a <knowledge></knowledge> xml block.
"""
        agent = AgentWrapper[str].create(
            name="KnowledgeNormalizer",
            instructions=PROMPT,
            model=self.model,
        )

        transcript = format_trajectory_transcript(trajectory)

        result = await agent.run(
            f"""\
Extract normalized knowledge from the following trajectory:
<Trajectory>
{transcript}
</Trajectory>
""",
            run_config=RunConfig(tracing_disabled=True),
        )

        response_text = result.final_output()
        match = re.search(r"<knowledge>(.*?)</knowledge>", response_text, re.DOTALL)
        if not match:
            logger.warning(
                "Could not find <knowledge> block in KnowledgeNormalizer response."
            )
            return response_text.strip()

        return match.group(1).strip()


async def verify_normalized_knowledge(
    runtime: Runtime,
    normalized_knowledge: str,
    model: AgentsSDKModel,
) -> VerificationResult:
    """
    Verifies that the Rust code embedded in the normalized knowledge is runnable
    and correctly reflects the described information.
    """
    code = extract_rust_code(normalized_knowledge)
    if not code:
        return VerificationResult(
            success=False,
            reasoning="No Rust code block found in the normalized knowledge.",
        )

    # 1. Run the code
    try:
        await runtime.set_content("src/main.rs", code)
        execution_output, run_success = await runtime.run_cargo()
    except Exception as e:
        return VerificationResult(
            success=False,
            reasoning=f"Failed to execute the Rust code: {e}",
        )

    if not run_success:
        return VerificationResult(
            success=False,
            reasoning=f"Rust code failed to compile or run:\n{execution_output}",
        )

    # 2. Semantic verification via AI Agent
    PROMPT = """\
<Role>
You are a Quality Assurance engineer for Rust knowledge bases.
Your task is to verify that a piece of "Normalized Knowledge" is correctly demonstrated by the provided Rust code and its execution output.
</Role>

<Context>
You are presented with:
1. The Normalized Knowledge: A technical summary of some Rust programming patterns or solutions.
2. The Source Code (main.rs): The code that is supposed to demonstrate this knowledge.
3. The Execution Output: The result of running the code.
</Context>

<HowTo>
You must:
1. Carefully read the Normalized Knowledge and the Source Code.
2. Analyze the Execution Output.
3. Determine if the Code and the Output together provide a valid and correct verification of the claims made in the Normalized Knowledge.
4. Report your finding using the `report_success` or `report_failure` tool.
</HowTo>

<Guidelines>
- If the output contradicts the knowledge, report failure.
- If the code is missing key elements mentioned in the knowledge as part of the solution/pattern, report failure.
- If the code and output successfully demonstrate the knowledge, report success.
- Be technical and precise in your reasoning.
</Guidelines>
"""
    agent = AgentWrapper.create(
        name="KnowledgeVerifier",
        instructions=PROMPT,
        model=model,
        mcp_servers=[],
        tools=[
            report_success,
            report_failure,
        ],
        model_settings=ModelSettings(tool_choice="required", parallel_tool_calls=True),
        tool_use_behavior=StopAtTools(
            stop_at_tool_names=[report_success.name, report_failure.name]
        ),
        reset_tool_choice=False,
    )

    context = VerifierContext()
    input_prompt = f"""\
<Normalized Knowledge>
{normalized_knowledge}
</Normalized Knowledge>

<Source Code (main.rs)>
{code}
</Source Code (main.rs)>

<Execution Output>
{execution_output}
</Execution Output>
"""

    try:
        await agent.run(input_prompt, max_turns=5, context=context)
        if context.result is None:
            return VerificationResult(
                success=False,
                reasoning="KnowledgeVerifier finished without reporting success or failure.",
            )
        return context.result
    except Exception as e:
        logger.error(f"Knowledge verification process failed: {e}")
        return VerificationResult(
            success=False,
            reasoning=f"Verification agent failed: {e}",
        )
