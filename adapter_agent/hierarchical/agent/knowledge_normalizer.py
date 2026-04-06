import logging
import re
from dataclasses import dataclass

from agents import ModelSettings, RunConfig, StopAtTools
from coder_mcp.runtime import Runtime
from litellm import InternalServerError
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.hierarchical.agent.rewirer import format_trajectory_transcript
from adapter_agent.hierarchical.agent.verifier import (
    VerificationResult,
    VerifierContext,
    report_failure,
    report_success,
)
from adapter_agent.rl.env.session_result import Knowledge
from adapter_agent.util.parsing import extract_rust_code

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class KnowledgeNormalizer[T: AgentsSDKModel](BaseAgent[T]):
    async def normalize(self, trajectory: list[TinkerMessage]) -> Knowledge:
        """
        Extracts "normalized knowledge" from an agent's trajectory log.
        This knowledge is a compact, necessary, and sufficient technical summary of the findings,
        patterns, and solutions discovered during the trial.
        """
        if not trajectory:
            raise ValueError("Trajectory cannot be empty")

        PROMPT = """\
You are a Senior Technical Writer and Knowledge Engineer responsible for distilling "Normalized Knowledge" from an AI agent's trajectory.

The goal is to produce a **self-contained, canonical, and single-themed** technical article (Markdown) that represents the core technical achievement of the task.

### Content Structure
Your article must be concise yet holistic, capturing the entire successful solution pattern. It should cover:

1. **Solution Pattern & Logic**: A clear, distilled explanation of the primary technical objective and the verified logic used to achieve it. Sub-techniques or implementation details used in the process should be described as part of this main goal, not as separate topics.
2. **Verified Implementation**: Provide exactly one self-contained, runnable Rust code block (using ```rust ... ```) that demonstrates the full solution pattern as verified in the trajectory.

### Rules
- **Single Primary Topic**: Identify the most significant technical goal or pattern achieved in the task. Focus the entire article on this one topic.
- **Holistic Integration**: Integrate all necessary steps, error resolutions, and logic into the main theme. Do not present the knowledge as a collection of unrelated fragments.
- **No Compound Titles**: Your title must represent a single, focused technical theme. Avoid titles that link multiple topics with "and" (e.g., avoid "Array initialization and Pi estimation"). Instead, use a title that captures the ultimate goal (e.g., "Pi estimation by Monte Carlo simulation").
- **Canonical and Standalone**: Other agents should be able to apply this pattern to similar problems without further searching.
- **No Verbosity**: Omit execution logs, internal reasoning steps, or background information not essential for understanding and reusing the pattern.
- **Self-contained Example**: The code block MUST be ready to be compiled in `main.rs`.

### Output Format
1. Write your brief reasoning for the extraction.
2. Output a **single, focused technical title** inside `<title>` XML tags.
3. Output the finalized technical article inside `<knowledge>` XML tags.
"""



        agent = AgentWrapper[str].create(
            name="KnowledgeNormalizer",
            instructions=PROMPT,
            model=self.model,
        )

        transcript = format_trajectory_transcript(trajectory)

        try:
            result = await agent.run(
                f"""\
Extract normalized knowledge from the following trajectory:
<Trajectory>
{transcript}
</Trajectory>
""",
                run_config=RunConfig(tracing_disabled=True),
            )
        except AgentRunFailure as e:
            msg = f"Knowledge extraction failed due to agent run failure: {e.cause}. Original error: {e}"
            logger.error(msg)
            return Knowledge(title="Extraction Failed", content=msg)
        except InternalServerError as e:
            msg = f"Knowledge extraction failed due to internal server error: {e}"
            logger.error(msg)
            return Knowledge(title="Extraction Failed", content=msg)

        response_text = result.final_output()
        title_match = re.search(r"<title>(.*?)</title>", response_text, re.DOTALL)
        knowledge_match = re.search(
            r"<knowledge>(.*?)</knowledge>", response_text, re.DOTALL
        )

        title = title_match.group(1).strip() if title_match else "Untitled Knowledge"
        content = (
            knowledge_match.group(1).strip()
            if knowledge_match
            else response_text.strip()
        )

        return Knowledge(title=title, content=content)


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
