import logging
import re
from dataclasses import dataclass

from agents import RunConfig
from oai_utils.agent import AgentsSDKModel, AgentWrapper

from adapter_agent.data import QRA
from adapter_agent.hierarchical.agent.base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class KnowledgeSummarizer[T: AgentsSDKModel](BaseAgent[T]):
    async def summarize(self, knowledges: list[QRA], task_instruction: str) -> str:
        """
        Summarizes a list of QRAs into a concise knowledge base for the Solver,
        tailored to a specific task.
        """
        if not knowledges:
            return ""

        PROMPT = f"""
You are a Senior Engineer specializing in technical documentation and knowledge extraction.
Your goal is to summarize a collection of Question-Reasoning-Answer (QRA) blocks into a concise, high-density knowledge base.
This knowledge base will be used by another AI agent (the Solver) to help it solve the following coding task in a Rust environment:

<TASK>
{task_instruction}
</TASK>

Guidelines:
1. Extract unique insights, common pitfalls, and specific library usage patterns that are PARTICULARLY RELEVANT TO THE TASK ABOVE.
2. Avoid redundancy. If multiple QRAs cover the same concept, synthesize them.
3. Keep it technical and concise. Use bullet points or short paragraphs.
4. If there are specific code snippets that are universally useful, include them.
5. Focus on what was learned from the attempts that can help solve the target task.

OUTPUT FORMAT:
Provide your reasoning, and then output the summarized knowledge base inside a <knowledge></knowledge> xml block.
"""
        agent = AgentWrapper[str].create(
            name="KnowledgeSummarizer",
            instructions=PROMPT,
            model=self.model,
        )

        knowledges_text = "\n\n".join(
            [
                f"Question: {qra.question}\nReasoning: {qra.reasoning}\nAnswer: {qra.answer}"
                for qra in knowledges
            ]
        )

        result = await agent.run(
            f"""\
Summarize the following QRAs into a knowledge base specifically for the task: "{task_instruction}"
<QRAs>
{knowledges_text}
</QRAs>
""",
            run_config=RunConfig(tracing_disabled=True),
        )

        response_text = result.final_output()
        match = re.search(r"<knowledge>(.*?)</knowledge>", response_text, re.DOTALL)
        if not match:
            # Fallback to the whole response if xml block is missing,
            # but usually it should be there.
            logger.warning(
                "Could not find <knowledge> block in KnowledgeSummarizer response."
            )
            return response_text.strip()

        return match.group(1).strip()
