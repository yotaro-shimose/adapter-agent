import logging
from typing import Any

from oai_utils.agent import AgentWrapper

from adapter_agent.hierarchical.types import Task

logger = logging.getLogger(__name__)


class KnowledgeInternalizer:
    def __init__(self, model: Any):
        self.model = model
        self.agent = AgentWrapper[str].create(
            name="KnowledgeInternalizer",
            instructions=(
                "You are an expert Rust developer.\n"
                "The user will provide a technical task and a segment of documentation knowledge.\n"
                "Your goal is to internalize this knowledge as your own memory.\n\n"
                "**Instructions**:\n"
                "1. **Structured Reasoning**: Your output must follow a three-part structure:\n"
                "   a. **Task Understanding**: Start with a sentence like \"Okay, I understand the task is to...\" and briefly state your understanding of the requirement.\n"
                "   b. **Internalized Knowledge**: Follow with technical facts from your memory (e.g., \"I recall that the `numrs2` library uses... \"). Focus on API signatures, types, and core rules from the provided documentation.\n"
                "   c. **Strategic Conclusion**: End with a statement about your next move (e.g., \"Since I have this information, I can now proceed to write the implementation\" or \"I still need to search for X to complete my understanding\").\n"
                "2. **No Code Blocks**: Do NOT provide full `fn main()` or large boilerplate code. Focus on the \"mental model\" of the library.\n"
                "3. **No Headers**: Do NOT include markdown headers like \"## Task Understanding\". Just use a natural, seamless flow of text.\n\n"
                "Write from your own perspective as an experienced Rust developer whose thoughts are being captured at the start of a task."
            ),
            model=self.model,
        )

    async def internalize(
        self, task: Task, knowledge_title: str, knowledge_content: str
    ) -> str:
        prompt = (
            f"<Task>\n{task.instruction}\n</Task>\n\n"
            f"<Knowledge Title>\n{knowledge_title}\n</Knowledge Title>\n\n"
            f"<Knowledge Content>\n{knowledge_content}\n</Knowledge Content>\n\n"
            "Generate your initial internal reasoning based on the Task and the provided Knowledge Content. "
            "1. State your understanding of the task starting with 'Okay, I understand...'. "
            "2. State the technical facts from the Knowledge Content as memories. "
            "3. Conclude with your next strategic step based on whether this knowledge is sufficient."
        )

        # Run agent
        result = await self.agent.run(prompt)
        return result.final_output()
