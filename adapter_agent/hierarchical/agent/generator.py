import logging
from dataclasses import dataclass

from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from pydantic import BaseModel

from adapter_agent.data import QA, QRA
from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class GeneratorAgent[T: AgentsSDKModel](BaseAgent[T]):
    rust_doc_analyzer: AsyncRustDocAnalyzer

    async def generate_sft(self, topic_hint: str) -> QRA:
        """
        Generate a QRA triplet for SFT bootstrapping.
        """
        prompt = self._build_system_prompt(is_sft=True)
        agent = self._create_agent(prompt, output_type=QRA)

        input_prompt = self._build_input_prompt(topic_hint)

        try:
            result = await agent.run(input_prompt)
            output = result.final_output()
            return output
        except AgentRunFailure as e:
            logger.error(f"SFT Task generation failed: {e}")
            raise

    async def generate_rl(self, topic_hint: str | None = None) -> QA:
        """
        Generate a QA pair for RL.
        """
        prompt = self._build_system_prompt(is_sft=False)
        agent = self._create_agent(prompt, output_type=QA)

        input_prompt = self._build_input_prompt(topic_hint)

        try:
            result = await agent.run(input_prompt)
            output = result.final_output()
            return output
        except AgentRunFailure as e:
            logger.error(f"RL Task generation failed: {e}")
            raise

    def _create_agent[OUT: BaseModel](
        self, instructions: str, output_type: type[OUT]
    ) -> AgentWrapper[OUT]:
        return AgentWrapper[output_type].create(
            name="GeneratorAgent",
            instructions=instructions,
            model=self.model,
            output_type=output_type,
        )

    def _build_system_prompt(self, is_sft: bool) -> str:
        mode_desc = (
            "Question-Reasoning-Answer (QRA) triplet"
            if is_sft
            else "Question-Answer (QA) pair"
        )

        prompt = f"""\
<Role>
You are an expert Rust software architect and technical educator.
Your task is to generate a high-quality {mode_desc} based on the provided API documentation.
</Role>

<Objective>
- Create a realistic coding challenge (Question) that test the understanding of the API.
- The task should be solvable using only the information in the provided documentation.
- The code (Answer) should be correct, idiomatic Rust.
{"- The reasoning ($R$) should explain the thought process, API choices, and logic flow." if is_sft else "- Even in RL mode, you must provide an Answer to ensure the question is definitely solvable."}
</Objective>

<Guidelines>
1. **Focus on the provided API**. Do not create tasks that ignore the specific library being taught.
2. **Solvability**. Ensure all types and methods used in the Answer are present in the Crate Overview.
3. **Complexity**. The task should not be trivial (like "print hello"), but should demonstrate meaningful usage of the API.
4. **Reliability**. All code generated MUST be valid Rust.
</Guidelines>

<OutputFormat>
You must return the result in a structured JSON format.
</OutputFormat>
"""
        return prompt

    def _build_input_prompt(self, topic_hint: str | None = None) -> str:
        crate_overview = self.rust_doc_analyzer.get_overview()
        hint_text = f"\n<Topic Hint>\n{topic_hint}\n</Topic Hint>" if topic_hint else ""

        return f"""\
<Crate Overview>
{crate_overview}
</Crate Overview>
{hint_text}

Please generate a new task now.
"""
