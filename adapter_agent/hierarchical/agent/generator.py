import logging
from dataclasses import dataclass
from typing import Optional

from oai_utils.agent import AgentsSDKModel, AgentWrapper
from pydantic import BaseModel

from adapter_agent.data import QA, QRA
from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.rl.env.session_result import Knowledge

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class GeneratorAgent[T: AgentsSDKModel](BaseAgent[T]):
    rust_doc_analyzer: AsyncRustDocAnalyzer

    async def generate_sft(self, knowledge: Knowledge) -> Optional[QRA]:
        """
        Generate a QRA triplet for SFT bootstrapping based on a specific Knowledge item.
        """
        prompt = self._build_system_prompt(is_sft=True)
        agent = self._create_agent(prompt, output_type=QRA)

        input_prompt = self._build_input_prompt(knowledge)

        try:
            result = await agent.run(input_prompt, time_out_seconds=60.0)
            output = result.final_output()
            return output
        except Exception as e:
            logger.exception(
                f"Problem generation (SFT) failed for '{knowledge.title}': {e}"
            )
            return None

    async def generate_rl(self, knowledge: Knowledge) -> Optional[QA]:
        """
        Generate a QA pair for RL based on a specific Knowledge item.
        """
        prompt = self._build_system_prompt(is_sft=False)
        agent = self._create_agent(prompt, output_type=QA)

        input_prompt = self._build_input_prompt(knowledge)

        try:
            result = await agent.run(input_prompt, time_out_seconds=60.0)
            output = result.final_output()
            return output
        except Exception as e:
            logger.exception(
                f"Problem generation (RL) failed for '{knowledge.title}': {e}"
            )
            return None

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
You are an expert Rust software architect who has already mastered the library.
Your task is to generate a high-quality {mode_desc}.
</Role>

<Objective>
- Create a realistic coding challenge (Question) that tests the understanding of the API.
- The Answer should be a natural language explanation of the solution, which MUST include the code enclosed in a ```rust ... ``` code block. The code must be correct, idiomatic Rust, and the execution results should clearly demonstrate that it works as intended.
{"- The reasoning ($R$) should reflect your internalized thought process, API choices, and logic flow." if is_sft else "- Even in RL mode, you must provide an Answer to ensure the question is definitely solvable."}
</Objective>

<Persona & Knowledge State>
- **Internalized Expertise**: Speak and reason from the perspective of an expert who has already integrated the library's concepts. 
- **No Meta-References**: Avoid any mention of "the provided documentation," "the overview," or "the text." Use direct first-person reasoning (e.g., "I will use...", "I need to...").
- **Professional Intuition**: Your reasoning should reflect professional intuition and problem-solving flow, even if the technical facts are technically provided in the context.
</Persona & Knowledge State>

<Guidelines>
1. **Focus on the Library**. Design tasks that require meaningful application of the specific library's features.
2. **Solvability**. Ensure all types and methods used are technically accurate according to the provided overview.
3. **Verifiability**. Ensure the generated code produces clear output during execution so that its correctness can be easily verified from the resulting execution logs.
4. **Reliability**. All code generated MUST be valid, compilable Rust.
</Guidelines>

<OutputFormat>
You must return the result in a structured JSON format.
</OutputFormat>
"""
        return prompt

    def _build_input_prompt(self, knowledge: Knowledge) -> str:
        return f"""\
<Knowledge Item: {knowledge.title}>
{knowledge.content}
</Knowledge Item>

Please generate a new task based on the knowledge provided above.
"""
