import logging
from typing import Literal

from oai_utils.agent import AgentsSDKModel, AgentWrapper
from pydantic import BaseModel
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.hierarchical.agent.rewirer import format_trajectory_transcript

logger = logging.getLogger(__name__)


class Reflection(BaseModel):
    insight: str
    category: Literal["api_usage", "pattern"]
    evidence: str


class Reflections(BaseModel):
    reflections: list[Reflection]


class Reflector[T: AgentsSDKModel](BaseAgent[T]):
    async def reflect(self, trajectory: list[TinkerMessage]) -> list[Reflection]:
        """
        Analyzes an AI agent's trajectory to extract key insights, failures, and successes.
        These reflections serve as the foundation for generating fine-tuning data.
        """
        if not trajectory:
            raise ValueError("Trajectory cannot be empty")

        PROMPT = """\
You are an AI Experience Analyst and Knowledge Engineer. Your mission is to decompose a "Trajectory" (a log of an AI agent's actions and environment responses) into actionable technical insights.

These insights will be used to create training data (Fine-tuning) so that future agents can learn from this history and avoid repeating failures.

### Focus Areas
1. **API Usage Discovery**: Distill the correct way to use specific APIs, libraries, or tools that the agent discovered through trial and error or documentation.
2. **Patterns**: Identify approaches the agent found which you think is widely applicable to other tasks.

### Guidelines
- **Be Technical**: Use specific names of functions, modules, and error messages.
- **Evidence-Based**: Briefly reference the specific turn or action in the trajectory that supports the insight.
- **Atomicity**: Decompose insights into the smallest possible atomic units. Better to have 10 small, focused reflections than 2 large, combined ones. Each reflection should aim to cover exactly one technical fact or error recovery pattern.


### Output
Return a list of reflections, each categorized and linked to evidence.
"""

        agent = AgentWrapper[Reflections].create(
            name="Reflector",
            instructions=PROMPT,
            model=self.model,
            output_type=Reflections,
        )

        transcript = format_trajectory_transcript(trajectory, use_thinking=True)
        input_prompt = f"""\
Enumerate the key things to learn from the following trajectory:
<Trajectory>
{transcript}
</Trajectory>
"""

        try:
            result = await agent.run(input_prompt)
            return result.final_output().reflections
        except Exception as e:
            logger.error(f"Reflection extraction failed: {e}")
            return []
