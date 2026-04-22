import logging

from oai_utils.agent import AgentsSDKModel, AgentWrapper
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.data import QRA
from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.hierarchical.agent.rewirer import format_trajectory_transcript

logger = logging.getLogger(__name__)


class QRADistiller[T: AgentsSDKModel](BaseAgent[T]):
    """Condense a successful multi-turn study trajectory into a single-turn QRA.

    The resulting QRA is intended to be consumed as an SFT sample: the question
    is the original task instruction, the reasoning is a compact chain-of-thought
    that a one-shot solver could plausibly produce, and the answer is the final
    conclusion the study session arrived at.
    """

    async def distill(
        self, instruction: str, trajectory: list[TinkerMessage]
    ) -> QRA | None:
        if not trajectory:
            raise ValueError("Trajectory cannot be empty")

        PROMPT = """\
You turn a multi-turn agent trajectory into a single-turn supervised training sample (QRA).

The agent solved a problem through search, tool use, and iterative attempts. Your job is
to compress that process into what a competent single-turn solver *would* have produced:
a focused chain-of-thought (reasoning) followed by the final answer.

### Rules
- **question**: Copy the original instruction verbatim.
- **reasoning**: A self-contained chain-of-thought. Include the key facts/API details
  the agent discovered, the approach it settled on, and why. Exclude dead ends, tool
  invocation syntax, and anything a one-shot solver wouldn't have narrated. Write as
  if you are the solver thinking, not a summary of someone else's work.
- **answer**: The final deliverable (code, explanation, result) the trajectory concluded
  with. Be concrete and executable where applicable.

Keep reasoning and answer faithful to what the trajectory actually established; do not
invent APIs or facts that were not grounded in the trajectory.
"""

        agent = AgentWrapper[QRA].create(
            name="QRADistiller",
            instructions=PROMPT,
            model=self.model,
            output_type=QRA,
        )

        transcript = format_trajectory_transcript(trajectory, use_thinking=True)
        input_prompt = f"""\
Original instruction:
{instruction}

Trajectory to distill:
<Trajectory>
{transcript}
</Trajectory>
"""

        try:
            result = await agent.run(input_prompt)
            return result.final_output()
        except Exception as e:
            logger.error(f"QRA distillation failed: {e}")
            return None
