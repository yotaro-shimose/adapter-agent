import logging
import re

from agents import RunConfig
from oai_utils.agent import AgentWrapper
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.hierarchical.agent.knowledge_normalizer import AgentsSDKModel
from adapter_agent.hierarchical.agent.rewirer import format_trajectory_transcript

logger = logging.getLogger(__name__)


class OCRewriter[T: AgentsSDKModel](BaseAgent[T]):
    async def rewrite_trajectory(
        self,
        trajectory: list[TinkerMessage],
        knowledge_id: str,
        knowledge_content: str,
        citation_turn_idx: int,
    ) -> list[TinkerMessage]:
        """
        Rewrites a trajectory to merge a search-and-use triplet (Ai, Oi, Ai+1) into a single recall turn (Ai').

        Args:
            trajectory: The original list of messages.
            knowledge_id: The ID of the knowledge that was used.
            knowledge_content: The actual content of the knowledge.
            citation_turn_idx: The index of the turn (Ai) where the knowledge search was initiated.
                               Expected structure:
                               trajectory[citation_turn_idx]     -> Ai (Assistant search tool call)
                               trajectory[citation_turn_idx + 1] -> Oi (Tool result with knowledge content)
                               trajectory[citation_turn_idx + 2] -> Ai+1 (Assistant using knowledge)
        Returns:
            A new list of messages where the triplet is replaced by a single merged turn.
        """
        if citation_turn_idx + 2 >= len(trajectory):
            logger.warning("Trajectory too short to rewrite at the given index.")
            return trajectory

        Ai = trajectory[citation_turn_idx]
        Oi = trajectory[citation_turn_idx + 1]
        Ai_plus_1 = trajectory[citation_turn_idx + 2]

        PROMPT = """
You are an expert trajectory editor. Your goal is to perform "Open-to-Close (OC) Conversion" on an AI agent's trajectory.

### Task
You are given a segment of an agent's trajectory consisting of three steps:
1. $A_i$: The agent calls a search tool to find information.
2. $O_i$: The system returns the search results (which contains the knowledge provided below).
3. $A_{i+1}$: The agent uses the retrieved information to proceed with its reasoning AND possibly another action (like writing code or submitting).

### Knowledge Provided
Content that was retrieved during search:
{KNOWLEDGE_CONTENT}

### Goal
Merge $A_i$, $O_i$, and $A_{i+1}$ into a SINGLE assistant message ($A'_{merged}$) so that the agent appears to possess the knowledge inherently (Closed Book).

Rules for $A'_{merged}$:
- **Single Turn**: The output must be one single coherent assistant message.
- **Prune Search**: Do not mention calling search tools or receiving external documentation.
- **Natural Recall**: The agent should perform internal reasoning, "recalling" the knowledge naturally. Use phrases like "Wait, I remember...", "Thinking about this...", or "Actually, I know that...".
- **Action Continuity**: The merged turn MUST end with the EXACT same intent and tool calls/tags as the original $A_{i+1}$. If $A_{i+1}$ contained `<write_and_run>` tags or `<submit>` tags, the merged $A'_{merged}$ MUST also contain those same tags with the same content.
- **Consistency**: Ensure the flow from "remembering" to "acting" is seamless.

### Output Format
Output ONLY the final text of the new merged assistant message within the following XML tag:
<Ai_merged>
[The merged content for the single assistant turn, including any tool tags like <write_and_run> or <submit> if they were in Ai+1]
</Ai_merged>
"""
        agent = AgentWrapper[str].create(
            name="OCRewriter",
            instructions=PROMPT.replace("{KNOWLEDGE_CONTENT}", knowledge_content),
            model=self.model,
        )

        input_prompt = f"""
### Original Trajectory Segment to Merge
<Ai>
{format_trajectory_transcript([Ai])}
</Ai>
<Oi>
{format_trajectory_transcript([Oi])}
</Oi>
<Ai_plus_1>
{format_trajectory_transcript([Ai_plus_1])}
</Ai_plus_1>

Please merge these three into a single assistant turn using the knowledge provided. 
Ensure any tool calls (like <write_and_run> or <submit>) from Ai+1 are preserved at the end of your response.
"""
        result = await agent.run(
            input_prompt,
            run_config=RunConfig(tracing_disabled=True),
        )

        response_text = result.final_output()
        Ai_merged = self._extract_tag(response_text, "Ai_merged")

        if not Ai_merged:
            logger.warning("Failed to extract Ai_merged tag, using fallback logic.")
            # Fallback to a dumb merge if something goes wrong
            Ai_merged = response_text.strip()

        # Build the new trajectory by replacing the triplet with a single message
        new_trajectory = list(trajectory)

        new_msg = TinkerMessage(
            role="assistant",
            content=Ai_merged,
        )

        # Replacing 3 elements at index citation_turn_idx with 1 element
        new_trajectory[citation_turn_idx : citation_turn_idx + 3] = [new_msg]

        return new_trajectory

    def _extract_tag(self, text: str, tag: str) -> str | None:
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
