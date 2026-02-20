import logging
from dataclasses import dataclass
from typing import Any

from agents import ModelSettings, RunContextWrapper, StopAtTools, function_tool
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from pydantic import BaseModel
from tinker_cookbook.renderers.base import get_text_content

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.library.rust_doc_tools import (
    WithRustDocAnalyzer,
    search_docs,
    search_symbol,
)
from adapter_agent.rl.env import EnvState

logger = logging.getLogger(__name__)


class RewireResult(BaseModel):
    turn_to_replace: int
    reasoning: str


class RewirerContext(WithRustDocAnalyzer):
    result: RewireResult | None = None


@function_tool
def report_rewire_point(
    wrapper: RunContextWrapper[RewirerContext],
    turn_to_replace: int,
    reasoning: str,
) -> None:
    """
    Report the point in the trajectory to branch from with new reasoning.
    Args:
        turn_to_replace: The 1-indexed turn number to replace from.
        reasoning: The new reasoning to fill in for that turn.
    """
    wrapper.context.result = RewireResult(
        turn_to_replace=turn_to_replace,
        reasoning=reasoning,
    )


def format_trajectory_transcript(messages: list[dict[str, Any]]) -> str:
    """Formats a list of agent-environment messages into a readable transcript."""
    transcript_lines = []
    turn_count = 1
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            content = get_text_content(msg)
            transcript_lines.append(f"=== Turn {turn_count} (Assistant) ===")
            transcript_lines.append(content)

            tool_calls = msg.get("tool_calls", [])
            for call in tool_calls:
                transcript_lines.append(
                    f"[Tool Call: {call.function.name} - Args: {call.function.arguments}]"
                )

            transcript_lines.append("")
            turn_count += 1
        elif role == "tool":
            content = get_text_content(msg)
            transcript_lines.append("=== Tool Result ===")
            transcript_lines.append(content)
            transcript_lines.append("")
        elif role == "user":
            content = get_text_content(msg)
            transcript_lines.append("=== User / System ===")
            transcript_lines.append(content)
            transcript_lines.append("")

    return "\n".join(transcript_lines)


def _log_rewire_result_debug(
    original_transcript: str, rewire_result: RewireResult, rewired_transcript: str
) -> None:
    """Helper function to log the rewire result using rich if debugging is enabled."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    console.print(
        Panel(
            Text(original_transcript, style="dim white"),
            title="Original Trajectory",
            border_style="yellow",
        )
    )
    console.print(
        Panel(
            Text(
                f"Turn to Replace: {rewire_result.turn_to_replace}\nReasoning: {rewire_result.reasoning}",
                style="bold cyan",
            ),
            title="Rewirer Output",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            Text(rewired_transcript, style="bold green"),
            title="Rewired Trajectory (Trimmed)",
            border_style="green",
        )
    )


@dataclass(kw_only=True)
class Rewirer[T: AgentsSDKModel](BaseAgent[T]):
    rust_doc_analyzer: RustDocAnalyzer

    async def rewire(self, state: EnvState) -> EnvState:
        """
        Reads a trajectory of another AI agent interactions with environment and detects a wrong action to rewrite.
        Returns the point to restart with new reasoning content.
        """
        PROMPT = """\
<Role>
You are an expert AI agent supervisor and a Rust software engineer.
Your task is to review a trajectory of another AI agent's interactions with a coding environment and identify the exact sequence (turn) where the agent made a mistake or sub-optimal decision.
</Role>

<Context>
You will be provided with:
1. The Task the agent was trying to accomplish.
2. The Transcript of the agent's interaction trajectory, where each assistant turn is labeled with its 1-indexed turn number.
3. The Crate Overview, giving you context about the library the agent is using.
</Context>

<HowTo>
You have access to the following search tools to understand the library documentation:
- `search_docs`: Use this to find functionality keywords, concepts, or how-to guides in the documentation.
- `search_symbol`: Use this to find specific types, functions, or traits by name.

You must:
1. Carefully read through the trajectory. Use search tools if you need more information about the crate to judge the agent's actions.
2. Identify the *first* turn where the agent's action (or reasoning leading to the action) was incorrect, unfruitful, or generally a misstep in solving the task.
3. Determine what the agent *should* have reasoned at that exact point.
4. Report your findings using the `report_rewire_point` tool, providing the 1-indexed `turn_to_replace` and the new `reasoning` that should be injected to branch the state.
</HowTo>

<Guidelines>
- Only point out one turn to replace. This turn should be the one where the agent diverged from the optimal path.
- The `reasoning` you provide will replace the agent's reasoning from that turn onward, acting as a correction so the agent can restart its trajectory from there.
- Make sure the `reasoning` is clear and explicitly states what the agent should do instead.
</Guidelines>\
"""
        tools = [
            report_rewire_point,
            search_docs,
            search_symbol,
        ]

        instructions = PROMPT

        agent = AgentWrapper.create(
            name="Rewirer",
            instructions=instructions,
            model=self.model,
            mcp_servers=[],
            tools=tools,
            model_settings=ModelSettings(tool_choice="auto", parallel_tool_calls=True),
            tool_use_behavior=StopAtTools(
                stop_at_tool_names=[report_rewire_point.name]
            ),
            reset_tool_choice=False,
        )

        context = RewirerContext(rust_doc_analyzer=self.rust_doc_analyzer)
        crate_overview = self.rust_doc_analyzer.get_overview()

        # Format transcript
        messages = state.messages or []
        transcript = format_trajectory_transcript(messages)

        input_prompt = f"""\
<Task>
{state.task.instruction}
</Task>

<Transcript>
{transcript}
</Transcript>

<Crate Overview>
{crate_overview}
</Crate Overview>
"""

        try:
            await agent.run(input_prompt, max_turns=5, context=context)
            if context.result is None:
                raise ValueError("Rewirer finished without generating new state")
            rewire_result = context.result

        except AgentRunFailure as e:
            if e.cause == "MaxTurnsExceeded":
                rewire_result = RewireResult(
                    turn_to_replace=1,
                    reasoning="Rewirer exceeded maximum number of turns.",
                )
            else:
                logger.error(f"Rewirer process failed: {e}")
                raise

        original_messages = state.messages or []
        new_messages = []
        assistant_turn_count = 1
        cut_index = len(original_messages)
        for i, msg in enumerate(original_messages):
            if msg.get("role") == "assistant":
                if assistant_turn_count == rewire_result.turn_to_replace:
                    cut_index = i
                    break
                assistant_turn_count += 1

        new_messages = original_messages[:cut_index]
        new_messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": rewire_result.reasoning}],
                "tool_calls": [],
            }
        )

        new_env_state = EnvState(
            task=state.task,
            code=state.code,
            max_turns=state.max_turns,
            remaining_turns=state.max_turns - (rewire_result.turn_to_replace - 1),
            image_name=state.image_name,
            library_name=state.library_name,
            messages=new_messages,
        )

        _log_rewire_result_debug(
            original_transcript=transcript,
            rewire_result=rewire_result,
            rewired_transcript=format_trajectory_transcript(
                new_env_state.messages or []
            ),
        )

        return new_env_state
