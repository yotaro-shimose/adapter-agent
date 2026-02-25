import logging
from dataclasses import dataclass

from agents import ModelSettings, RunContextWrapper, StopAtTools, function_tool
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from pydantic import BaseModel
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.renderers.base import get_text_content

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.library.rust_doc_tools import (
    WithRustDocAnalyzer,
    search_docs,
    search_symbol,
)
from adapter_agent.rl.env.env import ResumedEnvState

logger = logging.getLogger(__name__)


class FeedbackResult(BaseModel):
    turn_to_replace: int
    reasoning: str


class FeedbackContext(WithRustDocAnalyzer):
    result: FeedbackResult | None = None


class RewireOutput(BaseModel):
    continuous_reasoning: str


@function_tool
def report_feedback(
    wrapper: RunContextWrapper[FeedbackContext],
    turn_to_replace: int,
    reasoning: str,
) -> None:
    """
    Report the point in the trajectory to branch from with new reasoning.
    Args:
        turn_to_replace: The 1-indexed turn number to replace from.
        reasoning: The description about what the student agent should have done instead.
    """
    wrapper.context.result = FeedbackResult(
        turn_to_replace=turn_to_replace,
        reasoning=reasoning,
    )


def format_trajectory_transcript(
    messages: list[TinkerMessage], use_thinking: bool = False
) -> str:
    """Formats a list of agent-environment messages into a readable transcript."""
    transcript_lines = []
    turn_count = 1
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            transcript_lines.append(f"=== Turn {turn_count} (Assistant) ===")
            if use_thinking:
                content_raw = msg.get("content", "")
                if isinstance(content_raw, str):
                    if content_raw:
                        transcript_lines.append(content_raw)
                elif isinstance(content_raw, list):
                    for part in content_raw:
                        if part.get("type") == "thinking":
                            thinking = part.get("thinking", "")
                            if isinstance(thinking, str):
                                transcript_lines.append("<think>")
                                transcript_lines.append(thinking.strip())
                                transcript_lines.append("</think>")
                        elif part.get("type") == "text":
                            text = part.get("text", "")
                            if isinstance(text, str) and text:
                                transcript_lines.append(text)
            else:
                content = get_text_content(msg)
                transcript_lines.append(content)
                transcript_lines.append("")

            tool_calls = msg.get("tool_calls", [])
            for call in tool_calls:
                transcript_lines.append(
                    f"[Tool Call: {call.function.name} - Args: {call.function.arguments}]"
                )

            unparsed_tool_calls = msg.get("unparsed_tool_calls", [])
            for call in unparsed_tool_calls:
                transcript_lines.append(
                    f"[Unparsed Tool Call: Raw Text: {call.raw_text} - Error: {call.error}]"
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
    original_transcript: str, rewire_result: FeedbackResult, rewired_transcript: str
) -> None:
    """Helper function to log the rewire result using rich if debugging is enabled."""

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

    async def rewire(self, state: ResumedEnvState) -> ResumedEnvState:
        """
        Reads a trajectory of another AI agent interactions with environment and detects a wrong action to rewrite.
        Returns the point to restart with new reasoning content.
        """
        # Format transcript
        original_transcript = format_trajectory_transcript(state.messages)
        rewire_result = await self.get_feedback(state)
        cut_index = self.get_cut_index(state, rewire_result)

        new_messages = []
        new_messages = state.messages[:cut_index]
        new_reasoning = await self.get_new_reasoning(state, rewire_result)

        new_env_state = ResumedEnvState(
            task=state.task,
            code_history=state.code_history[: rewire_result.turn_to_replace],
            max_turns=state.max_turns,
            remaining_turns=state.max_turns - (rewire_result.turn_to_replace - 1),
            image_name=state.image_name,
            library_name=state.library_name,
            messages=new_messages,
            prethink=new_reasoning,
        )

        if logger.isEnabledFor(logging.DEBUG):
            _log_rewire_result_debug(
                original_transcript=original_transcript,
                rewire_result=rewire_result,
                rewired_transcript=format_trajectory_transcript(
                    new_messages
                    + [TinkerMessage(role="assistant", content=new_reasoning)]
                ),
            )

        return new_env_state

    async def get_feedback(self, state: ResumedEnvState) -> FeedbackResult:
        PROMPT = """\
<Role>
You are a Rust coding teacher.
Your task is to review a student's interactions with a coding environment.
You should identify the first turn where the student made a mistake or sub-optimal decision and provide what the student should have done that instead.
</Role>

<Context>
You will be provided with:
1. The Crate Overview, giving you context about the library the agent is using.
2. The Transcript of the student's interaction trajectory, where each assistant turn is labeled with its 1-indexed turn number.
</Context>

<HowTo>
You have access to the following search tools to understand the library documentation:
- `search_docs`: Use this to find functionality keywords, concepts guides in the documentation.
- `search_symbol`: Use this to find specific types, functions, or traits by name.

You must:
1. Carefully read through the trajectory. Use search tools if you need more information about the crate to judge the agent's actions.
2. Identify the *first* turn where the agent's action (or reasoning leading to the action) was incorrect, unfruitful, or generally a misstep in solving the task.
3. Determine what the agent should have done instead at that exact point to get back on track.
4. Report your findings using the `report_rewire_point` tool, providing the 1-indexed `turn_to_replace` and the new `reasoning` that will be injected to branch the state.
</HowTo>

<Guidelines>
- Only point out one turn to replace. This turn should be the one where the agent diverged from the optimal path.
- Causality rule: The agent at turn `N` cannot see the future! Your `reasoning` for turn `N` must NOT mention or react to errors, tool results, or events that happen in turn `N` or later in the transcript.
- The student agent does not have search tools. So `student should have searched for X` is not an executable plan, therefore not a valid reasoning.
    - If you think the student does not have enough knowledge, you use search tools and provide the knowledge to the student agent. It is often helpful to provide the symbol's signature and a short description of its purpose.
- The feedback should be detailed enough to help the student agent correct its course. Including detailed and concrete examples when applicable is recommended.
</Guidelines>\
"""
        tools = [
            report_feedback,
            search_docs,
            search_symbol,
        ]

        instructions = PROMPT

        agent = AgentWrapper.create(
            name="Reviewer",
            instructions=instructions,
            model=self.model,
            mcp_servers=[],
            tools=tools,
            model_settings=ModelSettings(
                tool_choice="required", parallel_tool_calls=True
            ),
            tool_use_behavior=StopAtTools(stop_at_tool_names=[report_feedback.name]),
            reset_tool_choice=False,
        )

        context = FeedbackContext(rust_doc_analyzer=self.rust_doc_analyzer)
        crate_overview = self.rust_doc_analyzer.get_overview()
        transcript = format_trajectory_transcript(state.messages)

        input_prompt = f"""\
<Crate Overview>
{crate_overview}
</Crate Overview>
<Transcript>
{transcript}
</Transcript>

"""

        try:
            await agent.run(input_prompt, max_turns=10, context=context)
            if context.result is None:
                raise ValueError("Rewirer finished without generating new state")
            feedback_result = context.result
        except AgentRunFailure as e:
            logger.error(f"Rewirer process failed: {e}")
            raise
        logger.debug(f"Feedback result: {feedback_result}")
        return feedback_result

    async def get_new_reasoning(
        self, state: ResumedEnvState, feedback: FeedbackResult
    ) -> str:
        instructions = """\
<Role>
You are an expert at simulating an AI coding agent's inner monologue.
Your task is to take a reviewer's feedback about what an agent *should* have done, and translate it into the agent's own first-person reasoning process.
</Role>

<HowTo>
You will be provided with:
1. The Trajectory of the agent's interaction up to the point where it needs to make a new decision.
2. The Reviewer's Feedback describing what the agent should do next instead of its original mistake.

You must generate the continuous, first-person internal monologue the agent would have right before taking the corrected action. This reasoning should naturally bridge the agent's past observations to the new desired action, as if the agent realized the correct path on its own.
</HowTo>

<Guidelines>
- Write purely in the first-person ("I need to...", "Let me check...", "Ah, I see...").
- Do NOT sound like external feedback or a reviewer ("The student should have...").
- Do NOT mention anything that happens this turn or later. The agent cannot see the future.
- Focus strictly on the thought process that logically concludes with taking the action suggested by the reviewer.
- The reasoning should be detailed and often include examples of what the agent should do but in first person.
- The reasoning should be written as if he recalled the knowledge from his memory without reading documentation.
- The reasoning should start from `Okay, let's see.`
</Guidelines>\
"""
        agent = AgentWrapper[RewireOutput].create(
            name="Brancher",
            instructions=instructions,
            model=self.model,
            output_type=RewireOutput,
        )
        cut_index = self.get_cut_index(state, feedback)
        messages = state.messages[:cut_index]
        transcript = format_trajectory_transcript(messages, use_thinking=False)

        input_prompt = f"""\
<Trajectory>
{transcript}
</Trajectory>

<Reviewer Feedback>
{feedback.reasoning}
</Reviewer Feedback>

Generate the first-person reasoning that logically leads to the reviewer's suggested action as a JSON object with the key "continuous_reasoning".
"""
        ret = await agent.run(input_prompt)
        return ret.final_output().continuous_reasoning

    def get_cut_index(self, state: ResumedEnvState, feedback: FeedbackResult) -> int:
        assistant_turn_count = 1
        cut_index = len(state.messages)
        for i, msg in enumerate(state.messages):
            if msg.get("role") == "assistant":
                if assistant_turn_count == feedback.turn_to_replace:
                    cut_index = i
                    break
                assistant_turn_count += 1
        return cut_index
