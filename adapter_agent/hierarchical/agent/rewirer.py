import logging
import re

from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.renderers.base import get_text_content

logger = logging.getLogger(__name__)


def format_trajectory_transcript(
    messages: list[TinkerMessage],
    use_thinking: bool = False,
    flip_tag: bool = False,
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
        elif role in {"user", "system"}:
            content = get_text_content(msg)
            transcript_lines.append("=== User / System ===")
            transcript_lines.append(content)
            transcript_lines.append("")

    transcript = "\n".join(transcript_lines)

    # Pre-render XML code blocks into beautiful markdown before flip_tag breaks the brackets
    transcript = re.sub(
        r"<write_and_run>(.*?)</write_and_run>",
        r"**[TEST RUN]**\n```rust\n\1\n```",
        transcript,
        flags=re.DOTALL,
    )
    transcript = re.sub(
        r"<submit>(.*?)</submit>",
        r"**[FINAL SUBMISSION]**\n```rust\n\1\n```",
        transcript,
        flags=re.DOTALL,
    )

    if flip_tag:
        # Rewrite HTML tags: <Tag>Content</Tag> -> [Tag]Content[/Tag]
        # Using [Tag] prevents Markdown's blockquote (>) conflicts and keeps it readable.
        transcript = re.sub(
            r"<([^>]+)>(.*?)</\1>", r"[\1]\2[/\1]", transcript, flags=re.DOTALL
        )
    return transcript


def _log_rewire_result_debug(original_transcript: str, rewired_transcript: str) -> None:
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
            Text(rewired_transcript, style="bold green"),
            title="Rewired Trajectory (Trimmed)",
            border_style="green",
        )
    )
