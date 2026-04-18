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
                                transcript_lines.append("")
                        elif part.get("type") == "text":
                            text = part.get("text", "")
                            if isinstance(text, str) and text:
                                transcript_lines.append(text)
                                transcript_lines.append("")
            else:
                content = get_text_content(msg)
                transcript_lines.append(content)
                transcript_lines.append("")

            tool_calls = msg.get("tool_calls", [])
            for call in tool_calls:
                # Handle both object (from API) and dict (from JSON DB)
                if isinstance(call, dict):
                    fn = call.get("function", {})
                    name = fn.get("name", "unknown")
                    args = fn.get("arguments", "{}")
                else:
                    name = call.function.name
                    args = call.function.arguments

                transcript_lines.append(f"[Tool Call: {name} - Args: {args}]")

            unparsed_tool_calls = msg.get("unparsed_tool_calls", [])
            if unparsed_tool_calls is None:
                unparsed_tool_calls = []
            for call in unparsed_tool_calls:
                if isinstance(call, dict):
                    raw_text = call.get("raw_text", "n/a")
                    error = call.get("error", "n/a")
                else:
                    raw_text = call.raw_text
                    error = call.error

                transcript_lines.append(
                    f"[Unparsed Tool Call: Raw Text: {raw_text} - Error: {error}]"
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
        r"[write_and_run]\n```rust\n\1\n```\n[/write_and_run]",
        transcript,
        flags=re.DOTALL,
    )
    transcript = re.sub(
        r"<submit>(.*?)</submit>",
        r"[submit]\n```rust\n\1\n```\n[/submit]",
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


def _log_trajectory_rich(messages: list[TinkerMessage]) -> None:
    """Helper function to log the trajectory using rich with structured panels."""
    from rich.console import Console, Group
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    entities = []

    for msg in messages:
        role = msg.get("role", "unknown")
        content_raw = msg.get("content", "")

        # Determine Title and Color based on role
        if role == "system":
            title, color = "System", "dim white"
        elif role == "user":
            title, color = "User", "cyan"
        elif role == "assistant":
            title, color = "Assistant", "green"
        elif role == "tool":
            title, color = "Tool Result", "magenta"
        else:
            title, color = role.capitalize(), "white"

        parts = []

        # Handle content parts
        if isinstance(content_raw, str):
            if content_raw.strip():
                parts.append(Markdown(content_raw))
        elif isinstance(content_raw, list):
            for p in content_raw:
                if p["type"] == "thinking":
                    thinking_text = p["thinking"]
                    if thinking_text:
                        parts.append(
                            Panel(
                                Text(thinking_text, style="italic dim"),
                                title="[dim]Thinking[/dim]",
                                border_style="dim",
                                padding=(0, 1),
                            )
                        )
                elif p["type"] == "text":
                    text = p.get("text", "").strip()
                    if text:
                        parts.append(Markdown(text))

        # Handle tool calls in assistant messages
        tool_calls = msg.get("tool_calls") or []
        for call in tool_calls:
            if isinstance(call, dict):
                fn = call.get("function", {})
                name = fn.get("name", "unknown")
                args = fn.get("arguments", "{}")
            else:
                name = call.function.name
                args = call.function.arguments
            parts.append(
                Panel(
                    Text(f"Call: {name}\nArgs: {args}", style="yellow"),
                    title="Tool Call",
                    border_style="yellow",
                )
            )

        if parts:
            entities.append(
                Panel(
                    Group(*parts),
                    title=f"[bold]{title}[/bold]",
                    border_style=color,
                    expand=False,
                )
            )

    if entities:
        console.print("\n")
        console.print(
            Panel(
                Group(*entities),
                title="[bold blue]Trajectory Visualizer[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
        )
        console.print("\n")


def log_trajectory(messages: list[TinkerMessage], flip_tag: bool = False) -> None:
    # We still keep the logic to log to standard logger as well if needed.
    # But for the visual requirement, we use the rich version.
    try:
        _log_trajectory_rich(messages)
    except Exception as e:
        # Fallback to the old method if rich rendering fails
        logger.warning(f"Rich trajectory logging failed: {e}. Falling back to text.")
        transcript = format_trajectory_transcript(
            messages, use_thinking=True, flip_tag=flip_tag
        )
        print(transcript)
