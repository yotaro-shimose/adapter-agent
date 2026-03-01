from tinker_cookbook.renderers import Renderer, ToolSpec
from tinker_cookbook.renderers.base import Message as TinkerMessage


def _inject_tools_into_prompt(
    renderer: Renderer, tools: list[ToolSpec], system_prompt: str
) -> list[TinkerMessage]:
    if not tools:
        return [TinkerMessage(role="system", content=system_prompt)]
    prefix_messages = renderer.create_conversation_prefix_with_tools(
        tools, system_prompt
    )
    return prefix_messages
