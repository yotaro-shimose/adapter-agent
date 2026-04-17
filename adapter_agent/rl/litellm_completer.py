import logging
from typing import Sequence

from litellm import acompletion
from tinker_cookbook.completers import MessageCompleter
from tinker_cookbook.renderers.base import (
    Message,
    Renderer,
    ToolCall,
)
from tinker_cookbook.tool_use.types import Tool

logger = logging.getLogger(__name__)


class LitellmMessageCompleter(MessageCompleter):
    """A completer that uses litellm to generate responses."""

    def __init__(
        self,
        model: str,
        renderer: Renderer,
        temperature: float = 1.0,
        tools: Sequence[Tool] | None = None,
    ):
        self.model = model
        self.renderer = renderer
        self.temperature = temperature

        self.tools = []
        if tools:
            for t in tools:
                self.tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters_schema,
                        },
                    }
                )

    async def __call__(self, messages: list[Message]) -> Message:
        # Render the conversation for the model using LiteLLM/OpenAI format
        openai_messages = [self.renderer.to_openai_message(m) for m in messages]

        kwargs = {}
        if self.tools:
            kwargs["tools"] = self.tools

        response = await acompletion(  # type: ignore
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            **kwargs,
        )

        message = response.choices[0].message
        content_str = message.content or ""

        tool_calls: list[ToolCall] = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        function=ToolCall.FunctionBody(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    )
                )

        msg: Message = {
            "role": "assistant",
            "content": content_str,
        }

        if tool_calls:
            msg["tool_calls"] = tool_calls

        return msg
