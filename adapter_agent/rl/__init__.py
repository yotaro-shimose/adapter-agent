"""OpenHands Agent Package - A production-quality agent using openai-agents-sdk."""

from adapter_agent.agent import OpenHandsAgent
from adapter_agent.prompts import SYSTEM_PROMPT
from coder_mcp.runtime import Runtime, LocalRuntime

__all__ = [
    "OpenHandsAgent",
    "SYSTEM_PROMPT",
    "Runtime",
    "LocalRuntime",
]
