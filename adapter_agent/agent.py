"""OpenHands Agent implementation using openai-agents-sdk."""

from oai_utils.agent import AgentsSDKModel
from dataclasses import dataclass
from agents.mcp import MCPServerStreamableHttp
from agents.run import RunResult
from agents.model_settings import ModelSettings
from typing import Self
from adapter_agent.prompts import SYSTEM_PROMPT
from oai_utils.agent import AgentWrapper


@dataclass
class OpenHandsAgent:
    """OpenHands-compatible agent using openai-agents-sdk with MCP tools.

    This agent connects to a runtime to access file operations, bash execution,
    and other tools, using OpenHands-aligned system prompts for consistent behavior.

    Example with LocalRuntime (requires running MCP server):
        from adapter_agent import OpenHandsAgent, LocalRuntime

        async with LocalRuntime() as runtime:
            async with OpenHandsAgent(mcp_server=runtime.server) as agent:
                result = await agent.run("Create a hello world script")
                print(result.final_output)

    Example with DockerRuntime (self-contained):
        from docker_runtime import DockerRuntime
        from adapter_agent import OpenHandsAgent

        async with DockerRuntime(image_name="coder-mcp") as runtime:
            async with OpenHandsAgent(mcp_server=runtime.server) as agent:
                result = await agent.run("Create a hello world script")
                print(result.final_output)
    """

    mcp_server: MCPServerStreamableHttp
    agent: AgentWrapper[str]

    @classmethod
    def create(
        cls,
        model: AgentsSDKModel,
        mcp_server: MCPServerStreamableHttp,
    ) -> Self:
        return cls(
            mcp_server=mcp_server,
            agent=AgentWrapper[str].create(
                name="OpenHands Agent",
                instructions=SYSTEM_PROMPT,
                mcp_servers=[mcp_server],
                model=model,
                model_settings=ModelSettings(
                    tool_choice="auto", parallel_tool_calls=True
                ),
            ),
        )

    async def run(
        self,
        input: str | list,
        *,
        context=None,
        max_turns: int | None = None,
    ) -> RunResult:
        """Run the agent with a task.

        Args:
            input: The initial input to the agent.
            context: The context to run the agent with.
            max_turns: The maximum number of turns to run the agent for.
            hooks: An object that receives callbacks on various lifecycle events.
            run_config: Global settings for the entire agent run.
            previous_response_id: The ID of the previous response.
            auto_previous_response_id: Whether to automatically use the previous response ID.
            conversation_id: The conversation ID.
            session: A session for automatic conversation history management.

        Returns:
            RunResult containing the agent's output and execution details
        """
        if max_turns is None:
            max_turns = 30

        ret_wrapper = await self.agent.run(
            input,
            context=context,
            max_turns=max_turns,
        )
        return ret_wrapper.result
