"""Shared pytest fixtures for OpenHands agent tests.

Provides reusable fixtures for runtimes and configurations.
Uses LLM-as-a-judge pattern for testing (no mocks).
"""

import os
from pathlib import Path

import pytest
import pytest_asyncio
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp import MCPServerStreamableHttp
from agents.tracing import add_trace_processor
from dotenv import load_dotenv
from oai_utils.agent import AgentsSDKModel
from oai_utils.tracing import AgentContentPrinter

from adapter_agent import OpenHandsAgent
from coder_mcp.runtime import LocalRuntime
from coder_mcp.runtime import DockerRuntime

# Enable tracing for test visibility
add_trace_processor(AgentContentPrinter())

# Load environment variables (OPENAI_API_KEY, etc.)
load_dotenv()


@pytest.fixture
def model():
    return LitellmModel(
        model="gemini/gemini-3-flash-preview",
        api_key=os.getenv("GOOGLE_API_KEY"),
    )


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest_asyncio.fixture
async def docker_runtime(temp_workspace: Path):
    """Create a DockerRuntime with temporary workspace and an OpenHandsAgent."""
    async with DockerRuntime(workspace_dir=str(temp_workspace)) as runtime:
        yield runtime


@pytest_asyncio.fixture
async def local_runtime():
    """Create a LocalRuntime (requires running MCP server)."""
    async with LocalRuntime(
        url="http://localhost:3000",
        timeout=30,
    ) as runtime:
        yield runtime


async def llm_judge(
    model: AgentsSDKModel,
    mcp_server: MCPServerStreamableHttp,
    task_description: str,
    agent_output: str,
    criteria: list[str],
) -> tuple[bool, str]:
    """Use another OpenHandsAgent as judge to evaluate agent output.

    The judge agent has access to the same workspace via MCP server,
    allowing it to inspect files and verify task completion.

    Args:
        mcp_server: MCP server for workspace access
        task_description: What the agent was asked to do
        agent_output: The agent's final output
        criteria: List of success criteria to check
        config: Agent config (defaults to env-based config with gpt-4o-mini)

    Returns:
        Tuple of (passed: bool, explanation: str)
    """

    criteria_text = "\n".join(f"- {c}" for c in criteria)

    judge_task = f"""You are a test evaluator. Evaluate if the agent successfully completed its task.

TASK GIVEN TO AGENT:
{task_description}

SUCCESS CRITERIA:
{criteria_text}

AGENT'S OUTPUT:
{agent_output}

INSTRUCTIONS:
1. Use the file tools (read_file, list_files) to verify the workspace state if needed
2. Check if the criteria are met based on the agent output and workspace state
3. Respond with your evaluation in this exact format:

PASSED: yes/no
EXPLANATION: <brief explanation of your findings>"""

    judge = OpenHandsAgent.create(model=model, mcp_server=mcp_server)
    result = await judge.run(judge_task)

    output = result.final_output or ""
    passed = "passed: yes" in output.lower()

    return passed, output
