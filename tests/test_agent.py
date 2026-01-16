"""Tests for OpenHands agent core functionality.

Uses DockerRuntime for isolated testing with LLM-as-a-judge evaluation.
"""

from oai_utils.agent import AgentsSDKModel
from coder_mcp.runtime import DockerRuntime
import pytest
from pathlib import Path


from adapter_agent import OpenHandsAgent
from tests.conftest import llm_judge


@pytest.mark.asyncio
async def test_create_file(
    model: AgentsSDKModel, docker_runtime: DockerRuntime, temp_workspace: Path
):
    """Test that agent can create a simple file."""
    async with docker_runtime as mcp_server:
        agent = OpenHandsAgent.create(model=model, mcp_server=mcp_server)
        task = "Create a Python file called 'hello.py' that prints 'Hello, World!'"
        result = await agent.run(task)

        # Verify file exists on host
        hello_file = temp_workspace / "hello.py"
        assert hello_file.exists(), f"Expected {hello_file} to exist"

        # LLM-as-a-judge verification
        passed, explanation = await llm_judge(
            model=model,
            mcp_server=mcp_server,
            task_description=task,
            agent_output=result.final_output,
            criteria=[
                "Agent created a Python file named hello.py",
                "Agent indicated the task was completed successfully",
            ],
        )
        assert passed, f"LLM judge failed: {explanation}"


@pytest.mark.asyncio
async def test_run_script(
    model: AgentsSDKModel, docker_runtime: DockerRuntime, temp_workspace: Path
):
    """Test that agent can create and execute a script."""
    async with docker_runtime as mcp_server:
        agent = OpenHandsAgent.create(model=model, mcp_server=mcp_server)
        task = """
        1. Create a Python file called 'fib.py' with a function that returns the nth Fibonacci number
        2. Run the script to print fib(10)
        """
        result = await agent.run(task)

        # Verify file exists
        fib_file = temp_workspace / "fib.py"
        assert fib_file.exists(), f"Expected {fib_file} to exist"

        # LLM-as-a-judge verification
        passed, explanation = await llm_judge(
            model=model,
            mcp_server=mcp_server,
            task_description=task,
            agent_output=result.final_output,
            criteria=[
                "Agent created a Python file with Fibonacci function",
                "Agent executed the script",
                "Agent showed correct output for fib(10) which is 55",
            ],
        )
        assert passed, f"LLM judge failed: {explanation}"


@pytest.mark.asyncio
async def test_edit_file(
    model: AgentsSDKModel, docker_runtime: DockerRuntime, temp_workspace: Path
):
    """Test that agent can edit an existing file."""
    # Pre-create a file
    test_file = temp_workspace / "greeting.py"
    test_file.write_text('message = "Hello"\nprint(message)')
    async with docker_runtime as mcp_server:
        agent = OpenHandsAgent.create(model=model, mcp_server=mcp_server)
        task = (
            "Edit greeting.py to change the message from 'Hello' to 'Hello, OpenHands!'"
        )
        result = await agent.run(task)

        # Verify file was edited
        content = test_file.read_text()
        assert "OpenHands" in content, (
            f"Expected 'OpenHands' in file content: {content}"
        )

        # LLM-as-a-judge verification
        passed, explanation = await llm_judge(
            model=model,
            mcp_server=mcp_server,
            task_description=task,
            agent_output=result.final_output,
            criteria=[
                "Agent modified the greeting.py file",
                "Agent changed the message to include 'OpenHands'",
            ],
        )
        assert passed, f"LLM judge failed: {explanation}"
