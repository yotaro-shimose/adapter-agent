"""Tests for Rust coding environment with sccache.

Uses RustCodingEnvironment for isolated Rust development testing.
"""

from oai_utils.agent import AgentsSDKModel
import pytest
from pathlib import Path


from adapter_agent import OpenHandsAgent
from coder_mcp.runtime.rust_env import RustCodingEnvironment
from tests.conftest import llm_judge


@pytest.mark.asyncio
async def test_rust_project_creation(model: AgentsSDKModel, tmp_path: Path):
    """Test that agent can create and build a Rust project."""
    workspace = tmp_path / "rust_workspace"
    cache_dir = tmp_path / "sccache"
    cargo_cache = tmp_path / "cargo_cache"

    async with RustCodingEnvironment(
        workspace_dir=str(workspace),
        cache_dir=str(cache_dir),
        cargo_cache_dir=str(cargo_cache),
    ) as mcp_server:
        agent = OpenHandsAgent.create(model=model, mcp_server=mcp_server)
        task = """
        1. Create a new Rust project called 'hello_rust'
        2. Add a dependency on 'serde' in Cargo.toml
        3. Build the project with cargo build
        4. Show the sccache stats
        """
        result = await agent.run(task)

        # Verify project was created
        project_dir = workspace / "hello_rust"
        assert project_dir.exists(), f"Expected {project_dir} to exist"

        cargo_toml = project_dir / "Cargo.toml"
        assert cargo_toml.exists(), "Expected Cargo.toml to exist"

        # LLM-as-a-judge verification
        passed, explanation = await llm_judge(
            model=model,
            mcp_server=mcp_server,
            task_description=task,
            agent_output=result.final_output,
            criteria=[
                "Agent created a Rust project named hello_rust",
                "Agent added serde dependency",
                "Agent built the project successfully",
                "Agent showed sccache statistics",
            ],
        )
        assert passed, f"LLM judge failed: {explanation}"


@pytest.mark.asyncio
async def test_rust_compile_twice_for_cache(model: AgentsSDKModel, tmp_path: Path):
    """Test that sccache caches compilations across builds."""
    workspace = tmp_path / "rust_workspace"
    cache_dir = tmp_path / "sccache"
    cargo_cache = tmp_path / "cargo_cache"

    async with RustCodingEnvironment(
        workspace_dir=str(workspace),
        cache_dir=str(cache_dir),
        cargo_cache_dir=str(cargo_cache),
    ) as mcp_server:
        agent = OpenHandsAgent.create(model=model, mcp_server=mcp_server)
        task = """
            1. Create a new Rust project called 'cache_test'
            2. Add serde dependency
            3. Build with cargo build
            4. Clean with cargo clean
            5. Build again with cargo build
            6. Show sccache stats - we expect some cache hits on the second build
            """
        result = await agent.run(task)

        # LLM-as-a-judge verification
        passed, explanation = await llm_judge(
            model=model,
            mcp_server=mcp_server,
            task_description=task,
            agent_output=result.final_output,
            criteria=[
                "Agent created and built a Rust project",
                "Agent performed cargo clean and rebuilt",
                "Agent showed sccache statistics",
                "Output mentions cache hits or cache misses",
            ],
        )
        assert passed, f"LLM judge failed: {explanation}"
