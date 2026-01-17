from coder_mcp.runtime import RustCodingEnvironment
from coder_mcp.runtime.temp_workspace import TempWorkspace
from pathlib import Path
async def main():
    boilerplate_dir = Path("templates/rust_template").resolve()
    async with RustCodingEnvironment(workspace_dir=boilerplate_dir) as rust_env:
        pass


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
