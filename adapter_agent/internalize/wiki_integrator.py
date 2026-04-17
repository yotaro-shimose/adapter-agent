import logging
from typing import Optional

from agents import ModelSettings, function_tool
from coder_mcp.runtime import Runtime
from oai_utils.agent import AgentsSDKModel, AgentWrapper

from adapter_agent.hierarchical.agent.reflector import Reflection
from adapter_agent.library.wiki_manager import WikiManager

logger = logging.getLogger(__name__)


class WikiIntegrator:
    """
    An autonomous agent designed to integrate technical insights into a versioned Wiki.
    Ensures articles remain atomic, well-organized, and up-to-date.
    """

    def __init__(self, wiki_manager: WikiManager, model: AgentsSDKModel):
        self.wiki_manager = wiki_manager
        self.model = model

    async def integrate(self, reflection: Reflection, runtime: Runtime) -> None:
        """
        Runs the autonomous integration loop for a single reflection.
        """

        @function_tool
        async def ls(path: Optional[str] = None) -> str:
            """
            Lists items in the Wiki.
            If a path is provided, it lists items within that directory.
            Directories are indicated with a trailing slash (e.g., 'concepts/').
            """
            # Strip leading slash if provided
            search_path = path.lstrip("/") if path else None
            # Ensure path ends with / if provided for consistent prefix matching
            if search_path and not search_path.endswith("/"):
                search_path += "/"
            
            titles = await self.wiki_manager.ls(search_path)
            if not titles:
                return "No items found."

            base_path = search_path if search_path else ""
            results = set()
            for t in titles:
                # Remove the base path prefix to get relative path
                rel_path = t[len(base_path):]
                segments = rel_path.split("/")
                if len(segments) > 1:
                    results.add(segments[0] + "/") # It's a directory
                else:
                    results.add(segments[0]) # It's a file
            
            # Sort: directories first, then files
            sorted_results = sorted(list(results), key=lambda x: (not x.endswith("/"), x))
            return "\n".join(sorted_results)

        @function_tool
        async def read_file(path: str) -> str:
            """Reads a Wiki article by title."""
            content = await self.wiki_manager.read(path)
            return content if content else f"Error: Article '{path}' not found."

        @function_tool
        async def write_file(path: str, content: str) -> str:
            """Creates or overwrites a Wiki article."""
            await self.wiki_manager.write(path, content)
            return f"Successfully wrote to '{path}'"

        @function_tool
        async def str_replace(path: str, old_str: str, new_str: str) -> str:
            """Atomically replaces text in a Wiki article."""
            success = await self.wiki_manager.str_replace(path, old_str, new_str)
            if success:
                return f"Successfully updated '{path}' (Atomic)"
            return f"Error: Update failed. Target string not found in '{path}'."

        @function_tool
        async def finish() -> str:
            """Signals that the integration is complete."""
            return "Integration complete."

        @function_tool
        async def run_cargo(code: str) -> str:
            """
            Tests Rust code in a sandbox.
            This tool writes the provided code to 'src/main.rs' from scratch and runs 'cargo run'.
            Use this to verify technical insights or API patterns before adding them to the Wiki.
            """
            await runtime.set_content("src/main.rs", code)
            run_ret, run_success = await runtime.run_cargo()
            return f"### Cargo Run Result (Success: {run_success})\n```\n{run_ret}\n```"

        tools = [ls, read_file, write_file, str_replace, finish, run_cargo]

        agent = AgentWrapper[str].create(
            name="WikiIntegrator",
            instructions=(
                f"You are the Wiki Knowledge Integrator.\n"
                "Your goal is to integrate a new technical Insight into the existing Wiki database while ensuring articles remain **atomic** and well-organized.\n\n"
                "### Insight to Integrate\n"
                f"Insight: {reflection.insight}\n"
                f"Evidence: {reflection.evidence}\n\n"
                "### Principles\n"
                "1. **Atomicity**: Each wiki article should cover one specific technical insight or pattern. Avoid creating bloated or multi-topic articles. We can start one struct/module per file, and split them later if needed.\n"
                "2. **Organization**: Use hierarchical paths to organize knowledge (e.g., `concepts/logic.md`, `api/numrs2.md`).\n"
                "3. **Splitting**: If an existing article is becoming too general or covers multiple topics, split it into multiple smaller, more focused articles.\n\n"
                "4. **Uniqueness**: Do not create duplicate articles. If you find a similar article, update it or even do nothing.\n\n"
                "### Workflow\n"
                "1. **Audit**: Use `ls` and `read_file` to see if this information or a related topic already exists. `ls` shows immediate items and directories (ending in `/`). Use `ls(path='folder/')` to explore subdirectories.\n"
                "2. **Verify**: If the insight involves code patterns or APIs, Use `run_cargo` to verify the code works as expected. You can iteratively use the tool to test different code snippets. Edit wiki only if your code is verified.\n"
                "3. **Synthesize**: \n"
                "   - If it's a perfect match or minor improvement to an atomic topic, use `str_replace` to update existing articles.\n"
                "   - If it's a new sub-topic or requires splitting an existing note, create new articles with `write_file` using hierarchical paths.\n"
                "   - **MOC Maintenance**: You MUST keep 'MOC.md' (Map of Content) up to date whenever you create or update knowledge. For each link in 'MOC.md', do not just provide the title. You MUST include:\n"
                "     a) A concise summary of the knowledge provided by the article.\n"
                "     b) Usage Guidelines: Clearly state in what situations or for what types of problems an agent should refer to this article.\n"
                "   - This applies to both newly created articles and existing articles whose content you have significantly improved.\n"
                "4. **Finish**: Once you have integrated the knowledge and ensured the MOC is appropriately descriptive, you MUST call the `finish` tool to conclude.\n"
            ),
            model=self.model,
            model_settings=ModelSettings(
                parallel_tool_calls=True, tool_choice="required"
            ),
            tools=tools,
        )

        logger.info(f"Starting integration for insight: {reflection.insight[:50]}...")
        await agent.run(
            f"Please integrate this insight: {reflection.insight}",
            max_turns=20,
        )
        logger.info("Integration agent finished successfully.")
