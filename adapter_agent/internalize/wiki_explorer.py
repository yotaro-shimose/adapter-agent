import logging
from typing import Optional

from agents import ModelSettings, StopAtTools, function_tool
from coder_mcp.runtime import Runtime
from oai_utils.agent import AgentsSDKModel, AgentWrapper

from adapter_agent.library.wiki_manager import WikiManager

logger = logging.getLogger(__name__)


class WikiExplorer:
    """
    An autonomous agent designed to explore the Wiki and answer questions.
    Supports a read-only mode to prevent accidental modifications.
    """

    def __init__(
        self,
        wiki_manager: WikiManager,
        model: AgentsSDKModel,
        read_only: bool = True,
        qwen3_tool_prompt: bool = False,
    ):
        self.wiki_manager = wiki_manager
        self.model = model
        self.read_only = read_only
        self.qwen3_tool_prompt = qwen3_tool_prompt

    async def ask(self, query: str, runtime: Runtime) -> str:
        """
        Runs the autonomous exploration loop to answer a query.
        """

        @function_tool
        async def ls(path: Optional[str] = None) -> str:
            """
            Lists items in the Wiki.
            If a path is provided, it lists items within that directory.
            Directories are indicated with a trailing slash (e.g., 'concepts/').
            """
            search_path = path.lstrip("/") if path else None
            if search_path and not search_path.endswith("/"):
                search_path += "/"

            titles = await self.wiki_manager.ls(search_path)
            if not titles:
                return "No items found."

            base_path = search_path if search_path else ""
            results = set()
            for t in titles:
                rel_path = t[len(base_path) :]
                segments = rel_path.split("/")
                if len(segments) > 1:
                    results.add(segments[0] + "/")
                else:
                    results.add(segments[0])

            sorted_results = sorted(
                list(results), key=lambda x: (not x.endswith("/"), x)
            )
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
        async def run_cargo(code: str) -> str:
            """
            Tests Rust code in a sandbox.
            Use this to verify technical details or examples before providing an answer.
            """
            await runtime.set_content("src/main.rs", code)
            run_ret, run_success = await runtime.run_cargo()
            return f"### Cargo Run Result (Success: {run_success})\n```\n{run_ret}\n```"

        @function_tool
        async def finish(answer: str) -> str:
            """Signals that the exploration is complete and provides the final answer."""
            return answer

        # Filter tools based on read_only
        all_tools = [ls, read_file, write_file, str_replace, run_cargo, finish]
        if self.read_only:
            # Exclude write tools
            tools = [ls, read_file, run_cargo, finish]
        else:
            tools = all_tools

        instructions = (
            "You are the Wiki Knowledge Explorer.\n"
            "Your goal is to answer technical questions by thoroughly exploring the existing Wiki database.\n\n"
            "### Core Principles\n"
            "1. **Trust Only the Wiki**: Your internal knowledge of libraries like `numrs2` may be outdated or incomplete. The Wiki is the absolute source of truth.\n"
            "2. **Thorough Investigation**: Before providing any code or final answers, you MUST explore the hierarchical structure of the Wiki to find the most relevant and up-to-date documentation.\n"
            "3. **Verification**: Always use `run_cargo` to verify any implementation details, API usage, or technical facts found during your exploration.\n\n"
            "### Workflow\n"
            "1. **Deep Research**: Use `ls` to understand the Wiki structure and `read_file` to study specific articles in depth.\n"
            "2. **Identify Patterns**: Look for established coding patterns, initialization methods, and library constraints within the Wiki.\n"
            "3. **Verified Answer**: Only once you have gathered and verified all necessary information, use the `finish` tool to provide your final, detailed response.\n\n"
            + (
                "READ-ONLY MODE: You cannot modify any files. Focus purely on retrieval and verification.\n"
                if self.read_only
                else "You may update or clarify existing articles if you find inaccuracies during your investigation.\n"
            )
        )

        agent = AgentWrapper[str].create(
            name="WikiExplorer",
            instructions=instructions,
            model=self.model,
            model_settings=ModelSettings(
                parallel_tool_calls=True, tool_choice="required"
            ),
            tools=tools,
            tool_use_behavior=StopAtTools(stop_at_tool_names=[finish.name]),
        )

        logger.info(f"Starting exploration for query: {query}")
        result = await agent.run(
            f"Please answer this question: {query}",
            max_turns=15,
        )
        logger.info("Exploration agent finished.")
        return result.final_output() or "No answer provided."
