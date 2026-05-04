import logging
import re

from coder_mcp.runtime import Runtime
from oai_utils.agent import AgentsSDKModel, AgentWrapper

from adapter_agent.hierarchical.agent.reflector import Reflection
from adapter_agent.library.wiki_manager import WikiManager

logger = logging.getLogger(__name__)


MAX_TURNS = 20

TAG_NAMES = ("ls", "read_file", "write_file", "str_replace", "run_cargo", "finish")
# Matches either <tag /> (self-closing) or <tag>...</tag>. group(1) is the tag
# name; group(2) is the body or None for self-closing.
TAG_RE = re.compile(
    r"<(" + "|".join(TAG_NAMES) + r")\s*(?:/>|>(.*?)</\1>)",
    re.DOTALL,
)
SUBFIELD_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)


def _parse_subfields(body: str) -> dict[str, str]:
    """Extract sub-tag fields from a multi-arg tool body."""
    return {m.group(1): m.group(2).strip() for m in SUBFIELD_RE.finditer(body)}


class WikiIntegrator:
    """
    An autonomous agent designed to integrate technical insights into a versioned Wiki.
    Ensures articles remain atomic, well-organized, and up-to-date.
    """

    def __init__(
        self,
        wiki_manager: WikiManager,
        model: AgentsSDKModel,
        qwen_no_think: bool = False,
    ):
        self.wiki_manager = wiki_manager
        self.model = model
        self.qwen_no_think = qwen_no_think

    async def integrate(
        self,
        reflection: Reflection,
        task_instruction: str,
        final_answer: str,
        runtime: Runtime,
    ) -> None:
        """
        Runs the autonomous integration loop for a single reflection.

        `task_instruction` and `final_answer` are the Task and the agent's
        verified-working submission (the body of `<submit>...</submit>`)
        from the trajectory that produced this reflection. They are the
        sole context the integrator sees about the original problem —
        intermediate exploration is intentionally hidden.
        """
        instructions = (
            "You are the Wiki Knowledge Integrator. Integrate a single technical Insight into a versioned Wiki while keeping every article atomic, focused, and discoverable.\n\n"
            "### Source Material\n"
            f"Insight: {reflection.insight}\n"
            f"Evidence: {reflection.evidence}\n\n"
            "<Task>\n"
            f"{task_instruction}\n"
            "</Task>\n\n"
            "<FinalAnswer>\n"
            f"{final_answer}\n"
            "</FinalAnswer>\n\n"
            "The Insight above is what you must encode. The Final Answer is the agent's verified-working solution that produced it — treat it as your authoritative source of correct API usage. When the article needs a code example, draw it from the Final Answer (quote, simplify, or excerpt). Do NOT introduce APIs, function names, or module paths that are absent from the Final Answer; if you need to verify a variation, use `<run_cargo>`.\n\n"
            "### Wiki Structure\n"
            "The Wiki has two layers:\n"
            "- **Articles**: Each article is a self-contained knowledge unit that a future agent can lift and apply to a sub-problem. Scope is defined by what an agent would search for, not by the structure of the underlying code. Several related functions, structs, or patterns belong in one article when they together address one task; a single item gets its own article only when that item alone is the knowledge unit.\n"
            "- **MOC.md** (Map of Content): The sole navigation hub. It links to every article with a one-line summary and usage guidance. Library overviews, table-of-contents pages, and any other index-style writing live here, never as standalone articles.\n\n"
            "### Principles\n"
            "1. **Atomic by reuse, not by API**: An article covers one task-shaped knowledge unit — the granularity at which a future agent, scanning the MOC, says \"this is what I need.\" Group items that together solve one task; do not split into one article per struct or function unless each is independently the knowledge unit.\n"
            "2. **Title–content alignment**: An article's title must describe exactly what its body covers. A title broader than the content misleads readers about what they will find.\n"
            "3. **Hierarchical paths**: Place articles where readers will navigate to find them — e.g. `concepts/<topic>.md`, `api/<library>/<task>.md`. The path mirrors how the knowledge is sought, not how the source code is organized.\n"
            "4. **Uniqueness**: Audit before writing. Update an existing match rather than creating a duplicate; merge or skip near-duplicates rather than forking.\n"
            "5. **Splitting**: If an existing article has drifted into covering multiple unrelated tasks, split it into atomic articles as part of this integration.\n"
            "6. **Concrete code examples**: Every article that documents an API or pattern includes at least one runnable code example. Prefer lifting the example directly from the Final Answer; simplify it when the original carries unrelated noise. Articles that describe an API only in prose are insufficient — readers need to see the call site shape, imports, and types.\n\n"
            "### Tool Tags\n"
            "You drive this task by emitting exactly ONE XML tool tag per response. Free-form text outside the tag (analysis, plans, comments) is ignored — only the tag is executed.\n"
            "Available tags:\n"
            "- `<ls />` — list the Wiki root.\n"
            "- `<ls>folder/</ls>` — list the contents of a folder.\n"
            "- `<read_file>path/to/article.md</read_file>` — read an article.\n"
            "- `<write_file><path>path/to/article.md</path><content>...full markdown body...</content></write_file>` — create or overwrite an article.\n"
            "- `<str_replace><path>path/to/article.md</path><old_str>...</old_str><new_str>...</new_str></str_replace>` — atomic in-place edit.\n"
            "- `<run_cargo>...full Rust source for src/main.rs...</run_cargo>` — write the code to `src/main.rs` and run `cargo run`. Use this to verify a variation before recording it.\n"
            "- `<finish />` — end the integration once the article and its MOC.md entry are coherent.\n"
            "Rules:\n"
            "- Emit only ONE tag per response. Multiple tags will be rejected and you will be asked to retry.\n"
            "- Multi-argument tags (`write_file`, `str_replace`) require their named sub-tags. Do NOT use JSON, attributes, or any other format.\n"
            "- Tag bodies are taken verbatim. Inside `<run_cargo>`, write Rust source directly — do not escape quotes or wrap in JSON.\n"
            "- Bodies must not contain the closing tag of the enclosing tool (e.g., `<run_cargo>` body cannot contain the literal `</run_cargo>`).\n\n"
            "### Workflow\n"
            "1. **Audit**: Use `<ls />` and `<read_file>` to see whether the concept (or a related one) already exists.\n"
            "2. **Verify**: Code lifted directly from the Final Answer is already verified — you may use it as-is. If you adapt or invent any variation, confirm it with `<run_cargo>` before writing.\n"
            "3. **Synthesize**:\n"
            "   - Exact match or minor improvement → update with `<str_replace>`.\n"
            "   - New atomic concept, or a split of an over-broad article → create with `<write_file>` using a hierarchical path. The article must include a runnable code example.\n"
            "4. **Update MOC.md**: Whenever you create or significantly revise an article, update its MOC.md entry. Each entry contains:\n"
            "   a) A one-line summary of what the article documents.\n"
            "   b) Usage guidance — the situations or problem types in which an agent should consult it.\n"
            "5. **Finish**: Emit `<finish />` once the article and its MOC.md entry are coherent.\n"
        )

        agent = AgentWrapper[str].create(
            name="WikiIntegrator",
            instructions=instructions,
            model=self.model,
        )

        first_user = (
            f"Please integrate this insight.\n\n"
            f"Insight: {reflection.insight}\n"
            f"Evidence: {reflection.evidence}"
        )
        if self.qwen_no_think:
            first_user = "/no_think " + first_user

        history: list[dict] = [{"role": "user", "content": first_user}]

        logger.info(f"Starting integration for insight: {reflection.insight[:50]}...")
        for turn in range(MAX_TURNS):
            result = await agent.run(history)
            assistant_text = result.final_output()
            history.append({"role": "assistant", "content": assistant_text})

            matches = list(TAG_RE.finditer(assistant_text))

            if not matches:
                history.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM ERROR] NO_TOOL_TAG\n"
                        "No recognized tool tag was found in your response. Emit exactly one of: "
                        "<ls />, <ls>folder/</ls>, <read_file>...</read_file>, <write_file>...</write_file>, "
                        "<str_replace>...</str_replace>, <run_cargo>...</run_cargo>, <finish />."
                    ),
                })
                continue

            if len(matches) > 1:
                names = [m.group(1) for m in matches]
                history.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM ERROR] MULTIPLE_TOOL_TAGS\n"
                        f"Detected: {names}. Emit only ONE tool tag per response. Re-think and re-send a single action."
                    ),
                })
                continue

            match = matches[0]
            tag_name = match.group(1)
            body = (match.group(2) or "").strip()

            if tag_name == "finish":
                logger.info("Integration agent finished successfully.")
                return

            tool_result = await self._dispatch(tag_name, body, runtime)
            history.append({"role": "user", "content": tool_result})

        logger.warning(
            f"WikiIntegrator hit max_turns ({MAX_TURNS}) without <finish />."
        )

    async def _dispatch(self, tag_name: str, body: str, runtime: Runtime) -> str:
        if tag_name == "ls":
            return await self._tool_ls(body)
        if tag_name == "read_file":
            return await self._tool_read_file(body)
        if tag_name == "write_file":
            fields = _parse_subfields(body)
            return await self._tool_write_file(
                fields.get("path", ""), fields.get("content", "")
            )
        if tag_name == "str_replace":
            fields = _parse_subfields(body)
            return await self._tool_str_replace(
                fields.get("path", ""),
                fields.get("old_str", ""),
                fields.get("new_str", ""),
            )
        if tag_name == "run_cargo":
            return await self._tool_run_cargo(body, runtime)
        return f"<error>Unknown tool: {tag_name}</error>"

    async def _tool_ls(self, path: str) -> str:
        search_path = path.lstrip("/") if path else None
        if search_path and not search_path.endswith("/"):
            search_path += "/"

        titles = await self.wiki_manager.ls(search_path)
        if not titles:
            return "<ls_result>No items found.</ls_result>"

        base_path = search_path if search_path else ""
        results: set[str] = set()
        for t in titles:
            rel_path = t[len(base_path):]
            segments = rel_path.split("/")
            if len(segments) > 1:
                results.add(segments[0] + "/")
            else:
                results.add(segments[0])

        sorted_results = sorted(results, key=lambda x: (not x.endswith("/"), x))
        return "<ls_result>\n" + "\n".join(sorted_results) + "\n</ls_result>"

    async def _tool_read_file(self, path: str) -> str:
        if not path:
            return "<read_error>Missing path. Use <read_file>path/to/article.md</read_file>.</read_error>"
        content = await self.wiki_manager.read(path)
        if content is None:
            return f"<read_error>Article '{path}' not found.</read_error>"
        return f"<file path=\"{path}\">\n{content}\n</file>"

    async def _tool_write_file(self, path: str, content: str) -> str:
        if not path:
            return "<write_error>Missing <path> sub-tag.</write_error>"
        await self.wiki_manager.write(path, content)
        return f"<write_result>Wrote '{path}'.</write_result>"

    async def _tool_str_replace(self, path: str, old_str: str, new_str: str) -> str:
        if not path or not old_str:
            return "<str_replace_error>Missing required sub-tag (<path> and <old_str>).</str_replace_error>"
        success = await self.wiki_manager.str_replace(path, old_str, new_str)
        if success:
            return f"<str_replace_result>Updated '{path}' atomically.</str_replace_result>"
        return f"<str_replace_error>Target string not found in '{path}'.</str_replace_error>"

    async def _tool_run_cargo(self, code: str, runtime: Runtime) -> str:
        if not code:
            return "<cargo_error>Empty body. Place the full Rust source inside <run_cargo>...</run_cargo>.</cargo_error>"
        await runtime.set_content("src/main.rs", code)
        run_ret, run_success = await runtime.run_cargo()
        return (
            f"<cargo_result success=\"{str(run_success).lower()}\">\n"
            f"{run_ret}\n"
            f"</cargo_result>"
        )
