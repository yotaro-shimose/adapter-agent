from dataclasses import dataclass, field
from typing import Any, List

from pydantic import BaseModel
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.tool_use.types import Tool, ToolInput, ToolResult

from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer, SearchResult
from adapter_agent.library.wiki_manager import WikiManager


class AgenticSearchContext(BaseModel):
    final_answer: str | None = None
    knowledge_id: str | None = None
    knowledge_title: str | None = None


@dataclass
class SimplifiedSolverMutableState:
    remaining_turns: int
    total_turns: int
    search_count: int = 0
    seen_knowledge_ids: set[str] = field(default_factory=set)


class SearchTool(Tool):
    def __init__(
        self,
        analyzer: AsyncRustDocAnalyzer,
        wiki_manager: WikiManager,
        mutable_state: SimplifiedSolverMutableState,
    ):
        self.analyzer = analyzer
        self.wiki_manager = wiki_manager
        self.mutable_state = mutable_state

    @property
    def name(self) -> str:
        return "search_library_doc"

    @property
    def description(self) -> str:
        return (
            "Search the official Rust documentation. Returns raw technical snippets."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The exact concept, function or error code to query.",
                },
            },
            "required": ["query"],
        }

    async def run(self, input: ToolInput) -> ToolResult:
        assert input.call_id is not None
        query = input.arguments.get("query", "")
        
        final_ans, ctx = await self.search(query)
        
        search_limit = self.mutable_state.total_turns // 2
        remaining_search_quota = max(0, search_limit - self.mutable_state.search_count)
        status_suffix = f"\n\n[STATUS]\nRemainingSearchQuota: {remaining_search_quota}\nRemainingTurns: {self.mutable_state.remaining_turns}"
        
        output = f"{final_ans}{status_suffix}"

        msg = TinkerMessage(
            role="tool",
            content=output,
            tool_call_id=input.call_id,
        )
        return ToolResult(messages=[msg])

    def _format_search_results(self, results: List[SearchResult]) -> str:
        if not results:
            return "No official documentation found for this query."
        
        parts = ["## Official Documentation Search Results"]
        for res in results[:5]:  # Limit to top 5
            part = f"### {res.name} ({res.kind})\n"
            if res.signature:
                part += f"**Signature:**\n```rust\n{res.signature}\n```\n"
            if res.docs:
                part += f"**Description:**\n{res.docs}\n"
            if res.associated_methods:
                part += f"**Methods:** {', '.join(res.associated_methods)}\n"
            parts.append(part)
        
        return "\n\n---\n\n".join(parts)

    async def search(self, query: str) -> tuple[str, AgenticSearchContext]:
        search_limit = self.mutable_state.total_turns // 2
        
        if self.mutable_state.search_count >= search_limit:
            output = (
                f"[SYSTEM ERROR] SEARCH_LIMIT_EXCEEDED\n"
                f"You have already performed {self.mutable_state.search_count} searches (Limit: {search_limit}).\n"
                f"Further 'search_library_doc' calls will be IGNORED.\n"
                f"You MUST use the Wiki exploration tools or the information you already have."
            )
            return output, AgenticSearchContext()

        self.mutable_state.search_count += 1
        
        # In the new strategy, SearchTool focus strictly on Doc Fallback.
        # However, we can also perform a quick keyword search on the Wiki for convenience.
        wiki_hits = await self.wiki_manager.search(query, limit=3)
        wiki_part = ""
        if wiki_hits:
            wiki_part = "## Wiki Matches (Preliminary)\n"
            for hit in wiki_hits:
                wiki_part += f"- [[{hit['title']}]] (Use `<wiki_read>{hit['title']}</wiki_read>` for details)\n"
            wiki_part += "\n---\n\n"

        # Official Doc Search
        doc_results = await self.analyzer.search(query, limit=5)
        doc_part = self._format_search_results(doc_results)
        
        final_ans = f"{wiki_part}{doc_part}"
        return final_ans, AgenticSearchContext()
