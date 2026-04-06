import json
import asyncio
import litellm
from dataclasses import dataclass, field
from typing import Any, cast

from agents import RunContextWrapper, StopAtTools, function_tool
from oai_utils import AgentsSDKModel
from oai_utils.agent import AgentWrapper
from pydantic import BaseModel
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.tool_use.types import Tool, ToolInput, ToolResult

from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.knowledge_db import KnowledgeDB


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
        search_model: AgentsSDKModel,
        analyzer: AsyncRustDocAnalyzer,
        knowledge_db: KnowledgeDB,
        mutable_state: SimplifiedSolverMutableState,
        blocked_knowledge_ids: set[str] = field(default_factory=set),
    ):
        self.search_model = search_model
        self.analyzer = analyzer
        self.knowledge_db = knowledge_db
        self.mutable_state = mutable_state
        self.blocked_knowledge_ids = blocked_knowledge_ids

    @property
    def name(self) -> str:
        return "search_library_doc"

    @property
    def description(self) -> str:
        return (
            "Search the Rust documentation returning analyzed knowledge about target."
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
        # Adding knowledge_id and content as dynamic attributes to track citations
        cast(dict[str, Any], msg)["knowledge_id"] = ctx.knowledge_id
        cast(dict[str, Any], msg)["knowledge_title"] = ctx.knowledge_title
        cast(dict[str, Any], msg)["knowledge_content"] = ctx.final_answer
        return ToolResult(messages=[msg])

    async def search(self, query: str) -> tuple[str, AgenticSearchContext]:
        search_limit = self.mutable_state.total_turns // 2
        
        if self.mutable_state.search_count >= search_limit:
            output = (
                f"[SYSTEM ERROR] SEARCH_LIMIT_EXCEEDED\n"
                f"You have already performed {self.mutable_state.search_count} searches (Limit: {search_limit}).\n"
                f"Further 'search_library_doc' calls will be IGNORED.\n"
                f"You MUST proceed with the information you already have.\n"
                f"Use `<write_and_run>...</write_and_run>` to test code or `<submit>...</submit>` to finish."
            )
            return output, AgenticSearchContext()

        self.mutable_state.search_count += 1
        ctx = AgenticSearchContext()

        # Step 1: Direct Search Execution (KnowledgeDB -> Fallback to RustDocs)
        # Fetch more to ensure we have enough even after filtering
        db_results = await self.knowledge_db.search(query, limit=10)
        
        # Filter out seen knowledge pieces AND blocked knowledge pieces (internalized)
        unseen_results = [
            res for res in db_results 
            if res["id"] not in self.mutable_state.seen_knowledge_ids 
            and res["id"] not in self.blocked_knowledge_ids
        ]
        
        # Take the top 3 unseen ones
        selected_db_results = unseen_results[:3]
        
        # Mark them as seen because they will be shown to the selector agent
        for res in selected_db_results:
            self.mutable_state.seen_knowledge_ids.add(res["id"])

        final_ans: str | None = None

        if selected_db_results:
            # Flow A: Knowledge DB (Select and return as-is)
            numbered_results = {i: res for i, res in enumerate(selected_db_results)}

            @function_tool
            def select_knowledge(
                wrapper: RunContextWrapper[AgenticSearchContext], index: int | None
            ) -> None:
                """Select the most relevant knowledge snippet by its integer index. If none of the snippets are relevant, pass `None`."""
                if index is not None and index in numbered_results:
                    wrapper.context.final_answer = numbered_results[index]["content"]
                    wrapper.context.knowledge_id = numbered_results[index]["id"]
                    wrapper.context.knowledge_title = numbered_results[index].get("title")
                else:
                    wrapper.context.final_answer = None

            agent = AgentWrapper[str].create(
                name="KnowledgeSelector",
                instructions=(
                    "You are a knowledge selection assistant specializing in high-precision retrieval.\n"
                    f"User Query: '{query}'\n\n"
                    "### Goal\n"
                    "Evaluate the provided knowledge snippets and decide if one provides a direct, comprehensive solution. "
                    "Accuracy is your highest priority. It is always preferable to return nothing (None) than to provide a snippet that is only partially relevant.\n\n"
                    "### Selection Logic\n"
                    "- **Return index**: Use this ONLY if a snippet is an exact 'bullseye'—it contains the precise code, API usage, or explanation requested.\n"
                    "- **Return None**: Use this if snippets are merely related, incomplete, or if there is any ambiguity regarding their helpfulness.\n\n"
                    "### Decision Rule\n"
                    "When in doubt, default to `index=None`. This signals that a fresh search of the raw documentation is required for better accuracy."
                ),
                model=self.search_model,
                tools=[select_knowledge],
                tool_use_behavior=StopAtTools(
                    stop_at_tool_names=[select_knowledge.name]
                ),
            )

            input_list = [
                {
                    "index": i,
                    "learned_query": r["query"],
                    "content": r["content"],
                    "id": r["id"],
                }
                for i, r in numbered_results.items()
            ]
            input_prompt = f"<KnowledgeSnippets>\n{json.dumps(input_list, indent=2)}\n</KnowledgeSnippets>"

            # Retry loop for KnowledgeSelector
            for attempt in range(3):
                try:
                    await agent.run(input_prompt, context=ctx)
                    break
                except litellm.InternalServerError:
                    if attempt == 2:
                        # Fallback to Flow B if all retries fail
                        ctx.final_answer = None
                    else:
                        await asyncio.sleep(2 ** attempt)
                except Exception:
                    # Fallback to Flow B for any other agent errors
                    ctx.final_answer = None
                    break

            final_ans = ctx.final_answer

        if not final_ans:
            # Flow B: Rust Documentation (Synthesize and Summarize)
            docs = await self.analyzer.search(query, limit=5)
            raw_results = [d.model_dump() for d in docs]

            if not raw_results:
                final_ans = (
                    "No results found in either Knowledge DB or standard documentation."
                )
            else:

                @function_tool
                def report_summary(
                    wrapper: RunContextWrapper[AgenticSearchContext], summary: str
                ) -> None:
                    """Report your comprehensive technical summary back to the user."""
                    wrapper.context.final_answer = summary

                agent = AgentWrapper[str].create(
                    name="RustDocSummarizer",
                    instructions=(
                        "You are an expert technical documentation synthesizer specializing in Rust.\\n"
                        f"The user searched for: '{query}'. Your mission is to provide the user with the exact technical context they need to achieve their goal.\\n"
                        "### Instructions\\n"
                        "1. **Identify Relevance**: Select the single most relevant document from the provided search results that best addresses the user's likely intent.\\n"
                        "2. **Extract API Signatures**: Provide the EXACT and COMPLETE function/struct/trait signatures found in the documentation. Do not hallucinate or omit generic parameters or return types.\\n"
                        "3. **Synthesize Example Code**: If the documentation provides an example, or if you can confidently derive a minimal, working example from the signatures, include a clean Rust code block. Always prioritize accuracy.\\n"
                        "4. **Maintain Technical Context**: Focus on return values, common pitfalls, and safety requirements (e.g., Unsafe) if mentioned. Tailor the explanation to be practical and actionable.\\n"
                        "5. **Submission**: Use the `report_summary` tool to deliver your final report. Do NOT just return a citation ID; provide the full, synthesized technical content."
                    ),
                    model=self.search_model,
                    tools=[report_summary],
                    tool_use_behavior=StopAtTools(
                        stop_at_tool_names=["report_summary"]
                    ),
                )

                input_prompt = (
                    f"<SearchResults>\\n{json.dumps(raw_results, indent=2)}\\n</SearchResults>\\n\\n"
                    "Select the single best document from the list above and provide a detailed summary using `report_summary`."
                )

                # Retry loop for RustDocSummarizer
                for attempt in range(3):
                    try:
                        await agent.run(input_prompt, context=ctx)
                        break
                    except litellm.InternalServerError:
                        if attempt == 2:
                            ctx.final_answer = "Documentation search failed due to a transient model error. Please try again or refine your query."
                        else:
                            await asyncio.sleep(2 ** attempt)
                    except Exception as e:
                        ctx.final_answer = f"An unexpected error occurred during documentation search: {str(e)}"
                        break

        final_ans = ctx.final_answer or "Failed to summarize the document."
        return final_ans, ctx
