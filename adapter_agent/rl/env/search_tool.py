import json
from dataclasses import dataclass
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


@dataclass
class SimplifiedSolverMutableState:
    remaining_turns: int


class SearchTool(Tool):
    def __init__(
        self,
        search_model: AgentsSDKModel,
        analyzer: AsyncRustDocAnalyzer,
        knowledge_db: KnowledgeDB,
        mutable_state: SimplifiedSolverMutableState,
    ):
        self.search_model = search_model
        self.analyzer = analyzer
        self.knowledge_db = knowledge_db
        self.mutable_state = mutable_state

    @property
    def name(self) -> str:
        return "search"

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
        query = input.arguments.get("query", "")

        ctx = AgenticSearchContext()

        # Step 1: Direct Search Execution (KnowledgeDB -> Fallback to RustDocs)
        db_results = await self.knowledge_db.search(query, limit=3)
        final_ans: str | None = None

        if db_results:
            # Flow A: Knowledge DB (Select and return as-is)
            numbered_results = {i: res for i, res in enumerate(db_results)}

            @function_tool
            def select_knowledge(
                wrapper: RunContextWrapper[AgenticSearchContext], index: int | None
            ) -> None:
                """Select the most relevant knowledge snippet by its integer index. If none of the snippets are relevant, pass `None`."""
                if index is not None and index in numbered_results:
                    wrapper.context.final_answer = numbered_results[index]["content"]
                    wrapper.context.knowledge_id = numbered_results[index]["id"]
                else:
                    wrapper.context.final_answer = None

            agent = AgentWrapper[str].create(
                name="KnowledgeSelector",
                instructions=(
                    "You are a knowledge selection assistant.\n"
                    f"The user searched for: '{query}'.\n"
                    "Below are previously learned knowledge snippets indexed from 0. "
                    "Read them carefully and decide if ANY of them are truly relevant for solving the user's issue.\n"
                    f"1. If a snippet is highly relevant, use the `{select_knowledge.name}` tool to choose it by its index.\n"
                    f"2. If NONE of the snippets are relevant or helpful, call `{select_knowledge.name}` with `index=None` to signal a fallback to standard documentation search.\n"
                    "You only need to output the tool call."
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

            await agent.run(input_prompt, context=ctx)
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
                        "You are an expert technical summarizer.\\n"
                        f"The user searched for: '{query}'. Source: Official Rust Documentation.\\n"
                        "Below are the raw search results. Your task is to:\\n"
                        "1. Determine the EXACTLY ONE most relevant document from the results.\\n"
                        "2. Summarize its contents comprehensively. Do NOT lose any technical details, Rust code signatures, or logic patterns.\\n"
                        "3. Do not formulate the response entirely as a citation ID; instead, provide the raw summary.\\n"
                        "4. Use the `report_summary` tool to submit."
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

                await agent.run(input_prompt, context=ctx)
                final_ans = ctx.final_answer or "Failed to summarize the document."

        output = f"{final_ans}"

        assert input.call_id is not None
        msg = TinkerMessage(
            role="tool",
            content=output,
            tool_call_id=input.call_id,
        )
        # Adding knowledge_id as a dynamic attribute to track citation
        cast(dict[str, Any], msg)["knowledge_id"] = ctx.knowledge_id
        return ToolResult(messages=[msg])
