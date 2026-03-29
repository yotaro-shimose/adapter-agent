import logging
from dataclasses import dataclass

from agents import (
    ModelSettings,
    RunContextWrapper,
    StopAtTools,
    function_tool,
)
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from pydantic import BaseModel

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.knowledge_db import KnowledgeDB

logger = logging.getLogger(__name__)


class UniquenessCheckerContext(BaseModel):
    is_unique: bool = True
    reasoning: str = "No decision made yet."
    search_count: int = 0
    max_searches: int = 5
    hard_limit: int = 6


async def _search_knowledge_db_impl(
    wrapper: RunContextWrapper[UniquenessCheckerContext],
    knowledge_db: KnowledgeDB,
    query: str,
) -> str:
    """
    Implementation of search_knowledge_db.
    """
    wrapper.context.search_count += 1
    count = wrapper.context.search_count

    if count > wrapper.context.hard_limit:
        return "ERROR: Search limit reached (6 searches). You are now prohibited from further searching. You MUST make a final decision using the report_uniqueness tool based on the information you have already gathered."

    results = await knowledge_db.search(query, limit=5)

    suffix = f"\n\n[System Info: Search {count}/{wrapper.context.max_searches}. "
    if count == wrapper.context.max_searches:
        suffix += "This is your last 'free' search. One more search is allowed but you should decide now if possible."
    elif count > wrapper.context.max_searches:
        suffix += "CRITICAL: You have exceeded the recommended search limit. This is your LAST chance to search. After this, you will be blocked."
    else:
        suffix += f"{wrapper.context.max_searches - count} recommended searches remaining."
    suffix += "]"

    if not results:
        return "No similar knowledge found." + suffix

    formatted_results = []
    for i, res in enumerate(results):
        formatted_results.append(
            f"Result {i+1}:\nQuery: {res['query']}\nContent:\n{res['content']}\n---"
        )

    return "\n".join(formatted_results) + suffix


@function_tool
def report_uniqueness(
    wrapper: RunContextWrapper[UniquenessCheckerContext],
    is_unique: bool,
    reasoning: str,
) -> None:
    """
    Report whether the new knowledge is unique or redundant.
    """
    wrapper.context.is_unique = is_unique
    wrapper.context.reasoning = reasoning


@dataclass(kw_only=True)
class KnowledgeUniquenessChecker[T: AgentsSDKModel](BaseAgent[T]):
    async def check_uniqueness(
        self, new_knowledge: str, task: Task, knowledge_db: KnowledgeDB
    ) -> tuple[bool, str]:
        """
        Determines if the new knowledge is unique or redundant.
        Returns (is_unique, reasoning).
        """
        PROMPT = """\
You are a Knowledge Integrity Agent. Your goal is to keep the Knowledge Database free of redundant information while ensuring all novel technical insights are preserved.

### Input
- **Original Task**: The problem that was solved to generate this knowledge.
- **New Knowledge**: A technical summary (Markdown) extracted from the solution.

### Your Goal
Determine if this **New Knowledge** is already represented in the database.

### Guidelines
1. **Redundancy**: If a similar query already exists and its content covers the same technical patterns, APIs, and insights as the New Knowledge, it is **Redundant** (is_unique=False).
2. **Novelty**: If the New Knowledge introduces new APIs, handles different edge cases, or provides a more efficient logic not present in the DB, it is **Unique** (is_unique=True).
3. **Superset**: If the New Knowledge is a **Superset** of existing knowledge (e.g., it covers what's in the DB plus more), it should be considered **Unique** (is_unique=True). We prefer a more comprehensive entry.
4. **Minor variations**: If $X'$ is just a slight variation of $X$ with no new technical value, it is **Redundant**.

### Constraints
- Total turns: 8 turns.
- Search limit: You are recommended to stay within 5 searches. After the 6th search, you will be blocked from searching further.
- If you run out of turns, we will assume the knowledge is Unique.

Perform searches to confirm your decision.
"""
        context = UniquenessCheckerContext()

        @function_tool
        async def search_knowledge_db(
            wrapper: RunContextWrapper[UniquenessCheckerContext],
            query: str,
        ) -> str:
            """
            Search the existing knowledge database for similar entries.
            """
            return await _search_knowledge_db_impl(wrapper, knowledge_db, query)

        agent = AgentWrapper[str].create(
            name="KnowledgeUniquenessChecker",
            instructions=PROMPT,
            model=self.model,
            tools=[search_knowledge_db, report_uniqueness],
            tool_use_behavior=StopAtTools(stop_at_tool_names=[report_uniqueness.name]),
            model_settings=ModelSettings(tool_choice="auto"),
        )

        input_text = f"Original Task: {task.instruction}\n\nNew Knowledge:\n{new_knowledge}\n"

        try:
            await agent.run(input_text, context=context, max_turns=8)
            return context.is_unique, context.reasoning

        except AgentRunFailure as e:
            if e.cause == "MaxTurnsExceeded":
                logger.warning("Uniqueness checker exceeded max turns. Defaulting to Unique.")
                return True, "Max turns exceeded. Assuming unique to avoid data loss."
            raise e
        except Exception as e:
            logger.error(f"Error in uniqueness check: {e}")
            return True, f"Error occurred: {e}. Defaulting to Unique."
