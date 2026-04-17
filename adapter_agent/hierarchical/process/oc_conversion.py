import logging
from typing import cast

from tinker_cookbook.renderers.base import Message as TinkerMessage
from adapter_agent.hierarchical.agent.oc_rewriter import OCRewriter
from adapter_agent.library.knowledge_db import KnowledgeDB
from adapter_agent.hierarchical.agent.knowledge_normalizer import AgentsSDKModel

logger = logging.getLogger(__name__)

async def oc_convert_trajectory(
    messages: list[TinkerMessage],
    model: AgentsSDKModel,
    knowledge_db: KnowledgeDB,
) -> list[TinkerMessage]:
    """
    Scans a trajectory for knowledge retrieval turns and rewrites them using OCRewriter.
    This effectively converts an "Open Book" (search-based) trajectory into a
    "Closed Book" (recall-based) trajectory for SFT.
    """
    rewriter = OCRewriter(model=model)
    new_trajectory = list(messages)

    # 1. Identify roles and citation turns
    # We find tool results that have a knowledge_id.
    retrieval_indices = []
    for i, msg in enumerate(messages):
        # A TinkerMessage is a dict, and we added knowledge_id in SimplifiedSolverEnv.step (via ToolResult)
        if msg.get("role") == "tool" and msg.get("knowledge_id"):
            # O_i is msg, so A_i is i-1
            # We assume the turn before the tool result is the one that triggered the search.
            retrieval_indices.append(i - 1)

    # 2. Rewrite each identified triplet (Ai, Oi, Ai+1)
    # We iterate BACKWARDS to keep citation indices stable since we are shrinking the list
    for citation_turn_idx in sorted(retrieval_indices, reverse=True):
        # Get the knowledge_id from Oi
        Oi = new_trajectory[citation_turn_idx + 1]
        knowledge_id = cast(str, Oi.get("knowledge_id"))

        # Retrieve the actual knowledge content from DB
        knowledge_doc = await knowledge_db.get_knowledge_by_id(knowledge_id)
        if not knowledge_doc:
            logger.warning(
                f"Knowledge content not found for ID: {knowledge_id}. Skipping rewrite."
            )
            continue

        knowledge_content = knowledge_doc["content"]

        # Perform the rewrite which merges [Ai, Oi, Ai+1] into [Ai']
        new_trajectory = await rewriter.rewrite_trajectory(
            trajectory=new_trajectory,
            knowledge_id=knowledge_id,
            knowledge_content=knowledge_content,
            citation_turn_idx=citation_turn_idx,
        )

    return new_trajectory
