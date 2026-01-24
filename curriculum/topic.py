from dataclasses import dataclass
from pathlib import Path

from agents import RunContextWrapper, function_tool
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper

from curriculum.database import Topic, TopicDatabase


@dataclass
class ContextType:
    db: TopicDatabase
    target_file_rel: str


@function_tool
async def register_topic(
    wrapper: RunContextWrapper[ContextType],
    id: str,
    title: str,
    description: str,
    related_apis: list[str],
) -> str:
    """Register a new topic in the database."""
    ctx = wrapper.context
    topic = Topic(
        id=id,
        title=title,
        description=description,
        related_apis=related_apis,
        source_file=ctx.target_file_rel,
    )
    print(f"  -> Registering topic: {title} ({id})")
    await ctx.db.add_topic(topic)
    return f"Topic '{id}' registered successfully."


DETAILED_TOPIC_GENERATOR_PROMPT_TEMPLATE = """You are a specialized Topic Extraction Agent.
Your goal is to extract learning topics from the file: `{file_path}`.

CONTEXT:
**Library Summary**:
{library_summary}

**Current File**: `{file_path}`.

INSTRUCTIONS:
1. Read the current file.
2. Identify "Topics" that a user of this library should learn.
   - Distinguish between **Public API** (user-facing) and **Internal Implementation**.
   - ONLY extract public/user-facing topics.
   - **IGNORE** Trait Implementations (e.g. `impl MyTrait for MyType`) unless they add new public methods that are NOT part of the trait. Usage of standard traits should be covered under the Trait's topic or the Struct's main topic, not as a separate topic for the implementation file.
   - **IGNORE** purely internal helper functions.
3. For each topic found, use the `register_topic` tool to save it.
   - **ID**: A unique snake_case identifier (e.g., `tensor_creation`, `matrix_multiplication`).
   - **Title**: Human readable title.
   - **Related APIs**: List of functions/structs involved. IMPORTANT: Use the **FULL PATH** (e.g., `<crate_name>::module::struct::method` or `<crate_name>::module::struct::method`). Do not use short names.
   - **Description**: Brief explanation of what this concept is.
4. If no user-facing topics are found in this file (e.g., internal utility), that is fine. You can finish without registering any topics.
5. Say "finished" when done.
"""


async def run_topic_generation_for_file(
    model: AgentsSDKModel,
    coder_mcp,
    target_file: Path,
    lib_dest: Path,
    db: TopicDatabase,
    library_summary: str,
):
    """Run topic generation for a single file."""
    relative_path = target_file.relative_to(lib_dest)
    agent_file_path = Path("repos/library") / relative_path
    print(f"Processing: {agent_file_path}")

    prompt = DETAILED_TOPIC_GENERATOR_PROMPT_TEMPLATE.format(
        file_path=str(agent_file_path), library_summary=library_summary
    )

    topic_agent = AgentWrapper[str].create(
        name=f"TopicGen-{relative_path.name}",
        instructions=prompt,
        model=model,
        mcp_servers=[coder_mcp],
        tools=[register_topic],
    )

    ctx = ContextType(db=db, target_file_rel=str(relative_path))
    try:
        await topic_agent.run(
            f"Analyze {agent_file_path} and register topics.",
            context=ctx,
            max_turns=30,
        )
    except (Exception, AgentRunFailure) as e:
        print(f"Error processing {relative_path}: {e}")
