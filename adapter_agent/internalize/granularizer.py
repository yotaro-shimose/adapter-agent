import logging
from typing import Optional

from agents import ModelSettings, StopAtTools, function_tool
from coder_mcp.runtime import Runtime
from oai_utils.agent import AgentsSDKModel, AgentWrapper
from pydantic import BaseModel

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.hierarchical.types import Knowledge, KnowledgeSeed, SeedList

logger = logging.getLogger(__name__)


class WikiGranularizer[T: AgentsSDKModel](BaseAgent[T]):
    """
    A two-stage pipeline to transform coarse Wiki articles into granular, verified knowledge items.
    """

    async def identify_seeds(self, article_title: str, article_content: str) -> list[KnowledgeSeed]:
        """
        Stage 1: Uses an LLM to identify granular technical insights and API usage patterns.
        """
        PROMPT = f"""\
You are a Senior Technical Curriculum Designer. Your task is to analyze a Wiki article about a Rust library and identify all the granular technical "seeds" within it.

A "seed" is a specific, atomic technical insight, API usage pattern, or functionality that can be independently verified with a code example and taught to an AI model.

### Article Title: {article_title}

### Guidelines:
- **Granularity**: Break down complex articles into multiple focused seeds. For example, instead of "Arithmetic Operations", create seeds for "Matrix Addition", "Scalar Multiplication", etc.
- **Coverage**: Ensure all unique technical facts, constraints, and API methods mentioned in the article are covered by at least one seed.
- **Independence**: Each seed should be understandable and verifiable on its own.

### Output:
Return a list of seeds where each seed has:
1. **title**: A sharp, technical title.
2. **description**: A concise summary of exactly what technical fact or API usage pattern this seed represents.
"""
        agent = AgentWrapper[SeedList].create(
            name="SeedIdentifier",
            instructions=PROMPT,
            model=self.model,
            output_type=SeedList,
        )

        logger.info(f"Identifying seeds for article: {article_title}...")
        try:
            result = await agent.run(
                f"Identify seeds from the following article content:\n\n{article_content}",
                time_out_seconds=60.0,
            )
            seed_list = result.final_output()
            logger.info(f"Generated {len(seed_list.seeds)} seeds.")
            return seed_list.seeds
        except Exception as e:
            logger.error(f"Failed to identify seeds for '{article_title}': {e}")
            return []

    async def generate_knowledge(
        self, seed: KnowledgeSeed, article_title: str, article_content: str, runtime: Runtime
    ) -> Optional[Knowledge]:
        """
        Stage 2: Uses a multi-turn agent to generate a detailed, verified knowledge item.
        """

        PROMPT = f"""\
You are a Staff Software Engineer and Technical Educator specialized in Rust.
Your goal is to generate a high-quality, verified "Knowledge" entry based on a specific "Knowledge Seed" and its broader context.

### Knowledge Seed to Formalize
- **Title**: {seed.title}
- **Description**: {seed.description}
- **Source Article**: {article_title}

### Workflow
1. **Verification (Proof of Concept)**:
   - You MUST prove that the technical insight or API pattern works by writing a standalone Rust script.
   - Use the `run_cargo(code)` tool to test your code. 
   - Analyze compiler errors or execution output. Iterate and fix the code until it compiles and correctly demonstrates the pattern.
   - The code must be self-contained and ready for `main.rs`.
2. **Formalization**:
   - Once verified, generate a high-quality Markdown article.
   - **Structure**:
     - A clear explanation of the technical insight or API usage.
     - Constraints or "gotchas" mentioned in the source context.
     - The **successfully verified** Rust code block.
   - **Submission Tag**: You MUST enclose the verified code block inside a `<submit>` tag.
     - Example: `<submit>\n```rust\n...\n```\n</submit>`
3. **Completion**: Call the `finish(title, content)` tool with your finalized knowledge only when you are absolutely sure.

### Rules
- **Focused content**: Stick to the seed's topic. Do not generalize too much.
- **Verifiable code**: Only include code in the final article that you have actually successfully run via `run_cargo`.
- **Submission**: The `<submit>` block is critical for downstream internalization. Ensure it contains the EXACT code that provides the technical evidence.
"""

        class GranularKnowledgeResult(BaseModel):
            title: str
            content: str
            done: bool = False

        result_store = GranularKnowledgeResult(title="", content="", done=False)

        @function_tool
        async def run_cargo(code: str) -> str:
            """Writes code to src/main.rs and runs 'cargo run'."""
            logger.info(f"Agent is running PoC for seed: {seed.title}")
            try:
                await runtime.set_content("src/main.rs", code)
                output, success = await runtime.run_cargo()
                status = "SUCCESS" if success else "FAILURE"
                return f"[Cargo Run {status}]\nOutput:\n{output}"
            except Exception as e:
                return f"ERROR: Failed to execute code: {e}"

        @function_tool
        def finish(title: str, content: str):
            """Signals that the knowledge generation is complete."""
            result_store.title = title
            result_store.content = content
            result_store.done = True

        agent = AgentWrapper[str].create(
            name="KnowledgeGenerator",
            instructions=PROMPT,
            model=self.model,
            tools=[run_cargo, finish],
            tool_use_behavior=StopAtTools(stop_at_tool_names=[finish.name]),
            model_settings=ModelSettings(tool_choice="auto"),
        )

        input_context = f"""\
### Seed to generate knowledge for:
{seed.model_dump_json(indent=2)}

### Context from Source Article:
{article_content}
"""

        try:
            logger.info(f"Starting multi-turn knowledge generation for: {seed.title} (max_turns=25)")
            await agent.run(input_context, max_turns=25, time_out_seconds=300.0)
            
            if result_store.done:
                logger.info(f"Successfully generated knowledge for seed: {seed.title}")
                # Strip <submit> tags from the content
                clean_content = result_store.content.replace("<submit>", "").replace("</submit>", "").strip()
                return Knowledge(title=result_store.title, content=clean_content)
            else:
                logger.warning(f"Agent reached max turns or stopped without calling finish() for seed: {seed.title}")
                return None
        except Exception as e:
            logger.error(f"Error during knowledge generation for seed '{seed.title}': {e}", exc_info=True)
            return None
