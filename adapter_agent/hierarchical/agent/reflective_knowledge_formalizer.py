import logging
from typing import Optional

from agents import ModelSettings, RunContextWrapper, StopAtTools, function_tool
from coder_mcp.runtime import Runtime
from oai_utils.agent import AgentsSDKModel, AgentWrapper
from pydantic import BaseModel
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.hierarchical.agent.reflector import Reflection
from adapter_agent.hierarchical.agent.rewirer import format_trajectory_transcript
from adapter_agent.hierarchical.types import Knowledge
from adapter_agent.library.knowledge_db import KnowledgeDB

logger = logging.getLogger(__name__)


class FormalizationResult(BaseModel):
    is_unique: bool
    reasoning: str
    knowledge: Optional[Knowledge] = None


class FormalizerContext(BaseModel):
    result: Optional[FormalizationResult] = None
    search_count: int = 0
    max_searches: int = 5
    hard_limit: int = 6


class ReflectiveKnowledgeFormalizer[T: AgentsSDKModel](BaseAgent[T]):
    async def formalize(
        self,
        reflection: Reflection,
        trajectory: list[TinkerMessage],
        knowledge_db: KnowledgeDB,
        runtime: Optional[Runtime] = None,
        skip_uniqueness_check: bool = False,
    ) -> FormalizationResult:
        """
        Takes a single reflection and formalizes it into a Knowledge entry after iteratively verifying it with code.
        """
        if not trajectory:
            raise ValueError("Trajectory cannot be empty")

        if skip_uniqueness_check:
            PROMPT = """\
You are a Senior Knowledge Engineer and Rust Specialist. Your goal is to transform a verified technical "Reflection" into a high-quality, verified "Knowledge" entry.

### workflow
1. **Iterative Verification (Proof of Concept)**:
   - You MUST prove the technical insight works by writing a standalone Rust script.
   - Use `run_cargo(code)` to test your code.
   - Analyze compiler errors or execution output. Iterate and fix the code until it compiles and accurately demonstrates the insight.
   - The code must be self-contained (ready for `main.rs`).
2. **Formalization**:
   - Once verified, generate a high-quality Markdown article.
   - Include the **successfully verified** Rust code block.
   - **Atomicity**: The entry MUST focus on exactly ONE technical insight or one specific API pattern. If the reflection contains multiple ideas, pick the most important one.
   - **Title**: Must be sharp, technical, and focused. Forbid composite titles (e.g., avoid "X and Y"). Use "Using X for Y" or similar.
3. **Final Report**: Call `report_result(is_unique=True, ...)` with the finalized title and content.

### Rules
- Only use ONE tool call per turn.
- Ensure the code you put in the Knowledge article is the one you successfully ran.
- Atomicity: No "General Tips" or multi-pattern guides. One file = One fact.
"""
        else:
            PROMPT = """\
You are a Senior Knowledge Engineer and Rust Specialist. Your goal is to transform a "Reflection" into a high-quality, verified "Knowledge" entry.

### workflow
1. **Uniqueness Check**: Use `search_knowledge_db` to see if this insight is already in the database.
   - If **Redundant**: Immediately call `report_result(is_unique=False, reasoning=...)`.
2. **Iterative Verification (Proof of Concept)**:
   - If Unique, you MUST prove the technical insight works by writing a standalone Rust script.
   - Use `run_cargo(code)` to test your code.
   - Analyze compiler errors or execution output. Iterate and fix the code until it compiles and accurately demonstrates the insight.
   - The code must be self-contained (ready for `main.rs`).
3. **Formalization**:
   - Once verified, generate a high-quality Markdown article.
   - Include the **successfully verified** Rust code block.
   - **Atomicity**: The entry MUST focus on exactly ONE technical insight or one specific API pattern. If the reflection contains multiple ideas, pick the most important one.
   - **Title**: Must be sharp, technical, and focused. Forbid composite titles (e.g., avoid "X and Y"). Use "Using X for Y" or similar.
4. **Final Report**: Call `report_result(is_unique=True, ...)` with the finalized title and content.

### Rules
- Only use ONE tool call per turn.
- Ensure the code you put in the Knowledge article is the one you successfully ran.
- **Strict Atomicity**: No "General Tips" or multi-pattern guides. One file = One fact.
"""

        context = FormalizerContext()

        tools = []
        if not skip_uniqueness_check:

            @function_tool
            async def search_knowledge_db(
                wrapper: RunContextWrapper[FormalizerContext],
                query: str,
            ) -> str:
                """Search the existing knowledge database for similar entries."""
                print(
                    f"    [Formalizer] Searching KnowledgeDB for: '{query}'...",
                    flush=True,
                )
                # Re-implement search logic to satisfy type safety and avoid wrapper mismatch
                wrapper.context.search_count += 1
                count = wrapper.context.search_count
                if count > wrapper.context.hard_limit:
                    return "ERROR: Search limit reached. You MUST make a decision."

                results = await knowledge_db.search(query, limit=5)
                if not results:
                    return "No similar knowledge found."

                formatted = []
                for i, res in enumerate(results):
                    formatted.append(
                        f"Result {i + 1}:\nTitle: {res['title']}\nContent:\n{res['content']}\n---"
                    )
                return "\n".join(formatted)

            tools.append(search_knowledge_db)

        @function_tool
        async def run_cargo(
            wrapper: RunContextWrapper[FormalizerContext],
            code: str,
        ) -> str:
            """
            Write code to src/main.rs and run 'cargo run'.
            Returns the execution output and whether it succeeded.
            """
            print("    [Formalizer] Running PoC verification script...", flush=True)
            if runtime is None:
                return "ERROR: Runtime is not available in this environment. You must proceed with your best recollection without execution."

            try:
                await runtime.set_content("src/main.rs", code)
                output, success = await runtime.run_cargo()
                status = "SUCCESS" if success else "FAILURE"
                print(f"    [Formalizer] PoC Result: {status}", flush=True)
                return f"[Cargo Run {status}]\nOutput:\n{output}"
            except Exception as e:
                print(f"    [Formalizer] PoC Error: {e}", flush=True)
                return f"ERROR: Failed to execute code: {e}"

        tools.append(run_cargo)

        @function_tool
        def report_result(
            wrapper: RunContextWrapper[FormalizerContext],
            is_unique: bool,
            reasoning: str,
            title: Optional[str] = None,
            content: Optional[str] = None,
        ) -> None:
            """Report the final formalization or redundancy result."""
            print(
                f"    [Formalizer] Final Report: {'UNIQUE' if is_unique else 'REDUNDANT'}",
                flush=True,
            )
            knowledge = None
            if is_unique and title and content:
                knowledge = Knowledge(title=title, content=content)
            wrapper.context.result = FormalizationResult(
                is_unique=is_unique, reasoning=reasoning, knowledge=knowledge
            )

        tools.append(report_result)

        agent = AgentWrapper[str].create(
            name="ReflectiveKnowledgeFormalizer",
            instructions=PROMPT,
            model=self.model,
            tools=tools,
            tool_use_behavior=StopAtTools(stop_at_tool_names=[report_result.name]),
            model_settings=ModelSettings(tool_choice="auto"),
        )

        transcript = format_trajectory_transcript(trajectory, use_thinking=True)
        input_text = f"""\
### Reflection to Formalize
Category: {reflection.category}
Insight: {reflection.insight}
Evidence: {reflection.evidence}

### Full Trajectory Context
<Trajectory>
{transcript}
</Trajectory>
"""

        try:
            await agent.run(input_text, context=context, max_turns=15)
            if context.result:
                return context.result
            return FormalizationResult(
                is_unique=True, reasoning="Agent finished without calling report_result"
            )
        except Exception as e:
            logger.error(f"Formalization process failed: {e}")
            return FormalizationResult(is_unique=True, reasoning=f"Error: {e}")
