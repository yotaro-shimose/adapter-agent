import logging
import re
from dataclasses import dataclass

from oai_utils.agent import AgentsSDKModel, AgentWrapper
from pydantic import BaseModel
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.renderers.base import get_text_content

from adapter_agent.hierarchical.agent.base import BaseAgent

logger = logging.getLogger(__name__)


class Reflection(BaseModel):
    insight: str
    evidence: str


REFLECTION_RE = re.compile(
    r"<reflection>\s*<insight>(.*?)</insight>\s*<evidence>(.*?)</evidence>\s*</reflection>",
    re.DOTALL,
)
TASK_RE = re.compile(r"<Task>\s*(.*?)\s*</Task>", re.DOTALL)
SUBMIT_RE = re.compile(r"<submit>\s*(.*?)\s*</submit>", re.DOTALL)


def extract_task_instruction(trajectory: list[TinkerMessage]) -> str | None:
    """Pull <Task>...</Task> from the first user message; fall back to the
    raw user content if the wrapper tag is absent."""
    for msg in trajectory:
        if msg.get("role") != "user":
            continue
        content = get_text_content(msg)
        m = TASK_RE.search(content)
        if m:
            return m.group(1).strip()
        return content.strip() or None
    return None


def extract_submit_content(trajectory: list[TinkerMessage]) -> str | None:
    """Pull the body of the LAST <submit>...</submit> emitted by the agent."""
    last: str | None = None
    for msg in trajectory:
        if msg.get("role") != "assistant":
            continue
        content = get_text_content(msg)
        for m in SUBMIT_RE.finditer(content):
            last = m.group(1).strip()
    return last


@dataclass(kw_only=True)
class Reflector[T: AgentsSDKModel](BaseAgent[T]):
    library_name: str
    qwen_no_think: bool = False

    async def reflect(self, trajectory: list[TinkerMessage]) -> list[Reflection]:
        """
        Extracts study notes from the *task instruction* and the agent's *final
        submitted answer* (the content of the last `<submit>...</submit>`).
        Intermediate exploration in the trajectory is intentionally ignored —
        the submitted answer is the verified working solution and is the
        cleanest signal for "how to use {library} correctly".
        """
        if not trajectory:
            raise ValueError("Trajectory cannot be empty")

        task_instruction = extract_task_instruction(trajectory)
        submit_content = extract_submit_content(trajectory)

        if task_instruction is None:
            logger.warning("Reflector: could not extract task instruction from trajectory.")
            return []
        if submit_content is None:
            logger.warning("Reflector: trajectory has no <submit> block; nothing to reflect on.")
            return []

        PROMPT = f"""\
You are an AI Experience Analyst and Knowledge Engineer. The agent under study is currently learning the **{self.library_name}** library, and your role is to take study notes from its work. Your reflections are saved to a knowledge base that future agents consult when solving tasks with {self.library_name}.

You will be shown a Task Instruction and the agent's Final Answer (the verified working solution it submitted). These are the only signals — intermediate exploration, failures, and tool usage are deliberately hidden, because the Final Answer alone is what was verified to work. Your job is to mine the Final Answer for reusable {self.library_name} knowledge.

### Scope
Every reflection must be about {self.library_name} itself or its idiomatic use. Do not promote alternative crates (e.g., ndarray, nalgebra) as substitutes for {self.library_name} functionality. General Rust patterns are acceptable only when they appear in service of using {self.library_name}.

### Focus Areas (in priority order)
1. **Direct Answer to the Task**: If the Final Answer conclusively answers the Task, capture that resolution as the first reflection.
2. **API Usage Discovery**: Distill the correct way to use the specific {self.library_name} APIs that appear in the Final Answer (function names, module paths, type signatures, argument shapes).
3. **Patterns**: Identify approaches in the Final Answer that apply broadly to other {self.library_name} tasks.

### Guidelines
- **How-to framing**: Each insight must describe how to do something correctly — the working API, the right module path, the idiomatic pattern. Do not frame insights as troubleshooting entries or as cautionary tales about what fails.
- **Be Technical**: Use specific names of functions, modules, and error messages drawn directly from the Final Answer.
- **Evidence-Based**: Each reflection's evidence must point to a concrete fragment of the Final Answer (a function call, an import, a code line). Do not cite turns or tool actions — those are not visible to you.
- **Concept-level granularity**: One reflection per *concept*, not per *fact*. A function together with its return type, error handling, and typical call-site pattern is one concept and goes into one reflection — not three. Prefer fewer, denser reflections. If two reflections share a function or topic, merge them into one.
- **Uniqueness**: Each `<reflection>` block must cover a distinct concept. Do not emit a reflection if it is a subset, restatement, or summary of another reflection in your output.

### Output Format
Emit each reflection as an XML block exactly in this shape:

<reflection>
<insight>The technical fact or pattern, stated concisely.</insight>
<evidence>A brief quotation or reference to the relevant fragment of the Final Answer.</evidence>
</reflection>

You may emit any number of <reflection> blocks. Free-form text outside the tags (analysis, summaries, comments) is ignored — only the tagged blocks are extracted. Do not nest <reflection> blocks. If a direct answer to the Task exists, place that block first.
"""

        agent = AgentWrapper[str].create(
            name="Reflector",
            instructions=PROMPT,
            model=self.model,
        )

        input_prompt = f"""\
Extract the reusable {self.library_name} knowledge from the following Task and Final Answer.

<Task>
{task_instruction}
</Task>

<FinalAnswer>
{submit_content}
</FinalAnswer>
"""
        if self.qwen_no_think:
            input_prompt = "/no_think " + input_prompt

        try:
            result = await agent.run(input_prompt)
            text = result.final_output()
        except Exception as e:
            logger.error(f"Reflection extraction failed: {e}")
            return []

        reflections = [
            Reflection(insight=m.group(1).strip(), evidence=m.group(2).strip())
            for m in REFLECTION_RE.finditer(text)
        ]
        if not reflections:
            logger.warning(
                "Reflector produced no parseable <reflection> blocks. Raw output: %s",
                text[:500],
            )
        return reflections
