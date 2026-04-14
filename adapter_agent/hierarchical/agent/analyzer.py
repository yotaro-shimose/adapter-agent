import logging
import re
from dataclasses import dataclass

from agents import RunConfig
from oai_utils.agent import AgentsSDKModel, AgentWrapper
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.hierarchical.agent.rewirer import format_trajectory_transcript
from adapter_agent.hierarchical.types import Task

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Analyzer[T: AgentsSDKModel](BaseAgent[T]):
    async def analyze_trajectory(self, trajectory: list[TinkerMessage]) -> Task:
        """
        Trajectoryを分析してなぜSolverが失敗したのかを理解する。
        Solverが解けるであろうより小さな問題を生成する。
        """
        PROMPT = """
You are a Senior Architect analyzing a technical failure in a Rust project.
A Solver agent attempted a task and failed. Your role is to identify a fundamental, simpler prerequisite that must be achieved to resolve the impasse.

### Objectives
1. Identify the core conceptual or functional hurdle that led to the failure.
2. Define a "Definition of Done" for a new, smaller sub-task that addresses this hurdle.

### Sub-task Requirements
- Focus on outcomes: Describe the functional behavior or state that must be achieved.
- Implementation Agnostic: Do not specify any library APIs, specific function names, or agent tools (e.g., do not say "use grep" or "call XXX API"). 
- Logical Atomicity: The sub-task should be the smallest meaningful step towards understanding or fixing the original failure.

The Solver has full autonomy over implementation details and tool usage. Your sub-task must define *what* to solve, not *how* to solve it.

OUTPUT FORMAT:
Provide your reasoning, and then output the subtask instruction inside a <subtask></subtask> xml block.
"""

        # Analyzer doesn't necessarily need runtime tools, but consistency helps.
        # We can run it without coder_mcp if it's just text processing, but let's give it just in case.
        agent = AgentWrapper[str].create(
            name="Analyzer",
            instructions=PROMPT,
            model=self.model,
        )

        # Filter out system messages to avoid tool description pollution
        filtered_trajectory = [m for m in trajectory if m.get("role") != "system"]

        result = await agent.run(
            f"""\
Analyze the following trajectory and generate a sub-task.
<Trajectory>
{format_trajectory_transcript(filtered_trajectory)}
</Trajectory>
""",
            run_config=RunConfig(tracing_disabled=True),
        )
        response_text = result.final_output()
        match = re.search(r"<subtask>(.*?)</subtask>", response_text, re.DOTALL)
        if not match:
            raise ValueError(
                f"Could not find <subtask> block in Analyzer response: {response_text}"
            )

        task = Task.from_instruction(match.group(1).strip())
        return task
