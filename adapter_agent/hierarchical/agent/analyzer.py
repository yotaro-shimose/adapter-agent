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
You are a Senior Engineer analyzing a Junior Engineer's failure.
The Junior Engineer tried to solve a task but failed.
The workspace is a cargo-initialized project.

Your goal is to create a *simpler* sub-task that helps bridge the gap.
The sub-task should be:
1. Self-contained.
2. Easier than the original failed task.
3. Related to the specific failure point.

Return a new Task with a clear instruction.
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

        result = await agent.run(
            f"""\
Analyze the following trajectory and generate a sub-task.
<Trajectory>
{format_trajectory_transcript(trajectory)}
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
