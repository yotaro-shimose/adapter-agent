import logging
import re
from dataclasses import dataclass

from agents import RunConfig
from oai_utils.agent import AgentsSDKModel, AgentWrapper

from adapter_agent.data import QRA
from adapter_agent.hierarchical.agent.base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class KnowledgeSlicer[T: AgentsSDKModel](BaseAgent[T]):
    async def slice(self, normalized_knowledge: str) -> list[QRA]:
        """
        Generates practice problems (QRAs) from normalized knowledge.
        Each QRA should include:
        - Q: numrs2 implementation task.
        - R: library usage recall.
        - A: answer with a rust code block.
        """
        if not normalized_knowledge:
            return []

        PROMPT = """\
You are a Senior Technical Instructor specializing in Rust and the `numrs2` library.
Your goal is to generate high-quality practice problems (QRAs) based on the provided "Normalized Knowledge".

Each QRA must consist of:
1. Question (Q): A clear, self-contained implementation task that requires using the `numrs2` library to solve a specific problem mentioned or implied in the knowledge.
2. Recall (R): A brief, first-person reasoning process where you "recall" the necessary technical details, library usage patterns, and logic to solve the task. This should simulate how a knowledgeable engineer would think through the problem.
3. Answer (A): A detailed explanation of the solution followed by a complete, runnable Rust code block enclosed in ```rust ... ```. The code should demonstrate the correct usage of `numrs2` to fulfill the task.

OUTPUT FORMAT:
Provide your reasoning, and then output one or more QRAs inside <qra_list></qra_list> xml block.
Each QRA should be wrapped in <qra></qra> tags, with <question>, <reasoning>, and <answer> tags inside.

Example:
<qra_list>
<qra>
<question>
[Implementation task using numrs2]
</question>
<reasoning>
Okay, let's see. [Recall of library usage and logic]
</reasoning>
<answer>
[Explanation]
```rust
// Code here
```
</answer>
</qra>
...
</qra_list>
"""
        agent = AgentWrapper[str].create(
            name="KnowledgeSlicer",
            instructions=PROMPT,
            model=self.model,
        )

        result = await agent.run(
            f"""\
Generate QRAs from the following normalized knowledge:
<NormalizedKnowledge>
{normalized_knowledge}
</NormalizedKnowledge>
""",
            run_config=RunConfig(tracing_disabled=True),
        )

        response_text = result.final_output()
        match_list = re.search(r"<qra_list>(.*?)</qra_list>", response_text, re.DOTALL)
        if not match_list:
            logger.warning(
                "Could not find <qra_list> block in KnowledgeSlicer response."
            )
            return []

        qra_blocks = re.findall(r"<qra>(.*?)</qra>", match_list.group(1), re.DOTALL)
        qras = []
        for block in qra_blocks:
            q_match = re.search(r"<question>(.*?)</question>", block, re.DOTALL)
            r_match = re.search(r"<reasoning>(.*?)</reasoning>", block, re.DOTALL)
            a_match = re.search(r"<answer>(.*?)</answer>", block, re.DOTALL)

            if q_match and r_match and a_match:
                qras.append(
                    QRA(
                        question=q_match.group(1).strip(),
                        reasoning=r_match.group(1).strip(),
                        answer=a_match.group(1).strip(),
                    )
                )

        return qras
