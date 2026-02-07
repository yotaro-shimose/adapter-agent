import logging
from dataclasses import dataclass

from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper

from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.hierarchical.state import TaskList
from adapter_agent.hierarchical.types import Task, TaskInstructionList
from adapter_agent.qra import QA

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Augmentor[T: AgentsSDKModel](BaseAgent[T, QA, TaskList]):
    async def augment(self, qa: QA, num_augmentations: int = 3) -> list[Task]:
        """
        １つのQAからこれと同じトピックを違う観点で学習するための複数のTask(Instruction)を生成する.
        例えばQAがnp.matmulの使い方に関するものだったのであれば、掛け算を使った複数のトピック(e.g. 行列の掛け算を応用した複数のベクタの内積の計算や３個以上の行列の掛け算など)に関するQuestionsを生成する。
        ただし、生成するQuestionsは元のQAの知識に閉じていること. 例えば新たな関数の使い方など(e.g. log関数や逆行列関数)を含まないこと
        """

        PROMPT = f"""
<Role>
You are a Conversational Partner helping a user learn a new library.
</Role>

<Goal>
Your goal is to create {num_augmentations} conversation-style questions to test the user's understanding of the concept, based on the original QA.
</Goal>

<Constraints>
1. **Self-contained**: Each question must be understandable on its own.
2. **Explicit Context**: The question must explicitly mention the library name being practiced (e.g., "in numrs", "using pandera").
3. **Same Concept**: Must cover the same topic/concept as the original QA but from a different perspective.
4. **Knowledge Constraint**: Do NOT introduce new concepts or library functions (e.g., log, inverse matrix) that were NOT part of the original QA concept. The goal is to deepen understanding of *what was already learned*, not to expand scope.
5. **Conversational Tone**: Use natural, conversational language. Avoid textbook-style phrasing.
6. **NO API Usage**: The question MUST NOT show the API usage or code snippets. The user should recall how to use the library from their internal knowledge.
</Constraints>

<Examples>
- "How do I calculate the sum along the second dimension in numpy?"
- "Can you explain how to perform matrix multiplication in pytorch?"
- "I need to reshape a tensor in tensorflow. What's the function for that?"
- "What is the way to create a zero-filled tensor in numrs?"
- "Explain to me how to slice a tensor effectively using this library." (Avoid this if possible, prefer explicit name)
- "Is there a way to transpose a matrix in pandas?"
</Examples>

<OutputFormat>
You must return a `TaskInstructionList` object which contains a list of `TaskInstruction` objects.
Each `TaskInstruction` must have:
- `instruction`: The string instruction for the practice task.

IMPORTANT: Your output must be PURE JSON. Do NOT include any markdown formatting such as ```json ... ```.
</OutputFormat>
"""

        agent = AgentWrapper[TaskInstructionList].create(
            name="Augmentor",
            instructions=PROMPT,
            model=self.model,
            output_type=TaskInstructionList,
        )

        try:
            result = await agent.run(
                f"<Original Task>\n{qa.question}\n</Original Task>\n<Original Answer>\n{qa.answer}\n</Original Answer>",
                max_turns=10,
            )
            generated_list = result.final_output()

            new_tasks = []
            for instr in generated_list.tasks:
                new_tasks.append(Task.from_instruction(instr.instruction))

            task_list = TaskList(tasks=new_tasks)

            self.maybe_add_to_memory(qa, task_list)
            return new_tasks
        except AgentRunFailure as e:
            logger.error(f"Augmentor failed: {e}, skipping.")
            return []
