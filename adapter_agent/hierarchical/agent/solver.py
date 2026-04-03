import logging
from dataclasses import dataclass

from agents import ModelSettings, RunContextWrapper, StopAtTools, function_tool
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from pydantic import BaseModel

from adapter_agent.data import QRA
from adapter_agent.hierarchical.agent.base import BaseAgent
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer

logger = logging.getLogger(__name__)


CONTEXT = """\
<Memory>
I am an expert in the `{library_name}` library.
I have internalized the full documentation and best practices for this library.
I will solve the task by recalling the appropriate APIs and patterns from my intrinsic knowledge.
</Memory>
"""


class SolverContext(BaseModel):
    solution: QRA | None = None


@function_tool
def report_solution(
    wrapper: RunContextWrapper[SolverContext],
    reasoning: str,
    answer: str,
) -> None:
    """
    Report the generated reasoning and code solution for the task.
    Args:
        reasoning: The step-by-step reasoning process explaining the solution logic.
        answer: The self-contained Rust code solution.
    """
    # Note: question will be filled by the caller based on the input question
    wrapper.context.solution = QRA(
        question="",
        reasoning=reasoning,
        answer=answer,
    )


@dataclass(kw_only=True)
class SolverAgent[T: AgentsSDKModel](BaseAgent[T]):
    rust_doc_analyzer: AsyncRustDocAnalyzer

    async def solve(self, question: str) -> QRA:
        """
        Solve a given programming task and return the Reasoning and Answer.
        """
        crate_overview = self.rust_doc_analyzer.get_overview()
        
        prompt = f"""\
<Role>
You are an expert Rust developer.
Your task is to solve the given Question using the provided API documentation.
</Role>

<Objective>
- Understand the requirement in the Question.
- Develop a solution using the API found in the Crate Overview.
- Provide a step-by-step reasoning in $R$.
- Provide the final, self-contained Rust code in $A$.
</Objective>

<Crate Overview>
{crate_overview}
</Crate Overview>

<OutputFormat>
Use the `report_solution` tool to submit your reasoning and code.
</OutputFormat>
"""
        agent = AgentWrapper.create(
            name="SolverAgent",
            instructions=prompt,
            model=self.model,
            tools=[report_solution],
            model_settings=ModelSettings(
                tool_choice="required",
            ),
            tool_use_behavior=StopAtTools(
                stop_at_tool_names=[report_solution.name]
            ),
        )

        context = SolverContext()
        input_prompt = f"<Question>\n{question}\n</Question>\n\nPlease solve this task now."

        try:
            await agent.run(input_prompt, max_turns=5, context=context)
            if context.solution is None:
                raise ValueError("SolverAgent finished without reporting a solution.")
            
            # Fill the question from the input
            context.solution.question = question
            return context.solution
        except AgentRunFailure as e:
            logger.error(f"Solution generation failed: {e}")
            raise
