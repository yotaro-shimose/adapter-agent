import asyncio
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agents import (
    RunContextWrapper,
    StopAtTools,
    TResponseInputItem,
    add_trace_processor,
    function_tool,
)
from coder_mcp.runtime.runtime import Runtime
from coder_mcp.runtime.rust_env import RustCodingEnvironment
from coder_mcp.runtime.temp_workspace import TempWorkspace
from dotenv import load_dotenv
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper
from oai_utils.tracing import AgentContentPrinter
from pydantic import BaseModel

from adapter_agent.model_helper import get_gemini
from adapter_agent.qra import QA


class Task(BaseModel):
    id: str
    instruction: str


class Trajectory(BaseModel):
    input_list: list[TResponseInputItem]

    def as_str(self) -> str:
        # Convert input list to a readable string representation
        buffer = []
        for item in self.input_list:
            if isinstance(item, dict):
                role = item.get("role", "unknown")
                content = item.get("content", "")
                buffer.append(f"[{role}]: {content}")
            else:
                buffer.append(str(item))
        return "\n\n".join(buffer)

    def add_item(self, item: TResponseInputItem) -> None:
        self.input_list.append(item)


class TaskPool(BaseModel):
    tasks: dict[str, Task]

    def register(self, task: Task) -> None:
        print(f"Details: Registering task: {task.instruction}")
        self.tasks[task.id] = task

    def get(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def delete(self, task_id: str) -> None:
        if task_id in self.tasks:
            del self.tasks[task_id]

    def pop_random(self) -> Task:
        task_id = random.choice(list(self.tasks.keys()))
        task = self.tasks[task_id]
        self.delete(task_id)
        return task


class SFTDataset(BaseModel):
    items: list[QA] = []

    def register(self, qra: QA) -> None:
        print(f"Details: Registering QA: {qra.question}")
        self.items.append(qra)


class VerificationResult(BaseModel):
    success: bool
    reasoning: str


class SolverContext(BaseModel):
    qra: QA | None = None


@function_tool
def report_success(
    wrapper: RunContextWrapper[SolverContext],
    question: str,
    answer: str,
) -> None:
    """
    Report that the task has been successfully solved.
    Args:
        question: The original task instruction or a refined version of it.
        answer: The final solution (code and explanation).
    """
    wrapper.context.qra = QA(
        question=question,
        answer=answer,
    )


@function_tool
def report_failure() -> None:
    """
    Report that the task could not be solved.
    Args:
        reason: The reason for failure.
    """
    pass


class SolverResult(BaseModel):
    qa: QA | None = None
    trajectory: Trajectory


@dataclass
class Solver:
    model: AgentsSDKModel

    async def try_solve(
        self, task: Task, runtime: Runtime, library_name: str
    ) -> SolverResult:
        """
        タスクを解いてみる。
        もしタスクを解くことができたらSolutionを生成してReturnする。
        もしタスクを解くことができなければ、実行結果からTrajectoryを生成してReturnする。
        """
        print(f"Details: Solver attempting task: {task.instruction}")

        PROMPT = f"""
You are an expert Rust software engineer.
Your task is to solve the following problem:
{task.instruction}

You are working in a cargo-initialized project.
The `{library_name}` library source code is located at `workspace_dir/repos/{library_name}` in case you do not know its API usage.
The library is just for reference and is already installed in the workspace_dir, so you do not need to run `cargo add`.
You must not add the repository as path dependency. Stick with the version that is already installed.

You have access to a coding environment. You can write and run code to test your solution.
Once you have defined a solution and confirmed it works (to the best of your ability), you MUST call the `report_success` tool.
If you find that you cannot solve the problem, you MUST call the `report_failure` tool with a reason.

If you hit the turn limit without reporting success or failure, it will be considered a failure.
"""

        async with runtime.coder_mcp() as coder_mcp:
            agent = AgentWrapper.create(
                name="Solver",
                instructions=PROMPT,
                model=self.model,
                tools=[report_failure, report_success],
                mcp_servers=[coder_mcp],
                tool_use_behavior=StopAtTools(
                    stop_at_tool_names=[report_failure.name, report_success.name]
                ),
            )

            context = SolverContext()

            try:
                result = await agent.run(
                    "Please solve the task.", max_turns=30, context=context
                )
                trajectory = Trajectory(input_list=result.to_input_list())
                if context.qra is not None:
                    return SolverResult(qa=context.qra, trajectory=trajectory)
                else:
                    return SolverResult(trajectory=trajectory)

            except AgentRunFailure as e:
                # We have not yet decided what to do in this case.
                print(f"Solver failed (AgentRunFailure): {e}")
                raise NotImplementedError
            except Exception as e:
                print(f"Solver error: {e}")
                raise NotImplementedError


@dataclass
class Verifier:
    model: AgentsSDKModel

    async def verify(self, qra: QA, runtime: Runtime) -> VerificationResult:
        """
        Questionに対してAnswerが問題を解決できるものとなっているかどうかをコードの実行などを通じて検証して、QAが正しければTrueをリターンする。
        """
        print("Details: Verifying QA...")

        PROMPT = f"""
You are a Quality Assurance engineer for Rust code.
Your task is to verify the following Solution for the given Question.

Question:
{qra.question}

Answer to Verify:
{qra.answer}

You are starting from the state where the Solver agent finished its implementation.
The workspace is a cargo-initialized project.

You must:
1. Create a verification script (e.g., a Rust test or main function) that checks if the code in the Answer works as expected.
2. Run the verification script.
3. If it compiles and runs correctly producing the expected output, report success.
4. If it fails, report failure with reasoning in JSON.
"""
        async with runtime.coder_mcp() as coder_mcp:
            agent = AgentWrapper[VerificationResult].create(
                name="Verifier",
                instructions=PROMPT,
                model=self.model,
                mcp_servers=[coder_mcp],
                output_type=VerificationResult,
            )

            try:
                result = await agent.run("Verify the solution.", max_turns=30)
                return result.final_output()
            except AgentRunFailure as e:
                print(f"Verification process failed: {e}")
                raise


@dataclass
class Analyzer:
    model: AgentsSDKModel

    async def analyze_trajectory(
        self, trajectory: Trajectory, runtime: Runtime
    ) -> Task:
        """
        Trajectoryを分析してなぜSolverが失敗したのかを理解する。
        Solverが解けるであろうより小さな問題を生成する。
        """
        print("Details: Analyzing failure trajectory...")

        PROMPT = f"""
You are a Senior Engineer analyzing a Junior Engineer's failure.
The Junior Engineer tried to solve a task but failed.
You can see the current state of the workspace where the Junior Engineer finished execution.
The workspace is a cargo-initialized project.

Here is the execution log (Trajectory):
{trajectory.as_str()}

Your goal is to create a *simpler* sub-task that helps bridge the gap.
The sub-task should be:
1. Self-contained.
2. Easier than the original failed task.
3. Related to the specific failure point (e.g., if they failed to import `numrs::tensor`, the task should be "Create a basic tensor in numrs").

Return a new Task with a clear instruction.
"""
        # Analyzer doesn't necessarily need runtime tools, but consistency helps.
        # We can run it without coder_mcp if it's just text processing, but let's give it just in case.
        async with runtime.coder_mcp() as coder_mcp:
            agent = AgentWrapper[Task].create(
                name="Analyzer",
                instructions=PROMPT,
                model=self.model,
                mcp_servers=[coder_mcp],
                output_type=Task,
            )

            try:
                result = await agent.run(
                    "Analyze the trajectory and generate a sub-task.", max_turns=30
                )
                task = result.final_output()
                # Ensure new ID
                task.id = str(uuid.uuid4())
                return task
            except Exception as e:
                print(f"Analyzer failed: {e}")
                # Fallback task
                return Task(
                    id=str(uuid.uuid4()),
                    instruction="Read the numrs documentation and summarize basic usage.",
                )


@dataclass
class Agents:
    solver: Solver
    verifier: Verifier
    analyzer: Analyzer

    @classmethod
    def from_model(cls, model: AgentsSDKModel):
        return cls(
            solver=Solver(model=model),
            verifier=Verifier(model=model),
            analyzer=Analyzer(model=model),
        )


async def process_task(
    agents: Agents,
    task: Task,
    task_pool: TaskPool,
    sft_dataset: SFTDataset,
    workspace_template_location: Path,
    host_lib_dir: Path,
):
    """
    1. リファレンスも使いながらエージェントがとく。
    2. Taskを解くのに成功したとエージェントが判断した場合
        ...
    """
    print(f"Processing Task: {task.instruction}")

    # Inject the already prepared library
    injections = {host_lib_dir: f"repositories/{host_lib_dir.name}"}

    async with TempWorkspace(
        workspace_template_location, injections=injections
    ) as temp_workspace:
        async with RustCodingEnvironment(workspace_dir=temp_workspace) as rust_env:
            solver_result = await agents.solver.try_solve(
                task, rust_env, host_lib_dir.name
            )

            if isinstance(solver_result, QA):
                print("Solver produced a QA. Verifying...")
                verification_result = await agents.verifier.verify(
                    solver_result, rust_env
                )
                if verification_result.success:
                    print("Verification SUCCESS.")
                    sft_dataset.register(solver_result)
                    # Task is effectively done (popped from pool by caller or here?)
                    pass
                else:
                    print("Verification FAILED.")
                    print(verification_result.reasoning)
                    solver_result.trajectory.add_item(
                        {
                            "role": "user",
                            "content": f"We ran verification process with another agent, but verification failed: {verification_result.reasoning}",
                        }
                    )

                    analysis = await agents.analyzer.analyze_trajectory(
                        solver_result.trajectory, rust_env
                    )
                    print(f"Generated subtask: {analysis.instruction}")
                    task_pool.register(analysis)

            elif isinstance(solver_result, Trajectory):
                print("Solver failed to produce QA. Analyzing trajectory...")
                trajectory_analysis = await agents.analyzer.analyze_trajectory(
                    solver_result, rust_env
                )
                print(f"Generated subtask: {trajectory_analysis.instruction}")
                task_pool.register(trajectory_analysis)
            else:
                raise ValueError(f"Unexpected solver result: {solver_result}")


async def main():
    load_dotenv()
    add_trace_processor(AgentContentPrinter())

    model = get_gemini()

    # Setup Experiment Directory
    experiment_id = f"hh_exp_{int(time.time())}"
    base_dir = Path("experiments") / experiment_id
    workspace_template_location = Path("templates") / "rust-template"
    lib_path = Path("repositories") / "numrs"

    print(f"Experiment ID: {experiment_id}")
    print(f"Base Directory: {base_dir}")

    agents = Agents.from_model(model)

    task_pool = TaskPool(tasks={})
    sft_dataset = SFTDataset(items=[])

    # Initial Task
    # "Using numrs library, create a Conv2D layer"
    # User said: "initial task ... TaskPool ... instructions self-contained ... mention language and library"
    initial_task = Task(
        id=str(uuid.uuid4()),
        instruction="Using the Rust programming language and the `numrs` library, implement a 2D Convolution (Conv2D) layer. The implementation should include a struct for the layer and a forward pass method.",
    )
    task_pool.register(initial_task)

    # Process loop (simple version)
    # Just run until pool empty or max steps
    max_steps = 3
    step = 0

    while step < max_steps:
        current_task = task_pool.pop_random()
        if not current_task:
            print("Task pool empty.")
            break

        await process_task(
            agents=agents,
            task=current_task,
            task_pool=task_pool,
            sft_dataset=sft_dataset,
            host_lib_dir=lib_path,
            workspace_template_location=workspace_template_location,
        )
        step += 1


if __name__ == "__main__":
    asyncio.run(main())
