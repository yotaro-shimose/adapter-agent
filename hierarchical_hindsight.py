import asyncio
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Optional, TypeVar

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


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class MemoryItem(BaseModel, Generic[InputT, OutputT]):
    input: InputT
    output: OutputT
    timestamp: float


class Memory(BaseModel, Generic[InputT, OutputT]):
    items: list[MemoryItem[InputT, OutputT]] = []
    file_path: Optional[str] = None

    def add(self, input: InputT, output: OutputT) -> None:
        self.items.append(MemoryItem(input=input, output=output, timestamp=time.time()))

    def save(self) -> None:
        if self.file_path:
            with open(self.file_path, "w") as f:
                f.write(self.model_dump_json(indent=2))

    def load(self) -> None:
        if self.file_path and Path(self.file_path).exists():
            with open(self.file_path, "r") as f:
                data = f.read()
                # Pydantic's model_validate_json handles generics if TypeAdapter is used externally,
                # but direct method needs care.
                # Actually model_validate_json on the model instance or class works if type info is preserved.
                # However, for generic model parsing from JSON, it's safer to rely on internal structure matching.
                loaded = self.model_validate_json(data)
                self.items = loaded.items


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

    def pop_random(self) -> Task | None:
        if not self.tasks:
            return None
        task_id = random.choice(list(self.tasks.keys()))
        task = self.tasks[task_id]
        self.delete(task_id)
        return task


class SFTDataset(BaseModel):
    items: list[QA] = []

    def register(self, qra: QA) -> None:
        print(f"Details: Registering QA: {qra.question}")
        self.items.append(qra)


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
    is_max_turns_exceeded: bool = False


@dataclass
class Solver:
    model: AgentsSDKModel
    memory: Memory[Task, SolverResult]

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
                    result = SolverResult(qa=context.qra, trajectory=trajectory)
                else:
                    result = SolverResult(trajectory=trajectory)

                self.memory.add(task, result)
                self.memory.save()
                return result

            except AgentRunFailure as e:
                if e.cause == "MaxTurnsExceeded":
                    print("Details: Solver hit MaxTurnsExceeded.")
                    input_list = e.to_input_list()
                    if input_list:
                        trajectory = Trajectory(input_list=input_list)
                        result = SolverResult(
                            trajectory=trajectory, is_max_turns_exceeded=True
                        )
                        self.memory.add(task, result)
                        self.memory.save()
                        return result
                    else:
                        # Should not happen if to_input_list works, but fallback
                        trajectory = Trajectory(input_list=[])
                        result = SolverResult(
                            trajectory=trajectory, is_max_turns_exceeded=True
                        )
                        self.memory.add(task, result)
                        self.memory.save()
                        return result
                else:
                    raise
            except Exception as e:
                print(f"Solver error: {e}")
                raise NotImplementedError


class VerificationResult(BaseModel):
    success: bool
    reasoning: str


@dataclass
class Verifier:
    model: AgentsSDKModel
    memory: Memory[QA, VerificationResult]

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

The workspace is a cargo-initialized project.
You are starting from the state where the Solver agent finished its implementation.

You must:
1. First, check if the provided code works without error.
2. After that, read the source code to check if the solution is not cheating and satisfies the question.
3. Then, respond with JSON with the following fields:
    - success: bool
    - reasoning: str
Do not waste tool calls to use them as your scratchpad.
Include your deep analysis in the reasoning field.
Your are not supposed to write dedicated test code. Most of the time you just execute the provided code, or add debug print statements at maximum.
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
                final_output = result.final_output()
                self.memory.add(qra, final_output)
                self.memory.save()
                return final_output
            except AgentRunFailure as e:
                print(f"Verification process failed: {e}")
                raise


@dataclass
class Analyzer:
    model: AgentsSDKModel
    memory: Memory[Trajectory, Task]

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
3. Related to the specific failure point.

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
                self.memory.add(trajectory, task)
                self.memory.save()
                return task
            except Exception as e:
                print(f"Analyzer failed: {e}")
                # Fallback task
                return Task(
                    id=str(uuid.uuid4()),
                    instruction="Read the numrs documentation and summarize basic usage.",
                )


class TaskList(BaseModel):
    tasks: list[Task]


@dataclass
class DecomposerInput(BaseModel):
    trajectory: Trajectory
    original_instruction: str


@dataclass
class Decomposer:
    model: AgentsSDKModel
    memory: Memory[DecomposerInput, TaskList]

    async def decompose(
        self, trajectory: Trajectory, original_instruction: str, library_name: str
    ) -> list[Task]:
        """
        Solverが時間切れ(MaxTurnsExceeded)で失敗した場合に呼ばれる。
        元のタスクが難しすぎたため、より簡単でself-containedな練習問題（サブタスク）を作成する。
        """
        print("Details: Decomposing task due to MaxTurnsExceeded...")

        PROMPT = f"""
You are a Teacher mentoring a student (Agent).
The student failed to complete the following task because it was too difficult (Code limit exceeded).
Task: "{original_instruction}"

Here is the partial progress (Trajectory) of the student:
{trajectory.as_str()}

Your goal is to create 2-3 *conceptually simpler* practice tasks that isolate core concepts required for the original task.
The goal is NOT to solve the original task, but to "practice" specific parts of it.

Constraints for the new tasks:
1. **Completely Self-contained**: Each task must be a standalone assignment. It must NOT assume any prior files, state, or context from previous attempts.
2. **Fresh Start**: Assume the agent starts in a fresh, clean cargo project. The `{library_name}` library is already available. Do not ask the agent to create a sub-directory for the project.
3. **No Context Continuity**: Do NOT reference the failed task's context or imply continuity (e.g., AVOID phrases like "continue with...", "before implementing X...", "as part of the previous goal..."). The task must make sense to someone seeing it for the first time without any background knowledge.
4. **Simpler**: Must be significantly easier than the original task.
5. **Targeted**: Focus on specific skills or API usages that the agent struggled with.
6. **Immediate**: The agent should be able to start coding immediately.

Example:
If the original task was "Implement a complex Neural Network Layer", and they got stuck on matrix multiplication,
a good practice task would be:
"Write a function that performs simple matrix multiplication using the `{library_name}` library and prints the result."

Return a list of Tasks, each with a clear, self-contained instruction.
"""
        agent = AgentWrapper[TaskList].create(
            name="Decomposer",
            instructions=PROMPT,
            model=self.model,
            output_type=TaskList,
        )

        try:
            result = await agent.run("Create simplified practice tasks.", max_turns=10)
            task_list = result.final_output()
            for task in task_list.tasks:
                task.id = str(uuid.uuid4())

            self.memory.add(
                DecomposerInput(
                    trajectory=trajectory, original_instruction=original_instruction
                ),
                task_list,
            )
            self.memory.save()
            return task_list.tasks
        except Exception as e:
            print(f"Decomposer failed: {e}")
            return [
                Task(
                    id=str(uuid.uuid4()),
                    instruction=f"Research the basics required for: {original_instruction}",
                )
            ]


@dataclass
class Agents:
    solver: Solver
    verifier: Verifier
    analyzer: Analyzer
    decomposer: Decomposer

    @classmethod
    def from_model(cls, model: AgentsSDKModel, base_dir: Path):
        base_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            solver=Solver(
                model=model,
                memory=Memory[Task, SolverResult](
                    file_path=str(base_dir / "memory_solver.json")
                ),
            ),
            verifier=Verifier(
                model=model,
                memory=Memory[QA, VerificationResult](
                    file_path=str(base_dir / "memory_verifier.json")
                ),
            ),
            analyzer=Analyzer(
                model=model,
                memory=Memory[Trajectory, Task](
                    file_path=str(base_dir / "memory_analyzer.json")
                ),
            ),
            decomposer=Decomposer(
                model=model,
                memory=Memory[DecomposerInput, TaskList](
                    file_path=str(base_dir / "memory_decomposer.json")
                ),
            ),
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

            if solver_result.is_max_turns_exceeded:
                print("Solver timed out. Decomposing task...")
                new_tasks = await agents.decomposer.decompose(
                    solver_result.trajectory, task.instruction, host_lib_dir.name
                )
                for new_task in new_tasks:
                    print(f"Generated practice task: {new_task.instruction}")
                    task_pool.register(new_task)

            elif isinstance(solver_result.qa, QA):
                print("Solver produced a QA. Verifying...")
                verification_result = await agents.verifier.verify(
                    solver_result.qa, rust_env
                )
                if verification_result.success:
                    print("Verification SUCCESS.")
                    sft_dataset.register(solver_result.qa)
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

            else:
                # Normal failure (report_failure called or other implicit failure without timeout)
                print("Solver failed to produce QA. Analyzing trajectory...")
                trajectory_analysis = await agents.analyzer.analyze_trajectory(
                    solver_result.trajectory, rust_env
                )
                print(f"Generated subtask: {trajectory_analysis.instruction}")
                task_pool.register(trajectory_analysis)


async def main():
    load_dotenv()
    add_trace_processor(AgentContentPrinter())

    model = get_gemini()

    # Setup Experiment Directory
    experiment_id = f"hh_exp_{int(time.time())}"
    base_dir = Path("experiments") / experiment_id
    workspace_template_location = Path("templates") / "rust_template"
    lib_path = Path("repositories") / "numrs"
    assert workspace_template_location.exists()
    assert lib_path.exists()

    print(f"Experiment ID: {experiment_id}")
    print(f"Base Directory: {base_dir}")

    agents = Agents.from_model(model, base_dir)

    task_pool = TaskPool(tasks={})
    sft_dataset = SFTDataset(items=[])

    # Initial Task
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
