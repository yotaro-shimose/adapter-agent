import asyncio
import json
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Self, TypedDict, cast

import tinker
from agents import RunContextWrapper, StopAtTools, function_tool
from coder_mcp.runtime import (
    CoderMCPRuntimeError,
    Runtime,
)
from oai_utils import AgentsSDKModel
from oai_utils.agent import AgentWrapper
from pydantic import BaseModel
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.renderers.base import TextPart, ThinkingPart, ToolCall, ToolSpec
from tinker_cookbook.rl import types
from tinker_cookbook.rl.message_env import (
    EnvFromMessageEnv,
    MessageEnv,
    MessageStepResult,
)
from tinker_cookbook.tool_use.tools import handle_tool_call
from tinker_cookbook.tool_use.types import Tool, ToolInput, ToolResult
from typing_extensions import AsyncGenerator

from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.knowledge_db import KnowledgeDB
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.rl.env.conclusion import SSConclusion
from adapter_agent.rl.env.injection import _inject_tools_into_prompt
from adapter_agent.rl.env.reward import LLMAsAJudgeSingleTurn
from adapter_agent.util.exception import CodingEnvironmentError

CARGO_INIT_MAIN_RS = """\
fn main() {
    println!("Hello, world!");
}
"""


@dataclass
class SimplifiedSolverEnvState:
    task: Task
    library_name: str
    prethink: str | None
    messages: list[TinkerMessage]
    qwen_no_think: bool = False

    @classmethod
    def numrs2(
        cls,
        task: Task,
        prethink: str | None = None,
        qwen_no_think: bool = False,
    ) -> Self:
        return cls(
            task=task,
            library_name="numrs2",
            prethink=prethink,
            messages=[],
            qwen_no_think=qwen_no_think,
        )

    def with_messages(self, messages: list[TinkerMessage]) -> Self:
        return self.__class__(
            task=self.task,
            library_name=self.library_name,
            prethink=self.prethink,
            messages=messages,
            qwen_no_think=self.qwen_no_think,
        )


@dataclass
class SimplifiedSolverMutableState:
    remaining_turns: int


class AgenticSearchContext(BaseModel):
    final_answer: str | None = None


class SearchTool(Tool):
    def __init__(
        self,
        search_model: AgentsSDKModel,
        analyzer: RustDocAnalyzer,
        knowledge_db: KnowledgeDB,
        mutable_state: SimplifiedSolverMutableState,
    ):
        self.search_model = search_model
        self.analyzer = analyzer
        self.knowledge_db = knowledge_db
        self.mutable_state = mutable_state

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return (
            "Search the Rust documentation returning analyzed knowledge about target."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The exact concept, function or error code to query.",
                },
            },
            "required": ["query"],
        }

    async def run(self, input: ToolInput) -> ToolResult:
        query = input.arguments.get("query", "")

        ctx = AgenticSearchContext()

        # Step 1: Direct Search Execution (KnowledgeDB -> Fallback to RustDocs)
        db_results = await self.knowledge_db.search(query, limit=3)

        if db_results:
            # Flow A: Knowledge DB (Select and return as-is)
            numbered_results = {i: res for i, res in enumerate(db_results)}

            @function_tool
            def select_knowledge(
                wrapper: RunContextWrapper[AgenticSearchContext], index: int
            ) -> None:
                """Select the most relevant knowledge snippet by its integer index."""
                if index in numbered_results:
                    wrapper.context.final_answer = numbered_results[index]["content"]
                else:
                    wrapper.context.final_answer = "Invalid knowledge index selected."

            agent = AgentWrapper[str].create(
                name="KnowledgeSelector",
                instructions=(
                    "You are a knowledge selection assistant.\\n"
                    f"The user searched for: '{query}'.\\n"
                    "Below are previously learned knowledge snippets indexed from 0. "
                    "Read them carefully, select the single most relevant snippet for solving the user's issue, "
                    "and use the `select_knowledge` tool to choose it. You only need to output the index."
                ),
                model=self.search_model,
                tools=[select_knowledge],
                tool_use_behavior=StopAtTools(stop_at_tool_names=["select_knowledge"]),
            )

            input_list = [
                {"index": i, "learned_query": r["query"], "content": r["content"]}
                for i, r in numbered_results.items()
            ]
            input_prompt = f"<KnowledgeSnippets>\\n{json.dumps(input_list, indent=2)}\\n</KnowledgeSnippets>"

            await agent.run(input_prompt, context=ctx)
            final_ans = ctx.final_answer or "Failed to select knowledge."

        else:
            # Flow B: Rust Documentation (Synthesize and Summarize)
            docs = self.analyzer.search(query, limit=5)
            raw_results = [d.model_dump() for d in docs]

            if not raw_results:
                final_ans = (
                    "No results found in either Knowledge DB or standard documentation."
                )
            else:

                @function_tool
                def report_summary(
                    wrapper: RunContextWrapper[AgenticSearchContext], summary: str
                ) -> None:
                    """Report your comprehensive technical summary back to the user."""
                    wrapper.context.final_answer = summary

                agent = AgentWrapper[str].create(
                    name="RustDocSummarizer",
                    instructions=(
                        "You are an expert technical summarizer.\\n"
                        f"The user searched for: '{query}'. Source: Official Rust Documentation.\\n"
                        "Below are the raw search results. Your task is to:\\n"
                        "1. Determine the EXACTLY ONE most relevant document from the results.\\n"
                        "2. Summarize its contents comprehensively. Do NOT lose any technical details, Rust code signatures, or logic patterns.\\n"
                        "3. Do not formulate the response entirely as a citation ID; instead, provide the raw summary.\\n"
                        "4. Use the `report_summary` tool to submit."
                    ),
                    model=self.search_model,
                    tools=[report_summary],
                    tool_use_behavior=StopAtTools(
                        stop_at_tool_names=["report_summary"]
                    ),
                )

                input_prompt = (
                    f"<SearchResults>\\n{json.dumps(raw_results, indent=2)}\\n</SearchResults>\\n\\n"
                    "Select the single best document from the list above and provide a detailed summary using `report_summary`."
                )

                await agent.run(input_prompt, context=ctx)
                final_ans = ctx.final_answer or "Failed to summarize the document."

        output = f"{final_ans}"

        assert input.call_id is not None
        msg = TinkerMessage(
            role="tool",
            content=output,
            tool_call_id=input.call_id,
        )
        return ToolResult(messages=[msg])


# ReplaceAndRunTool has been replaced by XML tag parsing in the step method.


class SSMetrics(TypedDict):
    success: float
    context_length_exceeded: float
    no_code_found: float
    no_text_content: float
    code_did_not_compile: float
    verification_failed: float
    verification_error: float
    environment_error: float
    rewire_failed: float
    max_turns_exceeded: float
    not_finished: float
    parse_failed: float


def conclusion_to_metrics(conclusion: SSConclusion) -> dict[str, float]:
    return cast(
        dict[str, float],
        SSMetrics(
            success=1.0 if conclusion == "success" else 0.0,
            context_length_exceeded=1.0
            if conclusion == "context_length_exceeded"
            else 0.0,
            no_code_found=1.0 if conclusion == "no_code_found" else 0.0,
            no_text_content=1.0 if conclusion == "no_text_content" else 0.0,
            code_did_not_compile=1.0 if conclusion == "code_did_not_compile" else 0.0,
            verification_failed=1.0 if conclusion == "verification_failed" else 0.0,
            verification_error=1.0 if conclusion == "verification_error" else 0.0,
            environment_error=1.0 if conclusion == "environment_error" else 0.0,
            rewire_failed=1.0 if conclusion == "rewire_failed" else 0.0,
            max_turns_exceeded=1.0 if conclusion == "max_turns_exceeded" else 0.0,
            not_finished=1.0 if conclusion == "not_finished" else 0.0,
            parse_failed=1.0 if conclusion == "parse_failed" else 0.0,
        ),
    )


@dataclass
class SSStepResult(MessageStepResult):
    conclusion: SSConclusion = field(default="not_finished")


@dataclass
class SimplifiedSolverEnv(MessageEnv):
    initial_state: SimplifiedSolverEnvState
    rust_env: Runtime
    search_model: Any
    knowledge_db: KnowledgeDB
    rust_doc_analyzer: RustDocAnalyzer
    initial_messages: list[TinkerMessage]
    reward_fn: LLMAsAJudgeSingleTurn
    mutable_state: SimplifiedSolverMutableState
    history: list[TinkerMessage] = field(default_factory=list)

    def __post_init__(self):
        search_tool = SearchTool(
            self.search_model,
            self.rust_doc_analyzer,
            self.knowledge_db,
            self.mutable_state,
        )
        self.tools = {
            search_tool.name: search_tool,
        }

    async def initial_observation(self) -> list[TinkerMessage]:
        if not self.history:
            self.history = list(self.initial_messages)
        return self.history

    async def step(self, message: TinkerMessage) -> SSStepResult:
        try:
            self.history.append(message)
            self.mutable_state.remaining_turns -= 1

            tool_calls: list[ToolCall] = list(message.get("tool_calls", []))
            if tool_calls:
                tool_results = await asyncio.gather(
                    *[handle_tool_call(self.tools, tc) for tc in tool_calls]  # type: ignore
                )
                for tool_result in tool_results:
                    for msg in tool_result.messages:
                        self.history.append(msg)

                if self.mutable_state.remaining_turns <= 0:
                    return SSStepResult(
                        reward=0.0,
                        episode_done=True,
                        next_messages=self.history,
                        conclusion="max_turns_exceeded",
                    )

                return SSStepResult(
                    reward=0.0,
                    episode_done=False,
                    next_messages=self.history,
                    conclusion="not_finished",
                )

            else:
                if isinstance(message["content"], str):
                    text_content = message["content"]
                else:
                    text_content = "".join(
                        part["text"]
                        for part in message["content"]
                        if part["type"] == "text"
                    )

                # Check for XML code test submission
                write_match = re.search(
                    r"<write_and_run>(.*?)</write_and_run>", text_content, re.DOTALL
                )
                if write_match:
                    code_to_test = write_match.group(1).strip()
                    await self.rust_env.set_content("src/main.rs", code_to_test)

                    run_ret, run_success = await self.rust_env.run_cargo()
                    content = f"<FileWritten>\\nsrc/main.rs has been updated.\\n</FileWritten>\\n<CargoRunResult>\\n{run_ret}\\n</CargoRunResult>\\n<RemainingTurns>\\n{self.mutable_state.remaining_turns}\\n</RemainingTurns>"

                    new_message = TinkerMessage(role="user", content=content)
                    self.history.append(new_message)

                    if self.mutable_state.remaining_turns <= 0:
                        return SSStepResult(
                            reward=0.0,
                            episode_done=True,
                            next_messages=self.history,
                            conclusion="max_turns_exceeded",
                        )
                    return SSStepResult(
                        reward=0.0,
                        episode_done=False,
                        next_messages=self.history,
                        conclusion="not_finished",
                    )

                # Check for Final Answer Submission
                submit_match = re.search(r"<submit>(.*?)</submit>", text_content, re.DOTALL)
                if not submit_match:
                    new_message = TinkerMessage(
                        role="user",
                        content="No code found in the message. Make sure you use `<write_and_run>...</write_and_run>` to test, or wrap the final valid runnable code in `<submit>...</submit>` to submit.",
                    )
                    self.history.append(new_message)

                    if self.mutable_state.remaining_turns <= 0:
                        return SSStepResult(
                            reward=0.0,
                            episode_done=True,
                            next_messages=self.history,
                            conclusion="max_turns_exceeded",
                        )
                    return SSStepResult(
                        reward=0.0,
                        episode_done=False,
                        next_messages=self.history,
                        conclusion="no_code_found",
                    )

                code = submit_match.group(1).strip()
                await self.rust_env.set_content("src/main.rs", code)

                reward, conclusion = await self.reward_fn(self.history)

                return SSStepResult(
                    reward=reward,
                    episode_done=True,
                    next_messages=self.history,
                    conclusion=conclusion,
                )
        except CoderMCPRuntimeError as e:
            raise CodingEnvironmentError(f"Environment error during step: {e}") from e

    async def get_state(self) -> SimplifiedSolverEnvState:
        return self.initial_state.with_messages(self.history)


def get_simplified_solver_initial_messages(
    env_state: SimplifiedSolverEnvState,
    tree_structure: str,
    tools: list[ToolSpec],
    renderer: Renderer,
) -> list[TinkerMessage]:
    PROMPT = f"""<Role>
You are a Rust engineer.
Your goal is to create a solution to achieve the user's request through the iterative process.
You create a solution using simple playground to find the correct code.
</Role>

<Context>
You are working in a cargo-initialized project.
You need to solve the problem using `{env_state.library_name}` library which is already installed as a dependency (e.g. via `cargo add`).
Note `{env_state.library_name}` is a new library you should be unfamilier with.
</Context>

<HowTo>
You have a simplified coding environment with the following tools:
- `search`: A JSON tool. Use this to search both symbol names and documentation for functionality, concepts, or how-to guides.

To TEST your code implementation without ending the task, simply output your Rust code inside plain XML tags like this (do NOT use JSON!):
<write_and_run>
fn main() {{
    // your test code here
}}
</write_and_run>
The system will automatically extract the code, write it to `src/main.rs`, run `cargo run`, and give you the output to help you iterate.

Once you confirmed that the solution works, you must output your FINAL answer as a complete, fully functioning source code enclosed in a `<submit> ... </submit>` block. You can also provide any necessary explanation.
Note: Outputting a `<submit> ... </submit>` block will IMMEDIATELY SUBMIT your answer and run the final verification, which will END the task.
</HowTo>
"""

    guidelines = """\
Verification: Once again, you MUST verify your answer. You should make your best efforts to avoid hallucination and make sure your answer is correct.
Self-contained: Note your solution has to be fully self-contained including both fully functioning source code and explanation.
Testing Code: Before submitting the final answer, use the `<write_and_run>...</write_and_run>` tags to test code and see outputs. Avoid JSON syntax errors.
Code block inclusion: Your final answer MUST include exactly one `<submit>\\n<your_code_here>\\n</submit>` block. Its content will be pasted to main.rs and executed for final verification to END the task.
Simple Search Keyword: When using the `search` tool, try asking full questions or simple keywords to best determine library logic.
Error Reflection: If `<write_and_run>` test fails, analyze the compiler error carefully. When you find your understanding about the library is wrong, use the `search` tool again.
"""
    PROMPT += f"\n<Guidelines>\n{guidelines}\n</Guidelines>"

    initial_message = f"""<Task>
{env_state.task.instruction}
</Task>

<Current Directory Structure>
{tree_structure}
</Current Directory Structure>

<Current src/main.rs>
{CARGO_INIT_MAIN_RS}
</Current src/main.rs>
"""

    if env_state.qwen_no_think:
        initial_message = "/no_think " + initial_message

    system_prompt_with_tools = _inject_tools_into_prompt(renderer, tools, PROMPT)
    return system_prompt_with_tools + [
        TinkerMessage(role="user", content=initial_message),
    ]


class SimplifiedSolverTokenEnv(EnvFromMessageEnv):
    def __init__(
        self,
        renderer: Renderer,
        message_env: SimplifiedSolverEnv,
        prethink: str | None = None,
        failed_parse_reward: float = -1.0,
        terminate_on_parse_error: bool = True,
        max_trajectory_tokens: int | None = None,
    ):
        super().__init__(
            renderer=renderer,
            message_env=message_env,
            failed_parse_reward=failed_parse_reward,
            terminate_on_parse_error=terminate_on_parse_error,
            max_trajectory_tokens=max_trajectory_tokens,
        )
        self.prethink = prethink

    async def get_state(self) -> SimplifiedSolverEnvState:
        return await self.message_env.get_state()  # type: ignore

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        messages = await self.message_env.initial_observation()
        prefill = f"<think>{self.prethink}</think>" if self.prethink else None
        return self.renderer.build_generation_prompt(
            messages, prefill=prefill
        ), self._base_stop_condition

    async def step(self, action: types.Action) -> types.StepResult:
        assistant_message, parse_success = self.renderer.parse_response(action)

        if not parse_success:
            return types.StepResult(
                reward=self.failed_parse_reward,
                episode_done=self.terminate_on_parse_error,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics={"parse_error": 1.0},
            )

        if self.prethink:
            if isinstance(assistant_message["content"], str):
                assistant_message["content"] = [
                    TextPart(type="text", text=assistant_message["content"])
                ]
            assistant_content = [
                ThinkingPart(
                    type="thinking",
                    thinking=self.prethink,
                ),
                *assistant_message["content"],
            ]
            assistant_message["content"] = assistant_content
            self.prethink = None

        msg_step = await self.message_env.step(assistant_message)
        next_observation = self.renderer.build_generation_prompt(msg_step.next_messages)
        next_stop_condition = msg_step.next_stop_condition or self._base_stop_condition

        if (
            self.max_trajectory_tokens is not None
            and next_observation.length > self.max_trajectory_tokens
        ):
            return types.StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics={**msg_step.metrics, "context_overflow": 1.0},
            )

        return types.StepResult(
            reward=msg_step.reward,
            episode_done=msg_step.episode_done,
            next_observation=next_observation,
            next_stop_condition=next_stop_condition,
            metrics=msg_step.metrics,
        )


@asynccontextmanager
async def build_simplified_solver_env(
    env_state: SimplifiedSolverEnvState,
    renderer: Renderer,
    verifier: Verifier,
    rust_doc_analyzer: RustDocAnalyzer,
    runtime: Runtime,
    search_model: Any,
    knowledge_db: KnowledgeDB,
    max_trajectory_tokens: int = 32 * 1024,
    max_turns: int = 10,
) -> AsyncGenerator[SimplifiedSolverTokenEnv, None]:
    exclude = ["target", ".git"]
    tree_structure = await runtime.tree(".", exclude=exclude, truncate=20)

    mutable_state = SimplifiedSolverMutableState(remaining_turns=max_turns)
    tools_list = [
        SearchTool(search_model, rust_doc_analyzer, knowledge_db, mutable_state),
    ]

    msg_env = SimplifiedSolverEnv(
        initial_state=env_state,
        rust_env=runtime,
        search_model=search_model,
        knowledge_db=knowledge_db,
        rust_doc_analyzer=rust_doc_analyzer,
        initial_messages=get_simplified_solver_initial_messages(
            env_state=env_state,
            tree_structure=tree_structure,
            tools=[t.to_spec() for t in tools_list],
            renderer=renderer,
        ),
        reward_fn=LLMAsAJudgeSingleTurn(
            task=env_state.task,
            rust_env=runtime,
            verifier=verifier,
            tree_structure=tree_structure,
        ),
        mutable_state=mutable_state,
    )

    yield SimplifiedSolverTokenEnv(
        renderer=renderer,
        message_env=msg_env,
        prethink=env_state.prethink,
        failed_parse_reward=-1.0,
        max_trajectory_tokens=max_trajectory_tokens,
    )
