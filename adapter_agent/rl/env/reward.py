import logging
from dataclasses import dataclass
from itertools import chain
from typing import TypedDict

from coder_mcp.runtime import CoderMCPRuntimeError, Runtime
from pydantic import BaseModel, ValidationError
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.data import QA
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.env.conclusion import SSConclusion
from adapter_agent.util.exception import CodingEnvironmentError

logger = logging.getLogger(__name__)


class ReportSuccessArgument(BaseModel):
    answer: str


class LLMAsAJudgeMetrics(TypedDict):
    code_did_not_compile: float
    report_success_not_called: float
    report_success_called_invalid_args: float
    verifier_failed: float
    verifier_error: float


@dataclass
class LLMAsAJudge:
    rust_env: Runtime
    verifier: Verifier
    tree_structure: str
    task: Task

    async def __call__(
        self, history: list[TinkerMessage]
    ) -> tuple[float, dict[str, float]]:
        if len(history) == 0:
            raise ValueError("History for LLMAsAJudge cannot be empty")
        try:
            execution_output, success = await self.rust_env.run_cargo()
        except CoderMCPRuntimeError as e:
            raise CodingEnvironmentError(
                f"Environment error during run_cargo: {e}"
            ) from e
        if not success:
            return 0.0, dict(
                code_did_not_compile=1.0,
                report_success_not_called=0.0,
                report_success_called_invalid_args=0.0,
                verifier_failed=0.0,
                verifier_error=0.0,
            )

        report_success_calls = list(
            chain.from_iterable(
                [
                    [
                        tool_call
                        for tool_call in message["tool_calls"]
                        if tool_call.function.name == "report_success"
                    ]
                    for message in history
                    if "tool_calls" in message
                ]
            )
        )

        if len(report_success_calls) == 0:
            return 0.0, dict(
                code_did_not_compile=0.0,
                report_success_not_called=1.0,
                report_success_called_invalid_args=0.0,
                verifier_failed=0.0,
                verifier_error=0.0,
            )
        report_success_call = report_success_calls[-1]

        try:
            report_success_args = ReportSuccessArgument.model_validate_json(
                report_success_call.function.arguments
            )
        except ValidationError as e:
            logger.debug(f"Failed to parse report_success arguments: {e}")
            return 0.0, dict(
                code_did_not_compile=0.0,
                report_success_not_called=0.0,
                report_success_called_invalid_args=1.0,
                verifier_failed=0.0,
                verifier_error=0.0,
            )
        try:
            content = await self.rust_env.view_file("src/main.rs")
        except CoderMCPRuntimeError as e:
            raise CodingEnvironmentError(
                f"Environment error during view_file: {e}"
            ) from e
        try:
            verification_result = await self.verifier.verify(
                qa=QA(
                    question=self.task.instruction,
                    answer=report_success_args.answer,
                ),
                tree_structure=self.tree_structure,
                execution_output=execution_output,
                main_rs_content=content,
            )
        except Exception as e:
            logger.debug(f"Failed to verify: {e}")
            return 0.0, dict(
                code_did_not_compile=0.0,
                report_success_not_called=0.0,
                report_success_called_invalid_args=0.0,
                verifier_failed=0.0,
                verifier_error=1.0,
            )
        if verification_result.success:
            return 1.0, dict(
                code_did_not_compile=0.0,
                report_success_not_called=0.0,
                report_success_called_invalid_args=0.0,
                verifier_failed=0.0,
                verifier_error=0.0,
            )
        else:
            return 0.0, dict(
                code_did_not_compile=0.0,
                report_success_not_called=0.0,
                report_success_called_invalid_args=0.0,
                verifier_failed=1.0,
            )

    @classmethod
    def is_successful_reward(cls, reward: float) -> bool:
        return reward > 0.0


@dataclass
class LLMAsAJudgeSingleTurn:
    rust_env: Runtime
    verifier: Verifier
    tree_structure: str
    task: Task

    async def __call__(
        self, history: list[TinkerMessage]
    ) -> tuple[float, SSConclusion]:
        if len(history) == 0:
            raise ValueError("History for LLMAsAJudge cannot be empty")
        try:
            execution_output, success = await self.rust_env.run_cargo()
        except CoderMCPRuntimeError as e:
            raise CodingEnvironmentError(
                f"Environment error during run_cargo: {e}"
            ) from e
        if not success:
            return 0.0, "code_did_not_compile"
        try:
            content = await self.rust_env.view_file("src/main.rs")
        except CoderMCPRuntimeError as e:
            raise CodingEnvironmentError(
                f"Environment error during view_file: {e}"
            ) from e
        assistant_messages = [m for m in history if m["role"] == "assistant"]
        if len(assistant_messages) == 0:
            raise ValueError("No assistant messages in history")
        last_assistant_message = assistant_messages[-1]
        if isinstance(last_assistant_message["content"], str):
            answer = last_assistant_message["content"]
        else:
            answer = "\n".join(
                part["text"]
                for part in last_assistant_message["content"]
                if part["type"] == "text"
            )
        try:
            verification_result = await self.verifier.verify(
                qa=QA(
                    question=self.task.instruction,
                    answer=answer,
                ),
                tree_structure=self.tree_structure,
                execution_output=execution_output,
                main_rs_content=content,
            )
        except Exception as e:
            raise CodingEnvironmentError(f"Environment error during verify: {e}") from e

        if verification_result.success:
            return 1.0, "success"
        else:
            return 0.0, "verification_failed"

    @classmethod
    def is_successful_reward(cls, reward: float) -> bool:
        return reward > 0.0
