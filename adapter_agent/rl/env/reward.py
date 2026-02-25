import logging
from dataclasses import dataclass
from itertools import chain
from typing import TypedDict

from coder_mcp.runtime import RustCodingEnvironment
from pydantic import BaseModel, ValidationError
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.data import QA
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task

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
    rust_env: RustCodingEnvironment
    verifier: Verifier
    tree_structure: str
    task: Task

    async def __call__(
        self, history: list[TinkerMessage]
    ) -> tuple[float, dict[str, float]]:
        if len(history) == 0:
            raise ValueError("History for LLMAsAJudge cannot be empty")
        execution_output, success = await self.rust_env.run_cargo()
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
        content = await self.rust_env.view_file("src/main.rs")
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
