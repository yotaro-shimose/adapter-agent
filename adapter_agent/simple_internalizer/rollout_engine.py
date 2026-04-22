import asyncio
import logging
from dataclasses import dataclass

import tinker
from prisma import Prisma
from tinker_cookbook.renderers import Message, Renderer

from adapter_agent.rl.shared_sampling_client import IndexedSamplingClient

from .executor import InternalizeExecutor

logger = logging.getLogger(__name__)


def build_solver_system_prompt(library_name: str) -> str:
    return f"""<Role>
You are an expert Rust engineer.
Your task is to solve the programming challenge using the `{library_name}` library.
</Role>

<Guidelines>
1. Write high-quality, idiomatic Rust code.
2. Ensure your solution is complete and self-contained.
3. Ensure that your code produces clear output during execution so that its correctness can be easily verified from the execution results.
4. Your response should include a natural language explanation, and the complete code MUST be enclosed in a ```rust ... ``` code block.
</Guidelines>
"""


@dataclass
class RolloutOutcome:
    """1 rollout (= サンプリングされた 1 系列) の結果。"""

    tokens: list[int]
    logprobs: list[float] | None
    parsed: bool
    reasoning: str
    answer: str
    success: bool
    execution_output: str
    verification_output: str


@dataclass
class RolloutBatch:
    """1 問 (instruction) に対する N 個のロールアウト結果のまとまり。"""

    prompt: list[int]
    outcomes: list[RolloutOutcome]


class RolloutEngine:
    """指示文 (instruction) を受けて N 回サンプリングし、
    parse → executor 実行/検証 → Prisma への記録 までを一括実行する。

    集計 (成功数カウント、Trajectory 組み立て等) は呼び出し側の責務。
    エンジンは 1 rollout につき 1 つの RolloutOutcome を素直に返す。
    """

    def __init__(
        self,
        renderer: Renderer,
        executor: InternalizeExecutor,
        prisma_client: Prisma,
        system_prompt: str,
        simple_train_id: str,
    ) -> None:
        self.renderer = renderer
        self.executor = executor
        self.prisma_client = prisma_client
        self.system_prompt = system_prompt
        self.simple_train_id = simple_train_id

    async def run(
        self,
        *,
        sampling_client: IndexedSamplingClient,
        instruction: str,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
        source_id: str,
        source_title: str,
    ) -> RolloutBatch:
        prompt = self.renderer.build_generation_prompt(
            [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=instruction),
            ]
        )

        sample_results = await sampling_client.client.sample_async(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
        )

        outcomes = await asyncio.gather(
            *[
                self._process_sequence(
                    seq,
                    instruction=instruction,
                    version=sampling_client.version,
                    source_id=source_id,
                    source_title=source_title,
                )
                for seq in sample_results.sequences
            ]
        )

        return RolloutBatch(prompt=prompt, outcomes=list(outcomes))

    async def _process_sequence(
        self,
        seq,
        *,
        instruction: str,
        version: int,
        source_id: str,
        source_title: str,
    ) -> RolloutOutcome:
        tokens, logprobs = seq.tokens, seq.logprobs
        parsed = False
        reasoning = ""
        answer = ""
        success = False
        execution_output = ""
        verification_output = ""

        msg, ok = self.renderer.parse_response(tokens)
        content = msg.get("content") if ok else None

        if ok and content:
            parsed = True
            if isinstance(content, list):
                for part in content:
                    if part["type"] == "thinking":
                        reasoning += part["thinking"]
                    elif part["type"] == "text":
                        answer += part["text"]
            else:
                answer = str(content)

            outcome = await self.executor.run_execution_and_verification(
                instruction, reasoning, answer
            )
            success = outcome.success
            execution_output = outcome.execution_output
            verification_output = outcome.verification_output
        else:
            verification_output = "Parse failed."

        try:
            await self.prisma_client.simpletrajectory.create(
                data={
                    "simple_train_id": self.simple_train_id,
                    "knowledge_id": source_id,
                    "knowledge_title": source_title,
                    "step": version,
                    "question": instruction,
                    "reasoning": reasoning,
                    "answer": answer,
                    "success": success,
                    "execution_output": execution_output,
                    "verification_output": verification_output,
                }
            )
        except Exception as e:
            logger.error(f"Failed to record rollout trajectory: {e}")

        return RolloutOutcome(
            tokens=tokens,
            logprobs=logprobs,
            parsed=parsed,
            reasoning=reasoning,
            answer=answer,
            success=success,
            execution_output=execution_output,
            verification_output=verification_output,
        )
