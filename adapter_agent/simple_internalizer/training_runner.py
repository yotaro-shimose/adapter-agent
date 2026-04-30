"""共通 SFT/学習プリミティブ。SimplePipeline と STaRPipeline の両方で合成的に再利用する。

責務:
  - forward/backward + optim_step のルックアヘッド・パイプライニング
  - cross_entropy SFT ステップのラッパ (任意で trigger_eval)
  - QRA → Datum 変換 (user/assistant + thinking part 形式)
  - Datum → epoch-chunked batch iterator
  - 重み同期 (training_client → sampling_client; 任意で broadcast hook)
  - チェックポイント保存
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

import tinker
from oai_utils.tinker.model_helper import get_tokenizer_renderer
from tinker import Datum
from tinker.types.loss_fn_type import LossFnType
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.renderers import Message, TextPart, ThinkingPart, TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import QRA
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient

logger = logging.getLogger(__name__)


BroadcastHook = Callable[[tinker.SamplingClient], Awaitable[None]]


class TrainingRunner:
    """Tinker 学習クライアント上の共通学習プリミティブ集。"""

    def __init__(
        self,
        training_client: tinker.TrainingClient,
        shared_sampling_client: SharedSamplingClient,
        ml_logger: MLLogger,
        log_dir: Path,
        model_name: str,
        eval_trigger: asyncio.Event,
        broadcast_hook: BroadcastHook | None = None,
    ) -> None:
        self.training_client = training_client
        self.shared_sampling_client = shared_sampling_client
        self.ml_logger = ml_logger
        self.log_dir = log_dir
        self.model_name = model_name
        self.eval_trigger = eval_trigger
        self.broadcast_hook = broadcast_hook

    async def run_training_steps(
        self,
        batch_iter: Iterable[list[Datum]],
        prefix: str,
        loss_fn: LossFnType,
        adam_params: tinker.AdamParams,
        extra_metrics: dict[str, float] | None = None,
    ) -> None:
        """Forward/backward + optim_step をバッチ間でルックアヘッド・パイプライン化。

        次バッチの update を現バッチ結果の await より前に enqueue することで、
        クライアント側ログ処理中もサーバを遊ばせない。

        `extra_metrics` を渡すと最初のバッチのメトリクスと同じ step で記録する
        (別タイミングで log_metrics を呼ぶと W&B 上で別 step になり順序が崩れるため)。
        """
        iterator = iter(batch_iter)
        try:
            first_batch = next(iterator)
        except StopIteration:
            return

        pending_extra = dict(extra_metrics) if extra_metrics else {}

        fwd_future = await self.training_client.forward_backward_async(
            data=first_batch, loss_fn=loss_fn
        )
        opt_future = await self.training_client.optim_step_async(adam_params)

        for next_batch in iterator:
            next_fwd_future = await self.training_client.forward_backward_async(
                data=next_batch, loss_fn=loss_fn
            )
            next_opt_future = await self.training_client.optim_step_async(adam_params)

            fwd_res = await fwd_future.result_async()
            await opt_future.result_async()
            metrics = {f"{prefix}/{k}": v for k, v in fwd_res.metrics.items()}
            metrics.update(pending_extra)
            pending_extra = {}
            self.ml_logger.log_metrics(metrics)

            fwd_future = next_fwd_future
            opt_future = next_opt_future

        fwd_res = await fwd_future.result_async()
        await opt_future.result_async()
        metrics = {f"{prefix}/{k}": v for k, v in fwd_res.metrics.items()}
        metrics.update(pending_extra)
        self.ml_logger.log_metrics(metrics)

    async def run_sft_steps(
        self,
        batch_iter: Iterable[list[Datum]],
        adam_params: tinker.AdamParams,
        prefix: str = "sft",
        trigger_eval: bool = True,
    ) -> None:
        """cross_entropy で SFT ステップを回し、重み同期後に任意で評価トリガを打つ。"""
        await self.run_training_steps(
            batch_iter=batch_iter,
            prefix=prefix,
            loss_fn="cross_entropy",
            adam_params=adam_params,
        )

        logger.info(
            f"Synchronizing sampling weights after {prefix}"
            + (" and triggering evaluation..." if trigger_eval else "...")
        )
        await self.sync_sampling_weights()
        if trigger_eval:
            self.eval_trigger.set()

    async def sync_sampling_weights(self) -> tinker.SamplingClient:
        new_client = (
            await self.training_client.save_weights_and_get_sampling_client_async()
        )
        self.shared_sampling_client.update_client(new_client)
        if self.broadcast_hook is not None:
            await self.broadcast_hook(new_client)
        return new_client

    async def save_checkpoint(
        self,
        name: str,
        loop_state: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        await checkpoint_utils.save_checkpoint_async(
            training_client=self.training_client,
            name=name,
            log_path=str(self.log_dir),
            loop_state=loop_state,
            kind="both",
            ttl_seconds=ttl_seconds,
        )

    def qras_to_datums(
        self,
        qras: list[QRA],
        system_prompt: str | None = None,
    ) -> list[Datum]:
        """Convert QRA samples into Datums for cross-entropy SFT.

        Standard user/assistant shape: user holds the question, assistant
        emits a thinking part (reasoning) followed by the text answer.
        Training is restricted to the last assistant message.

        If `system_prompt` is given, inject it before the user turn so the
        format matches inference-time prompting (used e.g. by STaR where
        training data was generated under the same system prompt). The
        injected system message is left non-trainable; under
        LAST_ASSISTANT_MESSAGE that's automatic, so we don't set the flag
        explicitly (which would trip an internal assert).
        """
        _, renderer = get_tokenizer_renderer(self.training_client, self.model_name)
        datums: list[Datum] = []
        for q in qras:
            conversation: list[Message] = []
            if system_prompt is not None:
                # trainable は明示しない (LAST_ASSISTANT_MESSAGE 下では tinker が
                # assistant 以外を自動で non-trainable に扱う。trainable を付けると
                # build_supervised_example で assert に引っかかる)。
                conversation.append(Message(role="system", content=system_prompt))
            conversation.extend([
                Message(role="user", content=q.question),
                Message(
                    role="assistant",
                    content=[
                        ThinkingPart(type="thinking", thinking=q.reasoning),
                        TextPart(type="text", text=f"\n\n{q.answer}"),
                    ],
                ),
            ])
            datums.append(
                conversation_to_datum(
                    conversation=conversation,
                    renderer=renderer,
                    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                    max_length=None,
                )
            )
        return datums

    @staticmethod
    def chunk_into_batches(
        datums: list[Datum], batch_size: int, num_epochs: int
    ) -> Iterable[list[Datum]]:
        """Wrap-around chunking: replicate for epochs, then pad the tail."""
        if not datums:
            return iter([])
        all_datums = datums * num_epochs
        num_batches = (len(all_datums) + batch_size - 1) // batch_size
        batches: list[list[Datum]] = []
        for i in range(num_batches):
            batch = all_datums[i * batch_size : (i + 1) * batch_size]
            if len(batch) < batch_size:
                batch.extend(all_datums[: batch_size - len(batch)])
            batches.append(batch)
        logger.info(f"Created {len(batches)} batches for {num_epochs} epochs.")
        return iter(batches)
