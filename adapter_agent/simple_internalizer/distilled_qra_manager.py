import asyncio
import logging

from adapter_agent.data import QRA
from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.task_net import StudyTask

logger = logging.getLogger(__name__)


class DistilledQRAManager:
    """Study で蒸留された QRA の入出力と、RL all-fail 時のルーティングを担う。

    責務:
      - qra_in_queue を非ブロッキングで drain し、バッファに溜める
      - 溜まった QRA を SFT バッチ単位で払い出す
      - RL all-fail 時に replay するか study_task_queue へ依頼するか決める
    """

    def __init__(
        self,
        qra_in_queue: asyncio.Queue[tuple[str, QRA]] | None,
        study_task_queue: asyncio.Queue[StudyTask] | None,
    ) -> None:
        self.qra_in_queue = qra_in_queue
        self.study_task_queue = study_task_queue
        self._buffer: list[QRA] = []
        self._by_task: dict[str, list[QRA]] = {}
        self._study_enqueued_task_ids: set[str] = set()

    def ingest(self) -> int:
        """qra_in_queue から取れるだけ取ってバッファに追加 (非ブロッキング)。
        queue が None の場合は 0。drain した件数を返す。"""
        if self.qra_in_queue is None:
            return 0
        drained = 0
        while True:
            try:
                task_id, qra = self.qra_in_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._buffer.append(qra)
            self._by_task.setdefault(task_id, []).append(qra)
            self.qra_in_queue.task_done()
            drained += 1
        return drained

    def try_take_batch(self, size: int) -> list[QRA] | None:
        """バッファ長が size 以上ならバッチを取り出し、バッファから除く。
        足りなければ None。"""
        if len(self._buffer) < size:
            return None
        batch = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return batch

    def on_all_fail(self, task: Task, group: RLGroup) -> None:
        """RL で group.rewards が全て 0 だった場合に呼ばれる想定。
        すでに task.id 用の蒸留 QRA を持っていればバッファへ再投入、
        無ければ study_task_queue へ初回投入 (同一 task.id は一度のみ)。"""
        if not group.rewards or max(group.rewards) > 0:
            return

        cached = self._by_task.get(task.id)
        if cached:
            self._buffer.extend(cached)
            logger.info(
                f"Replayed {len(cached)} cached QRAs into SFT buffer for task {task.id}."
            )
            return

        if self.study_task_queue is None:
            return
        if task.id in self._study_enqueued_task_ids:
            return
        self._study_enqueued_task_ids.add(task.id)
        study_task = StudyTask(task=task, is_generation=False)
        try:
            self.study_task_queue.put_nowait(study_task)
            logger.info(f"Enqueued task {task.id} for study after all-fail group.")
        except asyncio.QueueFull:
            logger.warning(
                f"Study task queue full; dropping enqueue for task {task.id}."
            )
            self._study_enqueued_task_ids.discard(task.id)

    @property
    def buffered(self) -> int:
        """現在のバッファ長 (ログ用)。"""
        return len(self._buffer)
