import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Literal, Self


@dataclass
class TaskQueue[T]:
    """
    アイテム管理と終了検知機能を備えたカスタムQueue
    """

    queue: asyncio.Queue[T]

    @classmethod
    def create(cls, order: Literal["FIFO", "LIFO"], maxsize: int) -> Self:
        if order == "FIFO":
            return cls(queue=asyncio.Queue[T](maxsize=maxsize))
        elif order == "LIFO":
            return cls(queue=asyncio.LifoQueue[T](maxsize=maxsize))
        else:
            raise ValueError("Invalid order")

    @asynccontextmanager
    async def get_item_manager(self) -> AsyncIterator[T]:
        """
        アイテムを取得し、スコープを抜ける際に自動で task_done() を呼び出す
        コンテキストマネージャ。
        """
        item: T = await self.queue.get()
        try:
            yield item
        finally:
            self.queue.task_done()

    def is_done(self) -> bool:
        """
        キューが空であり、かつすべてのタスクの task_done() が完了しているかを確認する。
        ブロックせずに即座に bool 値を返す。
        """
        # empty() は未処理のアイテムがないことを確認
        # _unfinished_tasks は処理中（get済みだがtask_done未完了）の数を確認
        unfinished_tasks: int = self.queue._unfinished_tasks  # type: ignore
        return self.queue.empty() and unfinished_tasks == 0

    async def put(self, item: T) -> None:
        await self.queue.put(item)

    def qsize(self) -> int:
        return self.queue.qsize()

    @property
    def maxsize(self) -> int:
        return self.queue.maxsize
