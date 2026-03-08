import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class TaskQueue[T](asyncio.Queue[T]):
    """
    アイテム管理と終了検知機能を備えたカスタムQueue
    """

    _unfinished_tasks: int

    @asynccontextmanager
    async def get_item_manager(self) -> AsyncIterator[T]:
        """
        アイテムを取得し、スコープを抜ける際に自動で task_done() を呼び出す
        コンテキストマネージャ。
        """
        item: T = await self.get()
        try:
            yield item
        finally:
            self.task_done()

    def is_done(self) -> bool:
        """
        キューが空であり、かつすべてのタスクの task_done() が完了しているかを確認する。
        ブロックせずに即座に bool 値を返す。
        """
        # empty() は未処理のアイテムがないことを確認
        # _unfinished_tasks は処理中（get済みだがtask_done未完了）の数を確認
        return self.empty() and self._unfinished_tasks == 0
