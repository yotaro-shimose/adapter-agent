"""STaR 用の成功 QRA FIFO バッファ。

iteration 横断で成功サンプルを累積保持し、古いものから溢れる deque ベース。
`dropped_total` で溢れた件数を観測可能にし、ml_logger に流せるようにする。
"""

from __future__ import annotations

import collections
import random
from typing import Iterable

from adapter_agent.data import QRA


class SuccessfulQRABuffer:
    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self._buffer: collections.deque[QRA] = collections.deque(maxlen=max_size)
        self._dropped_total: int = 0

    def extend(self, qras: Iterable[QRA]) -> None:
        for q in qras:
            if len(self._buffer) == self._buffer.maxlen:
                self._dropped_total += 1
            self._buffer.append(q)

    def sample(self, n: int, rng: random.Random | None = None) -> list[QRA]:
        if n <= 0 or len(self._buffer) == 0:
            return []
        k = min(n, len(self._buffer))
        pool = list(self._buffer)
        if rng is None:
            return random.sample(pool, k)
        return rng.sample(pool, k)

    def peek_all(self) -> list[QRA]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def dropped_total(self) -> int:
        return self._dropped_total
