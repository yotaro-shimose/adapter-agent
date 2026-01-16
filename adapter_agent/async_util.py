from typing import Awaitable, Iterable
import asyncio

from tqdm.asyncio import tqdm_asyncio


async def gather_with_semaphore[T](
    awaitables: Iterable[Awaitable[T]], max_concurrent: int, progressbar: bool = False
) -> list[T]:
    """Gathers awaitables with a maximum concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def sem_awaitable(aw: Awaitable[T]) -> T:
        async with semaphore:
            return await aw

    wrapped = [sem_awaitable(aw) for aw in awaitables]

    if progressbar:
        return await tqdm_asyncio.gather(*wrapped)
    return await asyncio.gather(*wrapped)
