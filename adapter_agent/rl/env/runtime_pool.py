import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Awaitable, Callable

from coder_mcp.runtime import CoderMCPRuntimeError, Runtime

from adapter_agent.rl.env.runtime_settings import RuntimeSettings

logger = logging.getLogger(__name__)


class RuntimePool:
    """Provides a reusable pool of remote/local Runtimes to minimize startup latency.
    Handles graceful teardowns upon target resource crashes, and execution wrapping.
    """

    def __init__(self, settings: RuntimeSettings, max_size: int):
        self.settings = settings
        self.max_size = max_size
        self._pool: asyncio.Queue[Runtime] = asyncio.Queue()
        self._current_size = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Runtime, None]:
        """Safely acquire a runtime from the pool or create a new one."""
        runtime = await self._get_or_create_runtime()
        try:
            yield runtime
            # Normal completion: return to pool for reuse
            self._pool.put_nowait(runtime)
        except CoderMCPRuntimeError as e:
            # Drop the runtime ONLY if an uncontrolled unhandled crash occurred in the infrastructure
            logger.warning(
                f"Runtime crashed during usage execution. Discarding resource: {e}"
            )
            await self._discard_runtime(runtime)
            raise
        except BaseException:
            # For application-level errors (e.g. ValueError, Task Cancelled),
            # the remote container is still perfectly healthy. Return it to the pool!
            self._pool.put_nowait(runtime)
            raise

    async def execute_with_retry[T](
        self,
        operation: Callable[[Runtime], Awaitable[T]],
        max_attempts: int = 2,
    ) -> T:
        """Helper to run a callback against a runtime, automatically retrying on
        internal resource crashes.
        """
        last_exception: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                async with self.acquire() as runtime:
                    return await operation(runtime)
            except CoderMCPRuntimeError as e:
                logger.error(f"Runtime operation failed (attempt {attempt}): {e}")
                last_exception = e

        raise RuntimeError(
            f"All {max_attempts} attempts failed. Last error: {last_exception}"
        ) from last_exception

    async def _get_or_create_runtime(self) -> Runtime:
        """Fetch an existing idle runtime, wait for one, or spin up a new one."""
        try:
            return self._pool.get_nowait()
        except asyncio.QueueEmpty:
            pass

        create_new = False
        async with self._lock:
            if self._current_size < self.max_size:
                self._current_size += 1
                create_new = True

        if create_new:
            try:
                rt = self.settings.build_runtime()
                await rt.__aenter__()
                return rt
            except Exception:
                async with self._lock:
                    self._current_size -= 1
                raise
        else:
            return await self._pool.get()

    async def _discard_runtime(self, runtime: Runtime) -> None:
        """Gracefully handle the disconnection and destruction of a crashed runtime."""
        try:
            await runtime.__aexit__(None, None, None)
        except Exception as cleanup_err:
            logger.error(f"Failed to cleanly destroy crashed runtime: {cleanup_err}")
        finally:
            async with self._lock:
                self._current_size -= 1

    async def close_all(self) -> None:
        """Empty the pool and properly exit all instances. Call on system shutdown."""
        while not self._pool.empty():
            rt = self._pool.get_nowait()
            await self._discard_runtime(rt)
