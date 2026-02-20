import asyncio

from tinker import SamplingClient


class SharedSamplingClient:
    def __init__(self, client: SamplingClient):
        self._client = client
        self._lock = asyncio.Lock()

    async def get_client(self) -> SamplingClient:
        async with self._lock:
            return self._client

    async def update_client(self, new_client: SamplingClient) -> None:
        async with self._lock:
            self._client = new_client
