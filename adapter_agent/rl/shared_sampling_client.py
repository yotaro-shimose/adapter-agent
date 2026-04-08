import asyncio

from dataclasses import dataclass
from tinker import SamplingClient


@dataclass
class IndexedSamplingClient:
    client: SamplingClient
    version: int


class SharedSamplingClient:
    def __init__(self, client: SamplingClient):
        self._client = client
        self._version = 0
        self._lock = asyncio.Lock()

    async def get_client(self) -> IndexedSamplingClient:
        async with self._lock:
            return IndexedSamplingClient(client=self._client, version=self._version)

    async def update_client(self, new_client: SamplingClient) -> None:
        async with self._lock:
            self._client = new_client
            self._version += 1
