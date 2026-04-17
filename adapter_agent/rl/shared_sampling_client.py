from dataclasses import dataclass

from tinker import SamplingClient


@dataclass
class IndexedSamplingClient:
    client: SamplingClient
    version: int


class SharedSamplingClient:
    def __init__(self, client: SamplingClient):
        self._client = client
        self.version = 0

    def get_client(self) -> IndexedSamplingClient:
        return IndexedSamplingClient(client=self._client, version=self.version)

    def update_client(self, new_client: SamplingClient) -> None:
        self._client = new_client
        self.version += 1
