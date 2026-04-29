import asyncio

import tinker
from dotenv import load_dotenv

TINKER_PATHS = [
    "tinker://976a7c11-7e95-596e-9230-38bff6526aa1:train:0/weights/rl_0020",
    "tinker://976a7c11-7e95-596e-9230-38bff6526aa1:train:0/sampler_weights/rl_0020",
    "tinker://1872f8c3-bd95-553c-9931-060c460cd18e:train:0/weights/rl_0050",
    "tinker://1872f8c3-bd95-553c-9931-060c460cd18e:train:0/sampler_weights/rl_0050",
]
THIRTY_DAYS_SECONDS = 30 * 24 * 3600


async def main():
    load_dotenv()
    client = tinker.ServiceClient()
    rest = client.create_rest_client()
    results = await asyncio.gather(
        *[
            rest.set_checkpoint_ttl_from_tinker_path_async(
                path, ttl_seconds=THIRTY_DAYS_SECONDS
            )
            for path in TINKER_PATHS
        ],
        return_exceptions=True,
    )
    for path, result in zip(TINKER_PATHS, results):
        if isinstance(result, Exception):
            print(f"[FAIL] {path}: {result}")
        else:
            print(f"[OK]   {path} (TTL = {THIRTY_DAYS_SECONDS}s)")


if __name__ == "__main__":
    asyncio.run(main())
