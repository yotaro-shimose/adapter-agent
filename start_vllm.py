import time
from pathlib import Path

from oai_utils.vllm import VLLMSetup


async def main():
    data_parallel_size = 1
    # vllm_setup = VLLMSetup.qwen3(data_parallel_size=data_parallel_size)
    vllm_setup = VLLMSetup(
        model="Qwen/Qwen3-4B",
        lora_adapters={"numrs": Path("checkpoints/qwen3-4b-numrs-qra-2026-01-13")},
        reasoning_parser="deepseek_r1",
        data_parallel_size=data_parallel_size,
    )
    await vllm_setup.ensure_vllm_running()
    while True:
        time.sleep(100)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
