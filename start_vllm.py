from adapter_agent.model_helper import get_qwen30b_a3b
import os
import time


async def main():
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    data_parallel_size = 1

    vllm_setup = get_qwen30b_a3b(data_parallel_size, publish=True, quantization="fp8")

    await vllm_setup.ensure_vllm_running()
    while True:
        time.sleep(100)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
