import time

from oai_utils.vllm import RopeScaling, VLLMSetup
import os


async def main():
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    data_parallel_size = 1
    yarn = 4
    base_max_len = 32768
    vllm_setup = VLLMSetup(
        model="Qwen/Qwen3-8B",
        reasoning_parser="deepseek_r1",
        data_parallel_size=data_parallel_size,
        quantization="fp8",
        rope_scaling=RopeScaling(
            rope_type="yarn",
            factor=yarn,
            original_max_position_embeddings=base_max_len,
        ),
    )
    await vllm_setup.ensure_vllm_running()
    while True:
        time.sleep(100)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


# vllm serve Qwen/Qwen3-8B --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
# vllm serve Qwen/Qwen3-8B --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072
