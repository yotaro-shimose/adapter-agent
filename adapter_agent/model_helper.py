import os
from typing import Literal

from agents.extensions.models.litellm_model import LitellmModel
from oai_utils.vllm import RopeScaling, VLLMSetup


def get_qwen8b(
    data_parallel_size: int = 1,
    yarn_factor: int = 4,
    publish: bool = False,
    quantization: Literal["fp8"] | None = None,
) -> VLLMSetup:
    base_max_len = 32768
    vllm_setup = VLLMSetup(
        model="Qwen/Qwen3-8B",
        reasoning_parser="deepseek_r1",
        data_parallel_size=data_parallel_size,
        quantization=quantization,
        rope_scaling=RopeScaling(
            rope_type="yarn",
            factor=yarn_factor,
            original_max_position_embeddings=base_max_len,
        )
        if yarn_factor != 1
        else None,
        host="0.0.0.0" if publish else None,
    )
    return vllm_setup


def get_qwen4b_vl_instruct(
    data_parallel_size: int = 1, yarn_factor: int = 1, publish: bool = False, quantization: Literal["fp8"] | None = None,
) -> VLLMSetup:
    base_max_len = 32768
    vllm_setup = VLLMSetup(
        model="Qwen/Qwen3-VL-4B-Instruct",
        data_parallel_size=data_parallel_size,
        quantization=quantization,
        rope_scaling=RopeScaling(
            rope_type="yarn",
            factor=yarn_factor,
            original_max_position_embeddings=base_max_len,
        )
        if yarn_factor != 1
        else None,
        host="0.0.0.0" if publish else None,
    )
    return vllm_setup


def get_local_qwen() -> LitellmModel:
    return get_qwen8b_fp8().as_litellm_model()


def get_gemini() -> LitellmModel:
    model_name = "gemini/gemini-3-flash-preview"
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Error: GEMINI_API_KEY not set.")
    return LitellmModel(model=model_name, api_key=api_key)
