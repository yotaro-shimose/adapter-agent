from oai_utils.tinker import setup_tinkermodel
import asyncio

import tinker
from oai_utils import AgentWrapper

from adapter_agent.util.logger_util import setup_base_loglevel


async def main():
    setup_base_loglevel()
    service_client = tinker.ServiceClient()
    model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    path = (
        "tinker://c25c1802-91fa-5e5d-badf-d1ef12a28c32:train:0/sampler_weights/000030"
    )
    model, _tokenizer, _renderer = setup_tinkermodel(service_client, model_name, path)
    agent = AgentWrapper.create(
        name="tinker_chat",
        model=model,
        instructions="You are a helpful assistant.",
    )

    # user_input = "I'm using the sum function in numrs and I noticed my 2D array became a 1D array. How can I modify my function call so that the result stays 2D with a size of 1 on the reduced axis?"
    user_input = "I am calculating the sum of a 2D array using the `numrs` library. The result is a 1D array, but I want to keep it as a 2D array with size 1 for the summed dimension. What arguments should I change to achieve this?"
    print("\n--- User Input ---")
    print(user_input)
    print("------------------")

    result = await agent.run(user_input)
    print("\n--- Agent Output ---")
    print(result.final_output())
    print("-------------------")


if __name__ == "__main__":
    asyncio.run(main())
