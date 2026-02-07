from oai_utils.tinker import setup_tinkermodel
import asyncio

import tinker
from oai_utils import AgentWrapper, contents2params

from adapter_agent.util.logger_util import setup_base_loglevel


async def main():
    setup_base_loglevel()
    service_client = tinker.ServiceClient()
    model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    path = (
        "tinker://29f399c0-0ce4-5ef7-9178-688b7cf4f373:train:0/sampler_weights/000010"
    )
    model, _tokenizer, _renderer = setup_tinkermodel(service_client, model_name, path)
    agent = AgentWrapper.create(
        name="tinker_chat",
        model=model,
        instructions="You are a helpful assistant.",
    )
    user_input = input("User: ")
    result = await agent.run(user_input)

    while True:
        print(result.final_output())
        user_input = input("User: ")
        new_input = result.to_input_list() + contents2params(
            role="user", items=[user_input]
        )
        if user_input == "exit":
            break
        result = await agent.run(new_input)


if __name__ == "__main__":
    asyncio.run(main())
