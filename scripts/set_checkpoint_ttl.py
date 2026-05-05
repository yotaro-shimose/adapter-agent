import asyncio

import tinker
from dotenv import load_dotenv

TINKER_PATHS = [
    "tinker://976a7c11-7e95-596e-9230-38bff6526aa1:train:0/weights/rl_0020",
    "tinker://976a7c11-7e95-596e-9230-38bff6526aa1:train:0/sampler_weights/rl_0020",
    "tinker://1872f8c3-bd95-553c-9931-060c460cd18e:train:0/weights/rl_0050",
    "tinker://1872f8c3-bd95-553c-9931-060c460cd18e:train:0/sampler_weights/rl_0050",
    "tinker://01c17add-b415-5839-9ea7-2fe09d5748c7:train:0/weights/rl_0010",
    "tinker://01c17add-b415-5839-9ea7-2fe09d5748c7:train:0/sampler_weights/rl_0010",
    "tinker://c3ce4acc-191b-5e0b-8a98-27995fac5384:train:0/weights/init_sft",  # SIP 32B SFT final checkpoint
    "tinker://c3ce4acc-191b-5e0b-8a98-27995fac5384:train:0/sampler_weights/init_sft",  # SIP 32B SFT final checkpoint
    "tinker://4d530b2a-8dff-5335-a4fc-eb5e78fa797b:train:0/weights/rl_0040",  # SIP 32B Knowledge RL final checkpoint
    "tinker://4d530b2a-8dff-5335-a4fc-eb5e78fa797b:train:0/sampler_weights/rl_0040",  # SIP 32B Knowledge RL final checkpoint
    "tinker://c263af3f-acfd-5d93-a297-2dc732548b74:train:0/weights/rl_0010",  # SIP 32B Task RL final checkpoint
    "tinker://c263af3f-acfd-5d93-a297-2dc732548b74:train:0/sampler_weights/rl_0010",  # SIP 32B Task RL final checkpoint
    "tinker://25175663-6abf-5703-90ad-0a92081da02e:train:0/weights/init_sft",  # hisab 32B SFT checkpoint
    "tinker://25175663-6abf-5703-90ad-0a92081da02e:train:0/sampler_weights/init_sft",  # hisab 32B SFT checkpoint
    "tinker://a7e97833-7934-558e-842d-a29f8a2bd48f:train:0/weights/rl_0060",  # hisab 32B Knowledge RL checkpoint (1-pass run, terminated at iter 60)
    "tinker://a7e97833-7934-558e-842d-a29f8a2bd48f:train:0/sampler_weights/rl_0060",  # hisab 32B Knowledge RL checkpoint (1-pass run, terminated at iter 60)
    "tinker://b48e4aae-6e11-56ce-9078-c0cfd02db410:train:0/weights/rl_0010",  # hisab 32B Task RL mid checkpoint
    "tinker://b48e4aae-6e11-56ce-9078-c0cfd02db410:train:0/sampler_weights/rl_0010",  # hisab 32B Task RL mid checkpoint
    "tinker://b48e4aae-6e11-56ce-9078-c0cfd02db410:train:0/weights/rl_0012",  # hisab 32B Task RL final checkpoint (3 passes × 4 iter/pass)
    "tinker://b48e4aae-6e11-56ce-9078-c0cfd02db410:train:0/sampler_weights/rl_0012",  # hisab 32B Task RL final checkpoint (3 passes × 4 iter/pass)
    "tinker://9d9c8ae1-c805-5189-9ed2-ee3cc1dd6c16:train:0/weights/rl_0040",  # hisab 32B Task RL2 (10-pass) checkpoint
    "tinker://9d9c8ae1-c805-5189-9ed2-ee3cc1dd6c16:train:0/sampler_weights/rl_0040",  # hisab 32B Task RL2 (10-pass) checkpoint

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
