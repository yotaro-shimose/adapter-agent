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
    "tinker://f5e180b4-f01a-5d07-893c-91563cf39b3f:train:0/weights/init_sft",  # hisab 32B QRA-SFT checkpoint (HISAB_SFT_FROM_PIPELINE_RECIPE)
    "tinker://f5e180b4-f01a-5d07-893c-91563cf39b3f:train:0/sampler_weights/init_sft",  # hisab 32B QRA-SFT checkpoint (HISAB_SFT_FROM_PIPELINE_RECIPE)
    "tinker://18c3f7dd-57e3-5045-baea-8b7ccd9e4051:train:0/weights/rl_0015",  # hisab 32B QRA-RL checkpoint (HISAB_KNOWLEDGE_RL_FROM_QRA_RECIPE)
    "tinker://18c3f7dd-57e3-5045-baea-8b7ccd9e4051:train:0/sampler_weights/rl_0015",  # hisab 32B QRA-RL checkpoint (HISAB_KNOWLEDGE_RL_FROM_QRA_RECIPE)
    "tinker://1237cd7d-e163-5ffb-9ef9-82c98c281079:train:0/weights/init_sft",  # hisab 32B QRA-SFT v2 checkpoint (HISAB_SFT_FROM_PIPELINE_V2_RECIPE)
    "tinker://1237cd7d-e163-5ffb-9ef9-82c98c281079:train:0/sampler_weights/init_sft",  # hisab 32B QRA-SFT v2 checkpoint (HISAB_SFT_FROM_PIPELINE_V2_RECIPE)
    "tinker://45a766f4-ef41-59d5-bfe1-6c543daf02ed:train:0/weights/rl_0040",  # hisab 32B QRA-RL v2 checkpoint (HISAB_KNOWLEDGE_RL_FROM_QRA_V2_RECIPE)
    "tinker://45a766f4-ef41-59d5-bfe1-6c543daf02ed:train:0/sampler_weights/rl_0040",  # hisab 32B QRA-RL v2 checkpoint (HISAB_KNOWLEDGE_RL_FROM_QRA_V2_RECIPE)
    "tinker://45a766f4-ef41-59d5-bfe1-6c543daf02ed:train:0/weights/rl_0054",  # hisab 32B QRA-RL v2 final checkpoint (HISAB_KNOWLEDGE_RL_FROM_QRA_V2_RECIPE, used by HISAB_TASK_RL_FROM_QRA_V2_RECIPE)
    "tinker://45a766f4-ef41-59d5-bfe1-6c543daf02ed:train:0/sampler_weights/rl_0054",  # hisab 32B QRA-RL v2 final checkpoint (HISAB_KNOWLEDGE_RL_FROM_QRA_V2_RECIPE, used by HISAB_TASK_RL_FROM_QRA_V2_RECIPE)
    "tinker://ca15e826-2364-563b-916d-d0bb13b825db:train:0/weights/rl_0030",  # hisab 32B Task RL on DECOMPOSED qra_train (HISAB_TASK_RL_FROM_DECOMPOSED_RECIPE)
    "tinker://ca15e826-2364-563b-916d-d0bb13b825db:train:0/sampler_weights/rl_0030",  # hisab 32B Task RL on DECOMPOSED qra_train (HISAB_TASK_RL_FROM_DECOMPOSED_RECIPE)
    "tinker://4c6bb913-cf76-53c6-9d04-c6e1097e0cb0:train:0/weights/init_sft",  # numrs2 32B Knowledge SFT checkpoint (NUMRS2_KNOWLEDGE_SFT_RECIPE)
    "tinker://4c6bb913-cf76-53c6-9d04-c6e1097e0cb0:train:0/sampler_weights/init_sft",  # numrs2 32B Knowledge SFT checkpoint (NUMRS2_KNOWLEDGE_SFT_RECIPE)
    "tinker://caaa9922-b354-5e58-80f7-54262e4ca496:train:0/weights/rl_0040",  # numrs2 32B Knowledge RL checkpoint (NUMRS2_KNOWLEDGE_RL_RECIPE)
    "tinker://caaa9922-b354-5e58-80f7-54262e4ca496:train:0/sampler_weights/rl_0040",  # numrs2 32B Knowledge RL checkpoint (NUMRS2_KNOWLEDGE_RL_RECIPE)
    "tinker://be9e6178-ae8f-570d-a987-f2dfd357e565:train:0/weights/rl_0040",  # numrs2 32B Task RL final checkpoint (NUMRS2_TASK_RL_RECIPE, canonical paper checkpoint — supersedes SIP-era _TINKER_SIP2/3, 10 passes × 4 iter)
    "tinker://be9e6178-ae8f-570d-a987-f2dfd357e565:train:0/sampler_weights/rl_0040",  # numrs2 32B Task RL final checkpoint (NUMRS2_TASK_RL_RECIPE, canonical paper checkpoint — supersedes SIP-era _TINKER_SIP2/3, 10 passes × 4 iter)
    "tinker://35ead364-ce45-5f98-9342-bc78aa6bf23f:train:0/weights/rl_0030",  # numrs2 32B Restudy Task RL final checkpoint (NUMRS2_RESTUDY_TASK_RL_RECIPE, peak eval 54.5% at step 25)
    "tinker://35ead364-ce45-5f98-9342-bc78aa6bf23f:train:0/sampler_weights/rl_0030",  # numrs2 32B Restudy Task RL final checkpoint (NUMRS2_RESTUDY_TASK_RL_RECIPE, peak eval 54.5% at step 25)
    "tinker://7fd46c37-1169-5d9b-a2b2-def75ca3c354:train:0/weights/init_sft",  # hisab 32B Restudy KSFT (HISAB_RESTUDY_KSFT_RECIPE v3, 1:7 on-policy replay, gh_archive_eval=8.0%)
    "tinker://7fd46c37-1169-5d9b-a2b2-def75ca3c354:train:0/sampler_weights/init_sft",  # hisab 32B Restudy KSFT (HISAB_RESTUDY_KSFT_RECIPE v3, 1:7 on-policy replay, gh_archive_eval=8.0%)
    "tinker://1e72a865-5d46-5ffe-9499-d029e02ff6be:train:0/weights/rl_0030",  # hisab 32B Restudy KRL (HISAB_RESTUDY_KRL_RECIPE, 30-iter replay-mix RL, gh_archive_eval=18.0%)
    "tinker://1e72a865-5d46-5ffe-9499-d029e02ff6be:train:0/sampler_weights/rl_0030",  # hisab 32B Restudy KRL (HISAB_RESTUDY_KRL_RECIPE, 30-iter replay-mix RL, gh_archive_eval=18.0%)
    "tinker://1dcb5948-682d-58c9-8402-4bf1780013cf:train:0/weights/rl_0030",  # hisab 32B Restudy Task RL final (HISAB_RESTUDY_TASK_RL_RECIPE, 30 iter on gh_archive[0:150], peak eval 31% at step 25)
    "tinker://1dcb5948-682d-58c9-8402-4bf1780013cf:train:0/sampler_weights/rl_0030",  # hisab 32B Restudy Task RL final (HISAB_RESTUDY_TASK_RL_RECIPE, 30 iter on gh_archive[0:150], peak eval 31% at step 25)

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
