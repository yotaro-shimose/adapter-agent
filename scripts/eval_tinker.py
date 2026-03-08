import asyncio
import logging
from pathlib import Path

import tinker
from oai_utils import gather_with_semaphore
from oai_utils.tinker import TinkerModel, setup_tinkermodel

from adapter_agent.data import TinkerMessagesDataset, TinkerMessageTrajectory
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.process.rewire_session import (
    log_trajectory,
    mean_metrics,
    solve_verify_tinker,
)
from adapter_agent.hierarchical.process.rewire_session_single_turn import (
    solve_verify_tinker_single_turn,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import (
    EnvParams,
    ModelLoadingSettings,
    TrajectorySFTDataConfig,
)
from adapter_agent.rl.env.single_turn import SingleTurnEnvState
from adapter_agent.rl.env.standard import InitEnvState
from adapter_agent.util.logger_util import setup_base_loglevel

logger = logging.getLogger(__name__)


def setup_logging():
    # Setup logging
    setup_base_loglevel()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("adapter_agent.hierarchical.agent.rewirer").setLevel(logging.INFO)
    logging.getLogger("adapter_agent.hierarchical.process.rewire_session").setLevel(
        logging.INFO
    )


def setup_agents(
    service_client: tinker.ServiceClient,
    model_loading_settings: ModelLoadingSettings,
    rust_doc_analyzer: RustDocAnalyzer,
):
    logger.info(
        f"Setting up model {model_loading_settings.model_name} from {model_loading_settings.resume_sampler_path}..."
    )
    model, tokenizer, renderer = setup_tinkermodel(
        service_client,
        model_loading_settings.model_name,
        model_loading_settings.resume_sampler_path,
    )

    # Verifier
    verifier_model = get_gemini()
    verifier = Verifier(model=verifier_model, rust_doc_analyzer=rust_doc_analyzer)

    return model, verifier


async def solve_verify_variable(
    model: TinkerModel,
    verifier: Verifier,
    task: Task,
    single_turn: bool,
):
    if single_turn:
        env_state = SingleTurnEnvState.numrs2(task=task)
        result = await solve_verify_tinker_single_turn(model, env_state, verifier)
        has_broadcast = any(
            "broadcast" in message["content"][-1]["text"]
            if isinstance(message["content"], list)
            else "broadcast" in message["content"]
            for message in result.env_state.messages
        )
        if has_broadcast:
            log_trajectory(result.env_state.messages)
    else:
        env_state = InitEnvState.numrs2(task=task, max_turns=3)
        result = await solve_verify_tinker(model, env_state, verifier)
    return result


async def main():
    setup_logging()
    num_max_train_sample = 32

    service_client = tinker.ServiceClient()
    model_loading_settings = ModelLoadingSettings(
        model_name="Qwen/Qwen3-8B",
        resume_trainer_path=None,
        # resume_sampler_path=None,
        # resume_sampler_path="tinker://f3c9ee41-7102-53c8-938c-7b68b63991ba:train:0/sampler_weights/000003",  # 3 step 1 epoch lr 1e-5
        # resume_sampler_path="tinker://6a9e68b0-fb18-519e-9bfc-87a7724a1cbe:train:0/sampler_weights/000015",  # 3 step 5 epoch lr 1e-4
        # resume_sampler_path="tinker://c88af3af-884b-586d-8a0a-00f40b9d1178:train:0/sampler_weights/000045",  # 3 step 15 epoch lr 1e-4
        # resume_sampler_path="tinker://95ec9773-02fa-518e-8e78-39e2340b83e2:train:0/sampler_weights/000135",  # Diagonified 9 step 15 epoch lr 1e-4
        # resume_sampler_path="tinker://655486c7-9ed4-5a31-8f45-f66497517844:train:0/sampler_weights/000045",  # Diagonified 3 (256 batch size) step 15 epoch lr 1e-4
        # resume_sampler_path="tinker://16ad7900-7611-561d-bc26-6dc5b7b78ceb:train:0/sampler_weights/000015",  # 1 step 15 epoch lr 1e-3
        # resume_sampler_path="tinker://d20cd30d-1222-524b-a401-f50fbb7873af:train:0/sampler_weights/000015",  # 1 step 15 epoch lr 1e-4
        # resume_sampler_path="tinker://c0e75630-04da-509e-9b57-83271bd47ff2:train:0/sampler_weights/000030",  # 3 steps 30 epochs
        # resume_sampler_path="tinker://14d462dc-5657-56d7-885e-a31aa1bf8630:train:0/sampler_weights/000030",  # 3 steps 60 epochs (30 + 30)
        # resume_sampler_path="tinker://a5118e9a-c25f-5826-ac4f-46e93d1c5f76:train:0/sampler_weights/000030",
        # resume_sampler_path="tinker://718ad4f9-5026-55ce-a015-a22943f28a17:train:0/sampler_weights/000060",
        # resume_sampler_path="tinker://718ad4f9-5026-55ce-a015-a22943f28a17:train:0/sampler_weights/000045",
        # resume_sampler_path="tinker://b2482f1f-e118-5e75-91eb-d0e812ba7e1e:train:0/sampler_weights/000045",
        resume_sampler_path="tinker://b2482f1f-e118-5e75-91eb-d0e812ba7e1e:train:0/sampler_weights/000060",
        lora_rank=32,
    )
    env_params = EnvParams.numrs2(
        max_turns=3,
        r_min=0.5,
        dataset_path=Path("data/sft/gen_20260218_182450/sft_dataset.json"),
    )
    rust_doc_analyzer = RustDocAnalyzer.from_libdir(env_params.library.local_path)
    model, verifier = setup_agents(
        service_client, model_loading_settings, rust_doc_analyzer
    )
    data_config = TrajectorySFTDataConfig(
        data_path=Path(
            "logs/Adapter_Agent/Adapter Agent_20260225_054614/sft_trajectories.json"
        ),
        train_ratio=0.9,
        test_ratio=0.1,
    )
    train, test = data_config.train_test_split()
    logger.info(f"Train size: {len(train.items[:num_max_train_sample])}")
    logger.info(f"Test size: {len(test.items)}")

    results = await gather_with_semaphore(
        [
            solve_verify_variable(
                model=model,
                verifier=verifier,
                task=item.task,
                single_turn=True,
            )
            for item in train.items[:num_max_train_sample]
        ],
        max_concurrent=32,
    )
    train_metrics = mean_metrics([r.metrics for r in results])
    train_success_ratio = sum(1 for r in results if r.is_success()) / len(results)
    train_metrics["success_ratio"] = train_success_ratio
    logger.info(f"Train metrics: {train_metrics}")
    train_trajectory_dataset = TinkerMessagesDataset(
        items=[
            TinkerMessageTrajectory(
                task=ret.env_state.task,
                messages=ret.env_state.messages,
            )
            for ret in results
        ]
    )
    train_trajectory_dataset.save(Path("train_traj.json"))

    test_results = await gather_with_semaphore(
        [
            solve_verify_variable(
                model=model,
                verifier=verifier,
                task=item.task,
                single_turn=True,
            )
            for item in test.items
        ],
        max_concurrent=32,
    )
    test_metrics = mean_metrics([r.metrics for r in test_results])
    test_success_ratio = sum(1 for r in test_results if r.is_success()) / len(
        test_results
    )
    test_metrics["success_ratio"] = test_success_ratio
    logger.info(f"Test metrics: {test_metrics}")
    test_trajectory_dataset = TinkerMessagesDataset(
        items=[
            TinkerMessageTrajectory(
                task=ret.env_state.task,
                messages=ret.env_state.messages,
            )
            for ret in test_results
        ]
    )
    test_trajectory_dataset.save(Path("test_traj.json"))


if __name__ == "__main__":
    asyncio.run(main())
