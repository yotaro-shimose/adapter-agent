"""Probe WikiCurator (1-stage Reflector+Integrator) on multiple successful trajectories.

Pulls NUM_TRAJECTORIES successful trajectories, then runs WikiCurator
sequentially on each — letting the curator decide what to mine and how
to handle overlap with articles already written by previous runs.

The probe writes to a dedicated WIKI_VERSION which is reset to empty at
the start of every run, so the real experiment's Wiki is never touched.
The wiki accumulates across trajectories within a single run; reset
happens only once, before the first trajectory.

Run:
    uv run scripts/check_wiki_curator.py
"""

import asyncio
import logging

import tinker
from agents import add_trace_processor
from oai_utils.tinker import setup_tinkermodel
from oai_utils.tracing import AgentContentPrinter
from prisma import Prisma

from adapter_agent.data import PydanticTinkerBaseMessage
from adapter_agent.hierarchical.agent.reflector import (
    extract_submit_content,
    extract_task_instruction,
)
from adapter_agent.internalize.wiki_curator import WikiCurator
from adapter_agent.library.wiki_manager import WikiManager
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.util.logger_util import setup_base_loglevel

EXPERIMENT_NAME = "study_20260503_091450"
WIKI_VERSION = f"check_wiki_curator_{EXPERIMENT_NAME}"
LIBRARY_NAME = "numrs2"
SUPPORT_MODEL_NAME = "Qwen/Qwen3-32B"
NUM_TRAJECTORIES = 4

# Trajectory selection (mirrors check_reflector.py).
INCLUDE_KEYWORD: str | None = None
EXCLUDE_KEYWORD: str | None = "layout"

logger = logging.getLogger(__name__)


async def fetch_trajectories(prisma: Prisma, limit: int):
    where: dict = {
        "experiment_name": EXPERIMENT_NAME,
        "reward": {"gt": 0},
    }
    if INCLUDE_KEYWORD:
        where["instruction"] = {
            "contains": INCLUDE_KEYWORD,
            "mode": "insensitive",
        }
    if EXCLUDE_KEYWORD:
        where["NOT"] = {
            "instruction": {
                "contains": EXCLUDE_KEYWORD,
                "mode": "insensitive",
            }
        }
    return await prisma.trajectory.find_many(
        where=where,
        order={"id": "asc"},
        take=limit,
    )


async def print_wiki_state(wiki_manager: WikiManager, header: str):
    print(f"\n=== {header} ===")
    titles = await wiki_manager.ls(None) or []
    for t in sorted(titles):
        print(f"  - {t}")
    print()
    for t in sorted(titles):
        content = await wiki_manager.read(t)
        print(f"----- {t} -----")
        print(content if content is not None else "(empty)")
        print()


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    setup_base_loglevel()
    add_trace_processor(AgentContentPrinter())

    prisma = Prisma()
    await prisma.connect()
    try:
        traj_rows = await fetch_trajectories(prisma, NUM_TRAJECTORIES)
        if not traj_rows:
            print(
                f"No successful trajectories found "
                f"(include={INCLUDE_KEYWORD!r}, exclude={EXCLUDE_KEYWORD!r}, "
                f"in {EXPERIMENT_NAME!r})."
            )
            return

        print(f"Fetched {len(traj_rows)} trajectory(ies).")

        # Decode trajectories and extract task / final answer.
        prepared: list[tuple[int, str, str]] = []
        for row in traj_rows:
            raw = row.trials_json
            if not isinstance(raw, list) or not raw:
                print(f"  skip id={row.id}: empty trials_json")
                continue
            trajectory = [
                PydanticTinkerBaseMessage.model_validate(m).to_tinker_message()
                for m in raw
            ]
            task = extract_task_instruction(trajectory)
            answer = extract_submit_content(trajectory)
            if task is None or answer is None:
                print(
                    f"  skip id={row.id}: missing task={task is None}, "
                    f"answer={answer is None}"
                )
                continue
            prepared.append((row.id, task, answer))

        if not prepared:
            print("No usable trajectories.")
            return

        service_client = tinker.ServiceClient()
        support_model, _, _ = setup_tinkermodel(
            service_client=service_client,
            model_name=SUPPORT_MODEL_NAME,
        )

        wiki_manager = WikiManager(prisma, version=WIKI_VERSION)
        await wiki_manager.reset()
        curator = WikiCurator(
            wiki_manager=wiki_manager,
            model=support_model,
            library_name=LIBRARY_NAME,
            qwen_no_think=True,
        )
        runtime_settings = RuntimeSettings.docker_numrs2()

        print(
            f"\n=== Running WikiCurator on {len(prepared)} trajectories "
            f"(wiki={WIKI_VERSION}, reset once) ==="
        )

        async with runtime_settings.build_runtime() as runtime:
            for i, (traj_id, task, answer) in enumerate(prepared, 1):
                print(f"\n#### Trajectory {i}/{len(prepared)} (id={traj_id}) ####")
                print(f"instruction: {task}")
                print(f"<submit> first 300 chars:\n{answer[:300]}")
                if len(answer) > 300:
                    print(f"... ({len(answer) - 300} chars truncated)")
                print()

                try:
                    await curator.curate(
                        task_instruction=task,
                        final_answer=answer,
                        runtime=runtime,
                    )
                    print("  curator: done")
                except Exception:
                    logger.exception(f"WikiCurator failed for trajectory id={traj_id}")

                await print_wiki_state(
                    wiki_manager,
                    f"Wiki state after trajectory {i}/{len(prepared)} (id={traj_id})",
                )
    finally:
        await prisma.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
