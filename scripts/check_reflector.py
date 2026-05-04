"""Probe Reflector + WikiIntegrator behavior on a real successful trajectory.

Pulls one successful trajectory (reward > 0) from the given experiment, runs
Reflector to extract reflections, then feeds each reflection through
WikiIntegrator using the same Tinker Qwen3-32B support model that study.py
uses. Prints reflections, integration progress, and the resulting Wiki
snapshot.

The probe writes to a dedicated Wiki version (WIKI_VERSION) which is reset
to empty at the start of every run, so the real experiment's Wiki is never
touched.

Run:
    uv run scripts/check_reflector.py
"""

import asyncio
import logging

import tinker
from oai_utils.tinker import setup_tinkermodel
from prisma import Prisma

from adapter_agent.data import PydanticTinkerBaseMessage
from adapter_agent.hierarchical.agent.reflector import (
    Reflector,
    extract_submit_content,
    extract_task_instruction,
)
from adapter_agent.internalize.wiki_integrator import WikiIntegrator
from adapter_agent.library.wiki_manager import WikiManager
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.util.logger_util import setup_base_loglevel

EXPERIMENT_NAME = "study_20260503_091450"
WIKI_VERSION = f"check_reflector_{EXPERIMENT_NAME}"
LIBRARY_NAME = "numrs2"
SUPPORT_MODEL_NAME = "Qwen/Qwen3-32B"

# Trajectory selection:
# - TRAJECTORY_ID: pick that specific row, ignoring all filters.
# - Otherwise pick the first successful trajectory whose instruction
#   contains INCLUDE_KEYWORD (if set) and does NOT contain EXCLUDE_KEYWORD
#   (if set). Matching is case-insensitive.
TRAJECTORY_ID: int | None = None
INCLUDE_KEYWORD: str | None = None
EXCLUDE_KEYWORD: str | None = "layout"

logger = logging.getLogger(__name__)


async def fetch_trajectory(prisma: Prisma):
    if TRAJECTORY_ID is not None:
        return await prisma.trajectory.find_unique(where={"id": TRAJECTORY_ID})

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
        # Prisma's String filter does not allow `not: { contains: ... }`, so
        # negation must live at the top level via a NOT clause.
        where["NOT"] = {
            "instruction": {
                "contains": EXCLUDE_KEYWORD,
                "mode": "insensitive",
            }
        }
    return await prisma.trajectory.find_first(
        where=where,
        order={"id": "asc"},
    )


async def main():
    setup_base_loglevel()

    prisma = Prisma()
    await prisma.connect()
    try:
        traj_row = await fetch_trajectory(prisma)
        if traj_row is None:
            if TRAJECTORY_ID is not None:
                criteria = f"id={TRAJECTORY_ID}"
            else:
                criteria = (
                    f"include={INCLUDE_KEYWORD!r}, exclude={EXCLUDE_KEYWORD!r} "
                    f"in {EXPERIMENT_NAME!r}"
                )
            print(f"No trajectory found matching: {criteria}")
            return

        raw = traj_row.trials_json
        if not isinstance(raw, list) or not raw:
            print(f"Trajectory id={traj_row.id} has empty/invalid trials_json.")
            return

        trajectory = [
            PydanticTinkerBaseMessage.model_validate(m).to_tinker_message() for m in raw
        ]

        task_instruction = extract_task_instruction(trajectory)
        final_answer = extract_submit_content(trajectory)

        print("=== Trajectory ===")
        print(f"id:          {traj_row.id}")
        print(f"task_id:     {traj_row.task_id}")
        print(f"reward:      {traj_row.reward}")
        print(f"conclusion:  {traj_row.conclusion}")
        print(f"instruction: {traj_row.instruction or ''}")
        print(f"messages:    {len(trajectory)}")
        print()
        print("=== <submit> content ===")
        print(final_answer if final_answer is not None else "(none)")
        print()

        service_client = tinker.ServiceClient()
        support_model, _, _ = setup_tinkermodel(
            service_client=service_client,
            model_name=SUPPORT_MODEL_NAME,
        )

        reflector = Reflector(
            model=support_model,
            qwen_no_think=True,
            library_name=LIBRARY_NAME,
        )

        print(
            f"=== Running Reflector ({SUPPORT_MODEL_NAME}, library={LIBRARY_NAME}) ==="
        )
        reflections = await reflector.reflect(trajectory)
        print(f"Parsed {len(reflections)} reflection(s).\n")

        for i, r in enumerate(reflections, 1):
            print(f"--- Reflection {i} ---")
            print(f"insight:  {r.insight}")
            print(f"evidence: {r.evidence}")
            print()

        if not reflections:
            print("No reflections to integrate. Exiting.")
            return

        if task_instruction is None or final_answer is None:
            print(
                f"Cannot integrate: missing task_instruction={task_instruction is None}, "
                f"final_answer={final_answer is None}."
            )
            return

        wiki_manager = WikiManager(prisma, version=WIKI_VERSION)
        await wiki_manager.reset()
        integrator = WikiIntegrator(
            wiki_manager=wiki_manager,
            model=support_model,
            qwen_no_think=True,
        )
        runtime_settings = RuntimeSettings.docker_numrs2()

        print(
            f"=== Running WikiIntegrator (wiki version={WIKI_VERSION}, reset=True) ==="
        )
        async with runtime_settings.build_runtime() as runtime:
            for i, reflection in enumerate(reflections, 1):
                print(f"\n--- Integrating reflection {i}/{len(reflections)} ---")
                print(f"insight: {reflection.insight[:80]}")
                try:
                    await integrator.integrate(
                        reflection,
                        task_instruction=task_instruction,
                        final_answer=final_answer,
                        runtime=runtime,
                    )
                    print("  done")
                except Exception:
                    logger.exception(f"Failed to integrate reflection {i}")

        print(f"\n=== Wiki snapshot after integration (version={WIKI_VERSION}) ===")
        titles = await wiki_manager.ls(None) or []
        for t in sorted(titles):
            print(f"  - {t}")
    finally:
        await prisma.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
