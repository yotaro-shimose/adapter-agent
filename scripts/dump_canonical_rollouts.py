"""Dump 20 rollouts (10 success + 10 failure per gemini_lite) from the
canonical hisab TaskRL checkpoint (ca15e826/rl_0030) against
gh_archive[150:200], so a human can grade each one and compare to
gemini_lite's verdict.

Output: a single Markdown file at /tmp/canonical_rollouts_review.md with
per-rollout sections containing the task instruction, the model's answer,
the cargo execution output, gemini_lite's verdict + reasoning, and an
empty `Human verdict:` line for the user to fill in.

Run:
    uv run scripts/dump_canonical_rollouts.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import tinker
from agents import set_tracing_disabled
from dotenv import load_dotenv
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from tinker_cookbook.renderers import Message

from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.library_spec import LibrarySpec
from adapter_agent.model_helper import get_gemini_lite
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.simple_internalizer.data_sources import load_gh_archive_suite
from adapter_agent.simple_internalizer.executor import InternalizeExecutor
from adapter_agent.simple_internalizer.rollout_engine import build_solver_system_prompt

set_tracing_disabled(True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")


SOLVER_MODEL = "Qwen/Qwen3-32B"

# Switch LIBRARY to flip the whole dump between numrs2 and hisab.
LIBRARY: str = "numrs2"  # "numrs2" | "hisab"

_LIBRARY_VARIANTS: dict[str, dict] = {
    "hisab": {
        "library_spec": LibrarySpec.hisab(),
        # Canonical hisab Task RL checkpoint
        # (HISAB_TASK_RL_FROM_DECOMPOSED_RECIPE output, pre-restudy baseline).
        "sampler_path": "tinker://ca15e826-2364-563b-916d-d0bb13b825db:train:0/sampler_weights/rl_0030",
    },
    "numrs2": {
        "library_spec": LibrarySpec.numrs2(),
        # Canonical numrs2 Task RL checkpoint
        # (NUMRS2_TASK_RL_RECIPE output, pre-restudy baseline).
        "sampler_path": "tinker://be9e6178-ae8f-570d-a987-f2dfd357e565:train:0/sampler_weights/rl_0040",
    },
}
_VARIANT = _LIBRARY_VARIANTS[LIBRARY]
SAMPLER_PATH = _VARIANT["sampler_path"]
LIBRARY_SPEC = _VARIANT["library_spec"]

# Per-rollout markdown files land in
# `human_review/canonical_<library>/rollout_NN.md`, with a single raw JSON
# summary alongside them.
OUTPUT_DIR = Path("human_review") / f"canonical_{LIBRARY}"
RAW_JSON_PATH = OUTPUT_DIR / "rollouts_raw.json"

EVAL_SLICE = slice(150, 200)

CONCURRENCY = 30
RUNTIME_POOL_SIZE = 30
PICK_PER_BUCKET = 10
PICK_SEED = 42


@dataclass
class RolloutRecord:
    task_idx: int
    instruction: str
    answer: str
    success: bool
    execution_output: str
    verification_output: str


async def _sample_one(
    model: TinkerModel,
    instruction: str,
    system_prompt: str,
) -> str:
    prompt = model.renderer.build_generation_prompt(
        [
            Message(role="system", content=system_prompt),
            Message(role="user", content=instruction),
        ]
    )
    res = await model.sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(),
    )
    msg, ok = model.renderer.parse_response(res.sequences[0].tokens)
    text = ""
    if ok:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text += part.get("text", "")
        elif isinstance(content, str):
            text = content
    return text


async def _rollout_one(
    *,
    task_idx: int,
    task: Task,
    model: TinkerModel,
    system_prompt: str,
    executor: InternalizeExecutor,
    sem: asyncio.Semaphore,
) -> RolloutRecord:
    async with sem:
        answer = await _sample_one(model, task.instruction, system_prompt)
        outcome = await executor.run_execution_and_verification(
            task.instruction, reasoning="", answer_text=answer
        )
        return RolloutRecord(
            task_idx=task_idx,
            instruction=task.instruction,
            answer=answer,
            success=outcome.success,
            execution_output=outcome.execution_output,
            verification_output=outcome.verification_output,
        )


def _format_single_review(rollout_no: int, r: RolloutRecord) -> str:
    lines: list[str] = []
    lines.append(f"# Rollout {rollout_no:02d} — task_idx={r.task_idx}")
    lines.append("")
    lines.append(f"Library: `{LIBRARY}`   Solver: `{SAMPLER_PATH}`   Verifier: `gemini-flash-lite`")
    lines.append("")
    lines.append(f"**gemini_lite verdict:** {'SUCCESS' if r.success else 'FAILURE'}")
    lines.append("")
    lines.append("**Human verdict:** (fill in OK / NG)")
    lines.append("")
    lines.append("## Task")
    lines.append("```")
    lines.append(r.instruction)
    lines.append("```")
    lines.append("")
    lines.append("## Model answer")
    lines.append(r.answer)
    lines.append("")
    lines.append("## cargo execution output")
    lines.append("```")
    lines.append(r.execution_output or "(empty)")
    lines.append("```")
    lines.append("")
    lines.append("## gemini_lite verifier reasoning")
    lines.append("```")
    lines.append(r.verification_output or "(empty)")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


async def main() -> None:
    load_dotenv()

    suite = load_gh_archive_suite(
        name="gh_archive_eval",
        task_slice=EVAL_SLICE,
        for_rl=False,
        for_eval=True,
        csv_path=LIBRARY_SPEC.benchmark_csv,
        difficulty=LIBRARY_SPEC.default_difficulty,
    )
    logger.info(f"Loaded {len(suite.tasks)} tasks from gh_archive[150:200].")

    logger.info(f"Loading Tinker sampler base={SOLVER_MODEL} path={SAMPLER_PATH}...")
    tinker_model, _, _ = setup_tinkermodel(
        model_name=SOLVER_MODEL,
        path=SAMPLER_PATH,
    )
    system_prompt = build_solver_system_prompt(LIBRARY_SPEC.name)

    runtime_pool = RuntimePool(
        settings=LIBRARY_SPEC.cloudrun_runtime(),
        max_size=RUNTIME_POOL_SIZE,
    )
    verifier = Verifier(model=get_gemini_lite(), library_name=LIBRARY_SPEC.name)
    executor = InternalizeExecutor(runtime_pool=runtime_pool, verifier=verifier)
    sem = asyncio.Semaphore(CONCURRENCY)

    try:
        records: list[RolloutRecord] = await asyncio.gather(
            *[
                _rollout_one(
                    task_idx=i,
                    task=t,
                    model=tinker_model,
                    system_prompt=system_prompt,
                    executor=executor,
                    sem=sem,
                )
                for i, t in enumerate(suite.tasks)
            ]
        )
    finally:
        await runtime_pool.close_all()

    successes = [r for r in records if r.success]
    failures = [r for r in records if not r.success]
    logger.info(
        f"Rolled out {len(records)}: success={len(successes)}, failure={len(failures)}."
    )

    rng = random.Random(PICK_SEED)
    picked_success = rng.sample(successes, min(PICK_PER_BUCKET, len(successes)))
    picked_failure = rng.sample(failures, min(PICK_PER_BUCKET, len(failures)))
    picked = picked_success + picked_failure
    rng.shuffle(picked)  # interleave so the order doesn't leak the bucket

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(picked, start=1):
        (OUTPUT_DIR / f"rollout_{i:02d}.md").write_text(
            _format_single_review(i, r), encoding="utf-8"
        )

    RAW_JSON_PATH.write_text(
        json.dumps(
            [
                {
                    "rollout_no": i,
                    "task_idx": r.task_idx,
                    "instruction": r.instruction,
                    "answer": r.answer,
                    "success_gemini_lite": r.success,
                    "execution_output": r.execution_output,
                    "verification_output": r.verification_output,
                }
                for i, r in enumerate(picked, start=1)
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Wrote {len(picked)} per-rollout files to {OUTPUT_DIR}/")
    logger.info(f"Wrote raw JSON to {RAW_JSON_PATH}")
    logger.info(
        f"Picked {len(picked_success)} success + {len(picked_failure)} failure "
        f"(seed={PICK_SEED})."
    )


if __name__ == "__main__":
    asyncio.run(main())
