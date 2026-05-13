"""passatk_restudy.py — self-learnability scan for restudy QRA.

Drives the same machinery as `run_passatk.py` but pointed at the
`restudy_*_qra` cache (output of `restudy_pipeline.py`)
and a single solver — the TaskRL checkpoint — instead of the
Base/KSFT/KRL sweep.

Per-task buckets (n=16, proficient_threshold=0.6 per the routing plan):
  - c == 0          → SFT  (needs to learn from scratch — no positive signal)
  - 1 <= c <= 9     → RL   (has signal, RL can amplify)
  - c >= 10  (>=60%) → Proficient  (no further training needed)

Output:
  - Stdout: aggregate counts in each bucket + per-task table.
  - CSV:    `logs/passatk/restudy_self_learnability.csv`
            with `task_idx, instruction, success_count, success_rate, bucket`
            so the routing decision per task is recoverable.

Run with:
    uv run scripts/passatk_restudy.py
"""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from agents import set_tracing_disabled
from dotenv import load_dotenv

from adapter_agent.library.library_spec import LibrarySpec
from adapter_agent.model_helper import get_gemini, get_gemini_lite

# Cross-script reuse — run_passatk.py lives in scripts/ which is on sys.path
# when invoked via `uv run scripts/foo.py`. Pin the dir for robustness.
_SCRIPTS_DIR = "/root/workspace/adapter-agent/scripts"
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from run_passatk import (  # noqa: E402
    _TINKER_HISAB_TASK_RL,
    _TINKER_NUMRS2_TASK_RL,
    DecomposedTrain,
    LibrarySetup,
    NamedSolver,
    PassAtKConfig,
    _load_train_suite,
    _run_one_solver,
    _summarize,
    _print_summary,
)

set_tracing_disabled(True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")


# === Config ==========================================================
# Single library × single solver — re-uses run_passatk's heavy lifting
# but with a focused setup: the restudy QRA cache evaluated by the
# canonical TaskRL checkpoint of the selected library.
#
# Flip LIBRARY to switch between numrs2 and hisab — pairs (QRA cache,
# solver, CSV path) move together. Keep this in sync with
# `restudy_pipeline.py`'s LIBRARY constant: passatk reads the same QRA
# cache that pipeline wrote.

LIBRARY: str = "hisab"  # "numrs2" | "hisab"

_LIBRARY_VARIANTS: dict[str, dict] = {
    "numrs2": {
        "name": "numrs2_restudy",
        "library_spec": LibrarySpec.numrs2(),
        "qra_cache_id": "restudy_numrs2_qra",
        "task_rl_solver": _TINKER_NUMRS2_TASK_RL,
        # Existing path — referenced by NUMRS2_RESTUDY_KRL_RECIPE in
        # run_continue_rl.py. Don't rename without updating that too.
        "csv_path": Path("logs/passatk/restudy_self_learnability.csv"),
    },
    "hisab": {
        "name": "hisab_restudy",
        "library_spec": LibrarySpec.hisab(),
        "qra_cache_id": "restudy_hisab_qra",
        "task_rl_solver": _TINKER_HISAB_TASK_RL,
        "csv_path": Path("logs/passatk/restudy_hisab_self_learnability.csv"),
    },
}

_VARIANT = _LIBRARY_VARIANTS[LIBRARY]
QRA_CACHE_ID = _VARIANT["qra_cache_id"]

RESTUDY_SETUP = LibrarySetup(
    name=_VARIANT["name"],
    library_spec=_VARIANT["library_spec"],
    train_source=DecomposedTrain(cache_id=QRA_CACHE_ID),
    solvers=(
        NamedSolver(label="Task RL", solver=_VARIANT["task_rl_solver"]),
    ),
)


CONFIG = PassAtKConfig(
    libraries=(RESTUDY_SETUP,),
    n_samples=16,
    ks=(1, 16),
    proficient_threshold=0.6,  # ← the routing threshold: c/n >= 0.6 = no train
    csv_path=_VARIANT["csv_path"],
    # 18 tasks × 16 samples = 288 verifications. Existing concurrency
    # defaults from run_passatk are sized for 150-task sweeps; 18 tasks
    # is much lighter so we don't need to dial up.
    concurrency=100,
    runtime_pool_size=200,
    verifier_model="gemini_lite",
)


# === Per-task routing bucket =========================================


@dataclass(frozen=True)
class TaskRouting:
    task_idx: int
    instruction: str
    success_count: int
    n_samples: int

    @property
    def success_rate(self) -> float:
        return self.success_count / self.n_samples if self.n_samples else 0.0

    @property
    def bucket(self) -> str:
        if self.success_count == 0:
            return "SFT"
        if self.success_rate >= CONFIG.proficient_threshold:
            return "Proficient"
        return "RL"


# === Main ============================================================


async def main() -> None:
    load_dotenv()
    cfg = CONFIG

    verifier_model = (
        get_gemini() if cfg.verifier_model == "gemini" else get_gemini_lite()
    )

    setup = cfg.libraries[0]
    suite = await _load_train_suite(setup)
    if not suite.tasks:
        raise SystemExit(
            f"No tasks loaded from cache_id='{QRA_CACHE_ID}' — was the QRA "
            "pipeline run? Expected verified rows in that cache."
        )

    named = setup.solvers[0]
    logger.info(f"=== {setup.name} / {named.label} ===")
    results = await _run_one_solver(
        cfg=cfg,
        setup=setup,
        named=named,
        suite=suite,
        verifier_model=verifier_model,
    )

    # Existing aggregate summary (untouchable / self-learnable / proficient).
    summary = _summarize(results, cfg=cfg)
    _print_summary([summary], cfg=cfg)

    # Routing table — what the user actually cares about: per-task bucket.
    routings = [
        TaskRouting(
            task_idx=r.task_idx,
            instruction=r.instruction,
            success_count=r.success_count,
            n_samples=r.n_samples,
        )
        for r in results
    ]
    routings.sort(key=lambda r: r.success_count)

    n_sft = sum(1 for r in routings if r.bucket == "SFT")
    n_rl = sum(1 for r in routings if r.bucket == "RL")
    n_prof = sum(1 for r in routings if r.bucket == "Proficient")

    print()
    print("=" * 80)
    print(f"ROUTING DECISION  (threshold = {cfg.proficient_threshold:.0%})")
    print("=" * 80)
    print(f"  SFT         : {n_sft}/{len(routings)}  (c == 0)")
    print(f"  RL          : {n_rl}/{len(routings)}  (1 <= c < {int(cfg.proficient_threshold * cfg.n_samples)}/{cfg.n_samples})")
    print(f"  Proficient  : {n_prof}/{len(routings)}  (c/n >= {cfg.proficient_threshold:.0%})")
    print()
    print("Per-task (sorted by success_count):")
    print(f"  {'idx':>3} | {'c/n':>5} | {'rate':>6} | {'bucket':<10} | instruction")
    print("  " + "-" * 100)
    for r in routings:
        head = r.instruction.replace("\n", " ")
        if len(head) > 80:
            head = head[:77] + "..."
        print(
            f"  {r.task_idx:>3} | {r.success_count:>2}/{r.n_samples:<2} | "
            f"{r.success_rate:>5.0%} | {r.bucket:<10} | {head}"
        )

    # CSV — per-task routing rows.
    cfg.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "task_idx", "instruction", "n_samples",
            "success_count", "success_rate", "bucket",
        ])
        for r in routings:
            w.writerow([
                r.task_idx, r.instruction, r.n_samples,
                r.success_count, f"{r.success_rate:.6f}", r.bucket,
            ])
    print(f"\nWrote routing CSV: {cfg.csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
