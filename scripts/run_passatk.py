"""run_passatk.py — Self-learnability measurement on TaskRL training sets.

For the paper's "self-learnability" axis: how many tasks in the TaskRL
training set are at all reachable by the policy at each progress point
(Base / Knowledge SFT / Knowledge RL)?

Definition:
  - n samples drawn per task, single_turn (no tools, no RAG).
  - c = number of successes among the n samples for one task.
  - **untouchable**     : c == 0           — RL gets no positive signal,
                                              cannot bootstrap on this task.
  - **self-learnable**  : c >= 1           — at least 1 success in n; GRPO
                                              has a positive sample to push
                                              the policy toward.
  - **proficient**      : c / n >= threshold  — already mostly solved (default
                                              threshold = 0.5).
  - **success rate**    : c / n            — pass@1 estimator per task.

Aggregates printed:
  - pass@1            = mean(c / n)            ("成功率")
  - pass@k            = mean(1 if c >= 1 else 0) when k == n, otherwise the
                        Codex-paper unbiased estimator ("self-learnable率")

Coverage:
  - Libraries: numrs2 (gh_archive[0:150]) and hisab (pipeline_v3_decomposed_qra_train).
  - Solvers:   Base / Knowledge SFT / Knowledge RL (canonical paper checkpoints).
  - Out of scope (intentionally): Gemini, RAG, TaskRL.

Output:
  - Stdout: per-(library, model) aggregate table + histogram of c.
  - CSV:    per-task long format
            `library,model_label,task_idx,instruction,n_samples,success_count,
             success_rate,self_learnable,proficient`

Run with:
    uv run scripts/run_passatk.py
"""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Iterable, Literal

import tinker
from agents import set_tracing_disabled
from dotenv import load_dotenv
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from prisma import Prisma
from tinker_cookbook.renderers import Message
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.library_spec import LibrarySpec
from adapter_agent.model_helper import get_gemini, get_gemini_lite
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.simple_internalizer.data_sources import load_gh_archive_suite
from adapter_agent.simple_internalizer.executor import InternalizeExecutor
from adapter_agent.simple_internalizer.rollout_engine import build_solver_system_prompt
from adapter_agent.simple_internalizer.types import SeedSuite

set_tracing_disabled(True)


# === Solver definitions =============================================
# Mirrors of constants in scripts/run_eval.py — kept in sync to keep this
# script standalone (no fragile import from run_eval).


@dataclass(frozen=True)
class TinkerSolverConfig:
    model_name: str
    sampler_path: str | None


# Qwen3-32B base — Base point for both numrs2 and hisab.
_TINKER_QWEN32B = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=None,
)

# numrs2 KSFT / KRL canonical paper checkpoints.
_TINKER_NUMRS2_KSFT = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://4c6bb913-cf76-53c6-9d04-c6e1097e0cb0:train:0/sampler_weights/init_sft"
    ),
)
_TINKER_NUMRS2_KRL = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://caaa9922-b354-5e58-80f7-54262e4ca496:train:0/sampler_weights/rl_0040"
    ),
)

# numrs2 Task RL canonical paper checkpoint (NUMRS2_TASK_RL_RECIPE rl_0040).
_TINKER_NUMRS2_TASK_RL = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://be9e6178-ae8f-570d-a987-f2dfd357e565:train:0/sampler_weights/rl_0040"
    ),
)

# hisab Task RL (decomposed-trained) canonical final checkpoint —
# HISAB_TASK_RL_FROM_DECOMPOSED_RECIPE rl_0030. The "final" hisab model.
_TINKER_HISAB_TASK_RL = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://ca15e826-2364-563b-916d-d0bb13b825db:train:0/sampler_weights/rl_0030"
    ),
)

# hisab KSFT V2 / KRL V2 canonical paper checkpoints.
_TINKER_HISAB_KSFT = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://1237cd7d-e163-5ffb-9ef9-82c98c281079:train:0/sampler_weights/init_sft"
    ),
)
_TINKER_HISAB_KRL_V2 = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://45a766f4-ef41-59d5-bfe1-6c543daf02ed:train:0/sampler_weights/rl_0054"
    ),
)

# Restudy KRL checkpoints — output of NUMRS2/HISAB_RESTUDY_KRL_RECIPE
# (resumed from Restudy KSFT, replay-mix RL for 30 iters). These are the
# "post-knowledge-replay" checkpoints right before the final Restudy TaskRL
# stage, useful for measuring how self-learnable TaskRL train tasks are
# AFTER the restudy knowledge injection but BEFORE the task-specific RL.
_TINKER_NUMRS2_RESTUDY_KRL = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://51ee406e-f614-5831-b780-d718c7698f29:train:0/sampler_weights/rl_0030"
    ),
)
_TINKER_HISAB_RESTUDY_KRL = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://1e72a865-5d46-5ffe-9499-d029e02ff6be:train:0/sampler_weights/rl_0030"
    ),
)


@dataclass(frozen=True)
class NamedSolver:
    label: str
    solver: TinkerSolverConfig


# === Train-set sources =============================================


@dataclass(frozen=True)
class GhArchiveTrain:
    """Raw gh_archive slice. Used for numrs2 (no decomposed train cache exists)."""

    task_slice: slice


@dataclass(frozen=True)
class DecomposedTrain:
    """Verified rows from a sftcacheitem cache (decompose-pipeline or any
    pipeline_v* QRA cache). When `sample_size` is set, we deterministically
    pick N tasks via `random.Random(sample_seed).sample`. Pass the same seed
    next time to hit the same 100 tasks (analysis reproducibility).
    """

    cache_id: str
    sample_size: int | None = None
    sample_seed: int = 42


TrainSource = GhArchiveTrain | DecomposedTrain


# === Library setup =================================================


@dataclass(frozen=True)
class LibrarySetup:
    name: str
    library_spec: LibrarySpec
    train_source: TrainSource
    solvers: tuple[NamedSolver, ...]


NUMRS2_SETUP = LibrarySetup(
    name="numrs2",
    library_spec=LibrarySpec.numrs2(),
    # NUMRS2_TASK_RL_RECIPE trains on gh_archive[0:150] directly. No decomposed
    # train cache for numrs2.
    train_source=GhArchiveTrain(task_slice=slice(0, 150)),
    solvers=(
        NamedSolver(label="Base", solver=_TINKER_QWEN32B),
        NamedSolver(label="Knowledge SFT", solver=_TINKER_NUMRS2_KSFT),
        NamedSolver(label="Knowledge RL", solver=_TINKER_NUMRS2_KRL),
    ),
)


HISAB_SETUP = LibrarySetup(
    name="hisab",
    library_spec=LibrarySpec.hisab(),
    # HISAB_TASK_RL_FROM_DECOMPOSED_RECIPE trains on pipeline_v3_decomposed_qra_train
    # (verified mid-level sub-tasks decomposed from gh_archive[0:150]). Pairs
    # with the decomposed eval bench as the canonical hisab paper benchmark.
    train_source=DecomposedTrain(cache_id="pipeline_v3_decomposed_qra_train"),
    solvers=(
        NamedSolver(label="Base", solver=_TINKER_QWEN32B),
        NamedSolver(label="Knowledge SFT", solver=_TINKER_HISAB_KSFT),
        NamedSolver(label="Knowledge RL", solver=_TINKER_HISAB_KRL_V2),
    ),
)


# Restudy-lineage variants: evaluate the Restudy KRL checkpoint against the
# gh_archive[0:150] TaskRL training tasks. The hisab Restudy TaskRL recipe
# trains on the raw gh_archive slice (NOT the decomposed cache), so we
# measure self-learnability against gh_archive here for both libraries.
NUMRS2_RESTUDY_KRL_SETUP = LibrarySetup(
    name="numrs2_restudy_krl",
    library_spec=LibrarySpec.numrs2(),
    train_source=GhArchiveTrain(task_slice=slice(0, 150)),
    solvers=(
        NamedSolver(label="Restudy KRL", solver=_TINKER_NUMRS2_RESTUDY_KRL),
    ),
)
HISAB_RESTUDY_KRL_SETUP = LibrarySetup(
    name="hisab_restudy_krl",
    library_spec=LibrarySpec.hisab(),
    train_source=GhArchiveTrain(task_slice=slice(0, 150)),
    solvers=(
        NamedSolver(label="Restudy KRL", solver=_TINKER_HISAB_RESTUDY_KRL),
    ),
)


# === Top-level config ==============================================


@dataclass(frozen=True)
class PassAtKConfig:
    libraries: tuple[LibrarySetup, ...] = (NUMRS2_SETUP, HISAB_SETUP)
    # n samples per task. self-learnable = c >= 1 / n.
    n_samples: int = 16
    # ks for pass@k aggregate display (k <= n_samples).
    ks: tuple[int, ...] = (1, 16)
    # c / n >= threshold → proficient (already mostly solved).
    proficient_threshold: float = 0.5
    # Cap on generated tokens per sample. Required — without it some
    # rollouts hang indefinitely (model gets stuck in repetition with no
    # natural stop). 6000 matches `simple_internalizer.types.RolloutConfig`
    # default so behavior here is consistent with the RL training runs
    # that produced these checkpoints.
    max_output_tokens: int = 6000
    # CSV destination for per-task long-format rows.
    csv_path: Path = Path("logs/passatk/self_learnability.csv")
    # Concurrency dials. Sized for "Tinker is strong + cloudrun is the
    # bottleneck", roughly matching run_continue_rl.py conventions:
    #   - Task-RL there uses num_samples=16, worker_count=50, runtime_pool=50,
    #     generation_concurrency=400.
    #   - This script does single_turn (no multi-turn agent loop), so the
    #     Tinker phase is a single batched sample_async per task — much
    #     lighter than RL rollouts. We push concurrency up and grow the
    #     runtime_pool to absorb the verification burst (concurrency × n
    #     verifications run in parallel after each Tinker call returns).
    concurrency: int = 100
    runtime_pool_size: int = 200
    verifier_model: Literal["gemini", "gemini_lite"] = "gemini_lite"


CONFIG = PassAtKConfig(
    libraries=(NUMRS2_RESTUDY_KRL_SETUP, HISAB_RESTUDY_KRL_SETUP),
    csv_path=Path("logs/passatk/restudy_krl_self_learnability.csv"),
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")


# === Suite loading =================================================


async def _load_train_suite(setup: LibrarySetup) -> SeedSuite:
    src = setup.train_source
    if isinstance(src, GhArchiveTrain):
        suite = load_gh_archive_suite(
            name=f"{setup.name}_gh_archive_train",
            task_slice=src.task_slice,
            for_rl=True,
            for_eval=False,
            csv_path=setup.library_spec.benchmark_csv,
            difficulty=setup.library_spec.default_difficulty,
        )
        logger.info(
            f"[{setup.name}] Loaded {len(suite.tasks)} gh_archive train tasks "
            f"(slice={src.task_slice})."
        )
        return suite
    # DecomposedTrain
    prisma = Prisma()
    await prisma.connect()
    try:
        rows = await prisma.sftcacheitem.find_many(
            where={"cache_id": src.cache_id, "verified": True},
            order={"id": "asc"},
        )
    finally:
        await prisma.disconnect()
    tasks = [Task(instruction=r.question) for r in rows if r.question]
    n_total = len(tasks)
    if src.sample_size is not None and n_total > src.sample_size:
        import random

        rng = random.Random(src.sample_seed)
        # Sort indices so task_idx in the CSV is monotone within the sample
        # (ordering doesn't affect the seed since we sampled the indices first).
        indices = sorted(rng.sample(range(n_total), src.sample_size))
        tasks = [tasks[i] for i in indices]
        logger.info(
            f"[{setup.name}] Sampled {len(tasks)}/{n_total} tasks from "
            f"cache_id='{src.cache_id}' (seed={src.sample_seed})."
        )
    else:
        logger.info(
            f"[{setup.name}] Loaded {len(tasks)} verified tasks "
            f"from cache_id='{src.cache_id}'."
        )
    return SeedSuite(
        name=f"sft_cache_train:{src.cache_id}",
        tasks=tasks,
        for_rl=True,
        for_eval=False,
    )


# === Per-task sampling + result ====================================


@dataclass
class TaskResult:
    library: str
    model_label: str
    task_idx: int
    instruction: str
    n_samples: int
    success_count: int


async def _sample_with_tinker(
    model: TinkerModel,
    instruction: str,
    system_prompt: str,
    num_samples: int,
    max_output_tokens: int,
) -> list[str]:
    prompt = model.renderer.build_generation_prompt(
        [
            Message(role="system", content=system_prompt),
            Message(role="user", content=instruction),
        ]
    )
    sample_results = await model.sampling_client.sample_async(
        prompt=prompt,
        num_samples=num_samples,
        sampling_params=tinker.SamplingParams(max_tokens=max_output_tokens),
    )
    answers: list[str] = []
    for seq in sample_results.sequences:
        msg, ok = model.renderer.parse_response(seq.tokens)
        text = ""
        if ok:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        text += part.get("text", "")
            elif isinstance(content, str):
                text = content
        answers.append(text)
    return answers


async def _evaluate_task(
    *,
    library: str,
    model_label: str,
    task_idx: int,
    task: Task,
    tinker_model: TinkerModel,
    system_prompt: str,
    executor: InternalizeExecutor,
    num_samples: int,
    max_output_tokens: int,
) -> TaskResult:
    answers = await _sample_with_tinker(
        tinker_model, task.instruction, system_prompt, num_samples,
        max_output_tokens=max_output_tokens,
    )
    outcomes = await asyncio.gather(
        *[
            executor.run_execution_and_verification(
                task.instruction, reasoning="", answer_text=ans
            )
            for ans in answers
        ]
    )
    return TaskResult(
        library=library,
        model_label=model_label,
        task_idx=task_idx,
        instruction=task.instruction,
        n_samples=num_samples,
        success_count=sum(1 for o in outcomes if o.success),
    )


async def _gather_with_progress(
    coros: list[Awaitable[TaskResult]],
    *,
    desc: str,
    max_concurrent: int,
) -> list[TaskResult]:
    sem = asyncio.Semaphore(max_concurrent)
    success = 0
    rollouts = 0
    self_learnable = 0
    completed = 0

    with logging_redirect_tqdm():
        with tqdm(total=len(coros), desc=desc, dynamic_ncols=True) as pbar:

            async def _worker(coro: Awaitable[TaskResult]) -> TaskResult:
                nonlocal success, rollouts, self_learnable, completed
                async with sem:
                    r = await coro
                success += r.success_count
                rollouts += r.n_samples
                if r.success_count >= 1:
                    self_learnable += 1
                completed += 1
                ratio = (success / rollouts) if rollouts else 0.0
                head = (
                    (r.instruction[:50] + "..")
                    if len(r.instruction) > 50
                    else r.instruction
                ).replace("\n", " ")
                mark = "✓" if r.success_count > 0 else "✗"
                tqdm.write(
                    f"  [{completed}/{len(coros)}] {mark} "
                    f"({r.success_count}/{r.n_samples}) {head}"
                )
                pbar.set_postfix_str(
                    f"ok={success}/{rollouts} ({ratio:.1%}) "
                    f"learnable={self_learnable}/{completed}"
                )
                pbar.update(1)
                return r

            return await asyncio.gather(*[_worker(c) for c in coros])


# === Per (library × solver) runner =================================


async def _run_one_solver(
    *,
    cfg: PassAtKConfig,
    setup: LibrarySetup,
    named: NamedSolver,
    suite: SeedSuite,
    verifier_model,
) -> list[TaskResult]:
    logger.info(
        f"[{setup.name} / {named.label}] Loading Tinker sampler "
        f"(base={named.solver.model_name}, path={named.solver.sampler_path})..."
    )
    tinker_model, _, _ = setup_tinkermodel(
        model_name=named.solver.model_name,
        path=named.solver.sampler_path,
    )

    runtime_settings = setup.library_spec.cloudrun_runtime()
    verifier = Verifier(model=verifier_model, library_name=setup.library_spec.name)
    runtime_pool = RuntimePool(runtime_settings, max_size=cfg.runtime_pool_size)
    executor = InternalizeExecutor(runtime_pool=runtime_pool, verifier=verifier)
    system_prompt = build_solver_system_prompt(setup.library_spec.name)

    try:

        async def _one(idx: int, t: Task) -> TaskResult:
            return await _evaluate_task(
                library=setup.name,
                model_label=named.label,
                task_idx=idx,
                task=t,
                tinker_model=tinker_model,
                system_prompt=system_prompt,
                executor=executor,
                num_samples=cfg.n_samples,
                max_output_tokens=cfg.max_output_tokens,
            )

        return await _gather_with_progress(
            [_one(i, t) for i, t in enumerate(suite.tasks)],
            desc=f"[{setup.name} / {named.label}]",
            max_concurrent=cfg.concurrency,
        )
    finally:
        await runtime_pool.close_all()


# === pass@k math + summary ==========================================


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Codex-paper unbiased estimator: probability that at least one of k
    samples drawn (without replacement) from n total is correct, given c
    correct out of n.
    """
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if c >= n - k + 1:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


@dataclass
class CellSummary:
    library: str
    model_label: str
    n_tasks: int
    n_samples: int
    pass_at_k: dict[int, float]
    untouchable: int  # c == 0
    self_learnable: int  # c >= 1
    proficient: int  # c / n >= threshold
    histogram: list[int]  # c -> count


def _summarize(results: list[TaskResult], *, cfg: PassAtKConfig) -> CellSummary:
    n = cfg.n_samples
    n_tasks = len(results)
    histogram = [0] * (n + 1)
    pass_sums = {k: 0.0 for k in cfg.ks}
    untouchable = 0
    self_learnable = 0
    proficient = 0
    for r in results:
        c = r.success_count
        histogram[c] += 1
        if c == 0:
            untouchable += 1
        else:
            self_learnable += 1
        if (c / n) >= cfg.proficient_threshold:
            proficient += 1
        for k in cfg.ks:
            pass_sums[k] += _pass_at_k(n, c, k)
    pass_avg = {k: (pass_sums[k] / n_tasks if n_tasks else 0.0) for k in cfg.ks}
    return CellSummary(
        library=results[0].library if results else "",
        model_label=results[0].model_label if results else "",
        n_tasks=n_tasks,
        n_samples=n,
        pass_at_k=pass_avg,
        untouchable=untouchable,
        self_learnable=self_learnable,
        proficient=proficient,
        histogram=histogram,
    )


def _print_summary(summaries: list[CellSummary], *, cfg: PassAtKConfig) -> None:
    label_w = max(
        (len(s.model_label) for s in summaries),
        default=12,
    )
    label_w = max(label_w, 14)
    lib_w = max((len(s.library) for s in summaries), default=8)
    lib_w = max(lib_w, 8)
    n_str_w = 5

    header = (
        f"{'Library':<{lib_w}} | {'Model':<{label_w}} | "
        f"{'N tasks':>{n_str_w}} | {'n':>3}"
    )
    for k in cfg.ks:
        header += f" | {f'pass@{k}':>8}"
    header += f" | {'untouch':>8} | {'self-LN':>8} | {'profic':>8}"
    line = "=" * len(header)
    print("\n" + line)
    print(header)
    print("-" * len(header))
    for s in summaries:
        row = (
            f"{s.library:<{lib_w}} | {s.model_label:<{label_w}} | "
            f"{s.n_tasks:>{n_str_w}} | {s.n_samples:>3}"
        )
        for k in cfg.ks:
            row += f" | {s.pass_at_k[k]:>7.2%}"
        row += (
            f" | {s.untouchable:>3}/{s.n_tasks:<3}"
            f" | {s.self_learnable:>3}/{s.n_tasks:<3}"
            f" | {s.proficient:>3}/{s.n_tasks:<3}"
        )
        print(row)
    print(line)
    print(
        "  pass@1 = mean(c/n)  = 成功率 (per-sample success rate, averaged over tasks)\n"
        f"  pass@{cfg.n_samples} = mean(c≥1)  = self-learnable率 (fraction with ≥1 success)\n"
        "  untouch  = c == 0 (no positive RL signal)\n"
        "  self-LN  = c >= 1 (at least one positive RL signal)\n"
        f"  profic   = c/n >= {cfg.proficient_threshold:.0%} (already mostly solved)"
    )

    # Per-cell histogram of c.
    print("\nPer-cell histogram of correct count c:")
    for s in summaries:
        max_count = max(s.histogram) if s.histogram else 0
        bar_w = 30
        print(f"\n  {s.library} / {s.model_label}  (N={s.n_tasks}, n={s.n_samples})")
        for c, count in enumerate(s.histogram):
            ratio = (count / s.n_tasks) if s.n_tasks else 0.0
            bar_len = int(round(bar_w * (count / max_count))) if max_count else 0
            tag = (
                "  ← untouchable"
                if c == 0
                else ("  ← perfectly solved" if c == s.n_samples else "")
            )
            print(
                f"    c={c:>2}/{s.n_samples}: {'█' * bar_len:<{bar_w}}  "
                f"{count:>4} ({ratio:>6.1%}){tag}"
            )


# === CSV writer =====================================================


def _write_csv(
    *,
    results: list[TaskResult],
    cfg: PassAtKConfig,
) -> None:
    cfg.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "library",
                "model_label",
                "task_idx",
                "instruction",
                "n_samples",
                "success_count",
                "success_rate",
                "self_learnable",
                "proficient",
            ]
        )
        for r in results:
            rate = r.success_count / r.n_samples if r.n_samples else 0.0
            w.writerow(
                [
                    r.library,
                    r.model_label,
                    r.task_idx,
                    r.instruction,
                    r.n_samples,
                    r.success_count,
                    f"{rate:.6f}",
                    "true" if r.success_count >= 1 else "false",
                    "true" if rate >= cfg.proficient_threshold else "false",
                ]
            )
    logger.info(f"Wrote {len(results)} rows to {cfg.csv_path}")


# === Main ===========================================================


async def main() -> None:
    load_dotenv()
    cfg = CONFIG

    # Validate ks vs n_samples up front.
    for k in cfg.ks:
        if k > cfg.n_samples:
            raise ValueError(
                f"ks contains k={k} > n_samples={cfg.n_samples}; "
                "increase n_samples or shrink ks."
            )

    verifier_model = (
        get_gemini() if cfg.verifier_model == "gemini" else get_gemini_lite()
    )

    # Pre-load suites once per library (independent of solver).
    suites: dict[str, SeedSuite] = {}
    for setup in cfg.libraries:
        suites[setup.name] = await _load_train_suite(setup)

    all_results: list[TaskResult] = []
    summaries: list[CellSummary] = []
    for setup in cfg.libraries:
        suite = suites[setup.name]
        for named in setup.solvers:
            logger.info(f"=== {setup.name} / {named.label} ===")
            results = await _run_one_solver(
                cfg=cfg,
                setup=setup,
                named=named,
                suite=suite,
                verifier_model=verifier_model,
            )
            all_results.extend(results)
            summary = _summarize(results, cfg=cfg)
            summaries.append(summary)
            logger.info(
                f"[{setup.name} / {named.label}] "
                f"untouchable={summary.untouchable}/{summary.n_tasks}  "
                f"self-learnable={summary.self_learnable}/{summary.n_tasks}  "
                f"proficient={summary.proficient}/{summary.n_tasks}  "
                + "  ".join(
                    f"pass@{k}={summary.pass_at_k[k]:.2%}" for k in cfg.ks
                )
            )
            # Persist CSV after every cell — keeps partial data if the run is
            # interrupted (long sweeps have been killed mid-run before).
            _write_csv(results=all_results, cfg=cfg)

    _print_summary(summaries, cfg=cfg)


if __name__ == "__main__":
    asyncio.run(main())
