"""Standalone evaluation for numrs2 coding-task benchmark.

Reuses the same eval suite and execution+verification pipeline as
`scripts/run_continue_rl_with_study.py` (via `gh_archive_eval`,
`InternalizeExecutor`, `Verifier`) but detaches it from the RL loop.

Two orthogonal axes control what is evaluated:

MODEL_BACKEND — which model runs as the solver:
  - `"tinker"`:  a Tinker sampler loaded from a checkpoint path (use this
                 for evaluating the RL/SFT-trained model).
  - `"agents"`:  any AgentsSDKModel (`from oai_utils import AgentsSDKModel`).
                 Concretely this is any `agents.models.interface.Model` —
                 e.g. `LitellmModel` / `get_gemini()` / `get_gemini_lite()`.

EVAL_STRATEGY — how the model is invoked per task:
  - `"single_turn"`:       one-shot prompt → Rust code → exec + verify (same
                           shape as RL eval).
  - `"ss_solve_verify"`:   multi-turn agent loop via `ss_solve_verify` (the
                           same solver used by `scripts/study.py`), with wiki
                           search + rustdoc lookup tools. Success is decided
                           by the session's own reward.

Edit the `CONFIG` instance near the top of this module to pick axes and
point at a checkpoint or model.
"""

import asyncio
import logging
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Iterable, Literal

import tinker
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from oai_utils import AgentsSDKModel
from oai_utils.agent import AgentWrapper
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from prisma import Prisma
from tinker_cookbook.renderers import Message
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.process.rewire import ss_solve_verify
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.wiki_manager import WikiManager
from adapter_agent.model_helper import get_gemini, get_gemini_lite
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.simple_internalizer.data_sources import load_gh_archive_suite
from adapter_agent.simple_internalizer.executor import InternalizeExecutor
from adapter_agent.simple_internalizer.rollout_engine import build_solver_system_prompt
from adapter_agent.simple_internalizer.types import SeedSuite

# --- Dataset splits (reference; pick one for `EvalConfig.task_slice`) ---
STUDY_SLICE = slice(0, 50)
TRAIN_SLICE = slice(50, 150)
EVAL_SLICE = slice(150, 200)


@dataclass(frozen=True)
class TinkerSolverConfig:
    """Tinker sampler loaded from a checkpoint path (RL/SFT-trained model)."""

    model_name: str
    sampler_path: str | None


@dataclass(frozen=True)
class AgentsSolverConfig:
    """AgentsSDKModel-backed solver (e.g. LitellmModel via gemini)."""

    model: Literal["gemini", "gemini_lite"]


SolverConfig = TinkerSolverConfig | AgentsSolverConfig


@dataclass(frozen=True)
class SsSolveVerifyConfig:
    """Settings for `EVAL_STRATEGY == "ss_solve_verify"` (mirrors study.py)."""

    rust_libdir: Path
    wiki_version: str  # ignored when `use_wiki` is False.
    max_turns: int
    qwen_no_think: bool
    runtime_mode: Literal["docker", "cloudrun"]
    concurrency: int  # typically lower than EvalConfig.concurrency — each run spins its own runtime.
    use_wiki: bool  # If False, the solver runs with an empty wiki (no Prisma needed).


@dataclass(frozen=True)
class EvalConfig:
    task_slice: slice
    rollout: int
    concurrency: int
    runtime_pool_size: int
    library_name: str
    verifier_model: Literal["gemini", "gemini_lite"]
    strategy: Literal["single_turn", "ss_solve_verify"]
    solver: SolverConfig
    ss: SsSolveVerifyConfig


# --- Shared sub-configs (re-used across named variants below) ---
_TINKER_RL0020 = TinkerSolverConfig(
    model_name="Qwen/Qwen3-8B",
    sampler_path=(
        "tinker://976a7c11-7e95-596e-9230-38bff6526aa1:train:0/sampler_weights/rl_0020"
    ),
)

# Newer training run (2026-04-30 batch).
_TINKER_SIP2 = TinkerSolverConfig(
    model_name="Qwen/Qwen3-8B",
    sampler_path=(
        "tinker://01c17add-b415-5839-9ea7-2fe09d5748c7:train:0/sampler_weights/rl_0010"
    ),
)

# Latest training run (2026-04-30 batch, c263af3f).
_TINKER_SIP3 = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://c263af3f-acfd-5d93-a297-2dc732548b74:train:0/sampler_weights/rl_0010"
    ),
)

# Qwen3-8B base model — no fine-tuned sampler, used as the baseline.
_TINKER_BASE = TinkerSolverConfig(
    model_name="Qwen/Qwen3-8B",
    sampler_path=None,
)

# Qwen3-32B base model — larger reference policy.
_TINKER_QWEN32B = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=None,
)

# Gemini via AgentsSDK / LitellmModel — used as a stronger reference solver.
_AGENTS_GEMINI = AgentsSolverConfig(model="gemini")

_SS_NUMRS_NOWIKI = SsSolveVerifyConfig(
    rust_libdir=Path("repositories/numrs"),
    wiki_version="study_20260430_024306",  # 新版Easy だけど多様なタスクリスト solved by gemini
    # wiki_version="study_20260419_041136",  # Easy のやつ
    max_turns=10,
    qwen_no_think=True,
    runtime_mode="docker",
    concurrency=50,
    use_wiki=False,
)

_SS_NUMRS_WITHWIKI = SsSolveVerifyConfig(
    rust_libdir=Path("repositories/numrs"),
    wiki_version="study_20260430_024306",  # 新版Easy だけど多様なタスクリスト solved by gemini
    max_turns=10,
    qwen_no_think=True,
    runtime_mode="docker",
    concurrency=50,
    use_wiki=True,
)


# --- Named eval variants. Pick one and assign to CONFIG below. ---

# Tinker rl_0020 × ss_solve_verify (rustdoc tools, no wiki).
SIP_RAG = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    library_name="numrs2",
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_RL0020,
    ss=_SS_NUMRS_NOWIKI,
)

# Tinker rl_0020 × single_turn (no RAG, no tools — one-shot prompt).
SIP_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    library_name="numrs2",
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_RL0020,
    ss=_SS_NUMRS_NOWIKI,  # unused for single_turn
)

# Qwen3-8B base (no checkpoint) × ss_solve_verify — RAG baseline.
BASE_RAG = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    library_name="numrs2",
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_BASE,
    ss=_SS_NUMRS_NOWIKI,
)

# Qwen3-8B base × single_turn — bare baseline (no checkpoint, no RAG).
BASE_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    library_name="numrs2",
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_BASE,
    ss=_SS_NUMRS_NOWIKI,  # unused for single_turn
)

# Qwen3-8B base × ss_solve_verify with wiki — full RAG baseline.
BASE_WIKI = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    library_name="numrs2",
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_BASE,
    ss=_SS_NUMRS_WITHWIKI,
)

# Gemini × single_turn — one-shot prompt, no tools.
GEMINI_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    library_name="numrs2",
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_AGENTS_GEMINI,
    ss=_SS_NUMRS_NOWIKI,  # unused for single_turn
)

# Gemini × ss_solve_verify (rustdoc tools, no wiki).
GEMINI_RAG = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    library_name="numrs2",
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_AGENTS_GEMINI,
    ss=_SS_NUMRS_NOWIKI,
)

# Gemini × ss_solve_verify with wiki — full RAG with Gemini policy.
GEMINI_WIKI = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    library_name="numrs2",
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_AGENTS_GEMINI,
    ss=_SS_NUMRS_WITHWIKI,
)

# Tinker SIP2 (rl_0010, newer run) × single_turn.
SIP2_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    library_name="numrs2",
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_SIP2,
    ss=_SS_NUMRS_NOWIKI,  # unused for single_turn
)

# Tinker SIP2 × ss_solve_verify (rustdoc tools, no wiki).
SIP2_RAG = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    library_name="numrs2",
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_SIP2,
    ss=_SS_NUMRS_NOWIKI,
)

# Tinker SIP3 (rl_0010, latest run c263af3f) × single_turn.
SIP3_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    library_name="numrs2",
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_SIP3,
    ss=_SS_NUMRS_NOWIKI,  # unused for single_turn
)

# Qwen3-32B base × ss_solve_verify (rustdoc tools, no wiki) — larger-model reference.
QWEN32B_RAG = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    library_name="numrs2",
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_QWEN32B,
    ss=_SS_NUMRS_NOWIKI,
)

# Qwen3-32B base × ss_solve_verify with wiki — larger-model + full RAG.
QWEN32B_WIKI = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    library_name="numrs2",
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_QWEN32B,
    ss=_SS_NUMRS_WITHWIKI,
)


CONFIG = SIP3_SINGLE


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")


@dataclass
class TaskEvalResult:
    suite: str
    instruction: str
    success_count: int
    total_count: int
    response_lengths: list[int] = field(default_factory=list)
    response_token_lengths: list[int] | None = None


def _response_length_from_trials(trials: Iterable[Message]) -> int:
    total = 0
    for msg in trials:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    total += len(part.get("text", ""))
                elif part.get("type") == "thinking":
                    total += len(part.get("thinking", ""))
    return total


def _length_stats(values: Iterable[int]) -> tuple[float, float, int] | None:
    vs = list(values)
    if not vs:
        return None
    mean = sum(vs) / len(vs)
    var = statistics.pvariance(vs) if len(vs) > 1 else 0.0
    return mean, var, max(vs)


async def _gather_eval_with_progress(
    coros: list[Awaitable["TaskEvalResult"]],
    *,
    desc: str,
    max_concurrent: int,
) -> list["TaskEvalResult"]:
    """Run eval coroutines with a semaphore + live progress bar.

    The bar shows running ok/total rollouts and ratio so the wait isn't blind.
    """
    sem = asyncio.Semaphore(max_concurrent)
    success = 0
    rollouts = 0
    completed_tasks = 0

    with logging_redirect_tqdm():
        with tqdm(total=len(coros), desc=desc, dynamic_ncols=True) as pbar:

            async def _worker(coro: Awaitable[TaskEvalResult]) -> TaskEvalResult:
                nonlocal success, rollouts, completed_tasks
                async with sem:
                    r = await coro
                success += r.success_count
                rollouts += r.total_count
                completed_tasks += 1
                ratio = (success / rollouts) if rollouts else 0.0
                head = (
                    (r.instruction[:50] + "..")
                    if len(r.instruction) > 50
                    else r.instruction
                ).replace("\n", " ")
                mark = "✓" if r.success_count > 0 else "✗"
                tqdm.write(
                    f"  [{completed_tasks}/{len(coros)}] {mark} "
                    f"({r.success_count}/{r.total_count}) {head}"
                )
                pbar.set_postfix_str(f"ok={success}/{rollouts} ({ratio:.1%})")
                pbar.update(1)
                return r

            return await asyncio.gather(*[_worker(c) for c in coros])


async def _sample_with_tinker(
    model: TinkerModel,
    instruction: str,
    system_prompt: str,
    num_samples: int,
) -> tuple[list[str], list[int]]:
    prompt = model.renderer.build_generation_prompt(
        [
            Message(role="system", content=system_prompt),
            Message(role="user", content=instruction),
        ]
    )
    sample_results = await model.sampling_client.sample_async(
        prompt=prompt,
        num_samples=num_samples,
        sampling_params=tinker.SamplingParams(),
    )
    answers: list[str] = []
    token_lengths: list[int] = []
    for seq in sample_results.sequences:
        token_lengths.append(len(seq.tokens))
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
    return answers, token_lengths


async def _sample_with_agents(
    model: AgentsSDKModel,
    instruction: str,
    system_prompt: str,
    num_samples: int,
) -> list[str]:
    async def _one() -> str:
        agent = AgentWrapper.create(
            name="EvalSolver",
            instructions=system_prompt,
            model=model,
        )
        try:
            result = await agent.run(instruction, max_turns=2)
        except Exception as e:
            logger.warning(f"Agents SDK run failed: {e}")
            return ""
        final = result.final_output()
        return final if isinstance(final, str) else (str(final) if final is not None else "")

    return await asyncio.gather(*[_one() for _ in range(num_samples)])


async def _evaluate_task_single_turn(
    task: Task,
    suite_name: str,
    *,
    backend: Literal["tinker", "agents"],
    tinker_model: TinkerModel | None,
    agents_model: AgentsSDKModel | None,
    system_prompt: str,
    executor: InternalizeExecutor,
    num_samples: int,
) -> TaskEvalResult:
    instruction = task.instruction
    token_lengths: list[int] | None
    if backend == "tinker":
        assert tinker_model is not None
        answers, token_lengths = await _sample_with_tinker(
            tinker_model, instruction, system_prompt, num_samples
        )
    else:
        assert agents_model is not None
        answers = await _sample_with_agents(
            agents_model, instruction, system_prompt, num_samples
        )
        token_lengths = None

    outcomes = await asyncio.gather(
        *[
            executor.run_execution_and_verification(
                instruction, reasoning="", answer_text=ans
            )
            for ans in answers
        ]
    )
    success_count = sum(1 for o in outcomes if o.success)
    return TaskEvalResult(
        suite=suite_name,
        instruction=instruction,
        success_count=success_count,
        total_count=num_samples,
        response_lengths=[len(a) for a in answers],
        response_token_lengths=token_lengths,
    )


async def _evaluate_task_ss_solve_verify(
    task: Task,
    suite_name: str,
    *,
    solver_model: TinkerModel | LitellmModel,
    verifier_model: AgentsSDKModel,
    rust_doc_analyzer: AsyncRustDocAnalyzer,
    wiki_manager: WikiManager,
    runtime_settings: RuntimeSettings,
    max_turns: int,
    qwen_no_think: bool,
    num_samples: int,
) -> TaskEvalResult:
    is_tinker = isinstance(solver_model, TinkerModel)

    async def _one() -> tuple[bool, int, int]:
        try:
            ret = await ss_solve_verify(
                solver_model=solver_model,
                verifier_model=verifier_model,
                rust_doc_analyzer=rust_doc_analyzer,
                task=task,
                max_turns=max_turns,
                runtime_settings=runtime_settings,
                wiki_manager=wiki_manager,
                qwen_no_think=qwen_no_think,
            )
        except Exception as e:
            logger.warning(f"ss_solve_verify failed: {e}")
            return False, 0, 0
        trials = getattr(ret, "trials", None) or []
        char_len = _response_length_from_trials(trials)
        traj = getattr(ret, "trajectory", None)
        token_len = (
            sum(len(tr.ac.tokens) for tr in traj.transitions) if traj is not None else 0
        )
        return ret.is_successful(), char_len, token_len

    outcomes = await asyncio.gather(*[_one() for _ in range(num_samples)])
    return TaskEvalResult(
        suite=suite_name,
        instruction=task.instruction,
        success_count=sum(1 for ok, _, _ in outcomes if ok),
        total_count=num_samples,
        response_lengths=[cl for _, cl, _ in outcomes],
        response_token_lengths=(
            [tl for _, _, tl in outcomes] if is_tinker else None
        ),
    )


async def _run_single_turn(
    flattened: list[tuple[str, Task]],
    *,
    cfg: EvalConfig,
    tinker_model: TinkerModel | None,
    agents_model: AgentsSDKModel | None,
    verifier_model: AgentsSDKModel,
) -> list[TaskEvalResult]:
    runtime_settings = RuntimeSettings.cloudrun_numrs2()
    verifier = Verifier(model=verifier_model, library_name=cfg.library_name)
    runtime_pool = RuntimePool(runtime_settings, max_size=cfg.runtime_pool_size)
    executor = InternalizeExecutor(runtime_pool=runtime_pool, verifier=verifier)
    system_prompt = build_solver_system_prompt(cfg.library_name)
    backend: Literal["tinker", "agents"] = (
        "tinker" if isinstance(cfg.solver, TinkerSolverConfig) else "agents"
    )

    try:
        async def _one(suite_name: str, task: Task) -> TaskEvalResult:
            return await _evaluate_task_single_turn(
                task,
                suite_name,
                backend=backend,
                tinker_model=tinker_model,
                agents_model=agents_model,
                system_prompt=system_prompt,
                executor=executor,
                num_samples=cfg.rollout,
            )

        return await _gather_eval_with_progress(
            [_one(sn, t) for sn, t in flattened],
            desc="single_turn eval",
            max_concurrent=cfg.concurrency,
        )
    finally:
        await runtime_pool.close_all()


class _NullWikiManager:
    """Duck-typed WikiManager with no Prisma backend.

    Enables running `ss_solve_verify` without a wiki: `ls()` yields no
    titles, `read()` returns None for every title, and MOC.md comes back
    empty (which makes `build_simplified_solver_msg_env` skip the
    <MapOfContent> prompt block entirely).
    """

    def __init__(self, version: str = "null") -> None:
        self.version = version

    async def ls(self, path: str | None = None) -> list[str]:
        return []

    async def read(self, title: str) -> str | None:
        return None


async def _run_ss_solve_verify(
    flattened: list[tuple[str, Task]],
    *,
    cfg: EvalConfig,
    solver_model: TinkerModel | LitellmModel,
    verifier_model: AgentsSDKModel,
) -> list[TaskEvalResult]:
    ss = cfg.ss
    runtime_settings = (
        RuntimeSettings.docker_numrs2()
        if ss.runtime_mode == "docker"
        else RuntimeSettings.cloudrun_numrs2()
    )
    wiki_label = ss.wiki_version if ss.use_wiki else "<disabled>"
    logger.info(
        f"Building ss_solve_verify resources "
        f"(libdir={ss.rust_libdir}, wiki={wiki_label}, runtime={ss.runtime_mode})..."
    )
    rust_doc_analyzer = await AsyncRustDocAnalyzer.create_from_libdir(ss.rust_libdir)

    prisma: Prisma | None = None
    wiki_manager: WikiManager | _NullWikiManager
    if ss.use_wiki:
        prisma = Prisma()
        await prisma.connect()
        wiki_manager = WikiManager(prisma, version=ss.wiki_version)
    else:
        wiki_manager = _NullWikiManager()

    try:
        async with rust_doc_analyzer:
            async def _one(suite_name: str, task: Task) -> TaskEvalResult:
                return await _evaluate_task_ss_solve_verify(
                    task,
                    suite_name,
                    solver_model=solver_model,
                    verifier_model=verifier_model,
                    rust_doc_analyzer=rust_doc_analyzer,
                    wiki_manager=wiki_manager,
                    runtime_settings=runtime_settings,
                    max_turns=ss.max_turns,
                    qwen_no_think=ss.qwen_no_think,
                    num_samples=cfg.rollout,
                )

            return await _gather_eval_with_progress(
                [_one(sn, t) for sn, t in flattened],
                desc="ss_solve_verify eval",
                max_concurrent=ss.concurrency,
            )
    finally:
        if prisma is not None:
            await prisma.disconnect()


async def main() -> None:
    load_dotenv()
    cfg = CONFIG

    eval_suite = load_gh_archive_suite(
        name="gh_archive_eval",
        task_slice=cfg.task_slice,
        for_rl=False,
        for_eval=True,
    )
    suites: list[SeedSuite] = [eval_suite]

    verifier_model = (
        get_gemini() if cfg.verifier_model == "gemini" else get_gemini_lite()
    )

    tinker_model: TinkerModel | None = None
    agents_model: AgentsSDKModel | None = None
    if isinstance(cfg.solver, TinkerSolverConfig):
        logger.info(
            f"Loading Tinker sampler (base={cfg.solver.model_name}) "
            f"from {cfg.solver.sampler_path}..."
        )
        tinker_model, _, _ = setup_tinkermodel(
            model_name=cfg.solver.model_name,
            path=cfg.solver.sampler_path,
        )
        backend: Literal["tinker", "agents"] = "tinker"
    else:
        logger.info(f"Using AgentsSDKModel policy: {cfg.solver.model}")
        agents_model = (
            get_gemini() if cfg.solver.model == "gemini" else get_gemini_lite()
        )
        backend = "agents"

    flattened: list[tuple[str, Task]] = [
        (s.name, t) for s in suites for t in s.tasks
    ]
    slice_step = (
        f":{cfg.task_slice.step}" if cfg.task_slice.step is not None else ""
    )
    slice_repr = f"[{cfg.task_slice.start}:{cfg.task_slice.stop}{slice_step}]"
    logger.info(
        f"Evaluating {len(flattened)} tasks × {cfg.rollout} rollouts "
        f"(backend={backend}, strategy={cfg.strategy}, slice={slice_repr})..."
    )

    if cfg.strategy == "single_turn":
        results = await _run_single_turn(
            flattened=flattened,
            cfg=cfg,
            tinker_model=tinker_model,
            agents_model=agents_model,
            verifier_model=verifier_model,
        )
    elif cfg.strategy == "ss_solve_verify":
        solver_model: TinkerModel | LitellmModel
        if isinstance(cfg.solver, TinkerSolverConfig):
            assert tinker_model is not None
            solver_model = tinker_model
        else:
            assert agents_model is not None
            if not isinstance(agents_model, LitellmModel):
                raise ValueError(
                    "ss_solve_verify requires the agents solver to be a LitellmModel "
                    f"(got {type(agents_model).__name__}). Use get_gemini()/get_gemini_lite()."
                )
            solver_model = agents_model
        results = await _run_ss_solve_verify(
            flattened=flattened,
            cfg=cfg,
            solver_model=solver_model,
            verifier_model=verifier_model,
        )
    else:
        raise ValueError(f"Unknown strategy: {cfg.strategy}")

    suite_stats: dict[str, dict[str, int]] = {s.name: {"success": 0, "rollouts": 0} for s in suites}
    suite_lengths: dict[str, list[int]] = {s.name: [] for s in suites}
    suite_token_lengths: dict[str, list[int]] = {s.name: [] for s in suites}
    has_token_lengths = False
    for r in results:
        suite_stats[r.suite]["success"] += r.success_count
        suite_stats[r.suite]["rollouts"] += r.total_count
        suite_lengths[r.suite].extend(r.response_lengths)
        if r.response_token_lengths is not None:
            suite_token_lengths[r.suite].extend(r.response_token_lengths)
            has_token_lengths = True

    print("\n" + "=" * 82)
    print(f"{'Suite':<40} | {'Success':>8} | {'Total':>8} | {'Ratio':>8}")
    print("-" * 82)
    for sn, stats in suite_stats.items():
        ratio = stats["success"] / stats["rollouts"] if stats["rollouts"] else 0.0
        print(
            f"{sn:<40} | {stats['success']:>8} | {stats['rollouts']:>8} | {ratio:>8.2%}"
        )
    print("=" * 82)

    print("\nResponse length (characters) per suite:")
    print(f"{'Suite':<40} | {'N':>6} | {'Mean':>10} | {'Variance':>12} | {'Max':>8}")
    print("-" * 86)
    for sn, lengths in suite_lengths.items():
        stats = _length_stats(lengths)
        if stats is None:
            print(f"{sn:<40} | {0:>6} | {'-':>10} | {'-':>12} | {'-':>8}")
            continue
        mean, var, mx = stats
        print(f"{sn:<40} | {len(lengths):>6} | {mean:>10.1f} | {var:>12.1f} | {mx:>8}")
    all_lengths = [ln for lns in suite_lengths.values() for ln in lns]
    overall = _length_stats(all_lengths)
    if overall is not None:
        mean, var, mx = overall
        print("-" * 86)
        print(
            f"{'ALL':<40} | {len(all_lengths):>6} | {mean:>10.1f} | {var:>12.1f} | {mx:>8}"
        )

    if has_token_lengths:
        print("\nResponse length (tokens) per suite:")
        print(f"{'Suite':<40} | {'N':>6} | {'Mean':>10} | {'Variance':>12} | {'Max':>8}")
        print("-" * 86)
        for sn, lengths in suite_token_lengths.items():
            stats = _length_stats(lengths)
            if stats is None:
                print(f"{sn:<40} | {0:>6} | {'-':>10} | {'-':>12} | {'-':>8}")
                continue
            mean, var, mx = stats
            print(f"{sn:<40} | {len(lengths):>6} | {mean:>10.1f} | {var:>12.1f} | {mx:>8}")
        all_tokens = [ln for lns in suite_token_lengths.values() for ln in lns]
        overall_tok = _length_stats(all_tokens)
        if overall_tok is not None:
            mean, var, mx = overall_tok
            print("-" * 86)
            print(
                f"{'ALL':<40} | {len(all_tokens):>6} | {mean:>10.1f} | {var:>12.1f} | {mx:>8}"
            )

    print("\nPer-task breakdown:")
    for r in results:
        prefix = "✓" if r.success_count > 0 else "✗"
        head = (r.instruction[:70] + "..") if len(r.instruction) > 70 else r.instruction
        print(f"  {prefix} [{r.success_count}/{r.total_count}] {head}")


if __name__ == "__main__":
    asyncio.run(main())
