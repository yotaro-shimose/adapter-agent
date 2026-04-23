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

Edit the constants at the top of `main()` to pick axes and point at a
checkpoint or model.
"""

import asyncio
import logging
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import tinker
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from oai_utils import AgentsSDKModel
from oai_utils.agent import AgentWrapper
from oai_utils.async_utils import gather_with_semaphore
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from prisma import Prisma
from tinker_cookbook.renderers import Message

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

# --- Eval suite ---
EVAL_SLICE = slice(40, 60)
EVAL_ROLLOUT = 1
EVAL_CONCURRENCY = 48
RUNTIME_POOL_SIZE = 50
LIBRARY_NAME = "numrs2"
VERIFIER_MODEL: Literal["gemini", "gemini_lite"] = "gemini_lite"

# --- Solver model (which model runs as the policy) ---
MODEL_BACKEND: Literal["tinker", "agents"] = "tinker"

# Tinker backend: path to the trained sampler checkpoint.
TINKER_MODEL_NAME = "Qwen/Qwen3-8B"
TINKER_SAMPLER_PATH: str | None = (
    "tinker://976a7c11-7e95-596e-9230-38bff6526aa1:train:0/sampler_weights/rl_0020"
)

# Agents backend: which LiteLLM-backed model to use as the policy.
AGENTS_MODEL: Literal["gemini", "gemini_lite"] = "gemini"

# --- Eval strategy (how the solver is invoked per task) ---
EVAL_STRATEGY: Literal["single_turn", "ss_solve_verify"] = "single_turn"

# Settings for EVAL_STRATEGY == "ss_solve_verify" (mirrors study.py defaults).
SS_RUST_LIBDIR = Path("repositories/numrs")
SS_WIKI_VERSION = "study_20260419_041136"
SS_MAX_TURNS = 6
SS_QWEN_NO_THINK = True
SS_RUNTIME_MODE: Literal["docker", "cloudrun"] = "docker"
SS_CONCURRENCY = 20  # lower than EVAL_CONCURRENCY — each run spins its own runtime.
SS_USE_WIKI = True  # If False, the solver runs with an empty wiki (no Prisma needed).


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
    tinker_model: TinkerModel | None,
    agents_model: AgentsSDKModel | None,
    verifier_model: AgentsSDKModel,
) -> list[TaskEvalResult]:
    runtime_settings = RuntimeSettings.cloudrun_numrs2()
    verifier = Verifier(model=verifier_model)
    runtime_pool = RuntimePool(runtime_settings, max_size=RUNTIME_POOL_SIZE)
    executor = InternalizeExecutor(runtime_pool=runtime_pool, verifier=verifier)
    system_prompt = build_solver_system_prompt(LIBRARY_NAME)

    try:
        async def _one(suite_name: str, task: Task) -> TaskEvalResult:
            return await _evaluate_task_single_turn(
                task,
                suite_name,
                backend=MODEL_BACKEND,
                tinker_model=tinker_model,
                agents_model=agents_model,
                system_prompt=system_prompt,
                executor=executor,
                num_samples=EVAL_ROLLOUT,
            )

        return await gather_with_semaphore(
            [_one(sn, t) for sn, t in flattened],
            max_concurrent=EVAL_CONCURRENCY,
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
    solver_model: TinkerModel | LitellmModel,
    verifier_model: AgentsSDKModel,
) -> list[TaskEvalResult]:
    runtime_settings = (
        RuntimeSettings.docker_numrs2()
        if SS_RUNTIME_MODE == "docker"
        else RuntimeSettings.cloudrun_numrs2()
    )
    wiki_label = SS_WIKI_VERSION if SS_USE_WIKI else "<disabled>"
    logger.info(
        f"Building ss_solve_verify resources "
        f"(libdir={SS_RUST_LIBDIR}, wiki={wiki_label}, runtime={SS_RUNTIME_MODE})..."
    )
    rust_doc_analyzer = await AsyncRustDocAnalyzer.create_from_libdir(SS_RUST_LIBDIR)

    prisma: Prisma | None = None
    wiki_manager: WikiManager | _NullWikiManager
    if SS_USE_WIKI:
        prisma = Prisma()
        await prisma.connect()
        wiki_manager = WikiManager(prisma, version=SS_WIKI_VERSION)
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
                    max_turns=SS_MAX_TURNS,
                    qwen_no_think=SS_QWEN_NO_THINK,
                    num_samples=EVAL_ROLLOUT,
                )

            return await gather_with_semaphore(
                [_one(sn, t) for sn, t in flattened],
                max_concurrent=SS_CONCURRENCY,
            )
    finally:
        if prisma is not None:
            await prisma.disconnect()


async def main() -> None:
    load_dotenv()

    eval_suite = load_gh_archive_suite(
        name="gh_archive_eval",
        task_slice=EVAL_SLICE,
        for_rl=False,
        for_eval=True,
    )
    suites: list[SeedSuite] = [eval_suite]

    verifier_model = get_gemini() if VERIFIER_MODEL == "gemini" else get_gemini_lite()

    tinker_model: TinkerModel | None = None
    agents_model: AgentsSDKModel | None = None
    if MODEL_BACKEND == "tinker":
        logger.info(
            f"Loading Tinker sampler (base={TINKER_MODEL_NAME}) from {TINKER_SAMPLER_PATH}..."
        )
        tinker_model, _, _ = setup_tinkermodel(
            model_name=TINKER_MODEL_NAME,
            path=TINKER_SAMPLER_PATH,
        )
    elif MODEL_BACKEND == "agents":
        logger.info(f"Using AgentsSDKModel policy: {AGENTS_MODEL}")
        agents_model = get_gemini() if AGENTS_MODEL == "gemini" else get_gemini_lite()
    else:
        raise ValueError(f"Unknown MODEL_BACKEND: {MODEL_BACKEND}")

    flattened: list[tuple[str, Task]] = [
        (s.name, t) for s in suites for t in s.tasks
    ]
    logger.info(
        f"Evaluating {len(flattened)} tasks × {EVAL_ROLLOUT} rollouts "
        f"(backend={MODEL_BACKEND}, strategy={EVAL_STRATEGY})..."
    )

    if EVAL_STRATEGY == "single_turn":
        results = await _run_single_turn(
            flattened=flattened,
            tinker_model=tinker_model,
            agents_model=agents_model,
            verifier_model=verifier_model,
        )
    elif EVAL_STRATEGY == "ss_solve_verify":
        solver_model: TinkerModel | LitellmModel
        if MODEL_BACKEND == "tinker":
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
            solver_model=solver_model,
            verifier_model=verifier_model,
        )
    else:
        raise ValueError(f"Unknown EVAL_STRATEGY: {EVAL_STRATEGY}")

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
