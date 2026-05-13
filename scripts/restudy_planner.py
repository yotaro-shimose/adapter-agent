"""restudy_planner.py — failure-decomposition pipeline.

Refactored from a single whole-problem solver into a 2-stage,
`study2_pipeline.py`-style pipeline:

  Stage 1 (PLAN): given a FailedAttempt, ask Gemini (via `plan_with_tools`)
    to identify the specific *primitive* `{library}` API facts the agent got
    wrong — NOT to re-solve the full task. Each item is a small, focused
    investigation target like "binary operator support for `numrs2::Array<f64>`:
    which owned / borrowed operand combinations are implemented?" or
    "element-wise squaring of `Array<f64>` — does `.map(closure)` exist, or
    is `.pow(2.0)` the right pattern?". Output: list[str] of items.

  Stage 2 (INVESTIGATE per item, in parallel): for each item, run the same
    investigator as `study2_pipeline.py` (`solve_verify` with search +
    write_and_run + verifier) to produce a *tiny* runnable example
    demonstrating exactly that primitive. Output: list of (item,
    submit_code, verified) tuples.

The aggregated output is a "teaching memo" the original agent can read
back as: "Remember: when you tried X and got error Y, the correct pattern
is shown here (3-line code snippet)." This avoids the budget/multi-tag
death spiral that comes from trying to solve the whole 4D-array task in
one session with library-discovery interleaved.

Run with:
    uv run scripts/restudy_planner.py
"""

import asyncio
import logging
import re
from dataclasses import dataclass

from agents import set_tracing_disabled
from dotenv import load_dotenv
from oai_utils.agent import AgentWrapper, AgentRunFailure
from oai_utils.litellm import litellm_concurrent_limit

from adapter_agent.hierarchical.process.plan_with_tools import (
    InvestigationPlan,
    plan_with_tools,
)
from adapter_agent.hierarchical.process.solve_verify import solve_verify
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.library_spec import LibrarySpec
from adapter_agent.model_helper import get_gemini, get_gemini_lite
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.session_result import (
    RewireSessionResultSuccess,
)
from adapter_agent.util.logger_util import setup_base_loglevel

set_tracing_disabled(True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
setup_base_loglevel()
logger = logging.getLogger(__name__)


# === Failure case under investigation ================================


@dataclass(frozen=True)
class FailedAttempt:
    """One agent run that ended badly.

    `execution_output` is whatever the runtime saw last — cargo errors for
    a compile failure, program stdout for a "compiles but wrong" case.
    `verifier_feedback` is populated only when the code COMPILED and RAN
    but the verifier still rejected it (None otherwise — cargo's own
    output is informative enough).
    """

    question: str
    reasoning: str
    answer: str
    execution_output: str
    verifier_feedback: str | None = None


DEFAULT_FAILURE = FailedAttempt(
    question=(
        "Construct a 3D coordinate grid of dimensions 10x10x5 spanning the "
        "domain [0, 0.75] in the first dimension, [0, 1.0] in the second, "
        "and [0, 1.0] in the third using the `numrs2` library. For every "
        "point (x, y, z) in this grid, calculate a transformed coordinate "
        "set (x', y', z') based on the following mathematical mapping: "
        "first, define an angular parameter theta = (pi / 2) * x and a "
        "radial parameter r = y + 1.0; then, compute x' = r * cos(theta^2), "
        "y' = r * sin(theta^2), and z' = z. The calculation must utilize "
        "high-level `numrs2` array operations, specifically broadcasting "
        "and element-wise trigonometric and power functions. The final "
        "output must be a 4-dimensional array with shape (10, 10, 5, 3) "
        "containing the resulting transformed coordinates for the entire "
        "grid volume."
    ),
    reasoning=(
        "To construct a 3D coordinate grid using the numrs2 library, I will "
        "utilize the arange and reshape functions to generate individual "
        "axis arrays... Using broadcasting, I will expand these 1D arrays "
        "into the full 10x10x5 3D grid... I will compute theta = (pi/2)*x "
        "and r = y+1.0. The numrs2 library provides element-wise power "
        "(pow) and trigonometric (sin, cos) functions... Finally, I will "
        "concatenate the transformed x', y', and z' arrays along a new "
        "axis to produce the desired 4D output array."
    ),
    answer="""\
use numrs2::prelude::*;

fn main() {
    let x = arange(0.0f64, 0.75, 0.075).reshape(&[10, 1, 1]);
    let y = arange(0.0f64, 1.0, 0.1).reshape(&[1, 10, 1]);
    let z = arange(0.0f64, 1.0, 0.2).reshape(&[1, 1, 5]);

    let pi = std::f64::consts::PI;
    let theta = &x * (pi / 2.0);
    let r = &y + 1.0;

    let theta_squared = theta.map(|v| v * v);
    let x_prime = &r * theta_squared.cos();
    let y_prime = &r * theta_squared.sin();
    let z_prime = &z;

    let x_prime_reshaped = x_prime.reshape(&[10, 10, 5, 1]);
    // ... (concatenate along axis 3)
}
""",
    execution_output="""\
error[E0277]: cannot multiply `&numrs2::array::Array<f64>` by `numrs2::array::Array<f64>`
  --> src/main.rs:31:22
   |
31 |     let x_prime = &r * theta_squared.cos();
   |                      ^ no implementation for `&numrs2::array::Array<f64> * numrs2::array::Array<f64>`
   |
   = help: the trait `Mul<numrs2::array::Array<f64>>` is not implemented for `&numrs2::array::Array<f64>`

error[E0277]: cannot multiply `&numrs2::array::Array<f64>` by `numrs2::array::Array<f64>`
  --> src/main.rs:32:22

error[E0282]: type annotations needed
  --> src/main.rs:31:9 / 32:9

error: could not compile `workspace` (bin "workspace") due to 4 previous errors
""",
    verifier_feedback=None,
)


# === Config ==========================================================

LIBRARY_SPEC = LibrarySpec.numrs2()
FAILURE: FailedAttempt = DEFAULT_FAILURE

# Stage 1: planner — pure source-search, no runtime needed.
PLANNER_MAX_TURNS = 15
PLANNER_ITEMS_LIMIT = 6  # cap, in case the planner is over-enthusiastic

# Stage 2: per-primitive investigator — small focused demo, modest budget.
INVESTIGATOR_MAX_TURNS = 10
INVESTIGATOR_CONCURRENCY = 4

RUNTIME_POOL_MAX_SIZE = INVESTIGATOR_CONCURRENCY


# === Stage 1: planner task assembly ==================================


def _build_planner_task(f: FailedAttempt, library_name: str) -> str:
    """Wrap the failure as a "find the misused primitives" ask.

    Crucially, we do NOT ask the planner to re-solve the original task.
    Its job is to atomize the failure into small `{library_name}`-specific
    API questions, each of which a downstream investigator can demonstrate
    with a tiny code example.
    """
    blocks: list[str] = [
        "<OriginalTask>",
        f.question.strip(),
        "</OriginalTask>",
        "",
        "<FailedAttempt>",
        "An agent tried to solve the task above and produced this attempt:",
        "",
        "<Reasoning>",
        f.reasoning.strip(),
        "</Reasoning>",
        "",
        "<Answer>",
        f.answer.strip(),
        "</Answer>",
        "",
        "<ExecutionOutput>",
        f.execution_output.strip(),
        "</ExecutionOutput>",
    ]
    if f.verifier_feedback:
        blocks.extend([
            "",
            "<VerifierFeedback>",
            f.verifier_feedback.strip(),
            "</VerifierFeedback>",
        ])
    blocks.extend([
        "</FailedAttempt>",
        "",
        "<YourJob>",
        "For each rejection the failed agent hit, emit ONE plan item with "
        "TWO parts:",
        "  (a) WHAT THE AGENT DID WRONG — quote the compiler / runtime / "
        "verifier objection and the offending source line.",
        f"  (b) WHAT FEATURE THE AGENT SHOULD HAVE KNOWN — a short, "
        f"behavior-level description of the `{library_name}` capability "
        f"that addresses (a). It does NOT need to name a specific API; "
        "the downstream investigator will resolve the exact symbol via "
        "source search.",
        "",
        "Out of scope: hand-rolled-but-working logic, stylistic choices, "
        "'could've used the helper' observations — only behaviors the "
        "compiler / runtime / verifier rejected count.",
        "",
        "Example shape:",
        '  ✓ "(a) E0277 on line 31: `&Array * Array` — `Mul` not '
        "satisfied. (b) Reference-by-reference multiplication of "
        'arrays — the ownership combination the library actually '
        'implements."',
        '  ✗ "the agent should use multiplication of references"   '
        "← missing (a), no failure citation",
        "",
        'One focused angle per item. Typically 1-3 items; only as many '
        "as there are distinct rejected behaviors. Do not pad.",
        "</YourJob>",
    ])
    return "\n".join(blocks)


async def plan_one_shot(
    failure: FailedAttempt,
    library_name: str,
    library_summary: str,
    model,
) -> InvestigationPlan | None:
    """Single-call planner — for failure recovery where source-search is
    NOT needed because plan items describe behavior, not specific APIs.

    Uses `oai_utils.AgentWrapper` with `InvestigationPlan` as the
    structured output type, so the LLM returns a parsed Pydantic object
    directly — no JSON parsing / markdown stripping needed.

    Drop-in replacement for `plan_with_tools` for the restudy
    pipeline. Returns None on agent failure (timeout, model behaviour
    error, etc.) so callers can handle the same way as the tool-loop
    variant."""
    instructions = (
        f"You are diagnosing a failed agent attempt to use the "
        f"`{library_name}` library. Read the failure trace in the user "
        "message and output an InvestigationPlan listing the misused "
        "primitives as behavior-level descriptions (NOT specific API "
        "names — the downstream investigator will resolve those).\n"
        "\n"
        f"<LibrarySummary>\n{library_summary}\n</LibrarySummary>"
    )
    wrapper = AgentWrapper[InvestigationPlan].create(
        name="failure_planner",
        model=model,
        instructions=instructions,
        output_type=InvestigationPlan,
    )
    user_input = _build_planner_task(failure, library_name)
    try:
        result = await wrapper.run(user_input)
    except AgentRunFailure as e:
        logger.warning(f"plan_one_shot failed: {e.cause}: {e}")
        return None
    return result.final_output()


# === Stage 2: per-primitive investigation ============================


def _build_investigation_task(item: str, library_name: str) -> str:
    """Mirrors `study2_pipeline._build_investigation_task` verbatim — the
    output is a small standalone `fn main()` that demonstrates exactly the
    one primitive named in `<InvestigationTarget>`."""
    return f"""\
Investigate the following aspect of the `{library_name}` library and produce a
runnable Rust code example that demonstrates it.

<InvestigationTarget>
{item}
</InvestigationTarget>

Requirements for your final `<submit>`:
- A complete `fn main()` program (no missing pieces).
- It MUST exercise the actual `{library_name}` API named by the investigation
  target — not a hand-rolled equivalent in plain std.
- It MUST compile and run successfully via `cargo run`.
- It SHOULD print output that makes the demonstrated behavior visible.
- Keep it minimal — just enough to clearly show how the API is used.
"""


@dataclass
class PrimitiveResult:
    item: str
    verified: bool
    conclusion: str
    submit_code: str | None
    verifier_reasoning: str | None


async def _investigate_one(
    item: str,
    *,
    sem: asyncio.Semaphore,
    library_spec: LibrarySpec,
    library_summary: str,
    runtime_pool: RuntimePool,
    solver_model,
    verifier_model,
) -> PrimitiveResult:
    async with sem:
        try:
            result = await solve_verify(
                solver_model=solver_model,
                verifier_model=verifier_model,
                task=Task(
                    instruction=_build_investigation_task(item, library_spec.name)
                ),
                libdir=library_spec.libdir,
                library_name=library_spec.name,
                runtime_pool=runtime_pool,
                max_turns=INVESTIGATOR_MAX_TURNS,
                reference_knowledge=library_summary,
                enable_search_tools=True,
            )
        except Exception as e:
            logger.exception(f"investigator crashed for item: {item!r}")
            return PrimitiveResult(
                item=item,
                verified=False,
                conclusion="exception",
                submit_code=None,
                verifier_reasoning=str(e),
            )

        success = isinstance(result, RewireSessionResultSuccess)
        submit = None
        trials = getattr(result, "trials", None)
        if trials:
            submit = _extract_submit(trials)
        return PrimitiveResult(
            item=item,
            verified=success,
            conclusion=getattr(result, "conclusion", "unknown"),
            submit_code=submit,
            verifier_reasoning=getattr(result, "reasoning", None),
        )


# === Helpers =========================================================


def _flatten(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t == "thinking":
                out.append(f"<think>{p.get('thinking', '')}</think>")
            else:
                out.append(p.get("text", ""))
        return "".join(out)
    return "" if content is None else str(content)


def _extract_submit(trials) -> str | None:
    for msg in reversed(trials):
        content = msg.get("content") if isinstance(msg, dict) else None
        if content is None:
            continue
        text = _flatten(content)
        m = re.search(r"<submit>(.*?)</submit>", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def _print_separator(char: str = "=", n: int = 80) -> None:
    print(char * n)


# === Main ============================================================


async def main() -> None:
    load_dotenv()

    try:
        library_summary = LIBRARY_SPEC.read_summary()
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    planner_task = _build_planner_task(FAILURE, LIBRARY_SPEC.name)

    _print_separator()
    print(f"Library            : {LIBRARY_SPEC.name}")
    print(f"Planner turns      : {PLANNER_MAX_TURNS}")
    print(f"Investigator turns : {INVESTIGATOR_MAX_TURNS}")
    print(f"Items cap          : {PLANNER_ITEMS_LIMIT}")
    print(f"Verifier feedback  : {'yes' if FAILURE.verifier_feedback else 'no'}")
    _print_separator()

    planner_model = get_gemini()
    solver_model = get_gemini()
    verifier_model = get_gemini_lite()

    runtime_pool: RuntimePool | None = None
    async with litellm_concurrent_limit(max_concurrent=16):
        try:
            # --- Stage 1: PLAN ---
            print("\n[stage 1: PLAN — identifying misused primitives]", flush=True)
            plan: InvestigationPlan | None = await plan_with_tools(
                task_instruction=planner_task,
                library_name=LIBRARY_SPEC.name,
                libdir=LIBRARY_SPEC.libdir,
                library_summary=library_summary,
                solver_model=planner_model,
                max_turns=PLANNER_MAX_TURNS,
            )
            if plan is None or not plan.items:
                print("[stage 1] planner produced no items. Aborting.")
                return

            items = plan.items[:PLANNER_ITEMS_LIMIT]
            print(f"[stage 1] planner returned {len(plan.items)} items "
                  f"(using first {len(items)}):")
            for i, it in enumerate(items, start=1):
                print(f"  {i}. {it}")

            # --- Stage 2: INVESTIGATE per primitive, in parallel ---
            print(f"\n[stage 2: INVESTIGATE — {len(items)} primitives in "
                  f"parallel, concurrency={INVESTIGATOR_CONCURRENCY}]",
                  flush=True)
            runtime_pool = RuntimePool(
                settings=LIBRARY_SPEC.cloudrun_runtime(),
                max_size=RUNTIME_POOL_MAX_SIZE,
            )
            sem = asyncio.Semaphore(INVESTIGATOR_CONCURRENCY)
            results: list[PrimitiveResult] = await asyncio.gather(
                *[
                    _investigate_one(
                        it,
                        sem=sem,
                        library_spec=LIBRARY_SPEC,
                        library_summary=library_summary,
                        runtime_pool=runtime_pool,
                        solver_model=solver_model,
                        verifier_model=verifier_model,
                    )
                    for it in items
                ]
            )
        finally:
            if runtime_pool is not None:
                await runtime_pool.close_all()

    # --- Output: teaching memo ---
    print()
    _print_separator()
    print("TEACHING MEMO FOR THE FAILED AGENT")
    _print_separator()
    print()
    print(
        "Based on your failed attempt on the task above, here are the "
        f"`{LIBRARY_SPEC.name}` primitives we investigated. Remember the "
        "patterns shown in the verified examples — they reflect what the "
        "library actually implements, not what you assumed.\n"
    )

    n_ok = sum(1 for r in results if r.verified)
    print(f"({n_ok}/{len(results)} primitives verified)\n")

    for i, r in enumerate(results, start=1):
        _print_separator("-")
        status = "VERIFIED" if r.verified else f"UNVERIFIED ({r.conclusion})"
        print(f"PRIMITIVE {i}/{len(results)} — {status}")
        _print_separator("-")
        print(f"Investigation target:\n  {r.item}\n")
        if r.submit_code:
            print("Working example (use this pattern in future attempts):")
            print()
            print(r.submit_code)
        else:
            print("(no <submit> recovered)")
        if not r.verified and r.verifier_reasoning:
            print()
            print(f"Verifier note: {r.verifier_reasoning}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
