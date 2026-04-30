"""LLM-as-a-judge classifier that tags each benchmark task with the numpy/scipy
API categories it would actually need, plus a skeptical complexity verdict.

The goal is to detect tasks whose framing (e.g. "physics simulation",
"high-performance kernel") is grander than what the task actually requires.
We ask the judge to ignore narrative and look at the computation that would
be implemented in a normal numpy reference solution.
"""

import argparse
import asyncio
import os
from pathlib import Path
from typing import Literal

import polars as pl
from dotenv import load_dotenv
from oai_utils.agent import AgentsSDKModel, AgentWrapper
from oai_utils.async_utils import gather_with_semaphore
from pydantic import BaseModel, Field

from adapter_agent.model_helper import get_gemini

load_dotenv()


# Multi-label taxonomy. Kept small enough to be predictable, broad enough to
# cover what numpy/scipy users actually reach for.
ApiCategory = Literal[
    # --- Pure numpy core ---
    "array_creation",          # zeros/ones/arange/linspace/eye/full/empty
    "elementwise_math",        # +-*/, exp/log/sin/cos/sqrt/tanh/abs/...
    "reduction",               # sum/mean/min/max/prod over axes
    "reshape_transpose",       # reshape/transpose/expand_dims/squeeze/swapaxes
    "broadcasting",            # non-trivial use of broadcasting rules
    "advanced_indexing",       # boolean masks / fancy indexing / np.where
    "concat_stack_split_pad",  # concatenate/stack/split/tile/repeat/pad
    "sort_search_unique",      # sort/argsort/argmin/argmax/searchsorted/unique/bincount
    "cumulative_diff",         # cumsum/cumprod/diff/gradient
    # --- Stats / distributions ---
    "statistics",              # std/var/median/percentile/corrcoef/cov/histogram
    "random",                  # np.random / Generator: uniform/normal/choice/...
    # --- Linear algebra ---
    "linalg_basic",            # matmul/dot/norm/inv/solve/det/trace
    "linalg_decomp",           # eig/svd/qr/cholesky/lstsq/pinv
    # --- Signal-processing-ish ---
    "fft",                     # FFT family
    "convolution_correlate",   # np.convolve/correlate, signal.convolve
    # --- Beyond numpy core ---
    "interpolation",           # np.interp / scipy.interpolate (cubic spline, ...)
    "optimize_solver",         # BFGS, minimize, root finders (scipy.optimize)
    "sparse",                  # sparse matrix formats (scipy.sparse)
    "spatial_tree",            # KDTree/BallTree/distance_matrix (scipy.spatial)
    "clustering_ml",           # DBSCAN/KMeans/sklearn-style ML routines
    "datetime_units",          # datetime64/timedelta64/timezone-aware ops
    "iterative_python_loop",   # inherently sequential, hard to vectorise cleanly
    "io_or_external",          # file I/O, GPU/device handles, hardware mgmt
    "other",                   # anything else relevant
]


ActualComplexity = Literal[
    # The task reduces to a handful of straightforward numpy calls regardless
    # of how grandly it is framed (e.g. "physics simulation" that is just
    # vector add + boundary clip).
    "trivial_array_ops",
    # Typical numpy-flavoured workload: several array operations composed,
    # broadcasting, reductions, maybe a matmul. Real but not surprising.
    "routine_numpy",
    # Genuinely non-trivial: requires non-obvious composition, an iterative
    # algorithm, decompositions, or scipy-level functionality.
    "nontrivial_algorithm",
]


class ApiClassification(BaseModel):
    required_apis: list[ApiCategory] = Field(
        description=(
            "All API categories a normal numpy/scipy reference implementation "
            "would actually need. Only include categories that are clearly "
            "required by the computation; do not pad the list."
        )
    )
    actual_complexity: ActualComplexity = Field(
        description=(
            "Skeptical assessment of what the task ACTUALLY computes, ignoring "
            "domain framing. If the task is framed as a 'simulation' or "
            "'kernel' but boils down to a couple of array operations, label it "
            "trivial_array_ops."
        )
    )
    framing_inflation: bool = Field(
        description=(
            "True if the task uses heavy domain framing (physics, finance, "
            "signal processing, etc.) that is disproportionate to the actual "
            "computation required."
        )
    )
    rationale: str = Field(
        description=(
            "One or two sentences (max ~50 words) justifying the "
            "classification. Mention the concrete numpy operations that would "
            "dominate the implementation. Do not repeat yourself."
        ),
    )


SYSTEM_INSTRUCTIONS = """You are a strict, skeptical reviewer classifying benchmark problems by the numpy/scipy APIs an experienced practitioner would actually use to implement them.

You must IGNORE domain framing (physics, finance, ML, signal processing, etc.) and focus on the underlying computation. Many tasks are dressed up with fancy vocabulary while the actual work is a handful of array operations. Call those out.

Rules:
- `required_apis` is multi-label. Include every category that meaningfully participates in a reference implementation. Exclude categories that are not actually needed even if the task name suggests them. Be precise, not exhaustive.
- `actual_complexity` should reflect what is computed, not how it is described. A "high-performance numerical kernel for X" that boils down to elementwise tanh + add is `trivial_array_ops`.
- `framing_inflation` is True when the prose oversells the work (e.g. "particle simulation" that is just position += velocity * dt with bounds clipping).
- `rationale` must name the dominant operations concretely (e.g. "matmul + cumsum", "boolean mask + argsort"). Keep it under 50 words. Do not repeat phrases.
- Do not invent categories outside the provided list. Use `other` only if nothing else fits.
"""


async def classify_problem(
    agent: AgentWrapper[ApiClassification], problem: str
) -> ApiClassification | None:
    prompt = (
        "Classify the following benchmark problem statement.\n\n"
        f"Problem Statement:\n{problem}\n"
    )
    try:
        result = await agent.run(prompt, time_out_seconds=60)
        return result.final_output()
    except BaseException as e:  # noqa: BLE001
        print(f"[classify_problem] failed: {type(e).__name__}: {e}")
        return None


def _serialize(c: ApiClassification | None) -> dict:
    if c is None:
        return {
            "required_apis": None,
            "actual_complexity": None,
            "framing_inflation": None,
            "rationale": None,
        }
    rationale = c.rationale.strip()
    if len(rationale) > 500:
        rationale = rationale[:497] + "..."
    return {
        # store as comma-separated string so the resulting CSV stays flat
        "required_apis": ",".join(c.required_apis),
        "actual_complexity": c.actual_complexity,
        "framing_inflation": c.framing_inflation,
        "rationale": rationale,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add numpy/scipy API category labels and a skeptical complexity "
            "verdict to a benchmark CSV using an LLM-as-a-judge."
        ),
    )
    parser.add_argument("--input", default="benchmark_dataset.csv")
    parser.add_argument("--output", default="benchmark_dataset_classified.csv")
    parser.add_argument(
        "--filter-appropriate",
        action="store_true",
        help="Only classify rows where the 'appropriate' column is True.",
    )
    parser.add_argument(
        "--difficulty",
        default=None,
        help="If set, classify only rows whose 'difficulty' equals this value.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of rows to classify (for smoke tests).",
    )
    parser.add_argument("--max-concurrent", type=int, default=10)
    return parser.parse_args()


async def classify_benchmark_dataset(
    input_path: Path,
    output_path: Path,
    filter_appropriate: bool = False,
    difficulty: str | None = None,
    limit: int | None = None,
    max_concurrent: int = 10,
    model: AgentsSDKModel | None = None,
) -> int:
    if model is None:
        model = get_gemini()

    agent = AgentWrapper[ApiClassification].create(
        name="BenchmarkApiClassifier",
        instructions=SYSTEM_INSTRUCTIONS,
        model=model,
        output_type=ApiClassification,
    )

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found.")

    df = pl.read_csv(input_path)
    if filter_appropriate and "appropriate" in df.columns:
        df = df.filter(pl.col("appropriate"))
    if difficulty is not None and "difficulty" in df.columns:
        df = df.filter(pl.col("difficulty") == difficulty)
    if limit is not None:
        df = df.head(limit)

    print(f"Classifying {len(df)} problems with up to {max_concurrent} in flight...")

    tasks = [classify_problem(agent, s) for s in df["problem_statement"]]
    classifications = await gather_with_semaphore(
        tasks,
        max_concurrent=max_concurrent,
        progress_bar=True,
        desc="Classifying APIs",
    )

    serialized = [_serialize(c) for c in classifications]

    df = df.with_columns(
        pl.Series("required_apis", [s["required_apis"] for s in serialized]),
        pl.Series("actual_complexity", [s["actual_complexity"] for s in serialized]),
        pl.Series("framing_inflation", [s["framing_inflation"] for s in serialized]),
        pl.Series("classification_rationale", [s["rationale"] for s in serialized]),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    print(f"Saved classified dataset to {output_path}")
    return len(df)


async def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    await classify_benchmark_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        filter_appropriate=args.filter_appropriate,
        difficulty=args.difficulty,
        limit=args.limit,
        max_concurrent=args.max_concurrent,
    )


if __name__ == "__main__":
    asyncio.run(main())
