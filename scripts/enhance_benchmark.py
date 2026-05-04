import argparse
import asyncio
import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from oai_utils.agent import AgentsSDKModel, AgentWrapper
from oai_utils.async_utils import gather_with_semaphore
from pydantic import BaseModel

from adapter_agent.model_helper import get_gemini

load_dotenv()


class EnhancedProblem(BaseModel):
    problem_statement: str


async def enhance_problem(
    agent: AgentWrapper[EnhancedProblem], problem: str, target_name: str
) -> str:
    prompt = f"""Please rewrite the following benchmark problem statement so that:

1. It explicitly states that the agent must solve it using the `{target_name}` library.
2. Standard math / CS algorithm names (bisection, Newton-Raphson, QR decomposition, Simpson's rule,
   monotone cubic spline, FFT, etc.) MAY remain — they are textbook concepts the agent already
   knows.
3. The algorithm and `{target_name}` must be mentioned *separately*, not fused into a phrase
   that names a specific `{target_name}` feature/API/routine. Rewrite phrases that tie an
   algorithm to a `{target_name}` feature:
     - "use {target_name}'s bisection-based numerical solver"
         → "using {target_name}, find the root via bisection"
     - "apply {target_name}'s QR decomposition routine"
         → "using {target_name}, solve the linear system via QR decomposition"
     - "call {target_name}'s adaptive Simpson integrator"
         → "using {target_name}, compute the definite integral via Simpson's rule"
     - "use {target_name}'s monotone cubic spline"
         → "using {target_name}, interpolate with a monotone cubic spline"
4. Do NOT name specific `{target_name}` function names, module paths, or feature flags
   (e.g. "{target_name}::num::bisection", "{target_name}'s `qr_decompose` function").
5. Preserve all original inputs, parameter values, expected output format, and problem logic.

In short: the algorithm name is OK; tying that algorithm to `{target_name}` as if it were a
named library feature is NOT OK. The agent must independently map standard algorithm names to
`{target_name}`'s API on their own.

Problem Statement:
{problem}
"""
    try:
        result = await agent.run(prompt)
        return result.final_output().problem_statement
    except BaseException as e:
        # One bad model response shouldn't kill the whole batch — fall back to
        # the original statement so the row is preserved.
        print(f"[enhance_problem] failed, keeping original: {type(e).__name__}: {e}")
        return problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite benchmark problem statements to explicitly use a target library."
    )
    parser.add_argument("--input", default="data/easy_benchmark.csv")
    parser.add_argument("--output", default="data/easy_benchmark_enhanced.csv")
    parser.add_argument(
        "--target-name",
        default="numrs2",
        help="Name of the target library to mandate in rewritten problem statements.",
    )
    parser.add_argument(
        "--filter-appropriate",
        action="store_true",
        help="Drop rows where the 'appropriate' column is False before enhancing.",
    )
    parser.add_argument(
        "--difficulty",
        default=None,
        help="If set, keep only rows whose 'difficulty' equals this value (e.g. Easy).",
    )
    parser.add_argument("--max-concurrent", type=int, default=10)
    return parser.parse_args()


async def enhance_benchmark_dataset(
    input_path: Path,
    output_path: Path,
    filter_appropriate: bool = False,
    difficulty: str | None = None,
    max_concurrent: int = 10,
    model: AgentsSDKModel | None = None,
    target_name: str = "numrs2",
) -> int:
    if model is None:
        model = get_gemini()

    agent = AgentWrapper[EnhancedProblem].create(
        name="BenchmarkEnhancer",
        instructions=(
            f"You are an expert in numerical computing. "
            f"Your task is to update benchmark problem statements to specify the use of the `{target_name}` library "
            f"while preserving the original problem's meaning and text."
        ),
        model=model,
        output_type=EnhancedProblem,
    )

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found.")

    df = pl.read_csv(input_path)
    if filter_appropriate and "appropriate" in df.columns:
        df = df.filter(pl.col("appropriate"))
    if difficulty is not None and "difficulty" in df.columns:
        df = df.filter(pl.col("difficulty") == difficulty)

    print(f"Enhancing {len(df)} problems in parallel (target: {target_name})...")

    tasks = [
        enhance_problem(agent, statement, target_name)
        for statement in df["problem_statement"]
    ]

    enhanced_statements = await gather_with_semaphore(
        tasks,
        max_concurrent=max_concurrent,
        progress_bar=True,
        desc="Enhancing problems",
    )

    df = df.with_columns(pl.Series("problem_statement", enhanced_statements))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)

    print(f"Finished! Enhanced dataset saved to {output_path}")
    return len(df)


async def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    await enhance_benchmark_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        filter_appropriate=args.filter_appropriate,
        difficulty=args.difficulty,
        max_concurrent=args.max_concurrent,
        target_name=args.target_name,
    )


if __name__ == "__main__":
    asyncio.run(main())
