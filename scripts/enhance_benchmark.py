import argparse
import asyncio
import os

import polars as pl
from dotenv import load_dotenv
from oai_utils.agent import AgentWrapper
from oai_utils.async_utils import gather_with_semaphore
from pydantic import BaseModel

from adapter_agent.model_helper import get_gemini

load_dotenv()


class EnhancedProblem(BaseModel):
    problem_statement: str


async def enhance_problem(agent: AgentWrapper[EnhancedProblem], problem: str) -> str:
    prompt = f"""Please modify the following benchmark problem statement to explicitly state that it should be solved using the `numrs2` library.
Keep the original problem logic and text as much as possible, just add the library requirement naturally.

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
        description="Rewrite benchmark problem statements to explicitly use numrs2."
    )
    parser.add_argument("--input", default="data/easy_benchmark.csv")
    parser.add_argument("--output", default="data/easy_benchmark_enhanced.csv")
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


async def main():
    args = parse_args()

    litellm_model = get_gemini()

    agent = AgentWrapper[EnhancedProblem].create(
        name="BenchmarkEnhancer",
        instructions="You are an expert in numerical computing. Your task is to update benchmark problem statements to specify the use of the `numrs2` library while preserving the original problem's meaning and text.",
        model=litellm_model,
        output_type=EnhancedProblem,
    )

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    df = pl.read_csv(args.input)
    if args.filter_appropriate and "appropriate" in df.columns:
        df = df.filter(pl.col("appropriate"))
    if args.difficulty is not None and "difficulty" in df.columns:
        df = df.filter(pl.col("difficulty") == args.difficulty)

    print(f"Enhancing {len(df)} problems in parallel...")

    tasks = [enhance_problem(agent, statement) for statement in df["problem_statement"]]

    enhanced_statements = await gather_with_semaphore(
        tasks,
        max_concurrent=args.max_concurrent,
        progress_bar=True,
        desc="Enhancing problems",
    )

    df = df.with_columns(pl.Series("problem_statement", enhanced_statements))

    df.write_csv(args.output)

    print(f"Finished! Enhanced dataset saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
