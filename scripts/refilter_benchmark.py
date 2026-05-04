"""Re-run only the filter stage on an existing benchmark CSV.

Useful for validating filter prompt changes without re-running BigQuery and
the generation step. Reads ``problem_statement`` (and ``difficulty``) from the
input CSV, re-evaluates each row with the current ``filter_benchmark_case``,
and writes a new CSV with refreshed ``appropriate`` and ``reason`` columns
(plus ``appropriate_old`` / ``reason_old`` for diffing).
"""

import argparse
import asyncio
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from oai_utils.async_utils import gather_with_semaphore
from oai_utils.litellm import litellm_concurrent_limit

from adapter_agent.hierarchical.gh import (
    BenchmarkCase,
    Library,
    filter_benchmark_case,
)
from adapter_agent.model_helper import get_gemini

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target-name", required=True)
    p.add_argument("--target-summary", required=True)
    p.add_argument("--max-concurrent", type=int, default=10)
    return p.parse_args()


async def filter_one(model, library, statement, difficulty):
    case = BenchmarkCase(
        problem_statement=statement,
        difficulty=difficulty if difficulty in ("Easy", "Medium", "Hard") else "Easy",
    )
    try:
        result = await filter_benchmark_case(model, case, library)
        return result.appropriate, result.reason
    except BaseException as e:
        return None, f"refilter error: {type(e).__name__}: {e}"


async def main() -> None:
    args = parse_args()

    df = pl.read_csv(args.input)
    library = Library(name=args.target_name, local_path=Path(args.target_summary))
    model = get_gemini()

    statements = df["problem_statement"].to_list()
    difficulties = (
        df["difficulty"].to_list() if "difficulty" in df.columns else ["Easy"] * len(df)
    )

    print(f"Re-filtering {len(statements)} cases against '{args.target_name}'...")

    async with litellm_concurrent_limit(args.max_concurrent):
        coros = [
            filter_one(model, library, s, d)
            for s, d in zip(statements, difficulties)
        ]
        results = await gather_with_semaphore(
            coros, max_concurrent=args.max_concurrent, progress_bar=True
        )

    appropriates = [r[0] for r in results]
    reasons = [r[1] for r in results]

    out = df.with_columns(
        pl.Series("appropriate_old", df["appropriate"] if "appropriate" in df.columns else [None] * len(df)),
        pl.Series("reason_old", df["reason"] if "reason" in df.columns else [None] * len(df)),
        pl.Series("appropriate", appropriates),
        pl.Series("reason", reasons),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_path)
    print(f"Wrote {out_path}")

    # Quick summary
    n_total = len(out)
    n_appropriate = sum(1 for a in appropriates if a is True)
    n_inappropriate = sum(1 for a in appropriates if a is False)
    n_error = sum(1 for a in appropriates if a is None)
    print(
        f"Summary: {n_appropriate} appropriate, {n_inappropriate} rejected, "
        f"{n_error} errors (of {n_total})"
    )


if __name__ == "__main__":
    asyncio.run(main())
