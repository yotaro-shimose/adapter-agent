"""End-to-end pipeline that builds a numrs2 benchmark dataset.

Stage 1: pull numpy snippets from BigQuery and turn them into candidate
benchmark problems via Gemini (`gh.generate_benchmark_dataset`).
Stage 2: rewrite the kept problems to mandate `numrs2`
(`enhance_benchmark.enhance_benchmark_dataset`).

Outputs land under `data/benchmarks/<name>/{original,enhanced}.csv`.
"""

import argparse
import asyncio
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from adapter_agent.hierarchical.gh import Library, generate_benchmark_dataset
from enhance_benchmark import enhance_benchmark_dataset

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a numrs2 benchmark dataset end-to-end (BigQuery -> generate -> enhance).",
    )
    parser.add_argument(
        "--name",
        default=f"numrs2_{date.today().isoformat()}",
        help="Benchmark name; outputs go to data/benchmarks/<name>/.",
    )
    parser.add_argument(
        "--library-summary",
        default="repositories/numrs/SUMMARY.md",
        help="Path to the target library's documentation used as model context.",
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--generate-max-concurrent", type=int, default=100)
    parser.add_argument("--enhance-max-concurrent", type=int, default=10)
    parser.add_argument(
        "--difficulty",
        default="Easy",
        help="Difficulty kept during enhancement; pass empty string to disable.",
    )
    parser.add_argument(
        "--no-filter-appropriate",
        action="store_true",
        help="Keep rows even if the generator marked them inappropriate.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Reuse an existing original.csv instead of re-running BigQuery.",
    )
    parser.add_argument(
        "--skip-enhance",
        action="store_true",
        help="Stop after the generation stage.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    out_dir = Path("data/benchmarks") / args.name
    original_csv = out_dir / "original.csv"
    enhanced_csv = out_dir / "enhanced.csv"

    library = Library(name="numrs2", local_path=Path(args.library_summary))

    if args.skip_generate:
        if not original_csv.exists():
            raise FileNotFoundError(
                f"--skip-generate set but {original_csv} does not exist."
            )
        print(f"[generate] skipped, reusing {original_csv}")
    else:
        await generate_benchmark_dataset(
            library=library,
            output_path=original_csv,
            limit=args.limit,
            max_concurrent=args.generate_max_concurrent,
        )

    if args.skip_enhance:
        print("[enhance] skipped")
        return

    difficulty = args.difficulty if args.difficulty else None
    await enhance_benchmark_dataset(
        input_path=original_csv,
        output_path=enhanced_csv,
        filter_appropriate=not args.no_filter_appropriate,
        difficulty=difficulty,
        max_concurrent=args.enhance_max_concurrent,
    )

    print(f"Done. Benchmark '{args.name}' available at {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
