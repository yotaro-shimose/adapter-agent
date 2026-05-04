"""End-to-end pipeline that builds a Rust benchmark dataset for a target library.

Stage 1: pull source-library snippets (default: numpy) from BigQuery and turn
them into candidate benchmark problems via Gemini
(`gh.generate_benchmark_dataset`).
Stage 2: rewrite the kept problems to mandate the target library
(`enhance_benchmark.enhance_benchmark_dataset`).

Defaults reproduce the original numrs2 ← numpy pipeline. Pass
``--source-name`` / ``--source-import`` and ``--target-name`` /
``--target-summary`` to retarget (e.g. hisab ← scipy).

Outputs land under `data/benchmarks/<name>/{original,enhanced}.csv`.
"""

import argparse
import asyncio
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from adapter_agent.hierarchical.gh import (
    Library,
    SourceLibrary,
    generate_benchmark_dataset,
)
from enhance_benchmark import enhance_benchmark_dataset

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Rust benchmark dataset end-to-end "
            "(BigQuery -> generate -> enhance) for a target library."
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Benchmark name; outputs go to data/benchmarks/<name>/. "
            "Defaults to '<target-name>_<today>'."
        ),
    )
    parser.add_argument(
        "--target-name",
        default="numrs2",
        help="Name of the target Rust library that problems will be rewritten to use.",
    )
    parser.add_argument(
        "--target-summary",
        default="repositories/numrs/SUMMARY.md",
        help="Path to the target library's documentation used as model context.",
    )
    parser.add_argument(
        "--source-name",
        default="numpy",
        help="Name of the upstream library whose snippets we mine for benchmark seeds.",
    )
    parser.add_argument(
        "--source-import",
        action="append",
        default=None,
        help=(
            "SQL LIKE substring identifying source-library imports in BigQuery. "
            "Repeat for multiple patterns (OR-ed). "
            "Defaults to ['import numpy as np'] when --source-name is numpy."
        ),
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


def resolve_source_imports(name: str, explicit: list[str] | None) -> list[str]:
    """Pick import patterns for the source library.

    Falls back to a small built-in lookup when ``--source-import`` is omitted,
    so common cases (numpy, scipy) work without extra flags.
    """
    if explicit:
        return explicit
    builtin = {
        "numpy": ["import numpy as np"],
        "scipy": ["from scipy", "import scipy"],
    }
    if name in builtin:
        return builtin[name]
    raise ValueError(
        f"No default --source-import patterns known for '{name}'. "
        "Pass --source-import explicitly (one or more times)."
    )


async def main() -> None:
    args = parse_args()

    benchmark_name = args.name or f"{args.target_name}_{date.today().isoformat()}"
    out_dir = Path("data/benchmarks") / benchmark_name
    original_csv = out_dir / "original.csv"
    enhanced_csv = out_dir / "enhanced.csv"

    library = Library(name=args.target_name, local_path=Path(args.target_summary))
    source = SourceLibrary(
        name=args.source_name,
        import_patterns=resolve_source_imports(args.source_name, args.source_import),
    )

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
            source=source,
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
        target_name=args.target_name,
    )

    print(f"Done. Benchmark '{benchmark_name}' available at {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
