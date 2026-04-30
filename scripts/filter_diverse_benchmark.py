"""Filter a classified benchmark CSV down to a diverse subset, then rewrite
the kept problems to mandate `numrs2` via `enhance_benchmark_dataset`.

Reproducible end-to-end of the manual filtering session:

  classified.csv
      |  drop tasks whose required_apis is a subset of BASIC
      |  drop tasks where appropriate is False (optional)
      |  drop tasks whose difficulty is in EXCLUDE_DIFFICULTIES (optional)
      v
  diverse_filtered.csv
      |  enhance_benchmark_dataset (Gemini rewrite to mention numrs2)
      v
  diverse_enhanced.csv

The default BASIC set is the one we converged on for "tasks that need at least
one non-trivial numpy API beyond the boring core":
    array_creation, elementwise_math, random, reduction,
    reshape_transpose, broadcasting

`required_apis` is read as a comma-separated string (the format produced by
`scripts/classify_benchmark_apis.py`).
"""

import argparse
import asyncio
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from enhance_benchmark import enhance_benchmark_dataset

load_dotenv()


DEFAULT_BASIC_APIS = (
    "array_creation,elementwise_math,random,reduction,"
    "reshape_transpose,broadcasting"
)


def parse_csv_set(value: str) -> set[str]:
    return {x.strip() for x in value.split(",") if x.strip()}


def filter_dataset(
    df: pl.DataFrame,
    basic_apis: set[str],
    require_appropriate: bool,
    exclude_difficulties: set[str],
) -> pl.DataFrame:
    if "required_apis" not in df.columns:
        raise ValueError(
            "Input CSV is missing the 'required_apis' column. "
            "Run scripts/classify_benchmark_apis.py first."
        )

    df = df.filter(pl.col("required_apis").is_not_null())

    def has_non_basic(s: str) -> bool:
        apis = {a.strip() for a in s.split(",")}
        return not apis.issubset(basic_apis)

    df = df.filter(pl.Series([has_non_basic(s) for s in df["required_apis"]]))

    if require_appropriate and "appropriate" in df.columns:
        df = df.filter(pl.col("appropriate"))
    if exclude_difficulties and "difficulty" in df.columns:
        df = df.filter(~pl.col("difficulty").is_in(list(exclude_difficulties)))
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a classified benchmark CSV to a diverse subset and "
            "rewrite the kept problems to mandate numrs2."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the classified CSV (output of classify_benchmark_apis.py).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for diverse_filtered.csv / diverse_enhanced.csv. "
            "Defaults to the parent of --input."
        ),
    )
    parser.add_argument(
        "--basic-apis",
        default=DEFAULT_BASIC_APIS,
        help=(
            "Comma-separated API categories considered 'basic'. Tasks whose "
            f"required_apis is a subset of this set are dropped. "
            f"Default: {DEFAULT_BASIC_APIS}"
        ),
    )
    parser.add_argument(
        "--exclude-difficulties",
        default="Hard",
        help=(
            "Comma-separated difficulty labels to drop. Default: Hard. "
            "Pass an empty string to keep all difficulties."
        ),
    )
    parser.add_argument(
        "--no-filter-appropriate",
        action="store_true",
        help="Keep rows where appropriate is False (default: drop them).",
    )
    parser.add_argument(
        "--skip-enhance",
        action="store_true",
        help="Stop after writing diverse_filtered.csv.",
    )
    parser.add_argument(
        "--filtered-name",
        default="diverse_filtered.csv",
    )
    parser.add_argument(
        "--enhanced-name",
        default="diverse_enhanced.csv",
    )
    parser.add_argument("--max-concurrent", type=int, default=100)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found.")

    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = out_dir / args.filtered_name
    enhanced_path = out_dir / args.enhanced_name

    basic_apis = parse_csv_set(args.basic_apis)
    exclude_difficulties = parse_csv_set(args.exclude_difficulties)

    df = pl.read_csv(input_path)
    before = len(df)
    filtered = filter_dataset(
        df,
        basic_apis=basic_apis,
        require_appropriate=not args.no_filter_appropriate,
        exclude_difficulties=exclude_difficulties,
    )
    print(f"[filter] {before} -> {len(filtered)} rows")
    print(f"  basic_apis            = {sorted(basic_apis)}")
    print(f"  require_appropriate   = {not args.no_filter_appropriate}")
    print(f"  exclude_difficulties  = {sorted(exclude_difficulties) or '(none)'}")

    filtered.write_csv(filtered_path)
    print(f"[filter] wrote {filtered_path}")

    if args.skip_enhance:
        print("[enhance] skipped")
        return

    await enhance_benchmark_dataset(
        input_path=filtered_path,
        output_path=enhanced_path,
        # We've already filtered; do not let enhance_benchmark re-filter.
        filter_appropriate=False,
        difficulty=None,
        max_concurrent=args.max_concurrent,
    )
    print(f"[enhance] wrote {enhanced_path}")


if __name__ == "__main__":
    asyncio.run(main())
