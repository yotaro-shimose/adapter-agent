import asyncio
from pathlib import Path
from typing import Literal

import polars as pl
from dotenv import load_dotenv
from google.cloud import bigquery
from litellm import batch_completion
from oai_utils.agent import AgentsSDKModel, AgentWrapper
from oai_utils.async_utils import gather_with_semaphore
from oai_utils.litellm import litellm_concurrent_limit
from pydantic import BaseModel

from adapter_agent.hierarchical.types import Task
from adapter_agent.model_helper import get_gemini

batch_completion
load_dotenv()


class Library(BaseModel):
    """Target library that benchmark problems should be solved with."""

    name: str
    local_path: Path


class SourceLibrary(BaseModel):
    """Upstream library whose code snippets we mine for benchmark seeds.

    `import_patterns` are SQL ``LIKE`` substrings used to locate snippets
    on BigQuery. The patterns are OR-ed together.
    """

    name: str
    import_patterns: list[str]


# Default source: NumPy snippets with the conventional ``np`` alias.
DEFAULT_SOURCE = SourceLibrary(
    name="numpy",
    import_patterns=["import numpy as np"],
)


class BenchmarkCase(BaseModel):
    problem_statement: str
    difficulty: Literal["Easy", "Medium", "Hard"]


class FilterResult(BaseModel):
    appropriate: bool
    reason: str


class BenchmarkResult(BaseModel):
    problem_statement: str
    difficulty: Literal["Easy", "Medium", "Hard"]
    appropriate: bool
    reason: str


async def generate_benchmark_case(
    model: AgentsSDKModel,
    content: str,
    library: Library,
    source: SourceLibrary = DEFAULT_SOURCE,
) -> BenchmarkCase:
    # Read library README for context
    library_context = library.local_path.read_text()

    agent = AgentWrapper[BenchmarkCase].create(
        name=f"{library.name}BenchmarkGenerator",
        instructions=(
            f"You are an expert in {library.name}. "
            f"You are generating benchmark problems for the library '{library.name}'. Here is the library's documentation:\n\n{library_context}\n\n"
            f"Your task is to analyze Python code snippets that use the '{source.name}' library (and possibly related numerical/utility libraries) and extract the core mathematical or logical kernel. "
            f"Then, formulate a CONCRETE, self-contained benchmark problem statement for implementing this kernel using the '{library.name}' library. "
            "\n### Requirements for the Problem Statement:\n"
            "1. **Specificity of inputs and outputs**: Explicitly describe input data structures, types, "
            "and the required output format. The agent must know what to produce.\n"
            "2. **Algorithm Detail**: Describing the sequence of operations (including standard math/CS "
            "algorithm names like 'bisection', 'QR decomposition', 'Simpson\\'s rule', 'monotone cubic "
            "spline', 'FFT') is encouraged when natural — these are textbook concepts the agent should "
            "already know.\n"
            f"3. **Do not tie algorithms to {library.name}-specific features**: Phrase standard algorithms "
            "as general math/CS concepts that the agent then maps to a library function on its own. "
            "The library mention and the algorithm mention should be *separable*.\n"
            "   Bad → Good rewrites:\n"
            f"     - \"use {library.name}'s bisection-based numerical solver\" → \"using {library.name}, find the root via bisection\"\n"
            f"     - \"apply {library.name}'s QR decomposition routine\"        → \"using {library.name}, solve the linear system via QR decomposition\"\n"
            f"     - \"call {library.name}'s adaptive Simpson integrator\"      → \"using {library.name}, compute the definite integral via Simpson's rule\"\n"
            f"     - \"use {library.name}'s monotone cubic spline\"             → \"using {library.name}, interpolate with a monotone cubic spline\"\n"
            f"   In short: the algorithm is a math concept the agent knows; the {library.name} mention is "
            "the requirement to solve via that library. Don't fuse them.\n"
            f"4. **No API hallucination**: Do NOT name specific {library.name} functions, modules, types, "
            "or feature flags. Refer to capabilities by their math/CS name only.\n"
            "5. **No domain jargon**: Describe the task in terms of raw data and transformations; remove "
            "application-domain terminology (chemistry, finance, physics narratives, etc.).\n"
            "6. **Data generation**: Describe how the test data should be generated or initialized so the "
            "problem is self-contained.\n"
            "7. **Natural language**: The final statement should be clear, professional prose.\n\n"
            "Do NOT provide solution code. Just describe the task clearly in the 'problem_statement' field."
        ),
        model=model,
        # model_settings=ModelSettings(
        # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        # ),
        output_type=BenchmarkCase,
    )
    result = await agent.run(
        f"Analyze this code and generate a {library.name} benchmark problem statement:\n\n```python\n{content}\n```",
        time_out_seconds=60,
    )
    return result.final_output()


async def filter_benchmark_case(
    model: AgentsSDKModel,
    benchmark_case: BenchmarkCase,
    library: Library,
) -> FilterResult:
    # Read library README for context
    library_context = library.local_path.read_text()

    agent = AgentWrapper[FilterResult].create(
        name=f"{library.name}BenchmarkFilter",
        instructions=(
            f"You are a STRICT but FAIR quality-control agent for the '{library.name}' benchmark dataset.\n\n"
            f"Below is the '{library.name}' README. Treat it as a HIGH-LEVEL CATEGORY SUMMARY, not an exhaustive "
            "function list. Apply these reading rules:\n"
            "  - If the README lists a category (e.g. 'PCG32 RNG'), assume the standard methods of that category "
            "exist (uniform sampling, ranged sampling, Gaussian/normal, etc.).\n"
            "  - If the README mentions 're-exports glam' or similar, all standard methods of the re-exported "
            "library are available (e.g. Vec3 dot, cross, length, normalize, lerp).\n"
            "  - If the README lists a decomposition (QR, LU, Cholesky), back-substitution / solving the "
            "decomposed system is acceptable to ask even if not named.\n"
            "  - Auxiliary std operations (linspace via map, simple loops, manual normalization, distance "
            "variants like Manhattan / Chebyshev, hand-rolled window functions like Hann, simple thresholding) "
            "are FINE — those are a few lines of std code and do not require library APIs.\n\n"
            "=== README BEGIN ===\n"
            f"{library_context}\n"
            "=== README END ===\n\n"
            "Internal procedure (not output, but follow before deciding):\n"
            "  a. Identify the CORE non-trivial computation the problem requires "
            "(the algorithmic heart, not surrounding glue).\n"
            "  b. Locate the corresponding category in the README under DEFAULT features only.\n"
            "  c. If the core algorithm is covered, ACCEPT — even if surrounding glue needs std.\n"
            "  d. If the core algorithm is NOT covered, REJECT.\n\n"
            "REJECT (set appropriate=false) when ANY of the following hold:\n\n"
            "1. **Missing core algorithm**: The PRIMARY non-trivial computation requires an algorithm that the "
            "README does not document. Common pitfalls — verify the README before assuming:\n"
            "   - Special functions: Bessel, Legendre (full evaluation), Hermite, Chebyshev, gamma, erf, beta, "
            "elliptic integrals, etc.\n"
            "   - Statistics primitives: mean, std, variance, percentile, histogram, correlation "
            "(unless README lists a stats module).\n"
            "   - Linear/logistic regression, classification, clustering, or any ML primitive.\n"
            "   - Tabular ops: row filter, group-by, joins, arbitrary tensor reshape across non-matrix shapes.\n"
            "   - Pixel-level image processing on full RGB frames.\n"
            "   - Signal-processing pipeline beyond what the README enumerates (FFT/DCT/DST is fine if listed; "
            "filter design / wavelets / specialized transforms generally are not).\n\n"
            "2. **Non-default feature is load-bearing**: The README marks some features as opt-in "
            "(e.g. tensor for N-d arrays, autodiff, symbolic, interval, parallel, ai). Reject only when the "
            "CORE problem cannot be expressed without the opt-in feature (e.g. an N-dimensional tensor problem "
            "when tensor is opt-in is REJECT; a 2D matrix problem solvable with the default DenseMatrix is OK).\n\n"
            "3. **Library is decorative**: The library's role is purely cosmetic — the entire substantive "
            "computation is std arithmetic and the library type is just a Vec wrapper. "
            "(E.g. 'use the library to add two scalars' or 'clamp components of a 2D vector by hand' — REJECT.)\n\n"
            "4. **Vague framework / pipeline**: High-level 'pipeline', 'framework', 'simulation system', "
            "'stateful processor', or 'kernel' without concrete inputs, outputs, and operations anchored to "
            "specific README capabilities.\n\n"
            f"5. **API hallucination**: Specific {library.name} function or module names that do not appear in "
            "the README and cannot be inferred from a documented category.\n\n"
            "6. **External setup**: Implies external files / real datasets / hardware unless it describes how "
            "to generate or simulate them.\n\n"
            "7. **Unverifiable output**: Requires producing binary media (images, audio) or other outputs that "
            "cannot be verified by string / value / structured-data comparison.\n\n"
            "OUTPUT REQUIREMENTS:\n"
            "  - `appropriate`: true if the CORE non-trivial computation is covered by a default-feature "
            "capability in the README, even when some auxiliary glue (linspace, manual back-substitute, distance "
            "variants, window functions, etc.) is std.\n"
            "  - `reason`: cite the criterion number AND name the specific operation/concept that triggered the "
            "rejection. Example: '#1 Missing core algorithm: requires Bessel J0 evaluation; README has no "
            "Bessel/special-function support.' Avoid vague reasons like 'doesn't fit' or 'out of scope'."
        ),
        model=model,
        output_type=FilterResult,
    )

    result = await agent.run(
        f"Evaluate this problem statement for the library '{library.name}':\n\n{benchmark_case.problem_statement}",
        time_out_seconds=60,
    )
    return result.final_output()


async def process_row(
    content: str,
    model: AgentsSDKModel,
    library: Library,
    source: SourceLibrary = DEFAULT_SOURCE,
) -> BenchmarkResult | None:
    print("-" * 40)
    print("Analyzing snippet...")

    try:
        benchmark_case = await generate_benchmark_case(model, content, library, source)
        print(f"Generated Case: {benchmark_case.problem_statement}")

        print("Filtering case...")
        filter_result = await filter_benchmark_case(model, benchmark_case, library)
        print(
            f"Appropriate: {filter_result.appropriate}, Reason: {filter_result.reason}"
        )

        return BenchmarkResult(
            problem_statement=benchmark_case.problem_statement,
            difficulty=benchmark_case.difficulty,
            appropriate=filter_result.appropriate,
            reason=filter_result.reason,
        )

    except BaseException as e:
        print(f"Error generating/filtering case: {e}")
        return None


async def generate_benchmark_dataset(
    library: Library,
    output_path: Path,
    limit: int = 1000,
    max_concurrent: int = 100,
    model: AgentsSDKModel | None = None,
    bigquery_project: str = "dsat2-405406",
    source: SourceLibrary = DEFAULT_SOURCE,
) -> int:
    if model is None:
        model = get_gemini()

    client = bigquery.Client(project=bigquery_project)

    if not source.import_patterns:
        raise ValueError(
            f"SourceLibrary '{source.name}' has no import_patterns configured."
        )
    where_clause = " OR ".join(
        f"content LIKE '%{pattern}%'" for pattern in source.import_patterns
    )
    sql_query = f"""
    SELECT content, binary
    FROM `bigquery-public-data.github_repos.sample_contents`
    WHERE {where_clause}
    LIMIT {limit}
    """

    print(
        f"Running BigQuery for {library.name} benchmarks "
        f"(source: {source.name}, patterns: {source.import_patterns})..."
    )
    query_job = client.query(sql_query)
    results = query_job.result()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Top {limit} contents. Saving to {output_path}...")
    async with litellm_concurrent_limit(max_concurrent):
        tasks = [
            process_row(row.content, model, library, source)
            for row in results
            if not row.binary
        ]
        results_data = await gather_with_semaphore(tasks, max_concurrent=max_concurrent)

        data_rows = [res for res in results_data if res is not None]

        if data_rows:
            df = pl.DataFrame([res.model_dump() for res in data_rows])
            df.write_csv(output_path)
            print(f"Saved {len(data_rows)} rows to {output_path}")
        else:
            print("No data rows generated.")

    return len(data_rows)


async def main():
    library = Library(name="numrs2", local_path=Path("repositories/numrs/SUMMARY.md"))
    await generate_benchmark_dataset(
        library=library,
        output_path=Path("benchmark_dataset.csv"),
        limit=1000,
        max_concurrent=100,
    )


def load_gh_archive(
    difficulty: str | None = "Easy",
    csv_path: Path = Path("data/benchmarks/numrs2_2026-04-29/diverse_enhanced.csv"),
) -> list[Task]:
    """Load benchmark tasks from the given CSV.

    Args:
        difficulty: If set (default "Easy"), keep only rows whose
            ``difficulty`` column equals this value. Pass ``None`` to keep
            every difficulty.
        csv_path: Benchmark CSV to read. Defaults to the numrs2 source for
            back-compat; pass a per-library path (typically
            ``LibrarySpec.benchmark_csv``) to switch libraries.
    """
    import polars as pl

    primary_df = pl.read_csv(csv_path)
    if difficulty is not None and "difficulty" in primary_df.columns:
        primary_df = primary_df.filter(pl.col("difficulty") == difficulty)
    statements: list[str] = primary_df["problem_statement"].to_list()

    return [Task.from_instruction(s) for s in statements]


if __name__ == "__main__":
    asyncio.run(main())
