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
    name: str
    local_path: Path


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
    model: AgentsSDKModel, content: str, library: Library
) -> BenchmarkCase:
    # Read library README for context
    library_context = library.local_path.read_text()

    agent = AgentWrapper[BenchmarkCase].create(
        name=f"{library.name}BenchmarkGenerator",
        instructions=(
            f"You are an expert in {library.name}. "
            f"You are generating benchmark problems for the library '{library.name}'. Here is the library's documentation:\n\n{library_context}\n\n"
            "Your task is to analyze Python code snippets using common numerical or utility libraries (like numpy) and extract the core mathematical or logical kernel. "
            f"Then, formulate a CONCRETE, self-contained benchmark problem statement for implementing this kernel using the '{library.name}' library. "
            "\n### Requirements for the Problem Statement:\n"
            "1. **Specificity**: Explicitly describe the input and output data structures and types.\n"
            "2. **Algorithm Detail**: Describe the exact sequence of operations to perform.\n"
            "3. **Avoid API Hallucination**: Do NOT mention specific internal function or class names of the target library as you might hallucinate them. Refer to them generically.\n"
            "4. **No Domain Context**: Describe the task in terms of raw data and transformations, removing specific application-domain jargon.\n"
            "5. **Data Generation**: Describe how the test data should be generated or initialized.\n"
            "6. **Natural Language**: The final statement should be clear, professional natural language.\n\n"
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
            f"You are a strict quality control agent for the '{library.name}' benchmark dataset. "
            f"Here is the '{library.name}' library documentation:\n\n{library_context}\n\n"
            "Evaluate whether the following benchmark problem statement is appropriate based on these CRITICAL criteria:\n"
            f"1. **Scope Fit**: The task must reasonably fall within the scope of the library '{library.name}'.\n"
            f"2. **No API Hallucination**: Reject tasks that mention specific internal functions, modules, or types of '{library.name}' that are not explicitly documented in the provided README. We want natural language problems, not code-like specs.\n"
            "3. **Zero External Setup**: Reject tasks that imply external files, large datasets, or specialized hardware unless they describe how to generate/simulate them. The task should be self-contained.\n\n"
            "If the task violates any of these, set 'appropriate' to False and provide a mandatory 'reason' explaining which criterion was failed."
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
    content: str, model: AgentsSDKModel, library: Library
) -> BenchmarkResult | None:
    print("-" * 40)
    print("Analyzing snippet...")

    try:
        benchmark_case = await generate_benchmark_case(model, content, library)
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
) -> int:
    if model is None:
        model = get_gemini()

    client = bigquery.Client(project=bigquery_project)

    sql_query = f"""
    SELECT content, binary
    FROM `bigquery-public-data.github_repos.sample_contents`
    WHERE content LIKE '%import numpy as np%'
    LIMIT {limit}
    """

    print(f"Running BigQuery for {library.name} benchmarks...")
    query_job = client.query(sql_query)
    results = query_job.result()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Top {limit} contents. Saving to {output_path}...")
    async with litellm_concurrent_limit(max_concurrent):
        tasks = [
            process_row(row.content, model, library)
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


def load_gh_archive(difficulty: str | None = "Easy") -> list[Task]:
    """Load benchmark tasks from the latest enhanced CSV.

    Args:
        difficulty: If set (default "Easy"), keep only rows whose
            ``difficulty`` column equals this value. Pass ``None`` to keep
            every difficulty.
    """
    import polars as pl

    # Latest source: API-diverse, non-Hard, appropriate-only set rewritten to
    # mandate numrs2 (produced by scripts/filter_diverse_benchmark.py).
    primary_path = Path("data/benchmarks/numrs2_2026-04-29/diverse_enhanced.csv")

    # # Previous sources (kept for reference / fallback):
    # primary_path = Path("data/easy_benchmark_verified.csv")
    # secondary_paths = [
    #     Path("data/easy_benchmark_enhanced.csv"),
    #     Path("data/benchmark_dataset_enhanced.csv"),
    # ]

    primary_df = pl.read_csv(primary_path)
    if difficulty is not None and "difficulty" in primary_df.columns:
        primary_df = primary_df.filter(pl.col("difficulty") == difficulty)
    statements: list[str] = primary_df["problem_statement"].to_list()

    return [Task.from_instruction(s) for s in statements]


if __name__ == "__main__":
    asyncio.run(main())
