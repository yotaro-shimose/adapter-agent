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
    result = await agent.run(prompt)
    return result.final_output().problem_statement


async def main():
    input_path = "data/easy_benchmark.csv"
    output_path = "data/easy_benchmark_enhanced.csv"

    litellm_model = get_gemini()

    agent = AgentWrapper[EnhancedProblem].create(
        name="BenchmarkEnhancer",
        instructions="You are an expert in numerical computing. Your task is to update benchmark problem statements to specify the use of the `numrs2` library while preserving the original problem's meaning and text.",
        model=litellm_model,
        output_type=EnhancedProblem,
    )

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # Load dataset using polars
    df = pl.read_csv(input_path)

    print(f"Enhancing {len(df)} problems in parallel...")

    # Create tasks for gather_with_semaphore
    tasks = [enhance_problem(agent, statement) for statement in df["problem_statement"]]

    # Run in parallel with semaphore
    enhanced_statements = await gather_with_semaphore(
        tasks, max_concurrent=10, progress_bar=True, desc="Enhancing problems"
    )

    # Update the dataframe and save
    df = df.with_columns(pl.Series("problem_statement", enhanced_statements))

    df.write_csv(output_path)

    print(f"Finished! Enhanced dataset saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
