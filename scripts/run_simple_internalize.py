import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

import tinker
from dotenv import load_dotenv
from prisma import Prisma

from adapter_agent.hierarchical.types import Knowledge
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.wiki_manager import WikiManager
from adapter_agent.rl.config import ModelLoadingSettings
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.simple_internalizer import PipelineConfig, SimplePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger("adapter_agent.internalize").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

EVAL_TREE_TASKS = {
    "root": [
        """Using the numrs2 library, implement a sequence partitioning algorithm for numerical arrays. Given a 2D array of dimensions (N, M) and an integer K, first interpret the 2D array as a sequence of N 1D arrays of length M. Then, partition this sequence into K contiguous sub-sequences. The partitioning must handle cases where N is not evenly divisible by K by distributing the remainder elements across the sub-sequences such that the sizes of any two sub-sequences differ by at most one. For benchmarking, initialize a source 2D array using numrs2 of size (160, 4) with floating-point values where each row contains the index of that row (e.g., row 0 is [0,0,0,0], row 1 is [1,1,1,1]). Partition this into 80 sub-sequences and verify that each resulting sub-sequence contains exactly 2 rows matching the original ordering.""",
        """Implement a numerical routine using the `numrs2` library that generates a 2D grid based on specific coordinate intervals and a bivariate function. First, generate two 1D vectors using `numrs2`: one containing 100 equally spaced floating-point values from 0.0 to 1.0 (exclusive), and another containing 200 equally spaced floating-point values from 0.0 to 1.0 (exclusive). Using these vectors, construct a coordinate grid (2D mesh) representing all possible pairs of these coordinates. For every coordinate pair (x, y) in the grid, calculate the value using the expression x^2 + 2xy. The final result should be a 100x200 `numrs2` matrix of 64-bit floating-point values containing the results of this operation. The implementation should leverage `numrs2` broadcasting or vectorized operations for efficiency where possible.""",
        """Evaluate the computational efficiency of an element-wise mapping operation on a large one-dimensional array using the numrs2 library. Generate a random array of 1,000,000 double-precision floating-point numbers (f64) sampled from a uniform distribution between 0.0 and 1.0 using numrs2. The task is to apply a mathematical transformation to each element of this array. The transformation involves calculating the natural logarithm of each element, then finding the square root of the absolute value of the result, and finally summing all the transformed elements into a single scalar value. The benchmark should measure the total time taken for both the element-wise transformations and the final reduction step, performed exclusively with numrs2 functions.""",
    ],
    "intermediate": [
        """Create a 3D array with dimensions [2, 2, 3] containing the values 0 to 11. \nUse the `Array::index` method and the `IndexSpec` enum to slice this array and extract only the second channel (index 1) of the third dimension. \n\nYour solution must:\n1. Define the input array using `Array::from_vec(...).reshape(&[2, 2, 3])`.\n2. Use `IndexSpec::All` for the first two dimensions.\n3. Use `IndexSpec::Slice` with the correct arguments to select only the index 1 of the third dimension (Hint: `Slice(start, Some(end), Some(step))`.\n4. Print the resulting array shape and data to verify it is a [2, 2, 1] (or [2, 2]) array containing the expected values.""",
        """Implement a small Rust program using the `numrs2` library that performs the following basic operations:\n1. Initialize a 2D array of size 10x10 with zeros using `numrs2::prelude::zeros`. Ensure you pass the shape as a slice `&[usize]` and handle generic arguments correctly.\n2. Manually populate the array by setting the value at index (5, 5) to 1.0. Note: Since standard `[(i, j)]` indexing failed in previous attempts, you must find the correct method for setting/getting elements (e.g., `set`, `get_mut`, or a specific indexing trait).\n3. Compute the mean of this array using the `numrs2::stats::Statistics` trait.\n4. Print the value at (5, 5) and the calculated mean to the console.""",
        """Create a simple Rust program using the `numrs2` library that demonstrates the correct way to:\n1. Initialize a 1D array of integers (e.g., `[1, 2, 3]`).\n2. Initialize a 2D array of floats (e.g., a 2x2 identity matrix).\n3. Convert an array of integers (i64) to an array of 32-bit floats (f32).\n\nYou should investigate the library documentation or use `cargo expand` / `search_library_doc` specifically to find the actual constructor functions (e.g., `Array::new`, `from_vec`, or the correct macro path) if `use numrs2::array;` continues to fail as a macro. Print the resulting arrays and their shapes to the console.""",
        """Create a minimal Rust program using `numrs2` that performs a basic transformation on a small data buffer. \n1. Create a 1D `Vec<u8>` containing the sequence 0 to 11 (size 12).\n2. Initialize a `numrs2::array::Array` from this vector and reshape it into a 3D shape `[2, 2, 3]`. Note: Use `from_vec` and then the `reshape` method as identified in the documentation.\n3. Use the `slice` method to extract a sub-array. The `slice` method in `numrs2` typically takes `(axis, start, end)`. Slice the 3rd axis (index 2) to take only the first 2 elements (0 to 2).\n4. Verify the final shape is `[2, 2, 2]` and print the array.\nDo not use any external crates like `rand`; use a simple range or manual vector for input data.""",
    ],
    "leaf": [
        """Using the `numrs2` library, create a 1D array of 5 integers (0 through 4) using the `Array::from_vec` method. Apply a transformation to this array that adds 10 to each element and multiplies the result by 2. Print the final array. \n\nConstraints:\n1. Use `numrs2::array::Array`.\n2. Ensure you pass a `Vec<i32>` to `from_vec`, not a range.\n3. Use the library's built-in mapping or arithmetic capabilities to perform the transformation.""",
        """Use the `numrs2` library to perform a filtering operation on two related 1D arrays.\n1. Create a 1D array named `ids` of type `i32` containing values from 0 to 99.\n2. Create a 1D array named `values` of type `f64` containing 100 elements. You can initialize these with a simple mathematical sequence (e.g., a sine wave or alternating positive/negative values) instead of using external random crates.\n3. Identify the indices where the elements in `values` are negative.\n4. Use these indices to extract the corresponding elements from the `ids` array.\n5. Print the resulting array of IDs.\n\nNote: Ensure you use the correct `numrs2` API for boolean masking or fancy indexing (check `IndexSpec` variants or array comparison methods like `.lt()`).""",
        """Identify the correct public API for the `numrs2` library to perform matrix-vector multiplication. \n\nSpecifically, write a Rust program that:\n1. Creates a 2x2 matrix representing [[1.0, 2.0], [3.0, 4.0]] using the `numrs2::matrix::Matrix` type.\n2. Creates a 2x1 column vector representing [10.0, 20.0].\n3. Performs a single matrix-vector multiplication (A * v).\n4. Prints the result.\n\nNote: You must correctly resolve whether `Matrix::new` takes a 1D `Array` with shape metadata or a nested `Vec`. Use `write_and_run` to explore the `numrs2::matrix` and `numrs2::array` modules to find the valid public constructors and the `dot` or `*` operator implementation.""",
        """Using the `numrs2` library, implement a program that:\n1. Creates a 1D `Array` of 1,000 `f64` values. You can initialize this from a `Vec<f64>` using `Array::from_vec`.\n2. Generate a boolean mask identifying all elements in the array that are less than 0.5.\n3. Use the `.get_where(&mask)` method (or the equivalent indexing syntax confirmed in the documentation) to extract these values into a new array.\n4. Print the count of elements found.\n\nEnsure you use the correct imports: `use numrs2::array::Array;` and `use numrs2::prelude::*;`. Avoid using structured arrays or random number generators for this sub-task; focus purely on the masking and selection logic.""",
    ],
}


async def main():
    parser = argparse.ArgumentParser(description="Simplified Internalization Pipeline")
    parser.add_argument(
        "--version",
        type=str,
        # default="study_20260413_145727",
        default="lab_verification",
        help="Wiki version to load knowledge from",
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Number of knowledge items to load"
    )

    parser.add_argument(
        "--granular-id",
        type=str,
        default="granular_prep_20260414_085004",
        help="SimpleTrainRun ID for granular knowledge",
    )
    parser.add_argument(
        "--cpt",
        action="store_true",
        help="Use Continual Pre-training mode (single trainable user message)",
    )
    args = parser.parse_args()

    load_dotenv()
    json_path = Path("repositories/numrs/target/doc/numrs2.json")

    if not json_path.exists():
        logger.error(f"RustDoc JSON not found at {json_path}")
        return

    logger.info("Setting up simplified internalization pipeline...")

    # 1. Analyzer
    analyzer = await AsyncRustDocAnalyzer.create_from_json(json_path)

    # 2. Runtime Settings
    # Use standard cloudrun settings for the runtime environment.
    runtime_settings = RuntimeSettings.cloudrun_numrs2()

    # 3. DB and Knowledge
    db = Prisma()
    await db.connect()

    knowledge_list = []
    if args.granular_id:
        logger.info(f"Loading granular knowledge from run ID: {args.granular_id}")
        granulars = await db.granularknowledge.find_many(
            where={"simple_train_id": args.granular_id}
        )
        for g in granulars:
            knowledge_list.append(Knowledge(id=g.id, title=g.title, content=g.content))
    else:
        wiki_manager = WikiManager(db, version=args.version)

        # Fetch articles starting with "api/"
        titles = await wiki_manager.ls(path="api/")
        if args.limit:
            titles = titles[: args.limit]

        for title in titles:
            content = await wiki_manager.read(title)
            if content:
                knowledge_list.append(Knowledge(title=title, content=content))

    if not knowledge_list:
        logger.error(
            f"No knowledge found (options: granular={args.granular}, version={args.version})."
        )
        await db.disconnect()
        return

    logger.info(
        f"Loaded {len(knowledge_list)} knowledge items from Wiki version {args.version}."
    )

    # 4. Pipeline Configuration
    simple_train_id = f"simple_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = PipelineConfig(
        runtime_pool_size=100,
        rl_worker_count=48,
        eval_concurrency=48,
        generation_concurrency=400,
        simple_train_id=simple_train_id,
        knowledge_list=knowledge_list,
        library_name="numrs2",
        runtime_settings=runtime_settings,
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-8B",
            resume_trainer_path=None,
            resume_sampler_path=None,
            lora_rank=32,
        ),
        k_sft=32,
        k_eval=1,
        k_rl=4,
        sft_epochs=12,
        eval_rollout=4,
        rl_rollout=8,
        init_sft_steps=20,
        sft_batch_size=256,
        max_iterations=50,
        adam_params=tinker.AdamParams(learning_rate=1e-3),
        rl_adam_params=tinker.AdamParams(learning_rate=2e-4),
        rl_loss_fn="ppo",
        extra_eval_suites=EVAL_TREE_TASKS,
        stop_grpo=False,
        kl_penalty_coef=0.0,
        kl_discount_factor=0.0,
        cpt=args.cpt,
    )

    pipeline = await SimplePipeline.create(config=config, rust_doc_analyzer=analyzer)

    try:
        async with analyzer:
            logger.info("Starting simple pipeline execution.")
            await pipeline.run()
            logger.info("Pipeline executed successfully.")
    except Exception as e:
        logger.exception(f"Pipeline encountered an error: {e}")
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
