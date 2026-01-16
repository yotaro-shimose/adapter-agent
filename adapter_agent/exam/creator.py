import os

from agents import ModelSettings
from agents.items import TResponseInputItem
from loguru import logger
from oai_utils.agent import AgentsSDKModel, AgentWrapper
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from adapter_agent.exam.exam import CodingExam
from adapter_agent.exam.repository import GitRepository
from adapter_agent.exam.syllabus import LearningTopic
from coder_mcp.runtime.rust_env import RustCodingEnvironment
from coder_mcp.runtime.temp_workspace import TempWorkspace

EXAM_CREATOR_SYSTEM_PROMPT = """You are an exam creator agent, a specialized AI assistant that can interact with a computer to generate high-fidelity coding exercises.

<ROLE>
Your primary role is to create a comprehensive "Gold Standard" implementation of a coding task and then hollow it out to create a student-facing problem, while preserving evaluation infrastructure.
</ROLE>

<FILE_SYSTEM_GUIDELINES>
* When a user provides a file path, do NOT assume it's relative to the current working directory. First explore the file system to locate the file before working on it.
* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.
* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.
* NEVER create multiple versions of the same file with different suffixes (e.g., file_test.py, file_fix.py, file_simple.py). Instead:
  - Always modify the original file directly when making changes
  - If you need to create a temporary file for testing, delete it once you've confirmed your solution works
  - If you decide a file you created is no longer useful, delete it instead of creating a new version
* Do NOT include documentation files explaining your changes in version control unless the user explicitly requests it
* When reproducing bugs or implementing fixes, use a single file rather than creating multiple files with different versions
</FILE_SYSTEM_GUIDELINES>
"""
# Define mode-specific instructions
MODE_INSTRUCTIONS = {
    "functional": (
        "- **Implementation**: Create a functional module (e.g., `src/lib.rs` or `src/solver.rs`).\n"
        "- **Integration Tests**: Create `tests/verify.rs`. This file MUST include `use <crate_name>::<module>;` to import the definitions from your solution files.\n"
        "- **Verification**: The tests must verify the public interface of your implementation. The logic will be removed later, but the interface must remain so these tests can still compile."
    ),
    "imitation": (
        "- **Implementation**: Provide a 'Reference Pattern' in `question.md` and a matching skeleton in `src/`.\n"
        "- **Tests**: Optional. If created, they must import from the `src/` implementation just like functional tests.\n"
        "- **Verification**: Primarily focused on whether the student adapted the pattern idiomatic to the library."
    ),
    "conceptual": (
        "- **Implementation**: Define the necessary structs/traits in `src/` but focus on the 'why' in documentation.\n"
        "- **Tests**: Usually skipped. Evaluation is based on `cargo check` of the interface and the written response."
    ),
}


async def create_exam(
    model: AgentsSDKModel,
    project_repo: GitRepository,
    library_repo: GitRepository,
    topic: LearningTopic,
    image_name: str,
) -> CodingExam:
    """Create a new coding exam based on the provided project and topic.

    This function orchestrates a two-phase process using AgentWrapper:
    1.  **Generate Solution**: An agent explores the codebase and implements a full solution
        (including question.md, rubric.md, and tests) based on the topic.
    2.  **Generate Problem**: The agent reverts the solution code to a "problem" state,
        leaving scaffolding and failing tests, without touching question/rubric.

    The result is a git history where the "Problem" commit is the child of the "Solution" commit.
    """
    logger.info(
        f"Cloning template from {project_repo.local_dir} and library from {library_repo.local_dir}"
    )

    try:
        injections = {library_repo.local_dir: "repos/library"}

        # Use TempWorkspace context manager
        # Note: project_repo.local_dir is the template
        with TempWorkspace(
            template_dir=project_repo.local_dir,
            injections=injections,
            prefix="exam_creator_",
            copy_method="clone",
        ) as work_dir:
            logger.info(f"Created temp workspace at {work_dir}")

            # Configure git user for the temp repo (TempWorkspace initializes git, but we set user)
            workspace_repo = GitRepository(local_dir=work_dir)
            workspace_repo.run_git(["config", "user.email", "yosemat.beta@gmail.com"])
            workspace_repo.run_git(["config", "user.name", "yotaro-shimose"])

            # Initialize Runtime (Persistent for both phases)
            img_name = image_name or os.getenv("OPENHANDS_IMAGE_NAME", "coder-mcp")
            async with RustCodingEnvironment(
                workspace_dir=work_dir, image_name=img_name
            ) as runtime:
                # Initialize AgentWrapper with Specialized Prompt
                agent = AgentWrapper[str].create(
                    name="SyllabusWorker",
                    instructions=EXAM_CREATOR_SYSTEM_PROMPT,
                    model=model,
                    mcp_servers=[runtime.server],
                    model_settings=ModelSettings(
                        tool_choice="auto", parallel_tool_calls=True
                    ),
                )

                # Phase 1: Generate Solution
                logger.info("Phase 1: Generating Solution...")
                # Select the specific instruction
                mode_extra = MODE_INSTRUCTIONS.get(topic.eval_mode, "")

                # Construct the dynamic prompt
                solution_prompt = f"""\
**Role:** Senior Rust Engineer & Pedagogical Expert.
**Context:** Creating a '{topic.eval_mode}' Gold Standard exercise for Topic: {topic.title}.

**Pedagogical Requirements ({topic.eval_mode} mode):**
{mode_extra}

**Step 1: Context Retrieval**
Read `{topic.source_reference}`. Identify target APIs: {", ".join(topic.api_surface)}.

**Step 2: Generate Infrastructure Files**
1. `question.md`: Clear problem statement. (For `imitation`, include the code scaffold here).
2. `rubric.md`: The 'Ground Truth' for the Evaluator. List specific API usage requirements.

3. **Implementation (`src/`)**: 
   - Implement the perfect, idiomatic solution within the crate's `src/` directory.
   - **Crucial**: Ensure all target functions, structs, and traits are marked `pub` so they can be imported by external tests.

4. **Integration Tests (`tests/`)**: 
   - (If required) Create tests in the `tests/` directory.
   - **Crucial**: These files MUST import definitions from your `src/` implementation using the crate name.
   - The tests must verify that the public interface of your solution works as expected.

**Strict Constraints:**
- CPU-only (Ubuntu).
- Do not modify `repos/library`.
- **Dependency Management**: When adding dependencies to `Cargo.toml`, you MUST use the registry version (e.g., `numrs2 = "0.1.0"`) rather than a local path (e.g., `path = "..."`). Treat the library as if it is already published on crates.io.
- Ensure `Cargo.toml` is configured so that the library crate name is correctly defined for integration tests.
"""

                res_wrapper = await agent.run(solution_prompt, max_turns=30)
                history: list[TResponseInputItem] = res_wrapper.result.to_input_list()

                # 3.1 Commit Solution State
                logger.info("Committing Solution State...")
                workspace_repo.add(".")

                status = workspace_repo.run_git(["status"])
                logger.debug(f"Git Status before Solution commit:\n{status}")

                if not workspace_repo.run_git(["status", "--porcelain"]):
                    raise RuntimeError("Agent failed to produce solution changes")

                workspace_repo.commit("Exam Solution: Reference Implementation")
                solution_commit = workspace_repo.rev_parse("HEAD")
                logger.info(f"Solution Commit: {solution_commit}")

                # Phase 2: Generate Problem
                logger.info("Phase 2: Generating Problem...")
                problem_prompt = f"""\
**Current Task: Strip Solution (Mode: {topic.eval_mode})**

1. **Hollow Out Logic (src/)**: 
   - Replace the bodies of the functions in `src/` with `todo!()`.
   - **STRICT RULE**: Do NOT change function signatures, trait definitions, or `pub` visibility. 
   - The interface in `src/` must remain exactly as it was so that the `tests/` can still import them.

2. **Preserve Validation (tests/ & rubric.md)**: 
   - Do NOT touch the `tests/` directory or `rubric.md`.
   - The import statements in the tests (e.g., `use crate::...`) must remain valid.

3. **Verification**: 
   - Run `cargo check`. It must pass (proving the interface is intact).
   {"- Run `cargo test`. They must FAIL with 'not yet implemented' errors." if topic.eval_mode == "functional" else "- (Tests skipped)."}
"""
                # Continue the conversation by appending the new user message
                new_message: EasyInputMessageParam = {
                    "role": "user",
                    "content": problem_prompt,
                    "type": "message",
                }
                # history includes the initial prompt and the agent's response(s) from Phase 1
                await agent.run(history + [new_message], max_turns=30)

                # 3.2 Commit Problem State
                logger.info("Committing Problem State...")
                workspace_repo.add(".")

                status = workspace_repo.run_git(["status"])
                logger.debug(f"Git Status before Problem commit:\n{status}")

                if not workspace_repo.run_git(["status", "--porcelain"]):
                    raise RuntimeError("Agent failed to produce problem changes")

                workspace_repo.commit("Exam Problem: Initial State")
                problem_commit = workspace_repo.rev_parse("HEAD")
                logger.info(f"Problem Commit: {problem_commit}")

                # Retrieve question and rubric content
                question = (work_dir / "question.md").read_text()
                rubric = (work_dir / "rubric.md").read_text()

                # Construct Result
                sanitized_title = "".join(
                    c if c.isalnum() or c == "_" else "_" for c in topic.title.lower()
                ).replace("__", "_")
                exam_id = f"exam_{sanitized_title}_{problem_commit[:7]}"

                exam = CodingExam(
                    id=exam_id,
                    image_name=image_name
                    or os.getenv("OPENHANDS_IMAGE_NAME", "coder-mcp"),
                    project=GitRepository(local_dir=work_dir),
                    library=library_repo,
                    solution_commit=solution_commit,
                    problem_commit=problem_commit,
                    question=question,
                    eval_rubric=rubric,
                )

                # 3.3 Push to Original Repo
                logger.info("Pushing commits to original repository...")
                branch_name = f"exam-{exam.id}"
                workspace_repo.run_git(
                    ["push", "origin", f"HEAD:refs/heads/{branch_name}"]
                )
                logger.info(f"Pushed to branch {branch_name}")

                return exam

    except Exception as e:
        logger.error(f"Failed to create exam: {e}")
        raise e
