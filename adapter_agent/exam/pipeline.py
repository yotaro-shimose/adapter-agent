import hashlib
import os
import subprocess
import traceback
from pathlib import Path

from oai_utils.agent import AgentsSDKModel, AgentWrapper

from adapter_agent.exam.creator import create_exam
from adapter_agent.exam.exam import CodingExam
from adapter_agent.exam.repository import GitRepository
from adapter_agent.exam.syllabus import (
    SYLLABUS_WORKER_PROMPT,
    LearningTopic,
    RawTopics,
)
from coder_mcp.runtime import DockerRuntime
from adapter_agent.async_util import gather_with_semaphore


# --- Helpers ---
def generate_topic_id(source_file: Path, title: str) -> str:
    """Generate a systematic, deterministic ID for a topic."""
    unique_str = f"{source_file.stem}_{title.lower().replace(' ', '_')}"
    hash_suffix = hashlib.md5(unique_str.encode()).hexdigest()[:8]
    return f"topic_{source_file.stem}_{hash_suffix}"


def save_topic_as_json(topic: LearningTopic, output_dir: Path):
    """Save a LearningTopic as a JSON file."""
    output_path = output_dir / f"{topic.id}.json"
    output_path.write_text(topic.model_dump_json(indent=2))
    print(f"Saved topic to {output_path}")


def save_exam_metadata(exam: CodingExam, output_dir: Path):
    """Save CodingExam metadata to JSON."""
    output_path = output_dir / f"{exam.id}.json"
    output_path.write_text(exam.model_dump_json(indent=2))
    print(f"Saved exam metadata to {output_path}")


# --- Phases ---
async def generate_exercises(
    model: AgentsSDKModel,
    library_repository: GitRepository,
    work_dir: Path,
    limit: int = 3,
    image_name: str | None = None,
) -> list[LearningTopic]:
    """
    Phase 1: Discover Rust files, generate curriculum abstracts and exercises.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # Clone library into repositories/numrs (needed for agent context)
    lib_dir = work_dir / "repositories" / "numrs"
    if not lib_dir.exists() and library_repository.exists:
        lib_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            # We assume git is available in the environment

            subprocess.run(
                ["git", "clone", str(library_repository.local_dir), str(lib_dir)],
                check=True,
            )
        except Exception as e:
            print(f"Clone failed (might already exist or error): {e}")

    # Output directory for topics
    topics_dir = work_dir / "curriculum" / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discovery
    src_dir = library_repository.local_dir / "src"
    if not src_dir.exists():
        print(f"No src directory found in {library_repository.local_dir}")
        return []

    all_files = list(src_dir.rglob("*.rs"))
    print(f"Discovered {len(all_files)} Rust files.")

    # Filter/Limit for experiment
    target_files = [
        f for f in all_files if "mod.rs" not in f.name and "lib.rs" not in f.name
    ]
    if not target_files:
        target_files = all_files

    # Take 'limit' for experiment
    experiment_batch = target_files[:limit]
    print(
        f"Processing batch of {len(experiment_batch)} files: {[f.name for f in experiment_batch]}"
    )

    collected_exercises = []

    img_name = image_name or os.getenv("OPENHANDS_IMAGE_NAME", "coder-mcp")
    async with DockerRuntime(
        workspace_dir=str(work_dir), image_name=img_name
    ) as img_runtime:
        mcp_server = img_runtime.server
        for rust_file in experiment_batch:
            print(f"--- Analyzing {rust_file.name} ---")

            # Read content
            try:
                code_content = rust_file.read_text()
            except Exception as e:
                print(f"Skipping {rust_file.name}: {e}")
                continue

            # Prepare Input
            user_msg = f"File Path: {rust_file.relative_to(library_repository.local_dir)}\n\nCode:\n```rust\n{code_content}\n```"

            # Create Ephemeral Agent
            agent = AgentWrapper[RawTopics].create(
                name=f"SyllabusWorker-{rust_file.stem}",
                instructions=SYLLABUS_WORKER_PROMPT,
                model=model,
                mcp_servers=[mcp_server],
                output_type=RawTopics,
            )

            # Run
            try:
                result_wrapper = await agent.run(user_msg, max_turns=5)
                raw_topics = result_wrapper.result.final_output

                if isinstance(raw_topics, RawTopics):
                    for raw in raw_topics.topics:
                        topic_id = generate_topic_id(rust_file, raw.title)
                        learning_topic = LearningTopic(id=topic_id, **raw.model_dump())
                        save_topic_as_json(learning_topic, topics_dir)
                        collected_exercises.append(learning_topic)
                else:
                    print(
                        f"Agent failed to return structured output for {rust_file.name}"
                    )

            except Exception as e:
                print(f"Error processing {rust_file.name}: {e}")

    print(f"Generated {len(collected_exercises)} exercises.")
    return collected_exercises


async def generate_exams(
    model: AgentsSDKModel,
    exercises: list[LearningTopic],
    library_repository: GitRepository,
    exam_template: GitRepository,
    work_dir: Path,
    image_name: str,
    push_to_origin: bool = True,
    max_concurrent: int = 1,
) -> list[CodingExam]:
    """
    Phase 2: Generate CodingExams for the provided exercises.
    """
    specs_dir = work_dir / "exams"
    specs_dir.mkdir(parents=True, exist_ok=True)

    async def _create_single_exam(ex: LearningTopic) -> CodingExam | None:
        print(f"Creating exam for exercise: {ex.id} ({ex.title})")
        try:
            # Use exam_template as the base project_repo for exams
            exam = await create_exam(
                model=model,
                project_repo=exam_template,
                library_repo=library_repository,
                topic=ex,
                image_name=image_name,
            )
            print(f"Exam created successfully: {exam.id}")

            # Save Metadata
            save_exam_metadata(exam, specs_dir)

            if push_to_origin:
                # Push to Original Repo (origin)
                branch_name = f"exam/{ex.id}"
                print(f"Pushing result to branch '{branch_name}' in template repo...")

                # exam.project is the temp repo. Its 'origin' is the exam_template.
                try:
                    exam.project.run_git(
                        ["push", "origin", f"HEAD:refs/heads/{branch_name}"]
                    )
                    print(f"✅ Successfully pushed exam to branch: {branch_name}")
                except Exception as push_err:
                    print(f"❌ Failed to push to origin: {push_err}")

            return exam
        except Exception as exam_error:
            # Log error including traceback, then continue to the next exercise
            print(f"Failed to create exam for {ex.id}: {exam_error}")
            traceback.print_exc()
            return None

    # Run concurrently using semaphore
    results = await gather_with_semaphore(
        [_create_single_exam(ex) for ex in exercises],
        max_concurrent=max_concurrent,
        progressbar=True,
    )

    # Filter out None results
    generated_exams = [res for res in results if res is not None]
    return generated_exams
