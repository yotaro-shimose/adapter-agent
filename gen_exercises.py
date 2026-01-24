import asyncio
import warnings
from pathlib import Path

from agents.tracing import add_trace_processor
from dotenv.main import load_dotenv
from oai_utils.tracing import AgentContentPrinter

from adapter_agent.async_util import gather_with_semaphore
from adapter_agent.model_helper import get_local_qwen
from curriculum.database import ExerciseDatabase, TopicDatabase
from curriculum.exercise import run_exercise_generation_for_topic

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


async def main():
    load_dotenv()
    add_trace_processor(AgentContentPrinter())

    # Config
    model = get_local_qwen()
    experiment_name = "exp_1768614687"
    experiment_dir = Path("experiments").resolve() / experiment_name

    library_name = "numrs"
    topic_path = experiment_dir / f"{library_name}_topics.json"
    exercise_path = experiment_dir / f"{library_name}_exercises.json"
    workspace_dir = experiment_dir / "workspace"
    library_path = workspace_dir / "repos/library"
    boilerplate_dir = Path("templates/rust_template").resolve()

    if not workspace_dir.exists():
        print(
            f"Error: Workspace {workspace_dir} does not exist. Run new_curriculum.py first."
        )
        return

    topic_db = TopicDatabase(db_path=topic_path)
    exercise_db = ExerciseDatabase(db_path=exercise_path)

    summary_path = experiment_dir / "library_summary.md"
    if not summary_path.exists():
        print(f"Error: Library summary not found at {summary_path}")
        return
    library_summary = summary_path.read_text()

    if not topic_db.topics:
        print("No topics found in database.")
        return

    # Process all topics concurrently
    selected_topics = topic_db.topics

    # Filter out topics that already have exercises
    pending_topics = []
    skipped_count = 0
    for topic in selected_topics:
        existing = exercise_db.get_exercises_by_topic(topic.id)
        if existing:
            skipped_count += 1
        else:
            pending_topics.append(topic)

    if skipped_count > 0:
        print(f"Skipping {skipped_count} topics that already have exercises.")

    selected_topics = pending_topics
    print(
        f"Starting execution for {len(selected_topics)} topics (filtering + generation)..."
    )

    await gather_with_semaphore(
        [
            run_exercise_generation_for_topic(
                model=model,
                topic=topic,
                exercise_db=exercise_db,
                library_summary=library_summary,
                library_name=library_name,
                boilerplate_dir=boilerplate_dir,
                library_path=library_path,
            )
            for topic in selected_topics
        ],
        max_concurrent=20,
    )

    print("\n=== Exercises Generated ===")
    for e in exercise_db.exercises:
        print(f"- Topic: {e.topic_id}")
        print(f"  Q: {e.question[:50]}...")


if __name__ == "__main__":
    asyncio.run(main())
