import asyncio
import warnings
from pathlib import Path

from agents import add_trace_processor
from dotenv import load_dotenv
from oai_utils.tracing import AgentContentPrinter

from adapter_agent.model_helper import get_gemini
from curriculum.database import ExerciseDatabase, TopicDatabase
from curriculum.exercise import run_exercise_generation_for_topic

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


async def main():
    load_dotenv()
    add_trace_processor(AgentContentPrinter())

    try:
        model = get_gemini()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure GEMINI_API_KEY is set in your .env file.")
        return

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

    # Select a single topic for debugging
    # You can change the index or filter by ID to select a specific topic
    selected_topic = topic_db.topics[0]
    print(f"DEBUG: Selected topic: {selected_topic.title} ({selected_topic.id})")

    await run_exercise_generation_for_topic(
        model=model,
        topic=selected_topic,
        exercise_db=exercise_db,
        library_summary=library_summary,
        library_name=library_name,
        boilerplate_dir=boilerplate_dir,
        library_path=library_path,
    )


if __name__ == "__main__":
    asyncio.run(main())
