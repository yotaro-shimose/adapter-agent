import asyncio
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from agents import ModelSettings, RunContextWrapper, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from agents.tracing import add_trace_processor
from coder_mcp.runtime.rust_env import RustCodingEnvironment
from coder_mcp.runtime.temp_workspace import TempWorkspace
from dotenv.main import load_dotenv
from oai_utils.agent import AgentWrapper
from oai_utils.tracing import AgentContentPrinter

from topic_db import Topic, TopicDatabase, Exercise, ExerciseDatabase


@dataclass
class ContextType:
    db: ExerciseDatabase
    topic: Topic


@function_tool
async def register_exercise(
    wrapper: RunContextWrapper[ContextType],
    question: str,
    answer: str,
) -> str:
    """Register a new exercise for the current topic."""
    ctx = wrapper.context
    exercise_id = str(uuid.uuid4())
    exercise = Exercise(
        id=exercise_id,
        topic_id=ctx.topic.id,
        question=question,
        answer=answer,
    )
    print(f"  -> Registering exercise: {exercise_id}")
    ctx.db.add_exercise(exercise)
    return f"Exercise '{exercise_id}' registered successfully."


# --- PROMPTS ---

EXERCISE_GENERATOR_PROMPT_TEMPLATE = """You are an expert Rust Developer and Technical Writer.
Your task is to create exercises (Question/Answer pairs) for a specific learning topic in the `{library_name}` library.

<TOPIC>
ID: {topic_id}
Title: {topic_title}
APIs: {related_apis}
Description: {topic_description}
</TOPIC>

<CONTEXT>
Library Summary:
{library_summary}

Existing Exercises for this topic:
{existing_exercises}
</CONTEXT>

<TASK>
1. Read the relevant source files and examples in `repos/library` to understand the topic deeply.
2. Design a Question and Answer pair that helps a user learn this topic.
   - The question should be practical.
   - The answer should include a correct, idiomatic Rust code snippet.
3. **Verify Code**: If you include Rust code in the answer, you MUST verify it.
   - Write a standalone Rust program (e.g., `src/bin/verify_exercise.rs`) using `write_to_file`.
   - Run it with `cargo run --bin verify_exercise` using `run_command`.
   - Fix any errors until it works.
4. Once verified, use the `register_exercise` tool to save the exercise.
5. If you believe the topic is already well-covered by existing exercises, or if you have generated enough exercises (at least 1, usually 2-3 for complex topics), you can say "finished".
6. Do NOT stop until you have considered if more exercises are needed.
</TASK>

<GUIDELINES>
- Exercises should be self-contained.
- Mention the library name `{library_name}` explicitly in the question.
- Avoid duplicate exercises.
- Focus on public API usage.
</GUIDELINES>

ENVIRONMENT:
All commands should be executed in `/workspace`. The library is at `/workspace/repos/library`.
"""


async def get_existing_exercises_text(db: ExerciseDatabase, topic_id: str) -> str:
    exercises = db.get_exercises_by_topic(topic_id)
    if not exercises:
        return "None"

    text = ""
    for i, e in enumerate(exercises):
        text += f"Exercise {i + 1}:\nQ: {e.question}\nA: {e.answer}\n\n"
    return text


async def run_exercise_generation_for_topic(
    model,
    topic: Topic,
    exercise_db: ExerciseDatabase,
    library_summary: str,
    library_name: str,
    boilerplate_dir: Path,
    library_path: Path,
):
    print(f"\n>>> Generating exercises for topic: {topic.title} ({topic.id})")

    existing_exercises = await get_existing_exercises_text(exercise_db, topic.id)

    prompt = EXERCISE_GENERATOR_PROMPT_TEMPLATE.format(
        library_name=library_name,
        topic_id=topic.id,
        topic_title=topic.title,
        related_apis=", ".join(topic.related_apis),
        topic_description=topic.description,
        library_summary=library_summary,
        existing_exercises=existing_exercises,
    )

    injections = {
        library_path: "repos/library",
    }

    # Prepare temp workspace for each topic to avoid interference
    with TempWorkspace(
        template_dir=boilerplate_dir,
        injections=injections,
        prefix=f"gen_ex_{topic.id}_",
    ) as sandbox_dir:
        async with RustCodingEnvironment(workspace_dir=sandbox_dir) as rust_env:
            async with rust_env.coder_mcp() as coder_mcp:
                agent = AgentWrapper[str].create(
                    name=f"ExerciseGen-{topic.id}",
                    instructions=prompt,
                    model=model,
                    mcp_servers=[coder_mcp],
                    tools=[register_exercise],
                    model_settings=ModelSettings(parallel_tool_calls=True),
                )

                ctx = ContextType(db=exercise_db, topic=topic)

                try:
                    await agent.run(
                        f"Generate exercises for topic '{topic.id}'.",
                        context=ctx,
                        max_turns=30,
                    )
                except Exception as e:
                    print(f"Error during exercise generation for {topic.id}: {e}")


async def main():
    load_dotenv()
    add_trace_processor(AgentContentPrinter())

    # Config
    model_name = "gemini/gemini-3-flash-preview"
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set.")
        return

    library_name = "numrs"
    workspace_dir = Path("workspace_new_curriculum").resolve()
    library_path = Path("repositories/numrs").resolve()
    boilerplate_dir = Path("templates/rust_boilerplate").resolve()

    if not workspace_dir.exists():
        print(
            f"Error: Workspace {workspace_dir} does not exist. Run new_curriculum.py first."
        )
        return

    topic_db = TopicDatabase(db_path=str(workspace_dir / "topics.json"))
    exercise_db = ExerciseDatabase(db_path=str(workspace_dir / "exercises.json"))

    summary_path = workspace_dir / "library_summary.md"
    if not summary_path.exists():
        print(f"Error: Library summary not found at {summary_path}")
        return
    library_summary = summary_path.read_text()

    if not topic_db.topics:
        print("No topics found in database.")
        return

    # Process the first 3 topics
    selected_topics = topic_db.topics[:3]
    print(f"Starting with topics: {[t.id for t in selected_topics]}")

    model = LitellmModel(model=model_name, api_key=api_key)

    for topic in selected_topics:
        await run_exercise_generation_for_topic(
            model=model,
            topic=topic,
            exercise_db=exercise_db,
            library_summary=library_summary,
            library_name=library_name,
            boilerplate_dir=boilerplate_dir,
            library_path=library_path,
        )

    print("\n=== Exercises Generated ===")
    for e in exercise_db.exercises:
        print(f"- Topic: {e.topic_id}")
        print(f"  Q: {e.question[:50]}...")


if __name__ == "__main__":
    asyncio.run(main())
