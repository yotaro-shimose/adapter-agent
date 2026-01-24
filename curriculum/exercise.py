import uuid
from dataclasses import dataclass
from pathlib import Path

from agents import ModelSettings, RunContextWrapper, function_tool
from coder_mcp.runtime.rust_env import RustCodingEnvironment
from coder_mcp.runtime.temp_workspace import TempWorkspace
from oai_utils.agent import AgentRunFailure, AgentsSDKModel, AgentWrapper

from curriculum.database import Exercise, ExerciseDatabase, Topic
from curriculum.filter import judge_topic_usefulness

# --- PROMPTS ---

EXERCISE_GENERATOR_PROMPT_TEMPLATE = """You are an expert Rust Developer and benevolent Technical Educator.
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
   - The question should be practical and ask "How do I..." or "What is the best way to..."
   - The answer MUST be a polite, conversational explanation. Imagine you are teaching a student.
   - The answer must be in Markdown format.
3. **Verify Code**: If you include Rust code in the answer, you MUST verify it.
   - Write a standalone Rust program (e.g., `src/main.rs`).
   - Run it with `cargo run`.
   - Confirm the behavior is as expected.
4. Once verified, use the `register_exercise` tool to save the exercise.
5. If you believe the topic is already well-covered by existing exercises, or if you have generated enough exercises (at least 1, usually 2-3 for complex topics), you can say "finished".
6. Do NOT stop until you have considered if more exercises are needed.
</TASK>

<GUIDELINES>
- Exercises should be self-contained.
- Mention the library name `{library_name}` explicitly in the question.
- Avoid duplicate exercises.
- Focus on public API usage.
- **Tone**: Helpful, encouraging, and clear.
- **Format**:
  Q: How do I [topic]?
  A: To [topic], you should use the `[API]` function. This function allows you to...
     Here is an example:
     ```rust
     // code
     ```
     In this example, we first...
</GUIDELINES>

ENVIRONMENT:
All commands should be executed in `/workspace`. The library is at `/workspace/repos/library`.
"""


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
    await ctx.db.add_exercise(exercise)
    return f"Exercise '{exercise_id}' registered successfully."


async def get_existing_exercises_text(db: ExerciseDatabase, topic_id: str) -> str:
    exercises = db.get_exercises_by_topic(topic_id)
    if not exercises:
        return "None"

    text = ""
    for i, e in enumerate(exercises):
        text += f"Exercise {i + 1}:\nQ: {e.question}\nA: {e.answer}\n\n"
    return text


async def run_exercise_generation_for_topic(
    model: AgentsSDKModel,
    topic: Topic,
    exercise_db: ExerciseDatabase,
    library_summary: str,
    library_name: str,
    boilerplate_dir: Path,
    library_path: Path,
):
    print(f"\n>>> Evaluating topic: {topic.title} ({topic.id})")

    # 1. Judge topic usefulness
    judgment = await judge_topic_usefulness(
        model=model,
        topic=topic,
        library_summary=library_summary,
        library_name=library_name,
    )

    if not judgment.useful:
        print(f"[SKIP] Topic {topic.id}: {topic.title} - {judgment.description}")
        return

    print(f"[KEEP] Topic {topic.id}: {topic.title} - {judgment.description}")
    print(f"    -> Starting exercise generation for: {topic.title}")

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
    async with TempWorkspace(
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
                except AgentRunFailure as e:
                    print(f"Error during exercise generation for {topic.id}: {e}")
