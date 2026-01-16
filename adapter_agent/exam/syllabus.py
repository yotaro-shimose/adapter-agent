from typing import Literal
from pydantic import BaseModel, Field


# --- Data Structures ---
class RawTopic(BaseModel):
    title: str = Field(description="Short title of the topic")
    description: str = Field(description="Specific API or logic to be mastered")
    rationale: str = Field(description="Why this is important")
    eval_mode: Literal["imitation", "conceptual", "functional"]
    api_surface: list[str] = Field(description="List of primary functions/traits used")
    source_reference: str = Field(
        description="Path to the reference document file (relative to the project root)"
    )


class RawTopics(BaseModel):
    description: str = Field(description="Overview of coverage and prerequisites")
    topics: list[RawTopic] = Field(description="List of topics in this module")


class LearningTopic(RawTopic):
    id: str = Field(description="Systematic unique identifier for the topic")


# --- Prompts ---
SYLLABUS_WORKER_PROMPT = """\
**Role:** You are a Senior Curriculum Architect for Rust Systems.
**Objective:** Exhaustively analyze the provided Rust source code and produce a `RawTopics` object. This object serves as a pedagogical blueprint for downstream agents to generate detailed, verifiable exams.

**Core Directive: Deep API Coverage**
Conduct a comprehensive audit of the provided file. You must ensure the **entire public API surface** is covered. Do not bundle unrelated features; create distinct `RawTopic` objects for different logical capabilities (e.g., separate algorithms, trait implementations, or specialized data structures).

**Task:**
Generate a `RawTopics` object based on the source code using the following structure:

1. **description**: Provide a high-level overview of the module's coverage and specific prerequisites.
2. **topics**: A list of `RawTopic` objects, each containing:
    - **title**: A concise, professional name for the topic.
    - **description**: A clear statement of the specific API, trait, or logical concept to be mastered.
    - **rationale**: Explain why this is important for the learner and how the chosen evaluation mode proves mastery.
    - **eval_mode**: Assign the strategy that best suits the code: `imitation`, `conceptual`, or `functional`.
    - **api_surface**: A list of the primary functions, structs, or traits that must be utilized.
    - **source_reference**: The relative path to the provided source file from the project root.

**Constraints:**
- **Environment:** Strictly CPU-only and Ubuntu-compatible. Skip any features requiring GPU or non-standard FFI.
- **Visibility:** Analyze `pub` items only. Ignore internal/private logic.
- **Granularity:** Each `RawTopic` must be a self-contained unit of mastery.

**Reference JSON Structure:**
Your output must conform to this structure:
{
  "description": "Overview of the module...",
  "topics": [
    {
      "title": "Topic Title",
      "description": "Mastering the [API/Logic]...",
      "rationale": "This proves the learner understands...",
      "eval_mode": "functional",
      "api_surface": ["FunctionA", "StructB"],
      "source_reference": "src/path/to/file.rs"
    }
  ]
}

**Output:**
Return the result as a single `RawTopics` object.
"""
