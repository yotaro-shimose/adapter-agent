---
name: code_quality
description: Core principles for code quality in this repository, including DRY, structured objects, single responsibility, and modern type hinting.
---

# Code Quality Philosophy

This document outlines the core principles for code quality in this repository. All contributions should adhere to these guidelines to ensure maintainability, readability, and consistency.

## 1. Don't Repeat Yourself (DRY)
- Avoid redundant code. If you find yourself copying and pasting logic, abstract it into a reusable function, class, or module.

## 2. Object over Container
- Prefer structured objects over complex nested containers (e.g., avoid `tuple[A, B, dict[str, int | float]]`).
- **Pydantic BaseModel**: Use when fields are entirely JSON serializable.
- **Dataclass**: Use when fields are not necessarily JSON serializable.
- **Exceptions**: Use containers only when required by external APIs or when keys are inherently dynamic and unknown beforehand.

## 3. Single Responsibility
- Each function should perform one clear task.
- Breakdown large functions into smaller, well-named helper functions.
- Aim for "Natural English" readability—it should be clear what a function does just by reading its name and flow.

## 4. Avoid Inner Functions
- Inner (nested) functions are generally discouraged.
- **Exception**: Use them only when a function depends on a vast number of variables from the outer scope, where passing those variables as arguments would significantly degrade readability or performance.

## 5. Modern Type Hinting
- Prefer `T | None` over `Optional[T]`.
- Use built-in `list`, `dict`, and `tuple` (Python 3.9+) instead of importing `List`, `Dict`, `Tuple` from `typing`.

## 6. Standard Import Placement
- All `import` statements must be at the very top of the file unless there is a critical technical reason for a delayed import (e.g., avoiding circular dependencies in specific contexts).

## 7. Concise Documentation
- Avoid verbose, "chatty" comments.
- Do not include long internal monologues or thought processes within the code comments.
- Focus on *why* something is done if it's non-obvious, rather than *what* is being done (which should be clear from the code itself).

## 8. Factory Methods for Complex Initialization
- If an object has a complex initialization pattern (e.g., deriving fields from other objects or performing calculations), prefer using `@classmethod` factory methods over putting logic in `__init__` or externalizing it.
- This keeps the construction logic encapsulated and the `__init__` (or dataclass default) clean.
- Example: `RLStepResult.from_task_rollouts(task_results)`
