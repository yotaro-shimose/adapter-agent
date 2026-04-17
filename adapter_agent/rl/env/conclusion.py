from typing import Literal, TypedDict, cast


type SSConclusion = Literal[
    "success",
    "context_length_exceeded",
    "no_code_found",
    "no_text_content",
    "code_did_not_compile",
    "verification_failed",
    "verification_error",
    "environment_error",
    "rewire_failed",
    "max_turns_exceeded",
    "not_finished",
    "parse_failed",
    "multiple_tool_tags",
    "knowledge_normalization_failed",
    "redundant",
    "quota_exceeded",
]


class SSMetrics(TypedDict):
    success: float
    context_length_exceeded: float
    no_code_found: float
    no_text_content: float
    code_did_not_compile: float
    verification_failed: float
    verification_error: float
    environment_error: float
    rewire_failed: float
    max_turns_exceeded: float
    not_finished: float
    parse_failed: float
    multiple_tool_tags: float
    knowledge_normalization_failed: float
    redundant: float
    quota_exceeded: float


def conclusion_to_metrics(conclusion: SSConclusion) -> dict[str, float]:
    return cast(
        dict[str, float],
        SSMetrics(
            success=1.0 if conclusion == "success" else 0.0,
            context_length_exceeded=1.0
            if conclusion == "context_length_exceeded"
            else 0.0,
            no_code_found=1.0 if conclusion == "no_code_found" else 0.0,
            no_text_content=1.0 if conclusion == "no_text_content" else 0.0,
            code_did_not_compile=1.0 if conclusion == "code_did_not_compile" else 0.0,
            verification_failed=1.0 if conclusion == "verification_failed" else 0.0,
            verification_error=1.0 if conclusion == "verification_error" else 0.0,
            environment_error=1.0 if conclusion == "environment_error" else 0.0,
            rewire_failed=1.0 if conclusion == "rewire_failed" else 0.0,
            max_turns_exceeded=1.0 if conclusion == "max_turns_exceeded" else 0.0,
            not_finished=1.0 if conclusion == "not_finished" else 0.0,
            parse_failed=1.0 if conclusion == "parse_failed" else 0.0,
            multiple_tool_tags=1.0 if conclusion == "multiple_tool_tags" else 0.0,
            knowledge_normalization_failed=1.0
            if conclusion == "knowledge_normalization_failed"
            else 0.0,
            redundant=1.0 if conclusion == "redundant" else 0.0,
            quota_exceeded=1.0 if conclusion == "quota_exceeded" else 0.0,
        ),
    )

