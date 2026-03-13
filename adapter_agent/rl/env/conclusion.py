from typing import Literal

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
]
