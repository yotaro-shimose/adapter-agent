import re
from dataclasses import dataclass

from tinker_cookbook.renderers.base import Message as TinkerMessage

_SUBMIT_PATTERN = re.compile(r"<submit>(.*?)</submit>", re.DOTALL)


@dataclass
class SolvedSubtask:
    instruction: str
    submit_code: str


def _message_text(message: TinkerMessage) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content if part["type"] == "text")


def extract_submit_from_trials(trials: list[TinkerMessage]) -> str | None:
    """Pull the last assistant message's <submit>...</submit> body, or None."""
    for message in reversed(trials):
        if message["role"] != "assistant":
            continue
        match = _SUBMIT_PATTERN.search(_message_text(message))
        if match:
            return match.group(1).strip()
        return None
    return None
