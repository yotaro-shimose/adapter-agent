import re


def extract_rust_code(text: str) -> str:
    # Extract code from ```rust ... ``` block
    match = re.search(r"```rust\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback to any code block
    match = re.search(r"```\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

