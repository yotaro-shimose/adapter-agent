import re


def extract_rust_code(text: str) -> str | None:
    match = re.search(r"```rust\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
