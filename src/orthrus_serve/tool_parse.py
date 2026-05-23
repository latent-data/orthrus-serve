from __future__ import annotations

import json
import re
import secrets

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_tags(text: str) -> str:
    return THINK_PATTERN.sub("", text).strip()


def parse_tool_calls(text: str) -> tuple[list[dict], str]:
    """
    Extract <tool_call>...</tool_call> blocks from generated text.

    Returns (tool_calls, remaining_content) where tool_calls is a list of
    OpenAI-format tool call dicts and remaining_content is the text with all
    tool_call blocks removed (trimmed).

    Blocks that are not valid JSON or are missing name/arguments are silently
    skipped so a partially-broken response still yields what it can.
    """
    tool_calls = []
    last_end = 0
    prefix_parts = []

    for match in TOOL_CALL_PATTERN.finditer(text):
        prefix_parts.append(text[last_end : match.start()])
        last_end = match.end()

        raw = match.group(1)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue

        name = parsed.get("name")
        arguments = parsed.get("arguments")
        if not isinstance(name, str) or arguments is None:
            continue

        tool_calls.append(
            {
                "id": f"call_{secrets.token_hex(12)}",
                "type": "function",
                "function": {
                    "name": name,
                    # arguments must be a JSON-encoded string per OpenAI spec
                    "arguments": json.dumps(arguments),
                },
            }
        )

    remaining = ("".join(prefix_parts) + text[last_end:]).strip()
    return tool_calls, remaining
