import json
import pytest
from orthrus_serve.tool_parse import parse_tool_calls, strip_think_tags


def test_single_tool_call():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Oxford"}}\n</tool_call>'
    calls, content = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_weather"
    # arguments must be a JSON-encoded string, not a dict
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"location": "Oxford"}
    assert calls[0]["id"].startswith("call_")
    assert content == ""


def test_multiple_tool_calls():
    text = (
        '<tool_call>{"name": "fn1", "arguments": {"a": 1}}</tool_call>'
        " some text "
        '<tool_call>{"name": "fn2", "arguments": {"b": 2}}</tool_call>'
    )
    calls, content = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "fn1"
    assert calls[1]["function"]["name"] == "fn2"
    assert content == "some text"


def test_no_tool_calls():
    text = "This is a plain response with no tool calls."
    calls, content = parse_tool_calls(text)
    assert calls == []
    assert content == text


def test_prefix_text_preserved():
    text = 'Sure, I will check.\n<tool_call>{"name": "lookup", "arguments": {}}</tool_call>'
    calls, content = parse_tool_calls(text)
    assert len(calls) == 1
    assert "Sure, I will check." in content


def test_invalid_json_skipped():
    text = "<tool_call>not valid json</tool_call>"
    calls, content = parse_tool_calls(text)
    assert calls == []


def test_missing_name_skipped():
    text = '<tool_call>{"arguments": {"x": 1}}</tool_call>'
    calls, content = parse_tool_calls(text)
    assert calls == []


def test_arguments_is_json_string():
    text = '<tool_call>{"name": "fn", "arguments": {"key": "value"}}</tool_call>'
    calls, _ = parse_tool_calls(text)
    assert isinstance(calls[0]["function"]["arguments"], str)
    parsed = json.loads(calls[0]["function"]["arguments"])
    assert parsed["key"] == "value"


def test_strip_think_tags():
    text = "<think>\nsome reasoning\n</think>\n\nActual response."
    assert strip_think_tags(text) == "Actual response."


def test_strip_think_tags_before_tool_call():
    text = "<think>\nreasoning\n</think>\n\n<tool_call>{\"name\": \"fn\", \"arguments\": {}}</tool_call>"
    calls, content = parse_tool_calls(strip_think_tags(text))
    assert len(calls) == 1
    assert content == ""


def test_strip_think_tags_no_think():
    assert strip_think_tags("plain text") == "plain text"


def test_unique_ids():
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    )
    calls, _ = parse_tool_calls(text)
    ids = [c["id"] for c in calls]
    assert ids[0] != ids[1]
