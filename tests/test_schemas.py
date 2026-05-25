import json
import pytest
from orthrus_serve.openai_schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    FunctionCall,
    Message,
    ModelCard,
    ModelList,
    ResponseMessage,
    ToolCall,
    ToolDefinition,
    FunctionDefinition,
    Usage,
)


def test_chat_completion_request_defaults():
    req = ChatCompletionRequest(
        model="orthrus-qwen3-8b",
        messages=[Message(role="user", content="hello")],
    )
    assert req.model == "orthrus-qwen3-8b"
    assert req.max_tokens == 2048
    assert req.stream is False
    assert req.tools is None


def test_chat_completion_request_model_required():
    """model is required per OpenAI spec — schema should reject if missing."""
    with pytest.raises(Exception):  # pydantic ValidationError
        ChatCompletionRequest(messages=[Message(role="user", content="hi")])


def test_chat_completion_request_with_tools():
    req = ChatCompletionRequest(
        model="orthrus-qwen3-8b",
        messages=[Message(role="user", content="what is the weather?")],
        tools=[
            ToolDefinition(
                function=FunctionDefinition(
                    name="get_weather",
                    description="Get current weather",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                )
            )
        ],
        tool_choice="auto",
    )
    assert len(req.tools) == 1
    assert req.tools[0].function.name == "get_weather"


def test_chat_completion_response_round_trip():
    resp = ChatCompletionResponse(
        id="chatcmpl-abc",
        model="orthrus-qwen3-8b",
        choices=[
            Choice(
                message=ResponseMessage(content="Hello!"),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    data = json.loads(resp.model_dump_json())
    assert data["id"] == "chatcmpl-abc"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] == 15


def test_tool_call_response():
    resp = ChatCompletionResponse(
        id="chatcmpl-xyz",
        model="orthrus-qwen3-8b",
        choices=[
            Choice(
                message=ResponseMessage(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_abc123",
                            function=FunctionCall(
                                name="get_weather",
                                arguments='{"location": "Oxford"}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
    )
    data = json.loads(resp.model_dump_json())
    tc = data["choices"][0]["message"]["tool_calls"][0]
    # arguments must be a string
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"location": "Oxford"}


def test_model_list():
    ml = ModelList(
        data=[ModelCard(id="orthrus-qwen3-8b", max_model_len=40960)]
    )
    data = json.loads(ml.model_dump_json())
    assert data["object"] == "list"
    assert data["data"][0]["max_model_len"] == 40960
    assert data["data"][0]["owned_by"] == "latent-data"


def test_stop_string_normalization():
    req = ChatCompletionRequest(
        model="orthrus-qwen3-8b",
        messages=[Message(role="user", content="hi")],
        stop="<|endoftext|>",
    )
    assert req.stop == "<|endoftext|>"
