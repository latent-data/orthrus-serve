from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoiceObject(BaseModel):
    type: Literal["function"]
    function: ToolChoiceFunction


ToolChoice = Literal["none", "auto", "required"] | ToolChoiceObject


class Message(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "orthrus-qwen3-8b"
    messages: list[Message]
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = None
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    max_tokens: int | None = 2048
    stop: list[str] | str | None = None
    stream: bool = False
    chat_template_kwargs: dict[str, Any] | None = None


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON-encoded string


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop", "tool_calls", "length"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "orthrus-qwen3-8b"
    choices: list[Choice]
    usage: Usage


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "orthrus-qwen3-8b"
    choices: list[StreamChoice]


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "latent-data"
    created: int = Field(default_factory=lambda: int(time.time()))
    max_model_len: int = 40960


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
