import inspect
from typing import cast

from pydantic import TypeAdapter

from .types import AsyncFunction, Function, FunctionToolDefinition, Message


def create_message(
    role: str,
    content: str | None,
    *,
    name: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None
) -> Message:
    message = {"role": role, "content": content}

    if role in ["assistant", "system", "user"] and isinstance(name, str):
        message["name"] = name

    if role == "assistant" and isinstance(tool_calls, list):
        message["tool_calls"] = tool_calls

    if role == "tool" and isinstance(tool_call_id, str):
        message["tool_call_id"] = tool_call_id

    return cast(Message, message)


def function_to_json_schema(func: Function | AsyncFunction) -> FunctionToolDefinition:
    name = func.__name__
    description = inspect.getdoc(func) or ""

    return {
        "type": "function",
        "name": name,
        "description": description,
        "input_schema": TypeAdapter(func).json_schema(),
    }
