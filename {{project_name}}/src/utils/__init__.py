import inspect

from pydantic import TypeAdapter

from .types import AsyncFunction, Function, FunctionToolDefinition


def function_to_json_schema(func: Function | AsyncFunction) -> FunctionToolDefinition:
    name = func.__name__
    description = inspect.getdoc(func) or ""

    return {
        "type": "function",
        "name": name,
        "description": description,
        "input_schema": TypeAdapter(func).json_schema(),
    }
