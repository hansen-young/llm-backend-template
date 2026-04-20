import asyncio
import inspect
import json
from typing import Any, Awaitable, Callable, Literal, ParamSpec, TypeAlias, TypeVar
from typing_extensions import TypedDict, Required, Optional

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionFunctionToolParam,
)
from openai.types.chat.chat_completion import Choice

# from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import TypeAdapter

Message: TypeAlias = ChatCompletionMessageParam
Messages: TypeAlias = list[ChatCompletionMessageParam]
ChatResponse: TypeAlias = ChatCompletion
ChatResponseMessage: TypeAlias = ChatCompletionMessage
# todo: ChatResponseChunk


P = ParamSpec("P")  # function parameters
R = TypeVar("R")  # function return type
Function: TypeAlias = Callable[P, R]
AsyncFunction: TypeAlias = Callable[P, Awaitable[R]]

FunctionTool: TypeAlias = Function | AsyncFunction
Tool: TypeAlias = FunctionTool


class FunctionToolDefinition(TypedDict, total=False):
    type: Required[Literal["function"]]
    name: Required[str]
    description: str
    input_schema: Required[dict[str, Any]]
    # output_schema: dict[str, Any]  # todo: support output schema in the future


ToolDefinition: TypeAlias = FunctionToolDefinition


class Toolset:
    def __init__(self):
        self._definitions: list[ToolDefinition] = []
        self._mapping: dict[str, Tool] = {}

    def add(self, tool: Tool, definition: ToolDefinition):
        self._definitions.append(definition)
        self._mapping[definition["name"]] = tool

    def get(self, name: str) -> Tool | None:
        return self._mapping.get(name)

    async def invoke(self, name: str, kwargs: dict) -> str:
        if not (tool := self.get(name)):
            raise KeyError(f"Tool '{name}' not found in the toolset.")

        if inspect.iscoroutinefunction(tool):
            result = await tool(**kwargs)
        else:
            result = await asyncio.to_thread(tool, **kwargs)

        return json.dumps(result)

    @property
    def definitions(self) -> list[ToolDefinition]:
        return self._definitions

    def __bool__(self):
        return len(self._definitions) > 0
