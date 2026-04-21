from abc import ABC, abstractmethod
from time import time
from uuid import uuid4
from typing import Self

from openai import AzureOpenAI
from openai.types.chat import ChatCompletionFunctionToolParam

from utils import create_message, function_to_json_schema
from utils.types import (
    ChatResponse,
    ChatResponseMessage,
    Choice,
    Messages,
    Tool,
    Toolset,
)


class AgentConfig:
    def __init__(
        self,
        system_prompt: str | None = None,
        toolset: Toolset | None = None,
    ):
        self.system_prompt = system_prompt
        self.toolset = toolset or Toolset()


class BaseAgent(ABC):
    def __init__(self, name: str, config: AgentConfig | None = None):
        self.name = name
        self.config = config or AgentConfig()

    @abstractmethod
    async def run(self, messages: Messages) -> ChatResponse: ...

    # @abstractmethod
    # async def run_async(self, messages: Messages): ...

    @abstractmethod
    def compile(self) -> Self: ...

    # --- Decorators --- #
    def tool(self, tool: Tool):
        definition = function_to_json_schema(tool)
        self.config.toolset.add(tool, definition)


class AzureOpenAIAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        azure_client: AzureOpenAI,
        azure_deployment: str,
        config: AgentConfig | None = None,
    ):
        super().__init__(name, config)
        self.client = azure_client
        self.deployment = azure_deployment
        self.kwargs = {}

    def _adapt_toolset(self, toolset: Toolset):
        tools: list[ChatCompletionFunctionToolParam] = []

        if not toolset:
            return tools

        for definition in toolset.definitions:
            if definition["type"] == "function":
                tools.append(
                    {
                        "type": definition["type"],
                        "function": {
                            "name": definition["name"],
                            "description": definition.get("description", ""),
                            "parameters": definition["input_schema"],
                        },
                    }
                )

        return tools

    def compile(self):
        config = self.config

        if config.toolset and (tools := self._adapt_toolset(config.toolset)):
            self.kwargs["tools"] = tools

        return self

    async def run(self, messages: Messages) -> ChatResponse:
        if self.config.system_prompt:
            messages = [create_message("system", self.config.system_prompt), *messages]

        return self.client.chat.completions.create(
            model=self.deployment, messages=messages, **self.kwargs
        )


class EchoAgent(BaseAgent):
    async def run(self, messages: Messages) -> ChatResponse:
        if not messages:
            raise RuntimeError("No messages provided to the agent")

        if messages[-1]["role"] != "user":
            raise RuntimeError("The last message must be from the user")

        if "content" not in messages[-1]:
            raise RuntimeError("The last user message must have content")

        if not isinstance(messages[-1]["content"], str):
            raise RuntimeError("The content of the last user message must be a string")

        reply_content = "Echo: " + messages[-1]["content"]

        return ChatResponse(
            id=str(uuid4()),
            choices=[
                Choice(
                    index=0,
                    finish_reason="stop",
                    message=ChatResponseMessage(
                        role="assistant",
                        content=reply_content,
                    ),
                )
            ],
            created=int(time()),
            model="echo-agent",
            object="chat.completion",
        )

    def compile(self):
        return self
