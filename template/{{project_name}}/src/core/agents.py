from abc import ABC, abstractmethod
from time import time
from uuid import uuid4
from typing import AsyncGenerator, Self

from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionFunctionToolParam

from utils import create_message, function_to_json_schema
from utils.types import (
    ChatResponse,
    ChatResponseChunk,
    ChatResponseMessage,
    Choice,
    ChoiceChunk,
    ChoiceDeltaChunk,
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

    @abstractmethod
    def run_async(
        self, messages: Messages
    ) -> AsyncGenerator[ChatResponseChunk, None]: ...

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
        azure_client: AsyncAzureOpenAI,
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

        return await self.client.chat.completions.create(
            model=self.deployment, messages=messages, **self.kwargs
        )

    async def run_async(self, messages: Messages):
        if self.config.system_prompt:
            messages = [create_message("system", self.config.system_prompt), *messages]

        async for chunk in await self.client.chat.completions.create(
            model=self.deployment, messages=messages, stream=True, **self.kwargs
        ):
            yield chunk


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

    async def run_async(self, messages: Messages):
        if not messages:
            raise RuntimeError("No messages provided to the agent")

        if messages[-1]["role"] != "user":
            raise RuntimeError("The last message must be from the user")

        if "content" not in messages[-1]:
            raise RuntimeError("The last user message must have content")

        if not isinstance(messages[-1]["content"], str):
            raise RuntimeError("The content of the last user message must be a string")

        from itertools import islice

        reply_content = "Echo: " + messages[-1]["content"]
        iterator = iter(reply_content)
        chunk_size = 2
        response_id = str(uuid4())

        # Step 1: Send initial chunk with filter results
        yield ChatResponseChunk(
            id="",
            choices=[],
            created=0,
            model="",
            object="chat.completion.chunk",
            service_tier=None,
            system_fingerprint=None,
            usage=None,
            prompt_filter_results=[
                {
                    "prompt_index": 0,
                    "content_filter_results": {
                        "hate": {"filtered": False, "severity": "safe"},
                        "jailbreak": {"detected": False, "filtered": False},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "safe"},
                    },
                }
            ],
        )

        # Step 2: Send chunk containing the role
        yield ChatResponseChunk(
            id=response_id,
            choices=[ChoiceChunk(index=0, delta=ChoiceDeltaChunk(role="assistant"))],
            created=int(time()),
            model="echo-agent",
            object="chat.completion.chunk",
        )

        # Step 3: Send content chunks
        while chunk := tuple(islice(iterator, chunk_size)):
            yield ChatResponseChunk(
                id=str(uuid4()),
                choices=[
                    ChoiceChunk(
                        index=0,
                        finish_reason="stop",
                        delta=ChoiceDeltaChunk(content="".join(chunk)),
                    )
                ],
                created=int(time()),
                model="echo-agent",
                object="chat.completion.chunk",
            )

    def compile(self):
        return self
