import json
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Generator, cast

from core.agents import BaseAgent
from core.sessions import BaseSessionService, Session

from utils.stream import ChoiceAggregator
from utils.types import Choice, ChatCompletionMessageToolCallUnion


class BaseRunner(ABC):
    def __init__(self, agent: BaseAgent, session_service: BaseSessionService):
        self.agent = agent.compile()
        self.session_service = session_service

    @abstractmethod
    async def run(self, session_id: str, message: str) -> str: ...

    @abstractmethod
    def run_stream(self, session_id: str, message: str) -> AsyncGenerator: ...


class SimpleRunner(BaseRunner):
    async def _invoke_tool(
        self, session: Session, tool_call: ChatCompletionMessageToolCallUnion
    ):
        try:
            if tool_call.type == "function":
                arguments: dict = json.loads(tool_call.function.arguments)
                result = await self.agent.config.toolset.invoke(
                    tool_call.function.name, arguments
                )
            else:
                result = "Unsupported tool type: " + tool_call.type

        except Exception as e:
            result = f"Error invoking tool: {str(e)}"

        session.add_message("tool", result, tool_call_id=tool_call.id)

    async def _handle_stop_reason(self, session: Session, choice: Choice):
        if not choice.message.content:
            raise RuntimeError("Agent response is empty")
        session.add_message("assistant", choice.message.content)

    async def _handle_tool_calls_reason(self, session: Session, choice: Choice):
        if not choice.message.tool_calls:
            raise RuntimeError("Tool calls information is missing in the response")

        session.add_message(
            choice.message.role,
            choice.message.content,
            tool_calls=choice.message.tool_calls,
        )

        for tool_call in choice.message.tool_calls:
            await self._invoke_tool(session, tool_call)

    def handoff_condition(self, session: Session) -> bool:
        return bool(session.messages) and session.messages[-1]["role"] == "assistant"

    async def run(self, session_id: str, message: str) -> str:
        if not (session := await self.session_service.load(session_id)):
            session = await self.session_service.create(session_id)

        session.add_message("user", message)

        while not self.handoff_condition(session):
            response = await self.agent.run(session.messages)
            choice = response.choices[0]

            if choice.message.role != "assistant":
                raise RuntimeError(
                    f"Unexpected message role: {choice.message.role}, expected 'assistant'"
                )

            if choice.finish_reason == "stop":
                await self._handle_stop_reason(session, choice)
            elif choice.finish_reason == "tool_calls":
                await self._handle_tool_calls_reason(session, choice)
            else:
                raise RuntimeError(f"Unexpected finish reason: {choice.finish_reason}")

        if "content" not in session.messages[-1]:
            raise RuntimeError("No valid assistant response found")

        await self.session_service.save(session_id, session)

        return cast(str, session.messages[-1]["content"])

    async def run_stream(self, session_id: str, message: str):
        if not (session := await self.session_service.load(session_id)):
            session = await self.session_service.create(session_id)

        session.add_message("user", message)
        aggregator = ChoiceAggregator()

        while not self.handoff_condition(session):
            aggregator.reset()

            async for chunk in self.agent.run_async(session.messages):
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                aggregator.update(choice)

                if aggregator.choice.message.role == "assistant":
                    # todo: if tool_calls is not None, content might be the LLM thought process, we might not want
                    #       to yield it directly.
                    if choice.delta.content:
                        yield choice.delta.content

            if aggregator:
                session.add_message(
                    aggregator.choice.message.role,
                    aggregator.choice.message.content,
                    tool_calls=aggregator.choice.message.tool_calls,
                )

                for tc in aggregator.choice.message.tool_calls or []:
                    await self._invoke_tool(session, tc)

            else:
                raise RuntimeError(
                    "Invalid response from agent: no content or tool calls"
                )
