import json
from abc import ABC, abstractmethod
from typing import cast

from core.agents import BaseAgent
from core.sessions import BaseSessionService


class BaseRunner(ABC):
    def __init__(self, agent: BaseAgent, session_service: BaseSessionService):
        self.agent = agent.compile()
        self.session_service = session_service

    @abstractmethod
    async def run(self, session_id: str, message: str) -> str: ...

    # @abstractmethod
    # async def run_stream(self, session_id: str, message: str): ...


class SimpleRunner(BaseRunner):
    async def run(self, session_id: str, message: str) -> str:
        if not (session := await self.session_service.load(session_id)):
            session = await self.session_service.create(session_id)

        session.add_message("user", message)

        while session.messages[-1]["role"] != "assistant":
            response = await self.agent.run(session.messages)
            choice = response.choices[0]

            if choice.finish_reason == "stop":
                if not choice.message.content:
                    raise RuntimeError("Agent response is empty")
                session.add_message("assistant", choice.message.content)

            elif choice.finish_reason == "tool_calls":
                if not choice.message.tool_calls:
                    raise RuntimeError(
                        "Tool calls information is missing in the response"
                    )

                session.add_message(
                    choice.message.role,
                    choice.message.content,
                    tool_calls=choice.message.tool_calls,
                )

                for tool_call in choice.message.tool_calls:
                    try:
                        if tool_call.type == "function":
                            name = tool_call.function.name
                            arguments: dict = json.loads(tool_call.function.arguments)
                            result = await self.agent.config.toolset.invoke(
                                name, arguments
                            )
                        else:
                            result = "Unsupported tool type: " + tool_call.type

                        session.add_message("tool", result, tool_call_id=tool_call.id)
                    except Exception as e:
                        session.add_message(
                            "tool",
                            f"Error invoking tool: {str(e)}",
                            tool_call_id=tool_call.id,
                        )

            else:
                raise RuntimeError(f"Unexpected finish reason: {choice.finish_reason}")

        await self.session_service.save(session_id, session)

        return cast(str, session.messages[-1]["content"])
