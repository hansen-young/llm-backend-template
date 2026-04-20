from abc import ABC, abstractmethod

from pydantic import BaseModel

from utils.types import Message, Messages


class Session(BaseModel):
    id: str
    messages: Messages

    def add_message(self, role: str, content: str | None, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})  # type: ignore


class BaseSessionService(ABC):
    @abstractmethod
    async def create(self, session_id: str) -> Session: ...

    @abstractmethod
    async def load(self, session_id: str) -> Session | None: ...

    @abstractmethod
    async def save(self, session_id: str, session: Session): ...

    @abstractmethod
    async def list(self) -> list[str]: ...

    @abstractmethod
    async def delete(self, session_id: str): ...


class InMemorySessionService(BaseSessionService):
    def __init__(self):
        self.sessions: dict[str, Session] = {}

    async def create(self, session_id: str) -> Session:
        if session_id in self.sessions:
            raise ValueError(f"Session `{session_id}` already exists")

        session = Session(id=session_id, messages=[])
        await self.save(session_id, session)

        return session

    async def load(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    async def save(self, session_id: str, session: Session):
        self.sessions[session_id] = session

    async def list(self) -> list[str]:
        return list(self.sessions.keys())

    async def delete(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
