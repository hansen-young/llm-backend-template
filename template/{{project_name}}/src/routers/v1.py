import json
from uuid import uuid4
from typing import Annotated

from fastapi import Depends
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

from bot import get_runner
from core.runners import BaseRunner

V1Router = APIRouter(prefix="/api/v1", tags=["v1"])


# --- Schema --- #


class ChatRequest(BaseModel):
    message: str
    session_id: Annotated[str, Field(default_factory=lambda: str(uuid4()))]


class ChatResponse(BaseModel):
    session_id: str
    message: str


# --- Routes --- #


@V1Router.post("/chat", tags=["chat"])
async def chat(body: ChatRequest, runner: Annotated[BaseRunner, Depends(get_runner)]):
    message = await runner.run(session_id=body.session_id, message=body.message)
    return ChatResponse(session_id=body.session_id, message=message)


@V1Router.post("/chat/stream", tags=["chat"], response_class=EventSourceResponse)
async def chat_stream(
    body: ChatRequest, runner: Annotated[BaseRunner, Depends(get_runner)]
):
    async for chunk in runner.run_stream(
        session_id=body.session_id, message=body.message
    ):
        yield chunk

    yield ServerSentEvent(
        data=json.dumps({"session_id": body.session_id}),
        event="end",
    )


@V1Router.get("/session/{session_id}", tags=["session"])
async def get_session(
    session_id: str, runner: Annotated[BaseRunner, Depends(get_runner)]
):
    session = await runner.session_service.load(session_id)
    if not session:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Session {session_id} not found"},
        )
    return {"session_id": session.id, "messages": session.messages}
