from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI

from config import config
from routers.health import HealthRouter
from routers.v1 import V1Router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.server_start_time = datetime.now()
    yield


def create_app():
    app = FastAPI(title="My Chatbot API", lifespan=lifespan)
    app.include_router(HealthRouter)
    app.include_router(V1Router)
    return app
