from datetime import datetime

from fastapi import Request
from fastapi.routing import APIRouter


HealthRouter = APIRouter(prefix="/health", tags=["health"])


@HealthRouter.get("")
def health_check(req: Request):
    start_time: datetime = req.app.state.server_start_time
    return {"status": "ok", "duration": str(datetime.now() - start_time)}
