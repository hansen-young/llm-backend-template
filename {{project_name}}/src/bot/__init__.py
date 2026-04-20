from functools import lru_cache
from core.runners import BaseRunner


@lru_cache(maxsize=1)
def get_runner() -> BaseRunner:
    from core.agents import EchoAgent
    from core.sessions import InMemorySessionService
    from core.runners import SimpleRunner

    agent = EchoAgent()
    session_service = InMemorySessionService()

    return SimpleRunner(agent=agent, session_service=session_service)
