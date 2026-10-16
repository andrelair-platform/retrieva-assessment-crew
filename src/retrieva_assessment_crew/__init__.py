"""retrieva-assessment-crew — the CrewAI DORA-assessment crew, exposed over A2A (RTV-80).

RTV-80 ships the service skeleton: an A2A server that advertises the `assess-arrangement` skill and
returns a contract-valid MOCK. RTV-81 replaces the mock with the real CrewAI crew; the A2A plumbing
stays. Entry point `main` runs the ASGI app with uvicorn.
"""

from __future__ import annotations

import logging

from .app import build_app
from .settings import load_settings
from .tracing import init_tracing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrieva-assessment-crew")

__all__ = ["build_app", "main"]


def main() -> None:
    import uvicorn

    settings = load_settings()
    init_tracing(settings)
    logger.info(
        "starting retrieva-assessment-crew on %s:%s (public_url=%s)",
        settings.host,
        settings.port,
        settings.public_url,
    )
    uvicorn.run(build_app(settings), host=settings.host, port=settings.port)
