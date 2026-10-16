"""ASGI app factory (RTV-80).

Wires the A2A server (Agent Card + JSON-RPC handler + the assess-arrangement executor) and adds a plain
`/healthz` probe plus a `/.well-known/agent.json` alias (A2A v1.0 serves the card at
`/.well-known/agent-card.json`; the alias keeps older-path clients working).
"""

from __future__ import annotations

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .agent_card import build_agent_card
from .executor import AssessArrangementExecutor
from .settings import Settings, load_settings


def build_app(settings: Settings | None = None) -> Starlette:
    settings = settings or load_settings()
    card = build_agent_card(settings.public_url)

    handler = DefaultRequestHandler(
        agent_executor=AssessArrangementExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app: Starlette = A2AStarletteApplication(agent_card=card, http_handler=handler).build()

    async def healthz(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "retrieva-assessment-crew"})

    async def legacy_card(_request: Request) -> JSONResponse:
        return JSONResponse(card.model_dump(by_alias=True, exclude_none=True))

    # Public routes lists are stable across Starlette versions — safer than add_route().
    app.router.routes.append(Route("/healthz", healthz, methods=["GET"]))
    app.router.routes.append(Route("/.well-known/agent.json", legacy_card, methods=["GET"]))
    return app
