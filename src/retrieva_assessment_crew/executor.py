"""A2A AgentExecutor for the `assess-arrangement` skill (RTV-80).

Reads the structured request from the inbound A2A message (a DataPart), produces a response, validates it
against the wire contract, and emits it as a DataPart. RTV-80 returns the MOCK; RTV-81 swaps
`_assess` for the real CrewAI crew (the only change needed — the A2A plumbing stays).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import DataPart, Part
from a2a.utils import new_agent_parts_message

from .contract import validate_request, validate_response
from .crew import AssessmentCrew

logger = logging.getLogger("retrieva-assessment-crew")


def _extract_request(context: RequestContext) -> dict[str, Any]:
    """Pull the JSON request from the inbound message's DataPart (fallback: text parsed as JSON)."""
    message = context.message
    if message and message.parts:
        for part in message.parts:
            root = getattr(part, "root", part)
            if getattr(root, "kind", None) == "data" and getattr(root, "data", None) is not None:
                return dict(root.data)
    text = context.get_user_input()
    return json.loads(text) if text else {}


class AssessArrangementExecutor(AgentExecutor):
    """Stateless executor. Persists nothing; holds no DB credentials. The crew is injected so the A2A
    plumbing is testable without an LLM (pass a stub `crew` with a `.run(request) -> dict`)."""

    # `crew` is any object exposing run(request: dict) -> dict (real AssessmentCrew, or a test stub).
    def __init__(self, crew: Any | None = None) -> None:
        self._crew = crew or AssessmentCrew()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        request = _extract_request(context)
        validate_request(request)  # reject anything off-contract before doing work
        # CrewAI's kickoff() is SYNC and refuses to run inside a running event loop ("Agent execution
        # was invoked synchronously from within a running event loop"). The A2A executor IS async, so
        # run the (blocking) crew in a worker thread — off the loop (also keeps the loop responsive).
        response = await asyncio.to_thread(self._assess, request)
        validate_response(response)  # never emit an off-contract response
        part = Part(root=DataPart(data=response))
        await event_queue.enqueue_event(new_agent_parts_message([part]))

    def _assess(self, request: dict[str, Any]) -> dict[str, Any]:
        # RTV-81: the real CrewAI crew (clause-mapper -> gap-analyst -> control-scorer). Same
        # request-in / contract-response-out signature as the RTV-80 mock it replaces; degrades to
        # insufficient-evidence when no LLM is configured.
        return self._crew.run(request)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("assess-arrangement runs to completion; cancellation is not supported")
