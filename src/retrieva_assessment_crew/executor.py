"""A2A AgentExecutor for the `assess-arrangement` skill (RTV-80).

Reads the structured request from the inbound A2A message (a DataPart), produces a response, validates it
against the wire contract, and emits it as a DataPart. RTV-80 returns the MOCK; RTV-81 swaps
`_assess` for the real CrewAI crew (the only change needed — the A2A plumbing stays).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import DataPart, Part
from a2a.utils import new_agent_parts_message

from .contract import validate_request, validate_response
from .mock import build_mock_response

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
    """Deterministic, stateless executor. Persists nothing; holds no DB credentials."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        request = _extract_request(context)
        validate_request(request)  # reject anything off-contract before doing work
        response = self._assess(request)
        validate_response(response)  # never emit an off-contract response
        part = Part(root=DataPart(data=response))
        await event_queue.enqueue_event(new_agent_parts_message([part]))

    def _assess(self, request: dict[str, Any]) -> dict[str, Any]:
        # RTV-80: mock. RTV-81 replaces this with the CrewAI crew (clause-mapper -> gap-analyst ->
        # control-scorer), keeping the same request-in / contract-response-out signature.
        return build_mock_response(request)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("assess-arrangement runs to completion; cancellation is not supported")
