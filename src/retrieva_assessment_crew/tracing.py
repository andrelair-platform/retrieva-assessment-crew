"""LLMOps tracing — Langfuse only (platform standard; no LangSmith). RTV-80 wired the client; RTV-81b
adds the per-assessment instrumentation that makes crew quality observable + evaluable.

What RTV-81b closes:
  • the 3 crew agents (clause-mapper → gap-analyst → control-scorer) each become a Langfuse span with
    their task input + output + the run's token usage — so prompt tuning is watchable, not a black box;
  • the crew's trace is STITCHED to the supervisor's per-run trace via the session id = the A2A
    `runId` (the supervisor already traces with `sessionId = runId`), so one assessment reads as one
    connected session across the TS supervisor + the Python crew (not two disjoint traces).

Everything here is FAIL-SAFE: no keys → a no-op recorder; any tracing error is swallowed and never
breaks the A2A call (observability must not reduce availability).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .settings import Settings

logger = logging.getLogger("retrieva-assessment-crew")
_client: Any = None


def init_tracing(settings: Settings) -> Any:
    global _client
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        logger.info("Langfuse disabled (no keys) — tracing is a no-op")
        _client = None
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse tracing enabled (host=%s)", settings.langfuse_host or "cloud")
    except Exception as exc:  # noqa: BLE001 — tracing must never break the service
        logger.warning("Langfuse init failed — tracing disabled: %s", exc)
        _client = None
    return _client


def get_client() -> Any:
    return _client


def _trace_input(request: dict[str, Any]) -> dict[str, Any]:
    """A compact, non-sensitive summary of the request for the root span input."""
    return {
        "runId": request.get("runId"),
        "arrangementId": request.get("arrangementId"),
        "controlLibraryVersion": request.get("controlLibraryVersion"),
        "clauseCount": len(request.get("clauses") or []),
        "evidenceRefCount": len(request.get("evidenceRefs") or []),
    }


def _usage_details(crew_output: Any) -> dict[str, int] | None:
    """Map CrewAI's token_usage (UsageMetrics) → Langfuse usage_details. None when unavailable."""
    usage = getattr(crew_output, "token_usage", None)
    if usage is None:
        return None
    try:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
        if not (prompt or completion or total):
            return None
        return {"input": prompt, "output": completion, "total": total}
    except Exception:  # noqa: BLE001 — usage is best-effort
        return None


class _Recorder:
    """Yielded inside `assessment_trace`. Records per-agent spans + the final output on the root span.
    Every method is fail-safe: a null client (tracing off) or any error → silent no-op."""

    def __init__(self, client: Any, root: Any) -> None:
        self._client = client
        self._root = root

    def record_agents(self, crew_output: Any) -> None:
        """One Langfuse `agent` span per crew task output (clause-mapper / gap-analyst / control-scorer)."""
        if self._client is None:
            return
        try:
            for to in getattr(crew_output, "tasks_output", None) or []:
                name = str(getattr(to, "agent", None) or getattr(to, "name", None) or "agent")
                self._client.start_observation(
                    name=f"agent:{name}",
                    as_type="agent",
                    input=str(getattr(to, "description", "") or "")[:1000],
                    output=str(getattr(to, "raw", "") or "")[:4000],
                ).end()
        except Exception as exc:  # noqa: BLE001 — tracing must never break the run
            logger.warning("record_agents failed (non-fatal): %s", exc)

    def finish(self, output: dict[str, Any], crew_output: Any = None) -> None:
        """Set the final assessment output + token usage on the root span."""
        if self._client is None or self._root is None:
            return
        try:
            self._root.update(output=output, usage_details=_usage_details(crew_output))
        except Exception as exc:  # noqa: BLE001
            logger.warning("trace finish failed (non-fatal): %s", exc)


@contextmanager
def assessment_trace(request: dict[str, Any]) -> Iterator[_Recorder]:
    """Fail-safe Langfuse span for one crew assessment. No-op when tracing is disabled. Stitches to the
    supervisor via `session_id = runId`.

    Only tracing *setup* is guarded (→ no-op recorder on failure, so observability can't reduce
    availability). Exceptions from the CREW body propagate normally: the `with` blocks still end the
    spans (recording the error), and the crew's own try/except degrades to insufficient-evidence.
    """
    client = _client
    if client is None:
        yield _Recorder(None, None)
        return
    try:
        from langfuse import propagate_attributes

        run_id = str(request.get("runId") or "") or None
        attrs_cm = propagate_attributes(
            session_id=run_id, tags=["crew", "assess-arrangement"], trace_name="crew.assess-arrangement"
        )
    except Exception as exc:  # noqa: BLE001 — tracing setup must never break the crew
        logger.warning("crew tracing setup failed (non-fatal): %s", exc)
        yield _Recorder(None, None)
        return
    try:
        with (
            attrs_cm,
            client.start_as_current_observation(
                name="crew.assess-arrangement", as_type="chain", input=_trace_input(request)
            ) as root,
        ):
            yield _Recorder(client, root)
    finally:
        try:
            client.flush()
        except Exception:  # noqa: BLE001
            pass
