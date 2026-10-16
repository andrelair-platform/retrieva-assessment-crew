"""LLMOps tracing — Langfuse only (platform standard; no LangSmith). RTV-80.

Service-level wiring: initialise a Langfuse client at startup when keys are present, else a no-op. RTV-81
(the real crew) adds per-assessment spans; the scaffold only proves the client is wired.
"""

from __future__ import annotations

import logging
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
