"""Runtime settings (RTV-80). Env-driven so the same image runs in dev and prod (12-factor).

Secrets (LiteLLM / Langfuse keys) are delivered by ESO → Vault at deploy time (RTV-80 slice B); they are
read from the environment here, never baked into the image. The crew is STATELESS and holds NO DB creds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    # The externally-reachable base URL advertised in the Agent Card (per-overlay env).
    public_url: str
    # LLM gateway (LiteLLM) — consumed by the real crew in RTV-81; present here for parity.
    litellm_base_url: str | None
    litellm_api_key: str | None
    # LLMOps backbone — Langfuse only (no LangSmith, platform standard).
    langfuse_host: str | None
    langfuse_public_key: str | None
    langfuse_secret_key: str | None


def load_settings() -> Settings:
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    return Settings(
        host=host,
        port=port,
        public_url=os.getenv("A2A_PUBLIC_URL", f"http://localhost:{port}"),
        litellm_base_url=os.getenv("LITELLM_BASE_URL"),
        litellm_api_key=os.getenv("LITELLM_API_KEY"),
        langfuse_host=os.getenv("LANGFUSE_HOST"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    )
