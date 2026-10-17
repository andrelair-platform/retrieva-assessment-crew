# syntax=docker/dockerfile:1
# retrieva-assessment-crew (RTV-80) — A2A DORA-assessment crew. uv multi-stage, non-root.
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Install deps first (cached layer), then the project.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime
# No custom CA: the crew's egress is all IN-CLUSTER HTTP (LiteLLM/Langfuse/Qdrant via ClusterIP) — it
# never speaks TLS, so httpx/openai use certifi's own (valid) bundle. The old minicloud-CA injection
# appended the cert WITHOUT a trailing newline and forced SSL_CERT_FILE at the result → httpx's
# ssl.load_verify_locations raised `[X509] PEM lib` at LLM construction. Dropped entirely (RTV-81b QA).
# If internal TLS is ever introduced, inject a WELL-FORMED CA (printf '%s\n') + set SSL_CERT_FILE to it.
ENV PATH="/app/.venv/bin:$PATH" \
    PORT=8080 \
    HOST=0.0.0.0
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
# uid 10001 WITH a home dir — CrewAI + litellm write cache/storage under $HOME; the old `useradd -r`
# created no home (/home/appuser missing) → `Permission denied: '/home/appuser'` at run (RTV-81b QA).
RUN useradd -u 10001 -m -d /home/appuser appuser
USER appuser
EXPOSE 8080
# Console script defined in pyproject [project.scripts]; runs uvicorn via main().
CMD ["retrieva-assessment-crew"]
