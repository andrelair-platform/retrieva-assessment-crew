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
# Trust the minicloud CA for internal TLS (LiteLLM/Langfuse) — injected at build (ci-registry.md).
# NEVER commit the CA; pass --build-arg CA_CERT="$(cat ~/minicloud-ca.crt)" or from the CI secret.
ARG CA_CERT=""
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && if [ -n "$CA_CERT" ]; then \
         printf '%s' "$CA_CERT" > /usr/local/share/ca-certificates/minicloud-ca.crt && update-ca-certificates; \
       fi \
    && rm -rf /var/lib/apt/lists/*
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8080 \
    HOST=0.0.0.0
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
RUN useradd -r -u 10001 appuser
USER appuser
EXPOSE 8080
# Console script defined in pyproject [project.scripts]; runs uvicorn via main().
CMD ["retrieva-assessment-crew"]
