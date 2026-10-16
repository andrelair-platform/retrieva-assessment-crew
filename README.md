# retrieva-assessment-crew

[![CI](https://github.com/andrelair-platform/retrieva-assessment-crew/actions/workflows/ci.yml/badge.svg)](https://github.com/andrelair-platform/retrieva-assessment-crew/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![A2A](https://img.shields.io/badge/A2A-spec%20v1.0-green)](https://a2a-protocol.org/)

> The **CrewAI DORA ICT third-party assessment crew** for [Retrieva](https://github.com/andrelair-platform/retrieva), exposed over the **A2A protocol**. It maps contract clauses to DORA controls, finds gaps, and scores coverage — evidence-grounded, and its output is **proposed, not authoritative** (the deterministic supervisor in `retrieva-backend` keeps a human in the loop). Runs on the self-hosted ktayl-solution k3s platform.

**Member of the Retrieva product** (RNCP39583 certification, board #2). The supervisor half lives in [`retrieva-backend`](https://github.com/andrelair-platform/retrieva-backend); this is the collaborative crew it delegates to over A2A.

---

## Status (RTV-80)

This repo currently ships the **service scaffold**: an A2A server that advertises the `assess-arrangement` skill and returns a **contract-valid mock**. The real CrewAI crew (clause-mapper → gap-analyst → control-scorer) lands in **RTV-81** — it replaces one method (`AssessArrangementExecutor._assess`); the A2A plumbing stays.

## Architecture

```
retrieva-backend (supervisor, TS)  ──A2A (HTTP/JSON, cluster-internal)──▶  retrieva-assessment-crew (this, Python)
  deterministic StateGraph, gap node                                        A2A server:
                                                                              GET  /.well-known/agent-card.json  (Agent Card)
                                                                              POST /                             (JSON-RPC: assess-arrangement)
                                                                              GET  /healthz
```

| Component | Choice |
|---|---|
| A2A server | `a2a-sdk[http-server]~=0.3.10` (`A2AStarletteApplication` + `DefaultRequestHandler`) — same line `crewai[a2a]` pins (RTV-81) |
| Skill | `assess-arrangement` (the DORA assessment) |
| Crew (RTV-81) | CrewAI via `crewai[a2a]` |
| LLM | LiteLLM gateway (runtime env) |
| Tracing | Langfuse only |
| Contract | vendored JSON Schema from `retrieva-backend` (source of truth) + runtime validation |

The crew is **stateless**, holds **no database credentials**, and persists nothing.

## Getting started

```bash
uv sync
uv run pytest -q                   # L1 tests
uv run retrieva-assessment-crew    # serve on http://localhost:8080
curl -s localhost:8080/healthz
curl -s localhost:8080/.well-known/agent-card.json
```

### Docker
```bash
docker build -t retrieva-assessment-crew:dev .
docker run --rm -p 8080:8080 retrieva-assessment-crew:dev
```
Pass the platform CA for internal TLS (RTV-81): `--build-arg CA_CERT="$(cat ~/minicloud-ca.crt)"`.

## The A2A contract

`assess-arrangement` — request `{ runId, node, arrangementId, controlLibraryVersion, clauses[], evidenceRefs[], context? }` → response `{ findings[], verdicts[], citations[], confidence, insufficientEvidence[] }`. The authoritative schema is `retrieva-backend/contracts/a2a/` (Zod → JSON Schema); the copy in `src/retrieva_assessment_crew/_schema/` is vendored and validated at runtime. Guardrail: no evidence → `insufficient_evidence`, never an invented verdict.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` / `HOST` | `8080` / `0.0.0.0` | bind address |
| `A2A_PUBLIC_URL` | `http://localhost:$PORT` | URL advertised in the Agent Card |
| `LITELLM_BASE_URL` / `LITELLM_API_KEY` | — | LLM gateway (used by the crew, RTV-81) |
| `LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | — | LLMOps tracing (Langfuse only) |

## License

MIT — see [LICENSE](LICENSE).
