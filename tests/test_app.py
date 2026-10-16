"""RTV-80 — the A2A server serves the Agent Card + a health probe (reachable in-cluster)."""

from __future__ import annotations

from starlette.testclient import TestClient

from retrieva_assessment_crew.app import build_app
from retrieva_assessment_crew.contract import SKILL_ID


def client() -> TestClient:
    return TestClient(build_app())


def test_healthz_ok() -> None:
    res = client().get("/healthz")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_serves_agent_card_at_spec_path() -> None:
    res = client().get("/.well-known/agent-card.json")
    assert res.status_code == 200
    card = res.json()
    assert card["name"] == "retrieva-assessment-crew"
    assert any(s["id"] == SKILL_ID for s in card["skills"])


def test_legacy_agent_card_alias() -> None:
    res = client().get("/.well-known/agent.json")
    assert res.status_code == 200
    assert res.json()["skills"][0]["id"] == SKILL_ID
