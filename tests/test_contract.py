"""RTV-80 — the mock responder honours the A2A `assess-arrangement` wire contract (vendored schema)."""

from __future__ import annotations

import pytest
from jsonschema import ValidationError

from retrieva_assessment_crew.contract import (
    request_schema,
    response_schema,
    validate_request,
    validate_response,
)
from retrieva_assessment_crew.mock import build_mock_response

VALID_REQUEST = {
    "runId": "11111111-1111-4111-8111-111111111111",
    "node": "gap",
    "arrangementId": "22222222-2222-4222-8222-222222222222",
    "controlLibraryVersion": "dora-2026.1",
    "controls": [
        {
            "ref": "DORA-28.8-EXIT-STRATEGY",
            "doraArticleRef": "Article 28(8)",
            "title": "Exit strategy",
            "description": "Documented exit plan",
        }
    ],
    "clauses": [{"ref": "§12.3", "text": "The provider shall maintain an exit plan."}],
    "evidenceRefs": [
        {"id": "ev-1", "collection": "compliance_kb", "source": "MSA.pdf", "text": "exit plan text"}
    ],
}


def test_valid_request_passes() -> None:
    validate_request(VALID_REQUEST)  # must not raise


def test_bad_uuid_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_request({**VALID_REQUEST, "runId": "not-a-uuid"})


def test_missing_required_field_rejected() -> None:
    bad = {k: v for k, v in VALID_REQUEST.items() if k != "controlLibraryVersion"}
    with pytest.raises(ValidationError):
        validate_request(bad)


def test_unknown_property_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_request({**VALID_REQUEST, "surprise": 1})


def test_mock_response_is_contract_valid() -> None:
    resp = build_mock_response(VALID_REQUEST)
    validate_response(resp)  # must not raise
    assert resp["verdicts"][0]["verdict"] == "partial"


def test_mock_defaults_to_insufficient_evidence_without_evidence() -> None:
    resp = build_mock_response({**VALID_REQUEST, "evidenceRefs": []})
    validate_response(resp)
    assert resp["verdicts"][0]["verdict"] == "insufficient_evidence"
    assert resp["insufficientEvidence"]  # non-empty


def test_vendored_request_schema_declares_the_fields_the_crew_relies_on() -> None:
    # Drift guard: the vendored schema is a hand-copied snapshot of the backend's generated schema.
    # A stale/wrong copy that dropped a field the crew reads would silently break the integration —
    # assert the request schema still declares the fields crew.py consumes (controls, clauses,
    # evidenceRefs, controlLibraryVersion). (A true byte-match guard needs a shared schema package.)
    props = request_schema().get("properties", {})
    for field in ("controls", "clauses", "evidenceRefs", "controlLibraryVersion", "runId"):
        assert field in props, f"vendored request schema is missing '{field}' — stale copy?"
    # a control carries the ref the crew echoes + maps into the register
    control_props = props["controls"]["items"]["properties"]
    assert "ref" in control_props and "title" in control_props


def test_vendored_response_schema_declares_verdicts_and_findings() -> None:
    props = response_schema().get("properties", {})
    for field in ("findings", "verdicts", "citations", "confidence", "insufficientEvidence"):
        assert field in props, f"vendored response schema is missing '{field}' — stale copy?"
