"""Mock `assess-arrangement` responder (RTV-80).

Contract-first stand-in until RTV-81 wires the real CrewAI crew. Returns a schema-valid response that
mirrors the TS mock (`retrieva-backend/contracts/a2a/mock.ts`) and honours the product guardrail: no
evidence -> `insufficient_evidence`, never an invented verdict.
"""

from __future__ import annotations

from typing import Any

_CONTROL_REF = "DORA-ART28"


def build_mock_response(request: dict[str, Any]) -> dict[str, Any]:
    clauses = request.get("clauses") or []
    evidence = request.get("evidenceRefs") or []
    has_evidence = len(evidence) > 0
    return {
        "controlLibraryVersion": request.get("controlLibraryVersion", "unknown"),
        "findings": [
            {
                "controlRef": _CONTROL_REF,
                "clauseRefs": [c.get("ref", "") for c in clauses],
                "summary": (
                    "mock: clause maps to the control; evidence present"
                    if has_evidence
                    else "mock: clause maps to the control; no evidence supplied"
                ),
            }
        ],
        "verdicts": [
            {
                "controlRef": _CONTROL_REF,
                "verdict": "partial" if has_evidence else "insufficient_evidence",
                "confidence": 0.6 if has_evidence else 0.3,
            }
        ],
        "citations": (
            [{"evidenceId": evidence[0]["id"], "controlRef": _CONTROL_REF}] if has_evidence else []
        ),
        "confidence": 0.6 if has_evidence else 0.2,
        "insufficientEvidence": [] if has_evidence else [_CONTROL_REF],
    }
