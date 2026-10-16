"""RTV-81a — the crew wiring + safe fallback, verified WITHOUT a live LLM.

What this proves locally: (1) with no LLM configured the crew degrades to a contract-valid
insufficient-evidence result (never crashes, never guesses); (2) the executor runs through the
injected crew and emits a contract-valid response. The real crew+LLM run (tuned prompts, Qdrant
evidence, eval) is RTV-81b, exercised on live dev where LiteLLM is reachable.
"""

from __future__ import annotations

from retrieva_assessment_crew.contract import validate_response
from retrieva_assessment_crew.crew import AssessmentCrew
from retrieva_assessment_crew.executor import AssessArrangementExecutor

REQUEST = {
    "runId": "11111111-1111-4111-8111-111111111111",
    "node": "gap",
    "arrangementId": "22222222-2222-4222-8222-222222222222",
    "controlLibraryVersion": "dora-2026.1",
    "clauses": [{"ref": "§12.3", "text": "The provider shall maintain an exit plan."}],
    "evidenceRefs": [{"id": "ev-1"}],
}


def test_no_llm_degrades_to_insufficient_evidence() -> None:
    # No LITELLM_* in the test env → _make_llm() returns None → safe fallback (no crewai import needed).
    out = AssessmentCrew().run(REQUEST)
    validate_response(out)  # contract-valid
    assert out["verdicts"][0]["verdict"] == "insufficient_evidence"
    assert out["confidence"] == 0.0
    assert out["insufficientEvidence"]
    assert out["controlLibraryVersion"] == "dora-2026.1"


def test_executor_runs_through_the_injected_crew() -> None:
    canned = {
        "controlLibraryVersion": "dora-2026.1",
        "findings": [{"controlRef": "DORA-ART28", "clauseRefs": ["§12.3"], "summary": "covered"}],
        "verdicts": [{"controlRef": "DORA-ART28", "verdict": "partial", "confidence": 0.6}],
        "citations": [{"evidenceId": "ev-1", "controlRef": "DORA-ART28"}],
        "confidence": 0.6,
        "insufficientEvidence": [],
    }

    class StubCrew:
        def __init__(self) -> None:
            self.called_with: dict | None = None

        def run(self, request: dict) -> dict:
            self.called_with = request
            return canned

    stub = StubCrew()
    ex = AssessArrangementExecutor(crew=stub)
    out = ex._assess(REQUEST)
    validate_response(out)
    assert stub.called_with == REQUEST
    assert out["verdicts"][0]["verdict"] == "partial"
