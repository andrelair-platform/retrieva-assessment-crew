"""RTV-81a — the crew wiring + safe fallback, verified WITHOUT a live LLM.

What this proves locally: (1) with no LLM configured the crew degrades to a contract-valid
insufficient-evidence result (never crashes, never guesses); (2) the executor runs through the
injected crew and emits a contract-valid response. The real crew+LLM run (tuned prompts, Qdrant
evidence, eval) is RTV-81b, exercised on live dev where LiteLLM is reachable.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from a2a.types import DataPart, Part

from retrieva_assessment_crew.contract import validate_response
from retrieva_assessment_crew.crew import (
    AssessmentCrew,
    _canon_ref,
    _canonicalize,
    _ensure_all_controls,
)
from retrieva_assessment_crew.executor import AssessArrangementExecutor

REQUEST = {
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


def test_execute_runs_the_crew_off_the_event_loop() -> None:
    """The A2A `execute` is async; the crew's kickoff() is sync and refuses a running event loop, so
    execute must run it via asyncio.to_thread. This exercises the async path end-to-end (a stub crew —
    the real CrewAI-in-loop failure is an integration issue caught by the live QA gate)."""
    canned = {
        "controlLibraryVersion": "dora-2026.1",
        "findings": [{"controlRef": "DORA-ART30", "clauseRefs": ["§12.3"], "summary": "ok"}],
        "verdicts": [{"controlRef": "DORA-ART30", "verdict": "partial", "confidence": 0.6}],
        "citations": [],
        "confidence": 0.6,
        "insufficientEvidence": [],
    }

    class StubCrew:
        def run(self, request: dict) -> dict:
            return canned

    captured: list = []

    class StubQueue:
        async def enqueue_event(self, event: object) -> None:
            captured.append(event)

    inbound = SimpleNamespace(parts=[Part(root=DataPart(data=REQUEST))])
    ctx = SimpleNamespace(message=inbound, get_user_input=lambda: None)
    ex = AssessArrangementExecutor(crew=StubCrew())
    asyncio.run(ex.execute(ctx, StubQueue()))  # type: ignore[arg-type]
    assert len(captured) == 1  # a contract-valid response was emitted


def test_canon_ref_snaps_llm_formatting_to_the_catalog() -> None:
    # LLMs render unicode hyphens + abbreviate; the catalog taxonomy must survive that.
    assert _canon_ref("DORA‑ART30") == "DORA-ART30"  # non-breaking hyphen
    assert _canon_ref("dora art 28") == "DORA-ART28"
    assert _canon_ref("Article 29 (subcontracting)") == "DORA-ART29"
    assert _canon_ref("Something off-catalog") == "Something off-catalog"  # unknown → left as-is


def test_canonicalize_normalises_all_ref_fields() -> None:
    data = {
        "verdicts": [{"controlRef": "DORA‑ART30", "verdict": "partial"}],
        "findings": [{"controlRef": "art 28", "clauseRefs": [], "summary": ""}],
        "citations": [{"evidenceId": "ev-1", "controlRef": "ART‑29"}],
        "insufficientEvidence": ["dora-art28"],
    }
    out = _canonicalize(data)
    assert out["verdicts"][0]["controlRef"] == "DORA-ART30"
    assert out["findings"][0]["controlRef"] == "DORA-ART28"
    assert out["citations"][0]["controlRef"] == "DORA-ART29"
    assert out["insufficientEvidence"] == ["DORA-ART28"]


def test_canon_ref_snaps_to_passed_control_ids() -> None:
    # RTV-81b §2: when the supervisor passes real control ids, snap the model's refs to THOSE.
    refs = ["DORA-30.3e-AUDIT-RIGHTS", "DORA-28.8-EXIT-STRATEGY"]
    assert _canon_ref("DORA‑30.3e‑AUDIT‑RIGHTS", refs) == "DORA-30.3e-AUDIT-RIGHTS"  # unicode hyphens
    # model appended the title to the id → still snaps back
    assert _canon_ref("DORA-28.8-EXIT-STRATEGY (Exit strategy)", refs) == "DORA-28.8-EXIT-STRATEGY"
    assert _canon_ref("DORA-99-UNKNOWN", refs) == "DORA-99-UNKNOWN"  # not in the set → left as-is


def test_ensure_all_controls_fills_dropped_controls_as_insufficient() -> None:
    # A weak model may return only the addressed control; the assessment must still cover ALL passed
    # controls — the rest are filled as insufficient_evidence (deterministic completeness).
    refs = ["DORA-30.3e-AUDIT-RIGHTS", "DORA-28.2-STRATEGY", "DORA-30.SUBCONTRACTING"]
    data = {
        "verdicts": [{"controlRef": "DORA-30.3e-AUDIT-RIGHTS", "verdict": "partial", "confidence": 0.6}],
        "insufficientEvidence": [],
    }
    out = _ensure_all_controls(data, refs)
    by_ref = {v["controlRef"]: v["verdict"] for v in out["verdicts"]}
    assert by_ref["DORA-30.3e-AUDIT-RIGHTS"] == "partial"  # kept
    assert by_ref["DORA-28.2-STRATEGY"] == "insufficient_evidence"  # filled
    assert by_ref["DORA-30.SUBCONTRACTING"] == "insufficient_evidence"  # filled
    assert set(out["insufficientEvidence"]) == {"DORA-28.2-STRATEGY", "DORA-30.SUBCONTRACTING"}
    # the fill is VISIBLE (not a silent illusion): the model returned 1 of 3, we filled 2
    assert out["_modelCoverage"] == {"returned": 1, "filled": 2, "total": 3}
