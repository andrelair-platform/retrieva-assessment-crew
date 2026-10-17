"""RTV-81b — the crew eval scorer, the measurable target for prompt tuning.

Deterministic, dependency-free: these assertions pin exactly what each metric rewards + penalises, so a
prompt change that regresses (e.g. starts guessing verdicts, or fabricating citations) shows up as a
lower number, not a silent quality drift.
"""

from __future__ import annotations

from retrieva_assessment_crew.evaluation import aggregate, score_case

REQUEST = {
    "evidenceRefs": [{"id": "ev-1"}, {"id": "ev-2"}],
    "clauses": [{"ref": "§1", "text": "..."}],
}
EXPECTED = {
    "verdicts": [{"controlRef": "DORA-ART28", "verdict": "partial"}],
    "insufficientControls": [],
}


def _actual(verdict="partial", citations=None):
    return {
        "verdicts": [{"controlRef": "DORA-ART28", "verdict": verdict, "confidence": 0.6}],
        "citations": citations or [],
    }


def test_perfect_match_scores_one() -> None:
    s = score_case(REQUEST, EXPECTED, _actual())
    assert s == {
        "verdict_accuracy": 1.0,
        "control_coverage": 1.0,
        "honesty": 1.0,
        "citation_grounding": 1.0,
        "overall": 1.0,
    }


def test_wrong_verdict_drops_accuracy_but_keeps_coverage() -> None:
    s = score_case(REQUEST, EXPECTED, _actual(verdict="compliant"))
    assert s["verdict_accuracy"] == 0.0
    assert s["control_coverage"] == 1.0  # the control still got *a* verdict


def test_missing_control_drops_coverage_and_accuracy() -> None:
    s = score_case(REQUEST, EXPECTED, {"verdicts": [], "citations": []})
    assert s["control_coverage"] == 0.0
    assert s["verdict_accuracy"] == 0.0


def test_honesty_penalises_guessing_where_evidence_is_absent() -> None:
    expected = {
        "verdicts": [{"controlRef": "DORA-ART30", "verdict": "insufficient_evidence"}],
        "insufficientControls": ["DORA-ART30"],
    }
    # the crew GUESSED compliant on a control with no evidence → honesty 0
    guessed = {"verdicts": [{"controlRef": "DORA-ART30", "verdict": "compliant"}], "citations": []}
    assert score_case(REQUEST, expected, guessed)["honesty"] == 0.0
    # the crew correctly said insufficient_evidence → honesty 1
    honest = {"verdicts": [{"controlRef": "DORA-ART30", "verdict": "insufficient_evidence"}], "citations": []}
    assert score_case(REQUEST, expected, honest)["honesty"] == 1.0


def test_citation_grounding_penalises_fabricated_evidence() -> None:
    # one real citation (ev-1) + one fabricated (ev-999 not in the request) → 0.5
    cited = _actual(citations=[{"evidenceId": "ev-1"}, {"evidenceId": "ev-999"}])
    assert score_case(REQUEST, EXPECTED, cited)["citation_grounding"] == 0.5


def test_aggregate_means_each_metric() -> None:
    keys = ("verdict_accuracy", "control_coverage", "honesty", "citation_grounding", "overall")
    a = dict.fromkeys(keys, 1.0)
    b = dict.fromkeys(keys, 0.0)
    agg = aggregate([a, b])
    assert agg["overall"] == 0.5
    assert agg["verdict_accuracy"] == 0.5


def test_aggregate_empty_is_zero() -> None:
    assert aggregate([])["overall"] == 0.0
