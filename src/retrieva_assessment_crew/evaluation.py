"""Deterministic crew evaluation (RTV-81b) — score a crew response against a golden expectation.

Prompt tuning needs a measurable target, not vibes. This scorer turns an assessment into four honest,
DORA-relevant metrics in [0,1] so each prompt iteration is comparable (watch the trace in Langfuse,
read the score here):

  • verdict_accuracy    — of the controls the golden case pins a verdict for, how many did the crew get
                          right (exact verdict match, keyed by controlRef).
  • control_coverage    — did the crew produce a verdict for every expected control (no silent drops).
  • honesty             — for controls the case marks as having NO supporting evidence, did the crew say
                          `insufficient_evidence` rather than GUESS a verdict. The anti-hallucination metric.
  • citation_grounding  — every citation the crew emits must reference an evidence id that was actually
                          supplied in the request (no fabricated evidence). 1.0 when it cites nothing.

Pure + dependency-free so it runs in CI and is reusable from the Langfuse eval (eval/run_eval.py maps
these to Langfuse scores). `overall` is the mean of the four — a single number to track across runs.
"""

from __future__ import annotations

from typing import Any

METRICS = ("verdict_accuracy", "control_coverage", "honesty", "citation_grounding")


def _verdict_by_control(items: list[dict[str, Any]]) -> dict[str, str]:
    return {str(v.get("controlRef")): str(v.get("verdict")) for v in items or [] if v.get("controlRef")}


def score_case(request: dict[str, Any], expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, float]:
    """Score one crew response. `expected` keys: verdicts[{controlRef,verdict}], insufficientControls[].

    All metrics default to 1.0 when they are vacuously satisfied (nothing expected / nothing emitted),
    so an empty case never drags the mean down artificially.
    """
    exp_verdicts = _verdict_by_control(expected.get("verdicts", []))
    act_verdicts = _verdict_by_control(actual.get("verdicts", []))

    # verdict_accuracy — exact match over the expected controls.
    if exp_verdicts:
        hits = sum(1 for ref, v in exp_verdicts.items() if act_verdicts.get(ref) == v)
        verdict_accuracy = hits / len(exp_verdicts)
    else:
        verdict_accuracy = 1.0

    # control_coverage — every expected control got some verdict.
    if exp_verdicts:
        covered = sum(1 for ref in exp_verdicts if ref in act_verdicts)
        control_coverage = covered / len(exp_verdicts)
    else:
        control_coverage = 1.0

    # honesty — controls with no evidence must be insufficient_evidence, not a guessed verdict.
    insufficient_controls = [str(c) for c in expected.get("insufficientControls", []) or []]
    if insufficient_controls:
        honest = sum(1 for ref in insufficient_controls if act_verdicts.get(ref) == "insufficient_evidence")
        honesty = honest / len(insufficient_controls)
    else:
        honesty = 1.0

    # citation_grounding — no fabricated evidence: every citation id was supplied in the request.
    supplied = {str(e.get("id")) for e in request.get("evidenceRefs", []) or [] if e.get("id")}
    citations = actual.get("citations", []) or []
    if citations:
        grounded = sum(1 for c in citations if str(c.get("evidenceId")) in supplied)
        citation_grounding = grounded / len(citations)
    else:
        citation_grounding = 1.0

    scores = {
        "verdict_accuracy": round(verdict_accuracy, 4),
        "control_coverage": round(control_coverage, 4),
        "honesty": round(honesty, 4),
        "citation_grounding": round(citation_grounding, 4),
    }
    scores["overall"] = round(sum(scores[m] for m in METRICS) / len(METRICS), 4)
    return scores


def aggregate(case_scores: list[dict[str, float]]) -> dict[str, float]:
    """Mean each metric across cases. Empty → all-zero (no evidence of quality)."""
    keys = (*METRICS, "overall")
    if not case_scores:
        return {k: 0.0 for k in keys}
    return {k: round(sum(s.get(k, 0.0) for s in case_scores) / len(case_scores), 4) for k in keys}
