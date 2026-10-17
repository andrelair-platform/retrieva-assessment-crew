"""The CrewAI assessment crew (RTV-81a) — clause-mapper -> gap-analyst -> control-scorer.

Replaces the RTV-80 mock behind the executor's `_assess` seam. It is the ONLY place autonomy lives
(ADR-AGENT-01): a bounded, sequential crew whose output is PROPOSED, never authoritative — the
supervisor's human gate decides. Guardrails: evidence-grounded; **no evidence -> insufficient_evidence**,
never an invented verdict; output is validated against the vendored wire contract before it leaves.

LLM runs through the LiteLLM gateway (on-cluster). When no LLM is configured (e.g. local dev, or creds
absent) the crew DEGRADES SAFELY to an insufficient-evidence response rather than crashing — which also
makes the service runnable + testable without live LLM infra. Tuned prompts + Qdrant evidence retrieval +
the eval harness are RTV-81b.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from .settings import Settings, load_settings

logger = logging.getLogger("retrieva-assessment-crew")

VERDICTS = ("compliant", "partial", "non_compliant", "insufficient_evidence", "not_applicable")

# Compact DORA ICT third-party control catalog (RTV-81b). Until the supervisor passes the full control
# library in the A2A request, this canonical set gives the crew a STABLE ref taxonomy to map clauses to
# — so verdicts are comparable across runs (the eval keys on these refs) instead of free-text prose.
# Refs mirror the DORA (Reg. (EU) 2022/2554) articles governing ICT third-party arrangements.
CONTROL_CATALOG = (
    "DORA-ART28 — General principles: sound, documented, proportionate management of ICT third-party "
    "risk across the arrangement lifecycle (strategy, register of information, risk assessment).\n"
    "DORA-ART29 — Subcontracting & concentration: conditions, prior consent and oversight for "
    "subcontracting the ICT services that support critical or important functions, incl. a subcontractor "
    "register and ICT concentration-risk assessment.\n"
    "DORA-ART30 — Key contractual provisions: the mandatory contract terms — service description & SLAs, "
    "data access/processing/return & deletion on termination, access/inspection/audit rights for the "
    "client AND the competent authority, incident assistance, and documented exit strategies/transition."
)


# ── Structured output (mirrors the RTV-79 response contract) ─────────────────────────────────────
class Finding(BaseModel):
    controlRef: str
    clauseRefs: list[str] = Field(default_factory=list)
    summary: str = ""


class VerdictItem(BaseModel):
    controlRef: str
    verdict: Literal["compliant", "partial", "non_compliant", "insufficient_evidence", "not_applicable"]
    confidence: float = 0.0


class Citation(BaseModel):
    evidenceId: str
    controlRef: str | None = None
    snippet: str | None = None


class AssessmentOutput(BaseModel):
    controlLibraryVersion: str
    findings: list[Finding] = Field(default_factory=list)
    verdicts: list[VerdictItem] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    insufficientEvidence: list[str] = Field(default_factory=list)


_CANON = {"28": "DORA-ART28", "29": "DORA-ART29", "30": "DORA-ART30"}
# LLMs render refs with unicode hyphens (‑ U+2011, – U+2013) and sometimes append the title; snap each
# emitted ref back to a KNOWN control id so verdicts are comparable + map 1:1 into the register.
_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-"})


def _norm(s: Any) -> str:
    return str(s).translate(_DASHES).strip().lower()


def _canon_ref(ref: Any, valid_refs: list[str] | None = None) -> str:
    """Snap a model-emitted ref to a known control id. With `valid_refs` (the controls the supervisor
    passed) an exact hyphen/case-normalised match wins, else a substring match; otherwise — for the
    fallback catalog — the DORA-ARTnn heuristic; otherwise leave the ref untouched (dashes normalised)."""
    s = str(ref).translate(_DASHES)
    if valid_refs:
        by_norm = {_norm(r): r for r in valid_refs}
        exact = by_norm.get(_norm(s))
        if exact:
            return exact
        for nr, orig in by_norm.items():
            if nr and nr in _norm(s):  # model appended the title / wrapped the id
                return orig
        return s
    m = re.search(r"art\D*(28|29|30)\b", s, re.IGNORECASE)
    return _CANON[m.group(1)] if m else s


def _canonicalize(data: dict[str, Any], valid_refs: list[str] | None = None) -> dict[str, Any]:
    """Normalise control refs to known control ids (hyphen/format drift is the LLM's, not ours)."""
    for v in data.get("verdicts", []) or []:
        v["controlRef"] = _canon_ref(v.get("controlRef", ""), valid_refs)
    for f in data.get("findings", []) or []:
        f["controlRef"] = _canon_ref(f.get("controlRef", ""), valid_refs)
    for c in data.get("citations", []) or []:
        if c.get("controlRef"):
            c["controlRef"] = _canon_ref(c["controlRef"], valid_refs)
    data["insufficientEvidence"] = [
        _canon_ref(r, valid_refs) for r in data.get("insufficientEvidence", []) or []
    ]
    return data


def _ensure_all_controls(data: dict[str, Any], valid_refs: list[str]) -> dict[str, Any]:
    """Guarantee a verdict for EVERY passed control. A weaker model tends to drop controls that no
    clause addresses; a DORA assessment must cover them all, so fill any missing control as
    `insufficient_evidence` (the safe default — the supervisor halts + a human reviews). Deterministic,
    so completeness never depends on the model's diligence.

    Records how many controls the MODEL actually returned vs how many we FILLED
    (data["_modelCoverage"] = {"returned", "filled", "total"}) so the fill is visible, not silent: a
    high fill rate means the model engaged with few controls (the eval's honesty would otherwise look
    perfect because of the fill, not the model). The caller logs/traces it; it is NOT part of the wire
    response (stripped before validation)."""
    if not valid_refs:
        return data
    verdicts = data.setdefault("verdicts", [])
    returned = len({v.get("controlRef") for v in verdicts} & set(valid_refs))
    present = {v.get("controlRef") for v in verdicts}
    insufficient = set(data.get("insufficientEvidence", []) or [])
    filled = 0
    for ref in valid_refs:
        if ref not in present:
            verdicts.append({"controlRef": ref, "verdict": "insufficient_evidence", "confidence": 0.0})
            insufficient.add(ref)
            filled += 1
    data["insufficientEvidence"] = sorted(insufficient)
    data["_modelCoverage"] = {"returned": returned, "filled": filled, "total": len(valid_refs)}
    if filled:
        logger.info(
            "completeness fill: model returned %d/%d controls; filled %d as insufficient_evidence",
            returned,
            len(valid_refs),
            filled,
        )
    return data


def _insufficient_evidence(request: dict[str, Any], reason: str) -> dict[str, Any]:
    """Safe-by-default result: every control is insufficient-evidence (the supervisor halts on this)."""
    control = "DORA-ART28"
    logger.info("crew degraded to insufficient-evidence: %s", reason)
    return {
        "controlLibraryVersion": request.get("controlLibraryVersion", "unknown"),
        "findings": [
            {
                "controlRef": control,
                "clauseRefs": [c.get("ref", "") for c in request.get("clauses") or []],
                "summary": reason,
            }
        ],
        "verdicts": [{"controlRef": control, "verdict": "insufficient_evidence", "confidence": 0.0}],
        "citations": [],
        "confidence": 0.0,
        "insufficientEvidence": [control],
    }


def _no_clause_result(request: dict[str, Any]) -> dict[str, Any]:
    """All passed controls → insufficient_evidence, WITHOUT calling the LLM (RTV-81b §2a). Used when the
    request has no clauses — nothing to assess, so the safe + honest result is insufficient for every
    control (the supervisor halts for human review)."""
    refs = [str(c["ref"]) for c in (request.get("controls") or []) if c.get("ref")] or ["DORA-ART28"]
    reason = "no contract clauses to assess"
    return {
        "controlLibraryVersion": request.get("controlLibraryVersion", "unknown"),
        "findings": [{"controlRef": r, "clauseRefs": [], "summary": reason} for r in refs],
        "verdicts": [{"controlRef": r, "verdict": "insufficient_evidence", "confidence": 0.0} for r in refs],
        "citations": [],
        "confidence": 0.0,
        "insufficientEvidence": refs,
    }


class AssessmentCrew:
    """The real crew. Construction is cheap (no LLM built here); the LLM + agents are built at run()."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings()

    def _make_llm(self):  # type: ignore[no-untyped-def]
        s = self._settings
        if not (s.litellm_base_url and s.litellm_api_key):
            return None
        from crewai import LLM  # lazy: keep import + cold-start cost out of module load

        model = os.getenv("LITELLM_MODEL", "openai/gpt-4o-mini")
        # temperature 0: a compliance verdict must be reproducible — same inputs, same assessment.
        return LLM(model=model, base_url=s.litellm_base_url, api_key=s.litellm_api_key, temperature=0)

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        llm = self._make_llm()
        if llm is None:
            return _insufficient_evidence(request, "no LLM configured — human review required")
        # 0-clause short-circuit (RTV-81b §2a, defence in depth — the supervisor skips us too): with no
        # clauses there is nothing to map, and the LLM returns invalid output for N controls → wasted
        # call + noisy json_invalid. Mark every passed control insufficient_evidence, no kickoff.
        if not (request.get("clauses") or []):
            logger.info("crew skipped — no clauses; all controls → insufficient_evidence")
            return _no_clause_result(request)
        from .tracing import assessment_trace  # lazy: keep langfuse off the no-LLM/degraded path

        # One Langfuse trace per crew run, stitched to the supervisor by session_id = runId (RTV-81b).
        # Fail-safe: a no-op when tracing is disabled; never breaks the assessment.
        with assessment_trace(request) as trace:
            return self._run_crew(llm, request, trace)

    def _run_crew(self, llm: Any, request: dict[str, Any], trace: Any) -> dict[str, Any]:
        try:
            from crewai import Agent, Crew, Process, Task

            # The supervisor passes the REAL applicable DORA controls (RTV-81b §2); the crew assesses
            # against THESE and echoes each `ref` as the verdict/finding controlRef (→ 1:1 into the
            # register). With none passed, fall back to the compact embedded catalog (ART28/29/30).
            req_controls = request.get("controls") or []
            if req_controls:
                catalog = "\n".join(
                    f"- {c.get('ref')} ({c.get('doraArticleRef', '')}) — "
                    f"{c.get('title', '')}: {c.get('description', '')}"
                    for c in req_controls
                )
                valid_refs = [str(c["ref"]) for c in req_controls if c.get("ref")]
            else:
                catalog = CONTROL_CATALOG
                valid_refs = ["DORA-ART28", "DORA-ART29", "DORA-ART30"]

            clause_mapper = Agent(
                role="DORA clause mapper",
                goal=(
                    "Map each contract clause to the catalog control(s) it addresses, using the EXACT "
                    "catalog ref (e.g. DORA-ART30). Map only what the clause text actually supports."
                ),
                backstory="A meticulous compliance analyst who only maps what the clause text supports.",
                llm=llm,
                verbose=False,
                allow_delegation=False,
            )
            gap_analyst = Agent(
                role="DORA gap analyst",
                goal=(
                    "For EACH catalog control, judge whether the mapped clauses cover its requirement "
                    "and what is still missing. A control that NO clause addresses is a full gap."
                ),
                backstory="A sceptical auditor who flags missing coverage and never assumes.",
                llm=llm,
                verbose=False,
                allow_delegation=False,
            )
            control_scorer = Agent(
                role="DORA control scorer",
                goal=(
                    "Give every catalog control a calibrated verdict + confidence, grounded in the "
                    "clause text and any cited evidence — never a guess, never a fabricated citation."
                ),
                backstory="A calibrated reviewer who would rather say 'insufficient evidence' than guess.",
                llm=llm,
                verbose=False,
                allow_delegation=False,
            )

            ctx = (
                "DORA ICT third-party control catalog (assess against THESE, use the exact refs):\n"
                f"{catalog}\n\n"
                "Control library version: {controlLibraryVersion}\n"
                "Contract clauses (the primary basis for contractual coverage):\n{clauses}\n"
                "Evidence references — cite by id to support a finding; do NOT invent evidence:\n"
                "{evidenceRefs}"
            )
            map_task = Task(
                description="Map each clause to the catalog control(s) it addresses (use exact refs).\n"
                + ctx,
                expected_output="(control ref, clause refs) mappings grounded in the clause text.",
                agent=clause_mapper,
            )
            gap_task = Task(
                description=(
                    "For each control in the catalog, state whether ANY clause addresses it and, if so, "
                    "what the clause still leaves unspecified. Distinguish two cases clearly: (a) a clause "
                    "addresses the control but implementation is unproven = a PARTIAL gap; (b) NO clause "
                    "mentions the control at all = a FULL gap. Absence of a separate evidence document is "
                    "NOT a full gap when a clause addresses the control."
                ),
                expected_output="Per-control: addressed by a clause? partial-gap vs full-gap, + clause refs.",
                agent=gap_analyst,
                context=[map_task],
            )
            score_task = Task(
                description=(
                    "Produce the final assessment. Output a verdict object for EVERY control in the "
                    "catalog above — one per control, NONE omitted (if there are N controls, return N "
                    "verdicts). Each `controlRef` MUST be copied VERBATIM from the catalog's `ref` ids "
                    "(ASCII, unchanged — do not rename, abbreviate, renumber, or use a different hyphen). "
                    "A control that no clause addresses is NOT dropped — it gets `insufficient_evidence` "
                    "and its ref is added to insufficientEvidence.\n"
                    "DORA Art. 28-30 govern what the CONTRACT must contain, so the clause text is the "
                    "primary artefact. Decide each control's verdict by this table:\n"
                    "  • a clause addresses it AND cited evidence proves implementation → `compliant`\n"
                    "  • a clause addresses the control but implementation is unproven (no/weak evidence) → "
                    "`partial`  ← this is the DEFAULT when a clause is present; a missing evidence document "
                    "does NOT downgrade it to insufficient_evidence\n"
                    "  • a clause exists but explicitly fails/contradicts the requirement → `non_compliant`\n"
                    "  • NO clause addresses the control at all → `insufficient_evidence` (list its ref in "
                    "insufficientEvidence)\n"
                    "  • the control does not apply to this arrangement → `not_applicable`\n"
                    "Worked example: a clause granting the client AND the competent authority rights of "
                    "access, inspection and audit, with no separate evidence file → the audit/access-rights "
                    "control = `partial` (the required provision is in the contract; implementation is "
                    "simply unverified). It is NOT insufficient_evidence, because a clause addresses it.\n"
                    f"verdict ∈ {VERDICTS}; confidence ∈ [0,1], calibrated (high only with a direct clause "
                    "+ cited evidence; NEVER high with insufficient_evidence). Cite ONLY supplied evidence "
                    "ids; omit snippet if you have no exact quote. NEVER fabricate a verdict or a citation."
                ),
                expected_output="A single AssessmentOutput JSON object matching the schema.",
                agent=control_scorer,
                context=[map_task, gap_task],
                output_pydantic=AssessmentOutput,
            )

            crew = Crew(
                agents=[clause_mapper, gap_analyst, control_scorer],
                tasks=[map_task, gap_task, score_task],
                process=Process.sequential,
                verbose=False,
            )
            result = crew.kickoff(
                inputs={
                    "controlLibraryVersion": request.get("controlLibraryVersion", ""),
                    "clauses": request.get("clauses", []),
                    "evidenceRefs": request.get("evidenceRefs", []),
                }
            )
            trace.record_agents(result)  # one Langfuse span per agent's task output (fail-safe)
            output = getattr(result, "pydantic", None)
            if not isinstance(output, AssessmentOutput):
                data = _insufficient_evidence(request, "crew produced no structured output")
                trace.finish(data, result)
                return data
            # exclude_none: the wire contract's optional fields (e.g. citation.snippet) are OMITTABLE,
            # not NULLABLE — emitting `null` fails validate_response (would break the live A2A call).
            data = _canonicalize(output.model_dump(exclude_none=True), valid_refs)
            data = _ensure_all_controls(data, valid_refs)
            data["controlLibraryVersion"] = request.get(
                "controlLibraryVersion", data.get("controlLibraryVersion", "unknown")
            )
            trace.finish(data, result)  # final assessment + token usage + model-coverage on the root span
            # _modelCoverage is observability only — strip it before the wire response (strict schema).
            data.pop("_modelCoverage", None)
            return data
        except Exception as exc:  # noqa: BLE001 — a crew failure must halt safely, not crash the A2A call
            logger.warning("crew run failed — degrading to insufficient-evidence: %s", exc)
            data = _insufficient_evidence(request, "assessment could not complete — human review required")
            trace.finish(data)
            return data
