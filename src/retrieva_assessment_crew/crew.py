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
from typing import Any, Literal

from pydantic import BaseModel, Field

from .settings import Settings, load_settings

logger = logging.getLogger("retrieva-assessment-crew")

VERDICTS = ("compliant", "partial", "non_compliant", "insufficient_evidence", "not_applicable")


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
        return LLM(model=model, base_url=s.litellm_base_url, api_key=s.litellm_api_key, temperature=0.1)

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        llm = self._make_llm()
        if llm is None:
            return _insufficient_evidence(request, "no LLM configured — human review required")
        from .tracing import assessment_trace  # lazy: keep langfuse off the no-LLM/degraded path

        # One Langfuse trace per crew run, stitched to the supervisor by session_id = runId (RTV-81b).
        # Fail-safe: a no-op when tracing is disabled; never breaks the assessment.
        with assessment_trace(request) as trace:
            return self._run_crew(llm, request, trace)

    def _run_crew(self, llm: Any, request: dict[str, Any], trace: Any) -> dict[str, Any]:
        try:
            from crewai import Agent, Crew, Process, Task

            clause_mapper = Agent(
                role="DORA clause mapper",
                goal="Map each contract clause to the relevant DORA ICT third-party control(s).",
                backstory="A meticulous compliance analyst who only maps what the clause text supports.",
                llm=llm,
                verbose=False,
                allow_delegation=False,
            )
            gap_analyst = Agent(
                role="DORA gap analyst",
                goal=(
                    "Find where the mapped clauses fail to cover a control's requirements, "
                    "grounded in the supplied evidence."
                ),
                backstory="A sceptical auditor who flags missing coverage and never assumes.",
                llm=llm,
                verbose=False,
                allow_delegation=False,
            )
            control_scorer = Agent(
                role="DORA control scorer",
                goal=(
                    "Score each control with a calibrated verdict + confidence, defaulting to "
                    "insufficient_evidence when the evidence does not support a conclusion."
                ),
                backstory="A calibrated reviewer who would rather say 'insufficient evidence' than guess.",
                llm=llm,
                verbose=False,
                allow_delegation=False,
            )

            ctx = (
                "Control library version: {controlLibraryVersion}\n"
                "Clauses: {clauses}\n"
                "Evidence references (cite by id; do NOT invent evidence): {evidenceRefs}"
            )
            map_task = Task(
                description="Map the clauses to DORA controls.\n" + ctx,
                expected_output="A list of (control, clause refs) mappings grounded in the clause text.",
                agent=clause_mapper,
            )
            gap_task = Task(
                description=(
                    "Given the mappings, identify coverage gaps per control, grounded in the evidence."
                ),
                expected_output="Per-control gaps with the clause refs that support/miss each requirement.",
                agent=gap_analyst,
                context=[map_task],
            )
            score_task = Task(
                description=(
                    "Produce the final assessment. For each control: a finding, a verdict from "
                    f"{VERDICTS}, a calibrated confidence in [0,1], and citations to evidence ids. "
                    "List controls with no supporting evidence in insufficientEvidence. "
                    "NEVER fabricate a verdict or a citation — default to insufficient_evidence."
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
            data = output.model_dump()
            data["controlLibraryVersion"] = request.get(
                "controlLibraryVersion", data.get("controlLibraryVersion", "unknown")
            )
            trace.finish(data, result)  # final assessment + token usage on the root span
            return data
        except Exception as exc:  # noqa: BLE001 — a crew failure must halt safely, not crash the A2A call
            logger.warning("crew run failed — degrading to insufficient-evidence: %s", exc)
            data = _insufficient_evidence(request, "assessment could not complete — human review required")
            trace.finish(data)
            return data
