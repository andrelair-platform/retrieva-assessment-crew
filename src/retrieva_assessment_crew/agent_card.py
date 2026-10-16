"""Build the A2A Agent Card served at /.well-known/agent-card.json (RTV-80).

Constructed in Python from the shared contract constants (the authoritative descriptor is mirrored from
`retrieva-backend/contracts/a2a/assessArrangement.contract.ts`). The crew advertises exactly one skill,
`assess-arrangement`; its wire I/O is the vendored JSON Schema (see contract.py).
"""

from __future__ import annotations

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from .contract import A2A_PROTOCOL_VERSION, CONTRACT_VERSION, SKILL_ID


def build_agent_card(public_url: str) -> AgentCard:
    skill = AgentSkill(
        id=SKILL_ID,
        name="Assess ICT arrangement for DORA",
        description=(
            "Given clauses + evidence refs + a control-library version, return findings, verdicts, "
            "citations, an overall confidence, and the controls with insufficient evidence."
        ),
        tags=["dora", "ict-tprm", "assessment", "gap-analysis"],
        input_modes=["application/json"],
        output_modes=["application/json"],
    )
    return AgentCard(
        protocol_version=A2A_PROTOCOL_VERSION,
        name="retrieva-assessment-crew",
        description=(
            "DORA ICT third-party assessment crew — maps contract clauses to controls, finds gaps, "
            "scores coverage. Evidence-grounded; output is proposed, not authoritative (human-gated "
            "by the supervisor)."
        ),
        version=CONTRACT_VERSION,
        url=public_url,
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=[skill],
    )
