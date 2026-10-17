"""RTV-81b — crew Langfuse instrumentation, proven WITHOUT a live Langfuse / LLM.

Locks the deterministic wiring: `assessment_trace` opens a root span, stitches to the supervisor via
`session_id = runId`, emits one span per crew agent, and records the final output + token usage — and
is a strict no-op (never raises) when tracing is disabled. The spans actually landing in the Langfuse
UI is verified on live dev during the RTV-81b flip; here we assert the client is called correctly.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import langfuse
import pytest

from retrieva_assessment_crew import tracing


@pytest.fixture(autouse=True)
def _reset_client():
    """Never leak a test's mock client into another test via the module global."""
    saved = tracing._client
    yield
    tracing._client = saved


REQUEST = {
    "runId": "11111111-1111-4111-8111-111111111111",
    "arrangementId": "22222222-2222-4222-8222-222222222222",
    "controlLibraryVersion": "1.0.0",
    "clauses": [{"ref": "c1", "text": "The provider shall maintain an exit plan."}],
    "evidenceRefs": [{"id": "c1"}, {"id": "e1"}],
}

# A fake CrewOutput: 3 task outputs (the 3 agents) + token usage.
FAKE_RESULT = SimpleNamespace(
    tasks_output=[
        SimpleNamespace(agent="DORA clause mapper", description="Map clauses", raw="mapped c1 -> ART28"),
        SimpleNamespace(agent="DORA gap analyst", description="Find gaps", raw="gap: exit testing"),
        SimpleNamespace(agent="DORA control scorer", description="Score", raw="{...}"),
    ],
    token_usage=SimpleNamespace(prompt_tokens=120, completion_tokens=45, total_tokens=165),
)


def test_disabled_is_a_strict_noop() -> None:
    tracing._client = None  # tracing off
    with tracing.assessment_trace(REQUEST) as rec:
        rec.record_agents(FAKE_RESULT)  # must not raise
        rec.finish({"confidence": 0.0}, FAKE_RESULT)  # must not raise


def test_enabled_emits_root_agents_and_output(monkeypatch) -> None:
    client = MagicMock(name="langfuse")
    tracing._client = client

    captured = {}

    @contextmanager
    def fake_propagate(**kwargs):
        captured.update(kwargs)  # record session_id / tags / trace_name
        yield

    monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)

    with tracing.assessment_trace(REQUEST) as rec:
        rec.record_agents(FAKE_RESULT)
        rec.finish({"confidence": 0.7, "verdicts": []}, FAKE_RESULT)

    # stitched to the supervisor's trace by session_id = runId
    assert captured["session_id"] == REQUEST["runId"]
    assert "crew" in captured["tags"]

    # one root "chain" span carrying the compact request summary
    root_call = client.start_as_current_observation.call_args
    assert root_call.kwargs["name"] == "crew.assess-arrangement"
    assert root_call.kwargs["as_type"] == "chain"
    assert root_call.kwargs["input"]["clauseCount"] == 1
    assert root_call.kwargs["input"]["runId"] == REQUEST["runId"]

    # one span per crew agent (3 task outputs)
    assert client.start_observation.call_count == 3
    agent_names = [c.kwargs["name"] for c in client.start_observation.call_args_list]
    assert agent_names == ["agent:DORA clause mapper", "agent:DORA gap analyst", "agent:DORA control scorer"]
    assert all(c.kwargs["as_type"] == "agent" for c in client.start_observation.call_args_list)

    # final output + token usage recorded on the root span, and the client flushed
    root = client.start_as_current_observation.return_value.__enter__.return_value
    update_call = root.update.call_args
    assert update_call.kwargs["output"]["confidence"] == 0.7
    assert update_call.kwargs["usage_details"] == {"input": 120, "output": 45, "total": 165}
    client.flush.assert_called_once()


def test_setup_failure_degrades_to_noop(monkeypatch) -> None:
    client = MagicMock(name="langfuse")
    tracing._client = client

    def boom(**kwargs):
        raise RuntimeError("propagate failed")

    monkeypatch.setattr(langfuse, "propagate_attributes", boom)

    # setup failure → no-op recorder, no raise, no spans
    with tracing.assessment_trace(REQUEST) as rec:
        rec.record_agents(FAKE_RESULT)
        rec.finish({"confidence": 0.0}, FAKE_RESULT)
    client.start_as_current_observation.assert_not_called()
