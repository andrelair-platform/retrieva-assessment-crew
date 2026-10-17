#!/usr/bin/env python
"""Crew eval runner (RTV-81b) — run the golden dataset through the crew, score each case, print the
aggregate, and (optionally) push the dataset + scores to Langfuse.

This is the loop that makes prompt tuning measurable:
    1. edit the agents' prompts in crew.py
    2. run this against a crew that can reach LiteLLM (local with LITELLM_* set, or in-cluster on dev)
    3. read the per-metric aggregate here + watch the per-agent spans in Langfuse
    4. keep the change only if `overall` went up — then, when it holds, flip CREW_A2A_ENABLED on dev.

Usage:
    # offline (no LLM → every case degrades to insufficient_evidence; proves the harness runs):
    uv run python eval/run_eval.py

    # real run (LiteLLM reachable) + gate the pipeline on a minimum score:
    LITELLM_BASE_URL=... LITELLM_API_KEY=... uv run python eval/run_eval.py --min-overall 0.6

    # also upload the golden set + scores to Langfuse (needs LANGFUSE_* env):
    uv run python eval/run_eval.py --langfuse
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from retrieva_assessment_crew.contract import validate_response
from retrieva_assessment_crew.evaluation import METRICS, aggregate, score_case

logger = logging.getLogger("crew-eval")

DATASET = Path(__file__).parent / "dataset.jsonl"
LANGFUSE_DATASET_NAME = "rtv-81b-crew-eval"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def default_runner() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """In-process crew. Reaches LiteLLM when configured; degrades to insufficient_evidence otherwise."""
    from retrieva_assessment_crew.crew import AssessmentCrew

    crew = AssessmentCrew()
    return crew.run


def evaluate(
    cases: list[dict[str, Any]], runner: Callable[[dict[str, Any]], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        request = case["request"]
        actual = runner(request)
        try:
            validate_response(actual)  # the crew must always return a contract-valid shape
        except Exception as exc:  # noqa: BLE001 — a bad shape is a hard fail for that case, not a crash
            logger.warning("case %r produced an off-contract response: %s", case.get("name"), exc)
        scores = score_case(request, case.get("expected", {}), actual)
        rows.append({"name": case.get("name", "?"), "scores": scores, "actual": actual})
    return rows


def print_report(rows: list[dict[str, Any]], agg: dict[str, float]) -> None:
    cols = (*METRICS, "overall")
    width = max((len(r["name"]) for r in rows), default=4) + 2
    header = "case".ljust(width) + "".join(c[:12].rjust(14) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = r["name"].ljust(width) + "".join(f"{r['scores'][c]:.2f}".rjust(14) for c in cols)
        print(line)
    print("-" * len(header))
    print("AGGREGATE".ljust(width) + "".join(f"{agg[c]:.2f}".rjust(14) for c in cols))


def push_to_langfuse(cases: list[dict[str, Any]], agg: dict[str, float]) -> None:
    """Best-effort: upsert the golden set as a Langfuse dataset + log the aggregate as scores. Fail-safe
    — any error here must not fail the eval. Verified end-to-end on live dev (needs LANGFUSE_* + reach)."""
    try:
        from retrieva_assessment_crew.settings import load_settings
        from retrieva_assessment_crew.tracing import init_tracing

        client = init_tracing(load_settings())
        if client is None:
            print("\n[langfuse] disabled (no keys) — skipping upload")
            return
        client.create_dataset(name=LANGFUSE_DATASET_NAME, description="RTV-81b DORA crew golden cases")
        for case in cases:
            client.create_dataset_item(
                dataset_name=LANGFUSE_DATASET_NAME,
                id=case.get("name"),
                input=case["request"],
                expected_output=case.get("expected", {}),
            )
        for metric, value in agg.items():
            client.create_score(name=f"eval.{metric}", value=float(value), comment="crew eval aggregate")
        client.flush()
        print(
            f"\n[langfuse] uploaded {len(cases)} cases + {len(agg)} aggregate scores "
            f"to '{LANGFUSE_DATASET_NAME}'"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n[langfuse] upload skipped (non-fatal): {exc}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Run the crew eval dataset and score it.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--min-overall", type=float, default=0.0, help="exit non-zero if aggregate overall < this"
    )
    parser.add_argument("--langfuse", action="store_true", help="upload the dataset + scores to Langfuse")
    args = parser.parse_args(argv)

    cases = load_dataset(args.dataset)
    rows = evaluate(cases, default_runner())
    agg = aggregate([r["scores"] for r in rows])
    print_report(rows, agg)
    if args.langfuse:
        push_to_langfuse(cases, agg)

    if agg["overall"] < args.min_overall:
        print(f"\nFAIL: overall {agg['overall']:.2f} < required {args.min_overall:.2f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
