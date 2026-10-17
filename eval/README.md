# Crew eval (RTV-81b)

The measurable loop for tuning the crew's prompts before flipping `CREW_A2A_ENABLED=true` on dev.

## What's here

| File | Role |
|---|---|
| `dataset.jsonl` | Golden DORA cases — one per line: `{name, request, expected}`. `request` is a real A2A `assess-arrangement` request; `expected` pins the verdicts (+ the controls that must stay `insufficient_evidence`). |
| `run_eval.py` | Runs each case through the crew, scores it, prints the aggregate, optionally pushes to Langfuse. |
| `../src/retrieva_assessment_crew/evaluation.py` | The **scorer** (pure, unit-tested in `tests/test_scorer.py`). |

## The four metrics (each in `[0,1]`, `overall` = their mean)

- **verdict_accuracy** — exact verdict match over the expected controls.
- **control_coverage** — every expected control got *a* verdict (no silent drops).
- **honesty** — controls with no evidence are `insufficient_evidence`, not a guess. *The anti-hallucination metric.*
- **citation_grounding** — every citation references evidence actually supplied in the request (no fabrication).

## Run it

```bash
# offline — no LLM → every case degrades to insufficient_evidence (proves the harness runs end-to-end):
uv run python eval/run_eval.py

# real run — needs LiteLLM reachable (local env or in-cluster on dev); gate on a minimum score:
LITELLM_BASE_URL=https://litellm.10.0.0.200.nip.io LITELLM_API_KEY=... \
  uv run python eval/run_eval.py --min-overall 0.6

# also upsert the golden set + log aggregate scores to Langfuse (needs LANGFUSE_* env):
uv run python eval/run_eval.py --langfuse
```

`--min-overall` makes the runner exit non-zero below the threshold, so it can gate a pipeline.

## The tuning loop

1. Edit the agent prompts in `src/retrieva_assessment_crew/crew.py`.
2. `uv run python eval/run_eval.py --min-overall <target>` against a crew that can reach LiteLLM.
3. Read the per-metric aggregate here **and** watch the per-agent spans for each case in Langfuse
   (the run is traced + stitched to its `runId` session — RTV-81b instrumentation in `tracing.py`).
4. Keep the change only if `overall` went up. When it holds at the target, flip `CREW_A2A_ENABLED=true`
   on dev and run the adversarial QA gate — then promote to prod (Kargo).

> Extend `dataset.jsonl` as real arrangements surface — the golden set is the contract for "good".
> The control refs (`DORA-ART28/29/30`) are the DORA ICT third-party articles; align them with the
> live control-library refs as the library firms up.
