# Contributing

Member repo of the **Retrieva** product (RNCP39583 certification, board #2).

## Branching
Trunk-based: `main` is the only deploy branch. Work on `feat/…` `fix/…` `docs/…` `chore/…` branches → PR → `main`. Branches auto-delete on merge; keep them short-lived.

## Commits
[Conventional Commits](https://www.conventionalcommits.org/) — `feat:` / `fix:` / `docs:` / `chore:` / `refactor:` / `test:`. Lowercase subject, no trailing period. Releases are automated by release-please.

## Local dev
```bash
uv sync                 # install (incl. dev deps)
uv run ruff check .     # L0 lint
uv run ruff format .    # format
uv run pytest -q        # L1 tests
uv run retrieva-assessment-crew   # run the A2A server (http://localhost:8080)
```

## Tests
Every change keeps `ruff` + `pytest` green. New behaviour needs a test. The A2A `assess-arrangement` wire contract is pinned in `src/retrieva_assessment_crew/_schema/` (vendored from `retrieva-backend`, the source of truth) — keep responses contract-valid.

## PRs
Link the issue (`Closes andrelair-platform/retrieva#<n>`). CI (ruff + pytest + Docker build) must pass.
