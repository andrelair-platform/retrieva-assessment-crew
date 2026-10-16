"""A2A `assess-arrangement` contract loading + validation (RTV-79/80).

The contract's SINGLE SOURCE OF TRUTH is the Zod definition in `retrieva-backend`
(`contracts/a2a/assessArrangement.contract.ts`), which generates the JSON Schema committed there. The
files under `_schema/` here are a VENDORED COPY of those generated schemas so the crew can validate the
same shape the supervisor sends/expects. Keep them in sync with the backend (future: a cross-repo CI
drift check); never hand-edit them independently.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft7Validator

SKILL_ID = "assess-arrangement"
CONTRACT_VERSION = "1.0.0"
A2A_PROTOCOL_VERSION = "1.0"


def _load(name: str) -> dict[str, Any]:
    return json.loads(files(f"{__package__}._schema").joinpath(name).read_text("utf-8"))


@lru_cache(maxsize=1)
def request_schema() -> dict[str, Any]:
    return _load("assess-arrangement.request.json")


@lru_cache(maxsize=1)
def response_schema() -> dict[str, Any]:
    return _load("assess-arrangement.response.json")


@lru_cache(maxsize=2)
def _validator(kind: str) -> Draft7Validator:
    return Draft7Validator(request_schema() if kind == "request" else response_schema())


def validate_request(payload: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if the inbound request violates the wire contract."""
    _validator("request").validate(payload)


def validate_response(payload: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if our response would violate the wire contract."""
    _validator("response").validate(payload)
