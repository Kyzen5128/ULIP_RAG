"""Load the byte-for-byte mirrored authoritative Draft 2020-12 contracts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


CONTRACT_DIR = Path(__file__).resolve().parent / "contracts"


def _load(name: str) -> Any:
    path = CONTRACT_DIR / name
    value = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(value)
    return value


REQUEST_SCHEMA = _load("ulip_product_candidate_request_v1.0.schema.json")
RESPONSE_SCHEMA = _load("ulip_product_candidate_response_v1.0.schema.json")
READYZ_SCHEMA = _load("ulip_product_candidates_readyz_v1.0.schema.json")

REQUEST_VALIDATOR = Draft202012Validator(REQUEST_SCHEMA)
RESPONSE_VALIDATOR = Draft202012Validator(RESPONSE_SCHEMA)
READYZ_VALIDATOR = Draft202012Validator(READYZ_SCHEMA)
# Compatibility spelling used by the implementation checklist/tests.
READY_VALIDATOR = READYZ_VALIDATOR


def first_validation_error(validator: Draft202012Validator, value: Any) -> Any:
    """Return one deterministic validation error, or ``None``."""
    errors = sorted(
        validator.iter_errors(value),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            error.message,
        ),
    )
    return errors[0] if errors else None


__all__ = [
    "CONTRACT_DIR",
    "READYZ_SCHEMA",
    "READYZ_VALIDATOR",
    "READY_VALIDATOR",
    "REQUEST_SCHEMA",
    "REQUEST_VALIDATOR",
    "RESPONSE_SCHEMA",
    "RESPONSE_VALIDATOR",
    "first_validation_error",
]
