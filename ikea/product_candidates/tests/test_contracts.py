from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


PACKAGE = Path(__file__).resolve().parents[1]
CONTRACTS = PACKAGE / "contracts"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
AUTHORITATIVE = Path(
    "/home/kyzen/V2T/HANDOFF_5090_TO_3090_PLACEMENT_LOOP_2026-07-15/contracts"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Draft202012ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.request_schema = load_json(
            CONTRACTS / "ulip_product_candidate_request_v1.0.schema.json"
        )
        cls.response_schema = load_json(
            CONTRACTS / "ulip_product_candidate_response_v1.0.schema.json"
        )
        cls.ready_schema = load_json(
            CONTRACTS / "ulip_product_candidates_readyz_v1.0.schema.json"
        )
        Draft202012Validator.check_schema(cls.request_schema)
        Draft202012Validator.check_schema(cls.response_schema)
        Draft202012Validator.check_schema(cls.ready_schema)
        cls.request_validator = Draft202012Validator(cls.request_schema)
        cls.response_validator = Draft202012Validator(cls.response_schema)
        cls.ready_validator = Draft202012Validator(cls.ready_schema)

    def assertValid(self, validator: Draft202012Validator, value: object) -> None:
        errors = sorted(validator.iter_errors(value), key=lambda e: list(e.path))
        self.assertEqual([], errors, [e.message for e in errors])

    def assertInvalid(self, validator: Draft202012Validator, value: object) -> None:
        self.assertTrue(list(validator.iter_errors(value)))

    def test_authoritative_request_and_response_are_byte_identical(self) -> None:
        for name in (
            "ulip_product_candidate_request_v1.0.schema.json",
            "ulip_product_candidate_response_v1.0.schema.json",
        ):
            self.assertEqual(
                (AUTHORITATIVE / name).read_bytes(),
                (CONTRACTS / name).read_bytes(),
                name,
            )

    def test_positive_request_fixture(self) -> None:
        self.assertValid(
            self.request_validator, load_json(FIXTURES / "request.valid.json")
        )

    def test_request_negative_schema_matrix(self) -> None:
        valid = load_json(FIXTURES / "request.valid.json")
        cases = []

        case = copy.deepcopy(valid)
        case["unknown"] = True
        cases.append(case)
        for required in (
            "schema_version",
            "request_id",
            "query_text",
            "categories_primary",
            "categories_secondary",
            "top_k_per_tier",
        ):
            case = copy.deepcopy(valid)
            case.pop(required)
            cases.append(case)
        for key in ("request_id", "query_text"):
            case = copy.deepcopy(valid)
            case[key] = ""
            cases.append(case)
        case = copy.deepcopy(valid)
        case["categories_primary"] = []
        cases.append(case)
        case = copy.deepcopy(valid)
        case["categories_primary"] = ["couch", "couch"]
        cases.append(case)
        for bad_top_k in (True, False, 0, 101, 1.5, "2", None):
            case = copy.deepcopy(valid)
            case["top_k_per_tier"] = bad_top_k
            cases.append(case)

        for case in cases:
            with self.subTest(case=case):
                self.assertInvalid(self.request_validator, case)

    def test_positive_response_fixtures(self) -> None:
        for name in ("response.ok.valid.json", "response.no_match.valid.json"):
            with self.subTest(name=name):
                self.assertValid(self.response_validator, load_json(FIXTURES / name))

    def test_response_negative_schema_matrix(self) -> None:
        valid = load_json(FIXTURES / "response.ok.valid.json")
        cases = []

        case = copy.deepcopy(valid)
        case["unknown"] = 1
        cases.append(case)
        for required in (
            "schema_version",
            "request_id",
            "status",
            "query_metadata",
            "results",
            "warnings",
            "provenance",
        ):
            case = copy.deepcopy(valid)
            case.pop(required)
            cases.append(case)
        case = copy.deepcopy(valid)
        case["query_metadata"]["token_count"] = True
        cases.append(case)
        case = copy.deepcopy(valid)
        case["query_metadata"]["truncated"] = True
        cases.append(case)
        case = copy.deepcopy(valid)
        case["query_metadata"]["eot_present"] = False
        cases.append(case)
        case = copy.deepcopy(valid)
        case["results"][1]["retrieval_score"] = True
        cases.append(case)
        case = copy.deepcopy(valid)
        case["results"][1]["retrieval_score"] = 1.01
        cases.append(case)
        case = copy.deepcopy(valid)
        case["results"][1]["dimensions"]["width"] = 0
        cases.append(case)
        case = copy.deepcopy(valid)
        case["results"][1]["special_flags"] = ["FLAG", "FLAG"]
        cases.append(case)
        case = copy.deepcopy(valid)
        case["results"][1]["allowed_yaws_deg"] = [0, 0]
        cases.append(case)
        case = copy.deepcopy(valid)
        case["results"][1]["allowed_yaws_deg"] = [45]
        cases.append(case)
        case = copy.deepcopy(valid)
        case["results"][1]["dimension_status"] = "complete"
        cases.append(case)
        case = copy.deepcopy(valid)
        case["results"][1]["footprint_kind"] = "special_unmodeled"
        case["results"][1]["dimension_status"] = "special_unmodeled"
        case["results"][1]["hard_filter_eligible"] = True
        cases.append(case)
        case = copy.deepcopy(valid)
        case["status"] = "no_match"
        cases.append(case)
        case = copy.deepcopy(valid)
        case["results"] = []
        cases.append(case)
        case = copy.deepcopy(valid)
        case["warnings"] = ["DUPLICATE", "DUPLICATE"]
        cases.append(case)
        case = copy.deepcopy(valid)
        case["warnings"] = ["not_uppercase"]
        cases.append(case)

        for case in cases:
            with self.subTest(case=case):
                self.assertInvalid(self.response_validator, case)

    def test_ready_schema_positive_and_negative(self) -> None:
        provenance = load_json(FIXTURES / "response.ok.valid.json")["provenance"]
        ready = {
            "schema_version": "1.0",
            "service": "ulip-product-candidates",
            "status": "ready",
            "deployment_profile_id": "fixture-profile",
            "endpoint_schema_version": "1.0",
            "provenance": provenance,
            "checks": {
                "model_checkpoint": True,
                "vector_bundle": True,
                "catalog_manifest": True,
                "dimension_bundle": True,
                "tokenizer_bundle": True,
                "row_identity_join": True,
                "policy_versions": True,
            },
            "inventory": {
                "vector_shape": [733, 512],
                "vector_rows": 733,
                "catalog_rows": 733,
                "dimension_profile_rows": 733,
            },
            "failures": [],
        }
        self.assertValid(self.ready_validator, ready)

        not_ready = copy.deepcopy(ready)
        not_ready.update(
            status="not_ready",
            deployment_profile_id=None,
            provenance=None,
            failures=["VECTOR_BUNDLE_HASH_MISMATCH"],
        )
        not_ready["checks"]["vector_bundle"] = False
        self.assertValid(self.ready_validator, not_ready)

        bad_ready = copy.deepcopy(ready)
        bad_ready["checks"]["row_identity_join"] = False
        self.assertInvalid(self.ready_validator, bad_ready)
        bad_not_ready = copy.deepcopy(not_ready)
        bad_not_ready["failures"] = []
        self.assertInvalid(self.ready_validator, bad_not_ready)


if __name__ == "__main__":
    unittest.main()
