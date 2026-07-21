from __future__ import annotations

import asyncio
import copy
import json
import math
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from ikea.product_candidates.runtime import (
    ENABLED_CATEGORY_TERMS,
    ProductCandidateRuntime,
    QueryTooLongError,
    RuntimeUnavailableError,
)
from ikea.product_candidates.service import create_app


REFERENCE_V2T = Path(
    "/home/kyzen/V2T/HANDOFF_5090_TO_3090_PLACEMENT_LOOP_2026-07-15"
) / "reference_v2t_da3"
sys.path.insert(0, str(REFERENCE_V2T))
from placement_loop.models import ProductRetrievalRequest  # noqa: E402
from placement_loop.retrievers import HttpProductRetriever  # noqa: E402


TOKEN = "unit-test-bearer-token"
CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
READY_SCHEMA = json.loads(
    (CONTRACTS / "ulip_product_candidates_readyz_v1.0.schema.json").read_text()
)


def provenance() -> dict[str, str]:
    return {
        "model_checkpoint_sha256": "1" * 64,
        "vector_bundle_sha256": "2" * 64,
        "catalog_manifest_sha256": "3" * 64,
        "dimension_bundle_sha256": "4" * 64,
        "tokenizer_bundle_sha256": "5" * 64,
        "category_policy_version": "v2t36_to_ikea15_v1",
        "dimension_axis_policy_version": "dimension_axis_v1",
    }


def complete_dimensions() -> dict[str, Any]:
    return {"unit": "mm", "width": 1000.0, "depth": 500.0, "height": 700.0}


def product_row(
    product_id: str,
    category: str = "Office Desk",
    *,
    index: int = 0,
    dimensions: Any = ...,
    dimension_status: str = "complete",
    height_policy: str = "optional",
    quality: str = "supported",
    eligible: bool = True,
    footprint_kind: str = "rectangle",
    flags: Any = None,
    yaws: Any = ...,
) -> dict[str, Any]:
    if dimensions is ...:
        dimensions = complete_dimensions()
    if yaws is ...:
        yaws = [0, 90, 180, 270]
    return {
        "row_index": index,
        "product_id": product_id,
        "name": f"Fixture {product_id}",
        "category": category,
        "dimensions": dimensions,
        "dimension_status": dimension_status,
        "height_policy": height_policy,
        "catalog_quality_status": quality,
        "hard_filter_eligible": eligible,
        "footprint_kind": footprint_kind,
        "special_flags": [] if flags is None else flags,
        "allowed_yaws_deg": yaws,
    }


class FakeBaseTokenizer:
    eot_token_id = 49407

    def __init__(self, *, retain_eot: bool = True) -> None:
        self.retain_eot = retain_eot
        self.encode_calls = 0
        self.tensor_calls = 0
        self.encoder = {"<|endoftext|>": self.eot_token_id}

    def encode(self, text: str) -> list[int]:
        self.encode_calls += 1
        if text.startswith("TOKENS:"):
            count = int(text.partition(":")[2])
        else:
            count = max(1, len(text.split()))
        return [10] * count

    def __call__(self, texts: list[str], context_length: int = 77) -> np.ndarray:
        self.tensor_calls += 1
        content = self.encode(texts[0])
        output = np.zeros((1, context_length), dtype=np.int64)
        output[0, 0] = 49406
        output[0, len(content) + 1] = self.eot_token_id if self.retain_eot else 123
        return output


class FakeEncoder:
    def __init__(
        self,
        vector: tuple[float, float] = (1.0, 0.0),
        *,
        delay_s: float = 0.0,
        entered: threading.Event | None = None,
        release: threading.Event | None = None,
        error: Exception | None = None,
    ) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)
        self.delay_s = delay_s
        self.entered = entered
        self.release = release
        self.error = error
        self.calls = 0

    def __call__(self, tokens: Any) -> np.ndarray:
        del tokens
        self.calls += 1
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            self.release.wait(timeout=3.0)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        return self.vector.copy()


def fixture_components() -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = [
        product_row(
            "desk-ineligible", index=0, dimensions=None,
            dimension_status="missing", eligible=False, yaws=None,
        ),
        product_row("desk-z", index=1),
        product_row("desk-a", index=2),
        product_row("bench-1", "Bench", index=3),
        product_row(
            "tv-partial", "TV Stand", index=4,
            dimensions={"unit": "mm", "width": 900.0, "depth": 400.0, "height": None},
            dimension_status="partial", height_policy="optional",
        ),
        product_row("bookshelf-1", "Bookshelf", index=5, height_policy="required"),
        product_row(
            "sofa-special", "Sofa", index=6,
            dimension_status="special_unmodeled", quality="degraded",
            eligible=False, footprint_kind="special_unmodeled",
            flags=["FOOTPRINT_SHAPE_UNVERIFIED"], yaws=None,
        ),
        product_row(
            "bed-blocked", "Bed", index=7, quality="blocked",
            eligible=False, yaws=None,
        ),
        product_row(
            "desk-range", index=8, dimension_status="suspect",
            eligible=False, flags=["DIMENSION_RANGE_PRESENT"], yaws=None,
        ),
        product_row("ottoman-1", "Storage Ottoman", index=9),
        product_row("table-1", "Dining Table", index=10),
        product_row("wardrobe-1", "Wardrobe", index=11, height_policy="required"),
    ]
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.8, 0.6],
            [0.6, 0.8],
            [0.7, math.sqrt(1.0 - 0.49)],
            [0.5, math.sqrt(0.75)],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.4, math.sqrt(0.84)],
            [0.3, math.sqrt(0.91)],
            [0.2, math.sqrt(0.96)],
        ],
        dtype=np.float32,
    )
    return rows, vectors


def make_runtime(
    *,
    encoder: FakeEncoder | None = None,
    tokenizer: FakeBaseTokenizer | None = None,
    checks: dict[str, bool] | None = None,
    integrity_revalidator: Any = None,
) -> tuple[ProductCandidateRuntime, FakeBaseTokenizer, FakeEncoder]:
    rows, vectors = fixture_components()
    tokenizer = tokenizer or FakeBaseTokenizer()
    encoder = encoder or FakeEncoder()
    runtime = ProductCandidateRuntime.from_components(
        profile_id="unit-profile",
        provenance=provenance(),
        vectors=vectors,
        rows=rows,
        tokenizer=tokenizer,
        tokenizer_id="fixture-tokenizer-v1",
        text_encoder=encoder,
        checks=checks,
        integrity_revalidator=integrity_revalidator,
    )
    return runtime, tokenizer, encoder


def request_wire(
    *,
    query: str = "compact desk",
    primary: list[str] | None = None,
    secondary: list[str] | None = None,
    top_k: int = 2,
    request_id: str = "runtime-fixture-001",
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "request_id": request_id,
        "query_text": query,
        "categories_primary": ["desk"] if primary is None else primary,
        "categories_secondary": [] if secondary is None else secondary,
        "top_k_per_tier": top_k,
    }


class TokenizerAndRuntimeTests(unittest.TestCase):
    def test_token_75_passes_with_eot_and_76_rejects_before_encoder(self) -> None:
        base = FakeBaseTokenizer()
        encoder = FakeEncoder()
        runtime, _, _ = make_runtime(encoder=encoder, tokenizer=base)
        passed = runtime.retrieve(request_wire(query="TOKENS:75", top_k=1))
        self.assertEqual(77, passed["query_metadata"]["token_count"])
        self.assertIs(passed["query_metadata"]["eot_present"], True)
        self.assertIs(passed["query_metadata"]["truncated"], False)
        tensor_calls = base.tensor_calls
        encoder_calls = encoder.calls
        with self.assertRaises(QueryTooLongError):
            runtime.retrieve(request_wire(query="TOKENS:76", top_k=1))
        self.assertEqual(tensor_calls, base.tensor_calls)
        self.assertEqual(encoder_calls, encoder.calls)

    def test_missing_eot_fails_closed_before_encoder(self) -> None:
        base = FakeBaseTokenizer(retain_eot=False)
        encoder = FakeEncoder()
        runtime, _, _ = make_runtime(encoder=encoder, tokenizer=base)
        with self.assertRaisesRegex(RuntimeUnavailableError, "EOT"):
            runtime.retrieve(request_wire())
        self.assertEqual(0, encoder.calls)

    def test_full_pool_gate_precedes_k_and_order_is_deterministic(self) -> None:
        runtime, _, encoder = make_runtime()
        first = runtime.retrieve(request_wire(top_k=2))
        second = runtime.retrieve(request_wire(top_k=2))
        self.assertEqual(first, second)
        self.assertEqual(
            ["desk-a", "desk-z"],
            [row["product_id"] for row in first["results"]],
        )
        self.assertNotIn(
            "desk-ineligible", [row["product_id"] for row in first["results"]]
        )
        self.assertNotIn("desk-range", [row["product_id"] for row in first["results"]])
        self.assertEqual(2, encoder.calls)

    def test_per_tier_quota_and_secondary_label(self) -> None:
        runtime, _, _ = make_runtime()
        response = runtime.retrieve(
            request_wire(primary=["desk", "bench"], secondary=["tv stand"], top_k=1)
        )
        self.assertEqual(2, len(response["results"]))
        self.assertEqual(
            ["primary", "secondary"],
            [row["category_tier"] for row in response["results"]],
        )
        self.assertEqual("desk-a", response["results"][0]["product_id"])
        self.assertEqual("tv-partial", response["results"][1]["product_id"])

    def test_alias_mapping_merge_and_cross_tier_precedence(self) -> None:
        runtime, _, _ = make_runtime()
        tier = runtime._resolve_tier(["  media console  ", "tv stand"])
        self.assertEqual((("TV Stand", "media console"),), tier.pools)
        for alias, canonical in ENABLED_CATEGORY_TERMS.items():
            with self.subTest(alias=alias):
                self.assertEqual(
                    ((canonical, alias),), runtime._resolve_tier([alias]).pools
                )
        response = runtime.retrieve(
            request_wire(primary=["media console"], secondary=["tv stand"], top_k=2)
        )
        self.assertEqual(
            ["primary"], [row["category_tier"] for row in response["results"]]
        )
        self.assertEqual(
            ["CATEGORY_CANONICAL_POOL_DEDUPED"], response["warnings"]
        )

    def test_unsupported_bed_sofa_and_secondary_policies(self) -> None:
        runtime, _, encoder = make_runtime()
        unsupported = runtime.retrieve(request_wire(primary=["chair"]))
        self.assertEqual("no_match", unsupported["status"])
        self.assertEqual(["CATEGORY_UNSUPPORTED"], unsupported["warnings"])
        self.assertEqual(0, encoder.calls)
        bed = runtime.retrieve(request_wire(primary=["bed"]))
        self.assertEqual("no_match", bed["status"])
        self.assertEqual(["CATEGORY_DATA_QUALITY_BLOCKED"], bed["warnings"])
        self.assertEqual(0, encoder.calls)
        sofa = runtime.retrieve(request_wire(primary=["couch"]))
        self.assertEqual("no_match", sofa["status"])
        self.assertEqual(["CATEGORY_DATA_QUALITY_DEGRADED"], sofa["warnings"])
        self.assertEqual(1, encoder.calls)
        secondary = runtime.retrieve(
            request_wire(primary=["chair"], secondary=["desk"], top_k=1)
        )
        self.assertEqual("secondary", secondary["results"][0]["category_tier"])
        self.assertEqual(
            ["CATEGORY_PARTIALLY_UNSUPPORTED"], secondary["warnings"]
        )

    def test_unknown_category_rejects_without_encoder(self) -> None:
        runtime, _, encoder = make_runtime()
        with self.assertRaisesRegex(ValueError, "controlled vocabulary"):
            runtime.retrieve(request_wire(primary=["unreviewed furniture"]))
        self.assertEqual(0, encoder.calls)


class ProductRowMatrixTests(unittest.TestCase):
    def runtime_for_rows(self, rows: list[dict[str, Any]]) -> ProductCandidateRuntime:
        for index, row in enumerate(rows):
            row["row_index"] = index
        vectors = np.zeros((len(rows), 2), dtype=np.float32)
        vectors[:, 0] = 1.0
        return ProductCandidateRuntime.from_components(
            profile_id="row-matrix",
            provenance=provenance(),
            vectors=vectors,
            rows=rows,
            tokenizer=FakeBaseTokenizer(),
            tokenizer_id="fixture-tokenizer-v1",
            text_encoder=FakeEncoder(),
        )

    def test_complete_and_partial_optional_are_eligible(self) -> None:
        complete = product_row("complete")
        partial = product_row(
            "partial",
            dimensions={
                "unit": "mm", "width": 1000, "depth": 500, "height": None,
            },
            dimension_status="partial",
            height_policy="optional",
        )
        runtime = self.runtime_for_rows([complete, partial])
        self.assertTrue(runtime.rows[0].production_eligible)
        self.assertTrue(runtime.rows[1].production_eligible)

    def test_partial_required_and_unknown_fail_closed(self) -> None:
        for policy in ("required", "unknown"):
            row = product_row(
                f"partial-{policy}",
                dimensions={
                    "unit": "mm", "width": 1000, "depth": 500, "height": None,
                },
                dimension_status="partial",
                height_policy=policy,
            )
            with self.subTest(policy=policy), self.assertRaisesRegex(
                ValueError, "required height"
            ):
                self.runtime_for_rows([row])

    def test_missing_suspect_range_and_special_can_never_be_eligible(self) -> None:
        cases = [
            product_row("missing", dimensions=None, dimension_status="missing"),
            product_row("suspect", dimension_status="suspect"),
            product_row("range", dimension_status="range_collapsed"),
            product_row(
                "special",
                dimension_status="special_unmodeled",
                footprint_kind="special_unmodeled",
            ),
        ]
        for row in cases:
            with self.subTest(status=row["dimension_status"]), self.assertRaises(
                ValueError
            ):
                self.runtime_for_rows([row])

    def test_blocked_and_unknown_quality_can_never_be_eligible(self) -> None:
        for quality in ("blocked", "unknown"):
            with self.subTest(quality=quality), self.assertRaises(ValueError):
                self.runtime_for_rows([product_row(quality, quality=quality)])

    def test_dimension_bool_nan_inf_nonpositive_and_cross_field_rejected(self) -> None:
        cases: list[tuple[str, dict[str, Any]]] = []
        for key in ("width", "depth", "height"):
            for bad in (True, 0, -1, float("nan"), float("inf"), float("-inf")):
                dims = complete_dimensions()
                dims[key] = bad
                cases.append(
                    (
                        f"{key}={bad}",
                        product_row(f"bad-{key}-{len(cases)}", dimensions=dims),
                    )
                )
        null_height = complete_dimensions()
        null_height["height"] = None
        cases.append(
            ("complete-null-height", product_row("complete-null", dimensions=null_height))
        )
        cases.append(
            (
                "partial-with-height",
                product_row(
                    "partial-height",
                    dimensions=complete_dimensions(),
                    dimension_status="partial",
                ),
            )
        )
        for label, row in cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.runtime_for_rows([row])

    def test_duplicate_ids_flags_and_yaws_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "product IDs"):
            self.runtime_for_rows([product_row("same"), product_row("same")])
        with self.assertRaisesRegex(ValueError, "special_flags"):
            self.runtime_for_rows(
                [product_row("flags", flags=["FLAG", "FLAG"])]
            )
        with self.assertRaisesRegex(ValueError, "allowed_yaws"):
            self.runtime_for_rows([product_row("yaws", yaws=[0, 0])])
        with self.assertRaisesRegex(ValueError, "allowed_yaws"):
            self.runtime_for_rows([product_row("bool-yaw", yaws=[True])])

    def test_missing_key_and_nonfinite_vectors_fail_closed(self) -> None:
        missing = product_row("missing-name")
        missing.pop("name")
        with self.assertRaisesRegex(ValueError, "name"):
            self.runtime_for_rows([missing])
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ProductCandidateRuntime.from_components(
                    profile_id="bad-vector",
                    provenance=provenance(),
                    vectors=np.asarray([[bad, 0.0]], dtype=np.float32),
                    rows=[product_row("vector")],
                    tokenizer=FakeBaseTokenizer(),
                    text_encoder=FakeEncoder(),
                )


class ServiceContractTests(unittest.TestCase):
    def app_client(
        self,
        runtime: ProductCandidateRuntime,
        *,
        token: str = TOKEN,
        require_strong_token: bool = False,
        timeout: float = 25.0,
        max_queue: int = 1,
    ) -> TestClient:
        return TestClient(
            create_app(
                runtime=runtime,
                bearer_token=token,
                require_strong_token=require_strong_token,
                hard_timeout_s=timeout,
                max_queue=max_queue,
            )
        )

    @staticmethod
    def auth(token: str = TOKEN) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def test_health_ready_and_success_response_are_strict(self) -> None:
        runtime, _, _ = make_runtime()
        with self.app_client(runtime) as client:
            health = client.get("/healthz")
            self.assertEqual(200, health.status_code)
            self.assertEqual("no-store", health.headers["cache-control"])
            ready = client.get("/readyz")
            self.assertEqual(200, ready.status_code)
            self.assertEqual("no-store", ready.headers["cache-control"])
            Draft202012Validator(READY_SCHEMA).validate(ready.json())
            response = client.post(
                "/v2/product-candidates",
                headers=self.auth(),
                json=request_wire(top_k=2),
            )
        self.assertEqual(200, response.status_code, response.text)
        payload = response.json()
        self.assertEqual("ok", payload["status"])
        self.assertNotIn("catalog_metadata", payload["results"][0])
        self.assertEqual(sorted(payload["warnings"]), payload["warnings"])

    def test_surrounding_whitespace_is_trimmed_and_exact_consumer_accepts(self) -> None:
        runtime, _, _ = make_runtime()
        wire = request_wire(
            primary=["  DeSk  "], top_k=1, request_id="trimmed-category-001"
        )
        with self.app_client(runtime) as client:
            response = client.post(
                "/v2/product-candidates", headers=self.auth(), json=wire
            )
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(
            "DeSk", response.json()["results"][0]["matched_request_category"]
        )
        strict_request = ProductRetrievalRequest(
            request_id="trimmed-category-001",
            query_text=wire["query_text"],
            categories=("  DeSk  ",),
            top_k=1,
        )
        batch = HttpProductRetriever._parse_response(
            strict_request, response.content
        )
        self.assertEqual("desk-a", batch.products[0].product_id)

    def test_same_app_rebinds_idle_gate_across_two_testclient_loops(self) -> None:
        runtime, _, _ = make_runtime()
        app = create_app(runtime=runtime, bearer_token=TOKEN)
        statuses = []
        for _ in range(2):
            with TestClient(app) as client:
                statuses.append(
                    client.post(
                        "/v2/product-candidates",
                        headers=self.auth(),
                        json=request_wire(top_k=1),
                    ).status_code
                )
        self.assertEqual([200, 200], statuses)

    def test_auth_and_content_type_fail_closed(self) -> None:
        runtime, _, encoder = make_runtime()
        with self.app_client(runtime) as client:
            for headers in (
                {},
                {"Authorization": "Bearer wrong"},
                {"Authorization": TOKEN},
            ):
                with self.subTest(headers=headers):
                    response = client.post(
                        "/v2/product-candidates",
                        headers=headers,
                        json=request_wire(),
                    )
                    self.assertEqual(401, response.status_code)
            response = client.post(
                "/v2/product-candidates",
                headers={**self.auth(), "Content-Type": "text/plain"},
                content=json.dumps(request_wire()),
            )
            self.assertEqual(415, response.status_code)
        self.assertEqual(0, encoder.calls)

    def test_schema_semantic_json_and_duplicate_negative_matrix(self) -> None:
        runtime, _, encoder = make_runtime()
        valid = request_wire()
        cases: list[tuple[str, bytes]] = []
        unknown = copy.deepcopy(valid)
        unknown["unknown"] = 1
        cases.append(("unknown", json.dumps(unknown).encode()))
        missing = copy.deepcopy(valid)
        missing.pop("query_text")
        cases.append(("missing", json.dumps(missing).encode()))
        boolean = copy.deepcopy(valid)
        boolean["top_k_per_tier"] = True
        cases.append(("bool", json.dumps(boolean).encode()))
        cases.append(
            (
                "nan",
                b'{"schema_version":"1.0","request_id":"x","query_text":NaN}',
            )
        )
        cases.append(("inf", b'{"x":Infinity}'))
        whitespace_query = copy.deepcopy(valid)
        whitespace_query["query_text"] = "   "
        cases.append(("whitespace-query", json.dumps(whitespace_query).encode()))
        whitespace_term = copy.deepcopy(valid)
        whitespace_term["categories_primary"] = ["   "]
        cases.append(("whitespace-term", json.dumps(whitespace_term).encode()))
        duplicate_tier = copy.deepcopy(valid)
        duplicate_tier["categories_primary"] = ["desk", "Desk"]
        cases.append(("duplicate-case-tier", json.dumps(duplicate_tier).encode()))
        duplicate_cross = copy.deepcopy(valid)
        duplicate_cross["categories_secondary"] = ["DESK"]
        cases.append(
            ("duplicate-case-cross-tier", json.dumps(duplicate_cross).encode())
        )
        unknown_term = copy.deepcopy(valid)
        unknown_term["categories_primary"] = ["unknown furniture"]
        cases.append(("unknown-term", json.dumps(unknown_term).encode()))
        cases.append(
            (
                "duplicate-json-key",
                b'{"schema_version":"1.0","schema_version":"1.0","request_id":"x"}',
            )
        )

        with self.app_client(runtime) as client:
            for label, body in cases:
                with self.subTest(label=label):
                    response = client.post(
                        "/v2/product-candidates",
                        headers={
                            **self.auth(),
                            "Content-Type": "application/json",
                        },
                        content=body,
                    )
                    self.assertEqual(422, response.status_code, response.text)
                    self.assertEqual({"error"}, set(response.json()))
        self.assertEqual(0, encoder.calls)

    def test_invalid_utf8_and_body_over_4mib_are_rejected(self) -> None:
        runtime, _, encoder = make_runtime()
        headers = {**self.auth(), "Content-Type": "application/json"}
        with self.app_client(runtime) as client:
            invalid_utf8 = client.post(
                "/v2/product-candidates", headers=headers, content=b"\xff\xfe"
            )
            self.assertEqual(422, invalid_utf8.status_code)
            too_large = client.post(
                "/v2/product-candidates",
                headers=headers,
                content=b" " * (4 * 1024 * 1024 + 1),
            )
            self.assertEqual(413, too_large.status_code)
        self.assertEqual(0, encoder.calls)

    def test_response_over_4mib_is_503_without_partial_or_fallback(self) -> None:
        runtime, _, _ = make_runtime()
        original_retrieve = runtime.retrieve

        def oversized(wire: dict[str, Any]) -> dict[str, Any]:
            payload = original_retrieve(wire)
            payload["warnings"] = ["A" * (4 * 1024 * 1024)]
            return payload

        runtime.retrieve = oversized  # type: ignore[method-assign]
        with self.app_client(runtime) as client:
            response = client.post(
                "/v2/product-candidates",
                headers=self.auth(),
                json=request_wire(top_k=1),
            )
        self.assertEqual(503, response.status_code)
        self.assertEqual("RESPONSE_TOO_LARGE", response.json()["error"]["code"])
        self.assertNotIn("status", response.json())

    def test_http_token_76_rejects_before_encoder(self) -> None:
        runtime, tokenizer, encoder = make_runtime()
        with self.app_client(runtime) as client:
            response = client.post(
                "/v2/product-candidates",
                headers=self.auth(),
                json=request_wire(query="TOKENS:76", top_k=1),
            )
        self.assertEqual(422, response.status_code)
        self.assertEqual(
            "QUERY_TOKEN_LIMIT_EXCEEDED", response.json()["error"]["code"]
        )
        self.assertEqual(0, tokenizer.tensor_calls)
        self.assertEqual(0, encoder.calls)

    def test_not_ready_and_encoder_failure_are_503_without_fallback(self) -> None:
        checks = {
            "model_checkpoint": True,
            "vector_bundle": False,
            "catalog_manifest": True,
            "dimension_bundle": True,
            "tokenizer_bundle": True,
            "row_identity_join": True,
            "policy_versions": True,
        }
        runtime, _, _ = make_runtime(checks=checks)
        with self.app_client(runtime) as client:
            ready = client.get("/readyz")
            self.assertEqual(503, ready.status_code)
            Draft202012Validator(READY_SCHEMA).validate(ready.json())
            response = client.post(
                "/v2/product-candidates",
                headers=self.auth(),
                json=request_wire(),
            )
            self.assertEqual(503, response.status_code)
            self.assertEqual({"error"}, set(response.json()))

        failed_runtime, _, _ = make_runtime(
            encoder=FakeEncoder(error=RuntimeError("boom"))
        )
        with self.app_client(failed_runtime) as client:
            failed = client.post(
                "/v2/product-candidates",
                headers=self.auth(),
                json=request_wire(),
            )
            self.assertEqual(503, failed.status_code)
            self.assertEqual(
                "ULIP_ENCODER_FAILED", failed.json()["error"]["code"]
            )
            self.assertNotIn("results", failed.json())

    def test_strong_token_failure_has_schema_valid_not_ready_payload(self) -> None:
        runtime, _, _ = make_runtime()
        with self.app_client(
            runtime, token="short", require_strong_token=True
        ) as client:
            ready = client.get("/readyz")
            self.assertEqual(503, ready.status_code)
            Draft202012Validator(READY_SCHEMA).validate(ready.json())
            self.assertIn(
                "AUTH_CONFIGURATION_INVALID", ready.json()["failures"]
            )

    def test_post_start_integrity_drift_makes_readyz_and_post_fail_closed(self) -> None:
        integrity = {"healthy": True, "calls": 0}

        def revalidate() -> bool:
            integrity["calls"] += 1
            if not integrity["healthy"]:
                raise RuntimeError("synthetic member hash drift")
            return True

        runtime, _, encoder = make_runtime(integrity_revalidator=revalidate)
        with self.app_client(runtime) as client:
            initial = client.get("/readyz")
            self.assertEqual(200, initial.status_code)
            self.assertGreaterEqual(integrity["calls"], 1)
            integrity["healthy"] = False
            drifted = client.get("/readyz")
            self.assertEqual(503, drifted.status_code)
            self.assertIn(
                "DEPLOYMENT_INTEGRITY_REVALIDATION_FAILED",
                drifted.json()["failures"],
            )
            Draft202012Validator(READY_SCHEMA).validate(drifted.json())
            blocked = client.post(
                "/v2/product-candidates",
                headers=self.auth(),
                json=request_wire(),
            )
            self.assertEqual(503, blocked.status_code)
            self.assertEqual(0, encoder.calls)
            integrity["healthy"] = True
            recovered = client.get("/readyz")
            self.assertEqual(200, recovered.status_code)

    def test_hard_timeout_returns_503_without_fallback(self) -> None:
        runtime, _, _ = make_runtime(encoder=FakeEncoder(delay_s=0.12))
        with self.app_client(runtime, timeout=0.02) as client:
            started = time.monotonic()
            response = client.post(
                "/v2/product-candidates",
                headers=self.auth(),
                json=request_wire(),
            )
            elapsed = time.monotonic() - started
            self.assertEqual(503, response.status_code)
            self.assertEqual(
                "INFERENCE_TIMEOUT", response.json()["error"]["code"]
            )
            self.assertLess(elapsed, 0.1)
            time.sleep(0.14)

    def test_one_active_one_queued_excess_returns_503_retry_after(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        runtime, _, _ = make_runtime(
            encoder=FakeEncoder(entered=entered, release=release)
        )
        app = create_app(
            runtime=runtime,
            bearer_token=TOKEN,
            hard_timeout_s=2.0,
            max_queue=1,
        )

        async def scenario() -> tuple[
            httpx.Response, httpx.Response, httpx.Response
        ]:
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://testserver"
                ) as client:
                    kwargs = {
                        "headers": self.auth(),
                        "json": request_wire(top_k=1),
                    }
                    first = asyncio.create_task(
                        client.post("/v2/product-candidates", **kwargs)
                    )
                    entered_ok = await asyncio.to_thread(entered.wait, 1.0)
                    self.assertTrue(entered_ok)
                    second = asyncio.create_task(
                        client.post("/v2/product-candidates", **kwargs)
                    )
                    for _ in range(100):
                        if app.state.product_candidates.gate._queued == 1:
                            break
                        await asyncio.sleep(0.005)
                    self.assertEqual(
                        1, app.state.product_candidates.gate._queued
                    )
                    third = await client.post(
                        "/v2/product-candidates", **kwargs
                    )
                    release.set()
                    one, two = await asyncio.gather(first, second)
                    return one, two, third

        one, two, third = asyncio.run(scenario())
        self.assertEqual([200, 200], [one.status_code, two.status_code])
        self.assertEqual(503, third.status_code)
        self.assertEqual(
            "INFERENCE_QUEUE_FULL", third.json()["error"]["code"]
        )
        self.assertEqual("1", third.headers["retry-after"])


if __name__ == "__main__":
    unittest.main()
