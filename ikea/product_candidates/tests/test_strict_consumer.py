from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


HANDOFF = Path(
    "/home/kyzen/V2T/HANDOFF_5090_TO_3090_PLACEMENT_LOOP_2026-07-15"
)
REFERENCE = HANDOFF / "reference_v2t_da3"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(REFERENCE))

from placement_loop.models import ProductRetrievalRequest  # noqa: E402
from placement_loop.retrievers import (  # noqa: E402
    HttpProductRetriever,
    RetrieverProtocolError,
)


class Exact5090ConsumerCompatibilityTests(unittest.TestCase):
    def request(self) -> ProductRetrievalRequest:
        return ProductRetrievalRequest(
            request_id="fixture-product-candidates-001",
            query_text="fixture query",
            categories=("couch",),
            secondary_categories=("desk",),
            top_k=2,
        )

    def parse(self, payload: object):
        def transport(url, body, headers, timeout):
            del url, body, headers, timeout
            return 200, json.dumps(payload).encode("utf-8")

        return HttpProductRetriever(
            "http://fixture.invalid/v2/product-candidates",
            transport=transport,
            max_retries=0,
        ).retrieve(self.request())

    def test_ok_fixture_passes_exact_strict_consumer_and_converts_mm_once(self) -> None:
        wire = json.loads((FIXTURES / "response.ok.valid.json").read_text())
        batch = self.parse(wire)
        self.assertEqual("ok", batch.status)
        self.assertEqual(2, len(batch.products))
        self.assertEqual("fixture-sofa-001", batch.products[0].product_id)
        self.assertEqual(1.2, batch.products[0].dimensions.width_m)
        self.assertEqual(0.7, batch.products[0].dimensions.depth_m)
        self.assertEqual("secondary", batch.products[1].category_tier)
        self.assertIsNone(batch.products[1].dimensions.height_m)

    def test_no_match_fixture_passes_exact_strict_consumer(self) -> None:
        wire = json.loads((FIXTURES / "response.no_match.valid.json").read_text())
        batch = self.parse(wire)
        self.assertEqual("no_match", batch.status)
        self.assertEqual((), batch.products)

    def test_exact_consumer_rejects_duplicate_id_flag_and_yaw(self) -> None:
        base = json.loads((FIXTURES / "response.ok.valid.json").read_text())
        cases = []
        duplicate_id = json.loads(json.dumps(base))
        duplicate_id["results"][1]["product_id"] = duplicate_id["results"][0]["product_id"]
        cases.append(duplicate_id)
        duplicate_flag = json.loads(json.dumps(base))
        duplicate_flag["results"][0]["special_flags"] = ["FLAG", "FLAG"]
        cases.append(duplicate_flag)
        duplicate_yaw = json.loads(json.dumps(base))
        duplicate_yaw["results"][1]["allowed_yaws_deg"] = [0, 0]
        cases.append(duplicate_yaw)
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(RetrieverProtocolError):
                    self.parse(payload)

    def test_exact_consumer_rejects_nonfinite_and_boolean_numbers(self) -> None:
        base = json.loads((FIXTURES / "response.ok.valid.json").read_text())
        for bad in (True, float("nan"), float("inf"), float("-inf")):
            payload = json.loads(json.dumps(base))
            payload["results"][1]["retrieval_score"] = bad
            with self.subTest(bad=bad):
                with self.assertRaises(RetrieverProtocolError):
                    self.parse(payload)


if __name__ == "__main__":
    unittest.main()
