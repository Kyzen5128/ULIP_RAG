import json
import tempfile
import unittest
from pathlib import Path

from ikea.evaluate_rag_retrieval import (
    _corpus_dir_for,
    _validate_adapter_profile,
    atomic_write_json,
    build_parser,
    evaluate_adapter,
    select_caption_rows,
)
from ikea.rag_serving import RAGCompatibilityError


class FakeAdapter:
    def __init__(self, rankings, errors=None):
        self.rankings = rankings
        self.errors = set(errors or [])
        self.calls = []

    def search(self, query, *, top_k, category, mode, rag_top_k):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "category": category,
                "mode": mode,
                "rag_top_k": rag_top_k,
            }
        )
        if (query, mode) in self.errors:
            raise RuntimeError(f"forced {mode} error")
        ids = self.rankings[(query, mode)]
        return {"results": [{"id": product_id} for product_id in ids[:top_k]]}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": f"p{i}", "category": "Chair", "caption": f"caption {i}"}
            for i in range(12)
        ]

    def test_selection_is_deterministic_and_caption_only(self):
        rows = self.rows + [
            {"id": "empty", "category": "Chair", "caption": ""},
            {"id": "no-category", "category": "", "caption": "usable"},
        ]
        first, excluded = select_caption_rows(rows, limit=5, seed=7)
        second, _ = select_caption_rows(rows, limit=5, seed=7)
        other, _ = select_caption_rows(rows, limit=5, seed=8)
        self.assertEqual(first, second)
        self.assertNotEqual([row["id"] for row in first], [row["id"] for row in other])
        self.assertEqual(excluded["missing_caption"], 1)
        self.assertEqual(excluded["missing_category"], 1)
        self.assertEqual(len(first), 5)

    def test_duplicate_product_id_fails_closed(self):
        rows = self.rows[:1] + [dict(self.rows[0])]
        with self.assertRaisesRegex(ValueError, "duplicate product id"):
            select_caption_rows(rows, limit=2, seed=0)


class MetricTests(unittest.TestCase):
    def samples(self):
        return [
            {"row_index": 0, "id": "a", "category": "Chair", "caption": "query-a"},
            {"row_index": 1, "id": "b", "category": "Table", "caption": "query-b"},
        ]

    def test_metrics_use_same_id_exact_category_and_all_attempts(self):
        adapter = FakeAdapter(
            {
                ("query-a", "vanilla"): ["a", "x"],
                ("query-a", "rag"): ["x", "a"],
                ("query-b", "vanilla"): ["x", "b"],
                ("query-b", "rag"): ["b", "x"],
            }
        )
        result = evaluate_adapter(adapter, self.samples(), top_k=10, rag_top_k=5)
        self.assertEqual(result["metrics"]["vanilla"]["recall_at_1"], 0.5)
        self.assertEqual(result["metrics"]["rag"]["recall_at_1"], 0.5)
        self.assertEqual(result["metrics"]["vanilla"]["recall_at_5"], 1.0)
        self.assertEqual(result["metrics"]["rag"]["recall_at_5"], 1.0)
        self.assertEqual(result["metrics"]["vanilla"]["mrr_at_10"], 0.75)
        self.assertEqual(result["metrics"]["rag"]["mrr_at_10"], 0.75)
        self.assertEqual(result["comparison"]["mean_top_k_overlap"], 1.0)
        self.assertEqual(result["failures"]["count"], 0)
        self.assertEqual(
            {(call["query"], call["category"]) for call in adapter.calls},
            {("query-a", "Chair"), ("query-b", "Table")},
        )

    def test_execution_error_is_reported_and_counted_as_miss(self):
        adapter = FakeAdapter(
            {
                ("query-a", "vanilla"): ["a"],
                ("query-a", "rag"): ["a"],
                ("query-b", "vanilla"): ["b"],
                ("query-b", "rag"): ["b"],
            },
            errors={("query-b", "rag")},
        )
        result = evaluate_adapter(adapter, self.samples(), top_k=10, rag_top_k=5)
        self.assertEqual(result["metrics"]["rag"]["attempted"], 2)
        self.assertEqual(result["metrics"]["rag"]["completed"], 1)
        self.assertEqual(result["metrics"]["rag"]["execution_errors"], 1)
        self.assertEqual(result["metrics"]["rag"]["misses_at_10"], 1)
        self.assertEqual(result["metrics"]["rag"]["recall_at_1"], 0.5)
        self.assertEqual(result["failures"]["count"], 1)
        self.assertEqual(result["failures"]["items"][0]["type"], "execution_error")

    def test_requires_depth_for_mrr_at_10(self):
        with self.assertRaisesRegex(ValueError, "at least 10"):
            evaluate_adapter(FakeAdapter({}), self.samples(), top_k=5)


class OutputTests(unittest.TestCase):
    def test_atomic_json_output(self):
        payload = {"status": "ok", "metric": 0.5}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.json"
            atomic_write_json(destination, payload)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), payload)
            self.assertEqual(list(Path(directory).iterdir()), [destination])


class ProfileTests(unittest.TestCase):
    class StatusAdapter:
        def __init__(self, profile, validation):
            self.profile = profile
            self.validation = validation

        def status(self):
            return {
                "profile": self.profile,
                "profile_validation": self.validation,
            }

    def test_parser_supports_only_adapter_profiles(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--profile", "provenance", "--corpus-dir", "/tmp/corpus"]
        )
        self.assertEqual(args.profile, "provenance")
        self.assertEqual(args.corpus_dir, Path("/tmp/corpus"))
        profile_action = next(
            action for action in parser._actions if action.dest == "profile"
        )
        self.assertEqual(set(profile_action.choices), {"legacy1095", "provenance"})

    def test_provenance_requires_explicit_corpus_directory(self):
        with self.assertRaisesRegex(ValueError, "requires --corpus-dir"):
            _corpus_dir_for("provenance", None)

    def test_adapter_profile_must_match_and_report_validation(self):
        _validate_adapter_profile(
            self.StatusAdapter(
                "legacy1095", {"immutable_bundle_validated": True}
            ),
            "legacy1095",
        )
        _validate_adapter_profile(
            self.StatusAdapter("provenance", {"schema_version": "test/1"}),
            "provenance",
        )
        with self.assertRaises(RAGCompatibilityError):
            _validate_adapter_profile(
                self.StatusAdapter("legacy1095", {"immutable_bundle_validated": True}),
                "provenance",
            )


if __name__ == "__main__":
    unittest.main()
