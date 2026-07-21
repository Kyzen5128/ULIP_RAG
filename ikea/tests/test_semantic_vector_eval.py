import tempfile
import unittest
from pathlib import Path

import numpy as np

from ikea.loop2.evaluate_semantic_vectors import load_modality, retrieval_metrics


class SemanticVectorEvaluationTests(unittest.TestCase):
    def test_metrics_use_exact_aligned_diagonal(self):
        gallery = np.eye(3, dtype=np.float32)
        query = gallery[[0, 2, 1]]
        metrics = retrieval_metrics(query, gallery[[0, 2, 1]])
        self.assertEqual(metrics["queries"], 3)
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["mrr_at_10"], 1.0)

    def test_vector_metadata_row_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "vectors_pc.npy", np.eye(2, dtype=np.float32))
            (root / "meta_pc.jsonl").write_text(
                '{"id":"one"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "vector/meta mismatch"):
                load_modality(root, "pc")


if __name__ == "__main__":
    unittest.main()
