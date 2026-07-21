import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


IKEA_DIR = Path(__file__).resolve().parents[1]
if str(IKEA_DIR) not in sys.path:
    sys.path.insert(0, str(IKEA_DIR))

from rag_training_utils import (  # noqa: E402
    PROVENANCE_SCHEMA,
    canonical_sha256,
    validate_stage1_checkpoint,
)
from main_ikea_rag import IkeaRAGDatasetAdapter, _set_train_modes  # noqa: E402


ENHANCER_SHAPES = {
    "attention.in_proj_weight": (1536, 512),
    "attention.in_proj_bias": (1536,),
    "attention.out_proj.weight": (512, 512),
    "attention.out_proj.bias": (512,),
    "norm1.weight": (512,),
    "norm1.bias": (512,),
    "norm2.weight": (512,),
    "norm2.bias": (512,),
    "ffn.0.weight": (1024, 512),
    "ffn.0.bias": (1024,),
    "ffn.2.weight": (512, 1024),
    "ffn.2.bias": (512,),
}


class RAGTrainingUtilsTests(unittest.TestCase):
    def _checkpoint(self, path: Path, corpus="corpus", index="index", dataset="data"):
        state = {
            f"module.rag_enhancer.{key}": torch.zeros(shape)
            for key, shape in ENHANCER_SHAPES.items()
        }
        torch.save(
            {
                "state_dict": state,
                "provenance": {
                    "schema_version": PROVENANCE_SCHEMA,
                    "pipeline": "ikea_rag",
                    "training_strategy": "stage_1",
                    "rag": {"corpus_sha256": corpus, "index_sha256": index},
                    "dataset": {"bundle_sha256": dataset},
                },
            },
            path,
        )

    def test_canonical_hash_does_not_depend_on_mapping_order(self):
        self.assertEqual(canonical_sha256({"a": 1, "b": 2}), canonical_sha256({"b": 2, "a": 1}))

    def test_stage1_checkpoint_is_bound_to_rag_and_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage1.pt"
            self._checkpoint(path)
            result = validate_stage1_checkpoint(
                path,
                expected_rag={"corpus_sha256": "corpus", "index_sha256": "index"},
                expected_dataset_bundle_sha256="data",
            )
            self.assertEqual(set(result["enhancer_state"]), set(ENHANCER_SHAPES))
            self.assertEqual(len(result["sha256"]), 64)

    def test_stage1_checkpoint_rejects_corpus_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage1.pt"
            self._checkpoint(path)
            with self.assertRaisesRegex(ValueError, "corpus_sha256"):
                validate_stage1_checkpoint(
                    path,
                    expected_rag={"corpus_sha256": "changed", "index_sha256": "index"},
                    expected_dataset_bundle_sha256="data",
                )

    def test_stage1_checkpoint_rejects_missing_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage1.pt"
            self._checkpoint(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            del checkpoint["state_dict"]["module.rag_enhancer.norm2.bias"]
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "missing RAG enhancer"):
                validate_stage1_checkpoint(path)

    def test_ikea_adapter_preserves_raw_caption_for_retrieval(self):
        class FakeTokenizer:
            def __call__(self, text):
                self.last_text = text
                return torch.arange(77, dtype=torch.long)

        class FakeBase:
            render_pick = "fixed_000"
            tokenizer = FakeTokenizer()
            samples = [
                {
                    "id": "product-1",
                    "category": "Dining Chair",
                    "caption": "a pale oak dining chair",
                    "image_path": "fallback.png",
                    "render_paths": [],
                }
            ]

            def __len__(self):
                return 1

            def _load_image(self, _path):
                return torch.zeros((3, 224, 224))

        adapter = IkeaRAGDatasetAdapter(FakeBase(), include_pointcloud=False)
        category, product_id, text, pointcloud, image = adapter[0]
        self.assertEqual(category, "Dining Chair")
        self.assertEqual(product_id, "product-1")
        self.assertEqual(text[1], ["a pale oak dining chair"])
        self.assertEqual(tuple(text[0].shape), (1, 77))
        self.assertEqual(tuple(pointcloud.shape), (1, 3))
        self.assertEqual(tuple(image.shape), (3, 224, 224))

    def test_stage_modes_keep_frozen_targets_deterministic(self):
        class FakeRAGModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = torch.nn.Dropout()
                self.transformer = torch.nn.Dropout()
                self.point_encoder = torch.nn.Dropout()
                self.rag_enhancer = torch.nn.Dropout()

        model = FakeRAGModel()
        _set_train_modes(model, "stage1")
        self.assertTrue(model.training)
        self.assertFalse(model.visual.training)
        self.assertFalse(model.transformer.training)
        self.assertFalse(model.point_encoder.training)
        self.assertTrue(model.rag_enhancer.training)

        _set_train_modes(model, "stage2")
        self.assertFalse(model.visual.training)
        self.assertFalse(model.transformer.training)
        self.assertTrue(model.point_encoder.training)
        self.assertFalse(model.rag_enhancer.training)


if __name__ == "__main__":
    unittest.main()
