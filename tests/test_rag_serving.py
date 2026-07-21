import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ikea.rag_serving import (
    CompatibilityReport,
    RAGCompatibilityError,
    RAGServingAdapter,
    RetrievedDocument,
    tokenize_preserving_eot,
    validate_corpus_profile,
    validate_exact_base_prefixes,
    validate_provenance_profile,
    validate_vector_bundle,
)


class FakeTokenizer:
    encoder = {"<|startoftext|>": 98, "<|endoftext|>": 99}

    def encode(self, text):
        return [1 + (ord(char) % 30) for char in text]


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.token_embedding = nn.Embedding(100, 4)
        self.positional_embedding = nn.Parameter(torch.zeros(77, 4))
        self.transformer = nn.Identity()
        self.ln_final = nn.Identity()
        self.text_projection = nn.Parameter(torch.eye(4))


class FakeEnhancer(nn.Module):
    def forward(self, original, documents, padding):
        del padding
        return original + torch.roll(documents.mean(dim=1), shifts=1, dims=-1)


class FakeRetriever:
    def retrieve(self, queries, top_k=None):
        del top_k
        return [
            [RetrievedDocument(1, 0, 0.75, f"knowledge for {query}", "chair")]
            for query in queries
        ]


def compatible_report():
    return CompatibilityReport(True, 1, (), (), (), ())


class CompatibilityTests(unittest.TestCase):
    def state(self):
        return {
            "token_embedding.weight": torch.ones(2, 2),
            "positional_embedding": torch.ones(3, 2),
            "transformer.block.weight": torch.ones(2, 2),
            "ln_final.weight": torch.ones(2),
            "text_projection": torch.ones(2, 2),
            "visual.block.weight": torch.ones(2, 2),
            "image_projection": torch.ones(2, 2),
            "logit_scale": torch.ones(1),
            "point_encoder.unrelated": torch.zeros(1),
        }

    def test_exact_prefixes_accept_identical_text_space(self):
        report = validate_exact_base_prefixes(self.state(), self.state())
        self.assertTrue(report.compatible)
        self.assertEqual(report.checked_tensors, 8)

    def test_exact_prefixes_reject_same_shape_different_value(self):
        left, right = self.state(), self.state()
        right["text_projection"] = torch.zeros(2, 2)
        with self.assertRaises(RAGCompatibilityError):
            validate_exact_base_prefixes(left, right)

    def test_named_corpus_profile_fails_on_hash_drift(self):
        expected = {
            "name": "fixture",
            "corpus_rows": 2,
            "faiss_rows": 2,
            "faiss_dimension": 3,
            "corpus_sha256": "a",
            "index_sha256": "b",
        }
        with self.assertRaises(RAGCompatibilityError):
            validate_corpus_profile(
                expected,
                corpus_rows=2,
                faiss_rows=2,
                faiss_dimension=3,
                corpus_sha256="drift",
                index_sha256="b",
            )

    def provenance_checkpoint(self, strategy, corpus="corpus", index="index"):
        checkpoint = {
            "args": {"model": "ULIP_PointBERT_RAG"},
            "provenance": {
                "schema_version": "ikea-rag-training-provenance/1.0",
                "pipeline": "ikea_rag",
                "training_strategy": strategy,
                "rag": {
                    "corpus_sha256": corpus,
                    "index_sha256": index,
                },
                "dataset": {"bundle_sha256": "dataset"},
                "base_ulip_checkpoint": {"sha256": "base-sha"},
            },
        }
        return checkpoint

    def test_provenance_profile_accepts_stage2_bound_to_stage1(self):
        stage1 = self.provenance_checkpoint("stage_1")
        stage2 = self.provenance_checkpoint("stage_2")
        stage2["provenance"]["stage1_checkpoint"] = {"sha256": "stage1-sha"}
        result = validate_provenance_profile(
            stage2,
            stage1,
            enhancer_checkpoint_sha256="stage1-sha",
            base_checkpoint_sha256="stage2-serving-sha",
            corpus_sha256="corpus",
            index_sha256="index",
        )
        self.assertTrue(result["stage2_validated"])
        self.assertEqual(result["base_training_strategy"], "stage_2")

    def test_provenance_profile_rejects_stage1_corpus_drift(self):
        stage1 = self.provenance_checkpoint("stage_1", corpus="old")
        vanilla = {"args": {"model": "ULIP_PointBERT"}}
        with self.assertRaisesRegex(RAGCompatibilityError, "corpus_sha256"):
            validate_provenance_profile(
                vanilla,
                stage1,
                enhancer_checkpoint_sha256="stage1-sha",
                base_checkpoint_sha256="base-sha",
                corpus_sha256="current",
                index_sha256="index",
            )

    def test_provenance_profile_rejects_wrong_stage1_chain(self):
        stage1 = self.provenance_checkpoint("stage_1")
        stage2 = self.provenance_checkpoint("stage_2")
        stage2["provenance"]["stage1_checkpoint"] = {"sha256": "other"}
        with self.assertRaisesRegex(RAGCompatibilityError, "different Stage 1"):
            validate_provenance_profile(
                stage2,
                stage1,
                enhancer_checkpoint_sha256="stage1-sha",
                base_checkpoint_sha256="stage2-serving-sha",
                corpus_sha256="corpus",
                index_sha256="index",
            )

    def test_provenance_profile_rejects_stage_dataset_lineage_drift(self):
        stage1 = self.provenance_checkpoint("stage_1")
        stage2 = self.provenance_checkpoint("stage_2")
        stage2["provenance"]["stage1_checkpoint"] = {"sha256": "stage1-sha"}
        stage2["provenance"]["dataset"]["bundle_sha256"] = "other-dataset"
        with self.assertRaisesRegex(RAGCompatibilityError, "dataset.bundle_sha256"):
            validate_provenance_profile(
                stage2,
                stage1,
                enhancer_checkpoint_sha256="stage1-sha",
                base_checkpoint_sha256="stage2-serving-sha",
                corpus_sha256="corpus",
                index_sha256="index",
            )

    def write_vector_bundle(self, directory, checkpoint_sha="base-sha"):
        directory = Path(directory)
        vectors = np.eye(3, dtype=np.float32)
        np.save(directory / "vectors_pc.npy", vectors)
        with (directory / "meta_pc.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(3):
                handle.write(json.dumps({"id": str(index)}) + "\n")
        (directory / "schema.json").write_text(
            json.dumps(
                {
                    "schema_version": "ikea-ulip-vectors/2.0",
                    "modalities": ["pc"],
                    "checkpoint": {"sha256": checkpoint_sha},
                    "text_embedding_mode": "vanilla",
                    "dims": {"pc": 3},
                }
            ),
            encoding="utf-8",
        )
        return vectors

    def test_provenance_vector_bundle_binds_base_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            vectors = self.write_vector_bundle(directory)
            result = validate_vector_bundle(
                Path(directory),
                corpus_profile="provenance",
                base_checkpoint_sha256="base-sha",
                vectors=vectors,
                metadata_rows=3,
                embedding_dimension=3,
            )
            self.assertEqual(result["schema_version"], "ikea-ulip-vectors/2.0")
            self.assertEqual(len(result["vectors_pc"]["sha256"]), 64)

    def test_provenance_vector_bundle_rejects_checkpoint_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            vectors = self.write_vector_bundle(directory, checkpoint_sha="other")
            with self.assertRaisesRegex(RAGCompatibilityError, "checkpoint.sha256"):
                validate_vector_bundle(
                    Path(directory),
                    corpus_profile="provenance",
                    base_checkpoint_sha256="base-sha",
                    vectors=vectors,
                    metadata_rows=3,
                    embedding_dimension=3,
                )

    def test_legacy_vector_bundle_rejects_unknown_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            vectors = self.write_vector_bundle(directory)
            with self.assertRaisesRegex(RAGCompatibilityError, "legacy1095 vector"):
                validate_vector_bundle(
                    Path(directory),
                    corpus_profile="legacy1095",
                    base_checkpoint_sha256="unused",
                    vectors=vectors,
                    metadata_rows=3,
                    embedding_dimension=3,
                )


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.base = FakeBase()
        self.tokenizer = FakeTokenizer()
        self.adapter = RAGServingAdapter(
            base_model=self.base,
            tokenizer=self.tokenizer,
            enhancer=FakeEnhancer(),
            retriever=FakeRetriever(),
            compatibility=compatible_report(),
            device="cpu",
            pc_vectors=np.eye(4, dtype=np.float32),
            pc_meta=[{"id": str(i), "category": "Chair"} for i in range(4)],
        )

    def test_tokenizer_preserves_eot_when_truncated(self):
        tokens, diag = tokenize_preserving_eot(self.tokenizer, ["x" * 100])
        self.assertTrue(diag[0]["truncated"])
        self.assertTrue(diag[0]["eot_preserved"])
        self.assertEqual(tokens[0, -1].item(), 99)

    def test_encode_supports_vanilla_and_rag_with_diagnostics(self):
        vanilla, vanilla_diag = self.adapter.encode("chair", mode="vanilla")
        rag, rag_diag = self.adapter.encode("chair", mode="rag")
        self.assertEqual(vanilla.shape, (4,))
        self.assertEqual(rag.shape, (4,))
        self.assertAlmostEqual(float(np.linalg.norm(vanilla)), 1.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(rag)), 1.0, places=6)
        self.assertEqual(vanilla_diag["retrieved_documents"], [])
        self.assertEqual(rag_diag["rag_top_k_returned"], 1)
        self.assertNotEqual(rag_diag["rag_vs_vanilla_l2"], 0.0)

    def test_search_returns_raw_ulip_similarity_and_status(self):
        result = self.adapter.search("chair", top_k=2, category="chair", mode="vanilla")
        self.assertEqual(result["count"], 2)
        self.assertIn("ulip_similarity", result["results"][0])
        self.assertTrue(self.adapter.ready)
        self.assertEqual(self.adapter.status()["modes"], ["vanilla", "rag"])


if __name__ == "__main__":
    unittest.main()
