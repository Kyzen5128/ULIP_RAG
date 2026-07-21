from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from bson import ObjectId
from fastapi.testclient import TestClient

from ikea.product_candidates.bundles import (
    BundleError,
    build_deployment_bundle,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    load_deployment_profile,
    sha256_bytes,
    sha256_file,
    write_deployment_bundle,
)
from ikea.product_candidates.runtime import ProductCandidateRuntime
from ikea.product_candidates.service import create_app


SOURCE_META = Path("/mnt/P300/data/ikea_data/vectors/meta_pc.jsonl")


class _FakeCollection:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents

    def find(self, query: dict) -> list[dict]:
        requested = {str(value) for value in query["_id"]["$in"]}
        return [row for row in self.documents if str(row["_id"]) in requested]

    def count_documents(self, query: dict) -> int:
        del query
        return len(self.documents)


class _FakeDatabase:
    def __init__(self, collections: dict[str, _FakeCollection]) -> None:
        self.collections = collections

    def command(self, command: str) -> dict:
        if command != "ping":
            raise AssertionError(f"write/non-ping command attempted: {command}")
        return {"ok": 1}

    def __getitem__(self, name: str) -> _FakeCollection:
        return self.collections[name]


class _FakeMongoClient:
    def __init__(self, database: _FakeDatabase) -> None:
        self.database = database

    def __getitem__(self, name: str) -> _FakeDatabase:
        self.last_database_name = name
        return self.database


class _RuntimeOnlyTokenizer:
    eot_token_id = 49407
    encoder = {"<|endoftext|>": 49407}

    def encode(self, text: str) -> list[int]:
        del text
        return [1]

    def __call__(self, texts: list[str], context_length: int = 77) -> np.ndarray:
        del texts
        result = np.zeros((1, context_length), dtype=np.int64)
        result[0, 0] = 49406
        result[0, 2] = 49407
        return result


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _resign_catalog_member(profile_dir: Path, rows: list[dict]) -> None:
    """Re-sign a deliberately bad catalog member so join checks are reached."""

    member_path = profile_dir / "catalog/catalog_rows.jsonl"
    member_bytes = canonical_jsonl_bytes(rows)
    member_path.write_bytes(member_bytes)

    manifest_path = profile_dir / "manifests/catalog_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["member"]["bytes"] = len(member_bytes)
    manifest["member"]["content_sha256"] = sha256_bytes(member_bytes)
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = sha256_bytes(manifest_bytes)

    profile_path = profile_dir / "deployment_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["manifests"]["catalog_manifest"]["sha256"] = manifest_sha
    profile["provenance"]["catalog_manifest_sha256"] = manifest_sha
    profile_path.write_bytes(canonical_json_bytes(profile))


class ImmutableBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="product-candidates-bundle-test-")
        cls.root = Path(cls.temp.name)
        cls.sources = cls.root / "sources"
        cls.sources.mkdir()

        cls.meta_path = cls.sources / "meta_pc.jsonl"
        shutil.copyfile(SOURCE_META, cls.meta_path)
        meta_rows = _read_jsonl(cls.meta_path)
        if len(meta_rows) != 733:
            raise AssertionError("unit fixture requires the reviewed 733-row metadata")

        vectors = np.zeros((733, 512), dtype=np.float32)
        vectors[:, 0] = 1.0
        cls.vectors_path = cls.sources / "vectors_pc.npy"
        np.save(cls.vectors_path, vectors)
        cls.checkpoint_path = cls.sources / "checkpoint.pt"
        cls.checkpoint_path.write_bytes(b"synthetic-checkpoint-unit-test\n")
        cls.schema_path = cls.sources / "schema.json"
        cls.schema_path.write_bytes(canonical_json_bytes({"shape": [733, 512]}))
        cls.tokenizer_path = cls.sources / "tokenizer.py"
        cls.tokenizer_path.write_text("# synthetic tokenizer fixture\n", encoding="utf-8")
        cls.bpe_path = cls.sources / "bpe.gz"
        cls.bpe_path.write_bytes(b"synthetic-bpe-fixture")

        quality = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "policies/quality_overrides_v1.json"
            ).read_text(encoding="utf-8")
        )
        range_rules = quality["range_evidence"]
        slash_ids = set(range_rules["string_parser_collapse"]["slash_multivalue_ids"])
        hyphen_ids = set(range_rules["string_parser_collapse"]["hyphen_ids"])
        paired_ids = set(range_rules["paired_dict_range"]["ids"])
        max_only_ids = set(range_rules["max_only"]["ids"])

        catalog_documents = []
        dimension_documents = []
        for meta in meta_rows:
            product_id = str(meta["id"])
            category = str(meta["category"])
            catalog_document = {
                "_id": ObjectId(product_id),
                "main_type": category,
                "name": f"Fixture {product_id}",
                "brand": "Fixture",
            }
            if product_id in slash_ids:
                catalog_document["size_options"] = "10/20 x 30"
            elif product_id in hyphen_ids:
                catalog_document["size_options"] = "10-20 x 30"
            elif product_id in paired_ids:
                catalog_document["size_options"] = {
                    "Min. Width": "10",
                    "Max. Width": "20",
                }
            elif product_id in max_only_ids:
                catalog_document["size_options"] = {"Max. Width": "20"}
            catalog_documents.append(catalog_document)
            dimension_documents.append(
                {
                    "_id": ObjectId(product_id),
                    "main_type": category,
                    "meta": {
                        "dimensions_mm": {
                            "length": 1000.0,
                            "width": 500.0,
                            "height": 700.0,
                        },
                        "dimensions_source": {
                            "method": "unit_fixture",
                            "source_format": "structured",
                            "confidence": "fixture",
                            "keys": ["length", "width", "height"],
                        },
                    },
                }
            )
        fake_client = _FakeMongoClient(
            _FakeDatabase(
                {
                    "ikea_product": _FakeCollection(catalog_documents),
                    "ikea_product_v2_2026q2": _FakeCollection(dimension_documents),
                }
            )
        )

        cls.source_hashes = {
            path: sha256_file(path)
            for path in (
                cls.meta_path,
                cls.vectors_path,
                cls.checkpoint_path,
                cls.schema_path,
                cls.tokenizer_path,
                cls.bpe_path,
            )
        }
        cls.build = build_deployment_bundle(
            checkpoint_path=cls.checkpoint_path,
            vectors_path=cls.vectors_path,
            meta_path=cls.meta_path,
            vector_schema_path=cls.schema_path,
            tokenizer_code_path=cls.tokenizer_path,
            tokenizer_bpe_path=cls.bpe_path,
            source_version="unit-fixture-v1",
            mongo_client=fake_client,
        )
        cls.base_profile = cls.root / "base-profile"
        write_deployment_bundle(cls.build, cls.base_profile)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def copy_profile(self, name: str) -> Path:
        target = self.root / name
        shutil.copytree(self.base_profile, target)
        return target

    def test_load_verifies_inventory_hashes_and_read_only_vectors(self) -> None:
        loaded = load_deployment_profile(self.base_profile)
        self.assertEqual([733, 512], list(loaded.inventory["vector_shape"]))
        self.assertEqual(733, len(loaded.rows))
        self.assertFalse(loaded.vectors.flags.writeable)
        self.assertEqual(
            {
                "model_checkpoint_sha256",
                "vector_bundle_sha256",
                "catalog_manifest_sha256",
                "dimension_bundle_sha256",
                "tokenizer_bundle_sha256",
                "category_policy_version",
                "dimension_axis_policy_version",
            },
            set(loaded.provenance),
        )
        self.assertTrue(all(loaded.checks.values()))

    def test_build_and_write_do_not_modify_sources_and_refuse_overwrite(self) -> None:
        self.assertEqual(
            self.source_hashes,
            {path: sha256_file(path) for path in self.source_hashes},
        )
        with self.assertRaises(BundleError):
            write_deployment_bundle(self.build, self.base_profile)

    def test_manifest_is_canonical_and_binds_each_member(self) -> None:
        profile = json.loads(
            (self.base_profile / "deployment_profile.json").read_text(encoding="utf-8")
        )
        for name, ref in profile["manifests"].items():
            data = (self.base_profile / ref["bundle_path"]).read_bytes()
            self.assertEqual(data, canonical_json_bytes(json.loads(data)))
            self.assertEqual(ref["sha256"], sha256_bytes(data), name)

    def test_missing_member_fails_closed(self) -> None:
        profile = self.copy_profile("missing-member")
        (profile / "dimensions/dimension_quality_rows.jsonl").unlink()
        with self.assertRaises(BundleError):
            load_deployment_profile(profile)

    def test_member_hash_fault_fails_closed(self) -> None:
        profile = self.copy_profile("hash-fault")
        path = profile / "catalog/catalog_rows.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(BundleError):
            load_deployment_profile(profile)

    def test_noncanonical_manifest_bytes_fail_closed(self) -> None:
        profile = self.copy_profile("manifest-canonical-fault")
        path = profile / "manifests/tokenizer_bundle.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        with self.assertRaises(BundleError):
            load_deployment_profile(profile)

    def test_row_order_fault_fails_even_when_member_is_resigned(self) -> None:
        profile = self.copy_profile("order-fault")
        rows = _read_jsonl(profile / "catalog/catalog_rows.jsonl")
        rows[0], rows[1] = rows[1], rows[0]
        _resign_catalog_member(profile, rows)
        with self.assertRaisesRegex(BundleError, "row identity join"):
            load_deployment_profile(profile)

    def test_row_index_fault_fails_even_when_member_is_resigned(self) -> None:
        profile = self.copy_profile("row-index-fault")
        rows = _read_jsonl(profile / "catalog/catalog_rows.jsonl")
        rows[0]["row_index"] = 99
        _resign_catalog_member(profile, rows)
        with self.assertRaisesRegex(BundleError, "row_index sequence"):
            load_deployment_profile(profile)

    def test_duplicate_join_id_fails_even_when_member_is_resigned(self) -> None:
        profile = self.copy_profile("duplicate-id-fault")
        rows = _read_jsonl(profile / "catalog/catalog_rows.jsonl")
        rows[1]["product_id"] = rows[0]["product_id"]
        _resign_catalog_member(profile, rows)
        with self.assertRaisesRegex(BundleError, "row identity join"):
            load_deployment_profile(profile)

    def test_actual_post_start_member_tamper_makes_readyz_and_post_503(self) -> None:
        profile = self.copy_profile("live-integrity-member-fault")
        loaded = load_deployment_profile(profile)
        runtime = ProductCandidateRuntime.from_components(
            profile_id=loaded.profile_id,
            provenance=loaded.provenance,
            vectors=loaded.vectors,
            rows=loaded.rows,
            tokenizer=_RuntimeOnlyTokenizer(),
            tokenizer_id=loaded.tokenizer_id,
            text_encoder=lambda tokens: np.ones(512, dtype=np.float32),
            checks=loaded.checks,
            inventory=loaded.inventory,
            integrity_revalidator=lambda: load_deployment_profile(profile),
        )
        app = create_app(runtime=runtime, bearer_token="bundle-test-token")
        with TestClient(app) as client:
            self.assertEqual(200, client.get("/readyz").status_code)
            member = profile / "dimensions/dimension_quality_rows.jsonl"
            member.write_bytes(member.read_bytes() + b" ")
            drifted = client.get("/readyz")
            self.assertEqual(503, drifted.status_code)
            self.assertIn(
                "DEPLOYMENT_INTEGRITY_REVALIDATION_FAILED",
                drifted.json()["failures"],
            )
            blocked = client.post(
                "/v2/product-candidates",
                headers={"Authorization": "Bearer bundle-test-token"},
                json={
                    "schema_version": "1.0",
                    "request_id": "bundle-drift-test",
                    "query_text": "desk",
                    "categories_primary": ["desk"],
                    "categories_secondary": [],
                    "top_k_per_tier": 1,
                },
            )
            self.assertEqual(503, blocked.status_code)


if __name__ == "__main__":
    unittest.main()
