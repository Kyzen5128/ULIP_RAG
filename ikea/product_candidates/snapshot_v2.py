"""Immutable IKEA 1,546-row Product Candidates deployment profiles.

Unlike the legacy 733-row builder, this module never reads MongoDB.  It joins
the frozen IKEA 2.0 catalog and spatial metadata to the point-vector row order,
copies every serving dependency into one profile, and verifies all bytes again
at load time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from .bundles import BundleError, DeploymentProfile, canonical_json_bytes, canonical_jsonl_bytes


SCHEMA_VERSION = "ulip-product-candidates-deployment/2.0"
SOURCE_VERSION = "ikea1546-product-candidates-v2-20260716"
EXPECTED_ROWS = 1546
EXPECTED_DIM = 512
TOKENIZER_ID = "ulip-simple-tokenizer-clip77-v2"
CATEGORY_POLICY_VERSION = "v2t36_to_ikea24_v2"
DIMENSION_POLICY_VERSION = "dimension_axis_ikea_snapshot_v2"
MEMBER_NAMES = {
    "model_checkpoint",
    "vectors_pc",
    "meta_pc",
    "vector_schema",
    "catalog_rows",
    "dimension_rows",
    "tokenizer_code",
    "tokenizer_bpe",
    "category_policy",
    "dimension_policy",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise BundleError(f"JSONL row {index} is not an object: {path}")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"invalid JSONL {path}: {exc}") from exc
    return rows


def _read_canonical_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = _read_json(path)
    if raw != canonical_json_bytes(value):
        raise BundleError(f"non-canonical JSON: {path}")
    return value


def _read_canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    rows = _read_jsonl(path)
    if raw != canonical_jsonl_bytes(rows):
        raise BundleError(f"non-canonical JSONL: {path}")
    return rows


def _descriptor(path: Path, bundle_path: str) -> dict[str, Any]:
    return {
        "bundle_path": bundle_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _safe_member(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise BundleError("bundle member path must be a non-empty string")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise BundleError(f"bundle member escapes profile: {relative!r}")
    return path


def _copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise BundleError(f"source file is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _write_canonical_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_jsonl_bytes(rows))


def _ordered_ids_sha256(ids: list[str]) -> str:
    return _sha256_bytes(canonical_json_bytes({"ordered_product_ids": ids}))


def _index_unique(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise BundleError(f"{label} row {index} has invalid {key}")
        if value in output:
            raise BundleError(f"duplicate {key} in {label}: {value}")
        output[value] = row
    return output


def _positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _build_rows(
    ordered_meta: list[dict[str, Any]],
    catalog_by_id: Mapping[str, dict[str, Any]],
    spatial_by_id: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catalog_rows: list[dict[str, Any]] = []
    dimension_rows: list[dict[str, Any]] = []
    for row_index, meta in enumerate(ordered_meta):
        product_id = meta.get("id")
        if not isinstance(product_id, str) or not product_id:
            raise BundleError(f"vector metadata row {row_index} has invalid id")
        catalog = catalog_by_id.get(product_id)
        spatial = spatial_by_id.get(product_id)
        if catalog is None or spatial is None:
            raise BundleError(f"missing catalog/spatial join for product {product_id}")
        category = catalog.get("canonical_category")
        name = catalog.get("name")
        if category != meta.get("category") or category != spatial.get("category"):
            raise BundleError(f"category drift across row {row_index}: {product_id}")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise BundleError(f"invalid catalog name for product {product_id}")

        catalog_quality = "degraded" if category == "Sofa" else "supported"
        catalog_rows.append({
            "row_index": row_index,
            "product_id": product_id,
            "name": name,
            "category": category,
            "type_name": catalog.get("type_name"),
            "url": catalog.get("url"),
            "catalog_quality_status": catalog_quality,
        })

        quality = spatial.get("dimension_quality_status")
        footprint_source = spatial.get("footprint_kind")
        raw_dimensions = spatial.get("dimensions_mm")
        reasons = list(spatial.get("reason_codes") or [])
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise BundleError(f"invalid reason code for product {product_id}")
        dimensions = None
        status = "missing"
        footprint_kind = {
            "rectangle_bbox": "rectangle",
            "circle": "round_bounding_box",
        }.get(footprint_source, "special_unmodeled")

        if footprint_kind == "special_unmodeled":
            status = "special_unmodeled"
            reasons.append("FOOTPRINT_UNMODELED")
        elif quality == "three_axes" and isinstance(raw_dimensions, Mapping) and all(
            _positive(raw_dimensions.get(axis)) for axis in ("width", "depth", "height")
        ):
            status = "complete"
            dimensions = {
                "unit": "mm",
                "width": float(raw_dimensions["width"]),
                "depth": float(raw_dimensions["depth"]),
                "height": float(raw_dimensions["height"]),
            }
        elif quality == "partial" and isinstance(raw_dimensions, Mapping) and all(
            _positive(raw_dimensions.get(axis)) for axis in ("width", "depth")
        ) and raw_dimensions.get("height") is None:
            status = "partial"
            dimensions = {
                "unit": "mm",
                "width": float(raw_dimensions["width"]),
                "depth": float(raw_dimensions["depth"]),
                "height": None,
            }
            reasons.append("MISSING_REQUIRED_DIMENSIONS")
        elif quality == "range_present":
            # Response v1 has no interval type. Preserve the true reason code,
            # omit the scalar dimensions, and fail closed rather than collapse.
            status = "suspect"
            reasons.append("DIMENSION_RANGE_PRESENT")
        else:
            status = "missing"
            reasons.append("MISSING_REQUIRED_DIMENSIONS")

        if category == "Sofa":
            reasons.append("FOOTPRINT_SHAPE_UNVERIFIED")
        reasons = sorted(set(reasons))
        hard_eligible = bool(
            spatial.get("hard_filter_eligible") is True
            and status == "complete"
            and footprint_kind != "special_unmodeled"
            and category != "Sofa"
        )
        dimension_rows.append({
            "row_index": row_index,
            "product_id": product_id,
            "category": category,
            "dimensions": dimensions,
            "dimension_status": status,
            "dimension_quality_status_source": quality,
            "height_policy": "required",
            "hard_filter_eligible": hard_eligible,
            "footprint_kind": footprint_kind,
            "special_flags": reasons,
            "allowed_yaws_deg": list(spatial.get("allowed_yaws_deg") or [0, 90, 180, 270]),
            "front_status": spatial.get("front_status", "unverified"),
            "clearance_status": spatial.get("clearance_status", "unknown"),
            "source_document_sha256": spatial.get("source_document_sha256"),
        })
    return catalog_rows, dimension_rows


def build_snapshot_deployment(
    dataset_root: Path,
    output_parent: Path,
    repo_root: Path,
) -> Path:
    """Build the deterministic 1,546-row profile and return its final path."""
    dataset_root = dataset_root.expanduser().resolve()
    output_parent = output_parent.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    vector_dir = dataset_root / "vectors/semantic_ulip_v2_best"
    vector_schema_source = vector_dir / "schema.json"
    vector_schema = _read_json(vector_schema_source)
    checkpoint_source = Path(vector_schema.get("checkpoint", {}).get("path", "")).resolve()
    expected_checkpoint_sha = vector_schema.get("checkpoint", {}).get("sha256")
    if not checkpoint_source.is_file() or _sha256_file(checkpoint_source) != expected_checkpoint_sha:
        raise BundleError("semantic vector schema/checkpoint pin is invalid")
    if vector_schema.get("text_embedding_mode") != "vanilla":
        raise BundleError("production profile must use vanilla semantic text embedding")

    vector_source = vector_dir / "vectors_pc.npy"
    meta_source = vector_dir / "meta_pc.jsonl"
    vectors = np.load(vector_source, mmap_mode="r")
    if vectors.shape != (EXPECTED_ROWS, EXPECTED_DIM) or str(vectors.dtype) != "float32":
        raise BundleError(f"unexpected production vector shape/dtype: {vectors.shape}/{vectors.dtype}")
    if not np.isfinite(vectors).all():
        raise BundleError("production vectors contain non-finite values")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-3, atol=1e-3):
        raise BundleError("production vectors are not L2 normalized")

    ordered_meta = _read_jsonl(meta_source)
    if len(ordered_meta) != EXPECTED_ROWS:
        raise BundleError("vector metadata row count is not 1,546")
    ids = [row.get("id") for row in ordered_meta]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != EXPECTED_ROWS:
        raise BundleError("vector metadata product IDs are invalid or duplicated")
    catalog_by_id = _index_unique(_read_jsonl(dataset_root / "catalog/products.jsonl"), "product_id", "catalog")
    spatial_by_id = _index_unique(
        _read_jsonl(dataset_root / "metadata/product_spatial_metadata.jsonl"),
        "product_id",
        "spatial metadata",
    )
    catalog_rows, dimension_rows = _build_rows(ordered_meta, catalog_by_id, spatial_by_id)

    category_policy_source = repo_root / "ikea/product_candidates/policies/category_v2t36_to_ikea24_v2.json"
    dimension_policy_source = repo_root / "ikea/product_candidates/policies/dimension_axis_ikea_snapshot_v2.json"
    category_policy = _read_json(category_policy_source)
    dimension_policy = _read_json(dimension_policy_source)
    if category_policy.get("policy_version") != CATEGORY_POLICY_VERSION:
        raise BundleError("category policy version drifted")
    if dimension_policy.get("policy_version") != DIMENSION_POLICY_VERSION:
        raise BundleError("dimension policy version drifted")
    if set(category_policy.get("canonical_categories", [])) != {
        row["category"] for row in catalog_rows
    }:
        raise BundleError("category policy and 1,546-row catalog taxonomy differ")

    output_parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".ikea1546-v2-", dir=output_parent))
    try:
        member_paths = {
            "model_checkpoint": "model/checkpoint_best.pt",
            "vectors_pc": "vectors/vectors_pc.npy",
            "meta_pc": "vectors/meta_pc.jsonl",
            "vector_schema": "vectors/schema.json",
            "catalog_rows": "manifests/catalog_rows.jsonl",
            "dimension_rows": "manifests/dimension_rows.jsonl",
            "tokenizer_code": "tokenizer/tokenizer.py",
            "tokenizer_bpe": "tokenizer/bpe_simple_vocab_16e6.txt.gz",
            "category_policy": "policies/category_v2t36_to_ikea24_v2.json",
            "dimension_policy": "policies/dimension_axis_ikea_snapshot_v2.json",
        }
        _copy_file(checkpoint_source, temp / member_paths["model_checkpoint"])
        _copy_file(vector_source, temp / member_paths["vectors_pc"])
        _write_canonical_jsonl(temp / member_paths["meta_pc"], ordered_meta)
        _write_canonical_json(temp / member_paths["vector_schema"], vector_schema)
        _write_canonical_jsonl(temp / member_paths["catalog_rows"], catalog_rows)
        _write_canonical_jsonl(temp / member_paths["dimension_rows"], dimension_rows)
        _copy_file(repo_root / "core/utils/tokenizer.py", temp / member_paths["tokenizer_code"])
        _copy_file(repo_root / "core/utils/bpe_simple_vocab_16e6.txt.gz", temp / member_paths["tokenizer_bpe"])
        _write_canonical_json(temp / member_paths["category_policy"], category_policy)
        _write_canonical_json(temp / member_paths["dimension_policy"], dimension_policy)
        members = {
            name: _descriptor(temp / relative, relative)
            for name, relative in sorted(member_paths.items())
        }
        ordered_hash = _ordered_ids_sha256(ids)  # type: ignore[arg-type]
        vector_material = {
            "vectors_pc_sha256": members["vectors_pc"]["sha256"],
            "meta_pc_sha256": members["meta_pc"]["sha256"],
            "schema_sha256": members["vector_schema"]["sha256"],
            "ordered_ids_sha256": ordered_hash,
            "shape": [EXPECTED_ROWS, EXPECTED_DIM],
            "dtype": "float32",
        }
        tokenizer_material = {
            "tokenizer_id": TOKENIZER_ID,
            "code_sha256": members["tokenizer_code"]["sha256"],
            "bpe_sha256": members["tokenizer_bpe"]["sha256"],
            "context_length": 77,
            "safe_content_tokens": 75,
        }
        provenance = {
            "model_checkpoint_sha256": members["model_checkpoint"]["sha256"],
            "vector_bundle_sha256": _sha256_bytes(canonical_json_bytes(vector_material)),
            "catalog_manifest_sha256": members["catalog_rows"]["sha256"],
            "dimension_bundle_sha256": members["dimension_rows"]["sha256"],
            "tokenizer_bundle_sha256": _sha256_bytes(canonical_json_bytes(tokenizer_material)),
            "category_policy_version": CATEGORY_POLICY_VERSION,
            "dimension_axis_policy_version": DIMENSION_POLICY_VERSION,
        }
        dimension_counts = Counter(row["dimension_status"] for row in dimension_rows)
        category_counts = Counter(row["category"] for row in catalog_rows)
        inventory = {
            "vector_shape": [EXPECTED_ROWS, EXPECTED_DIM],
            "vector_rows": EXPECTED_ROWS,
            "catalog_rows": EXPECTED_ROWS,
            "dimension_profile_rows": EXPECTED_ROWS,
            "hard_filter_eligible_rows": sum(row["hard_filter_eligible"] for row in dimension_rows),
            "category_counts": dict(sorted(category_counts.items())),
            "dimension_status_counts": dict(sorted(dimension_counts.items())),
        }
        rag_ab_paths = {
            "stage2_best": dataset_root / "evaluation/rag_val_ab_best.json",
            "stage2_last": dataset_root / "evaluation/rag_val_ab_last.json",
        }
        experimental = {
            "rag_promoted_to_production": False,
            "reason": "validation_semantic_vanilla_outperformed_stage2_rag",
            "training_corpus_sha256": _sha256_file(dataset_root / "corpus/product_corpus_train.jsonl"),
            "ab_artifacts": {
                name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
                for name, path in sorted(rag_ab_paths.items())
            },
        }
        identity_material = {
            "schema_version": SCHEMA_VERSION,
            "source_version": SOURCE_VERSION,
            "provenance": provenance,
            "ordered_ids_sha256": ordered_hash,
            "member_hashes": {name: value["sha256"] for name, value in sorted(members.items())},
            "retrieval_mode": "semantic_vanilla",
        }
        identity_sha = _sha256_bytes(canonical_json_bytes(identity_material))
        profile_id = f"{SOURCE_VERSION}-{identity_sha[:16]}"
        profile = {
            "schema_version": SCHEMA_VERSION,
            "source_version": SOURCE_VERSION,
            "deployment_profile_id": profile_id,
            "profile_id": profile_id,
            "deployment_identity_sha256": identity_sha,
            "retrieval_mode": "semantic_vanilla",
            "tokenizer_id": TOKENIZER_ID,
            "ordered_ids_sha256": ordered_hash,
            "provenance": provenance,
            "inventory": inventory,
            "members": members,
            "vector_bundle_material": vector_material,
            "tokenizer_bundle_material": tokenizer_material,
            "experimental_rag": experimental,
        }
        _write_canonical_json(temp / "deployment_profile.json", profile)
        final = output_parent / profile_id
        if final.exists():
            existing = final / "deployment_profile.json"
            if existing.is_file() and existing.read_bytes() == (temp / "deployment_profile.json").read_bytes():
                shutil.rmtree(temp)
                load_snapshot_deployment(final)
                return final
            raise BundleError(f"deployment profile path already exists with different bytes: {final}")
        os.replace(temp, final)
        load_snapshot_deployment(final)
        return final
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def _verify_member(root: Path, descriptor: Mapping[str, Any], label: str) -> Path:
    path = _safe_member(root, descriptor.get("bundle_path"))
    if not path.is_file():
        raise BundleError(f"missing profile member: {label}")
    if path.stat().st_size != descriptor.get("bytes"):
        raise BundleError(f"profile member size drifted: {label}")
    if _sha256_file(path) != descriptor.get("sha256"):
        raise BundleError(f"profile member hash drifted: {label}")
    return path


def load_snapshot_deployment(profile_dir: Path) -> DeploymentProfile:
    """Verify and load a snapshot-backed deployment without external reads."""
    root = profile_dir.expanduser().resolve()
    profile = _read_canonical_json(root / "deployment_profile.json")
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise BundleError("unsupported snapshot deployment schema")
    members = profile.get("members")
    if not isinstance(members, Mapping) or set(members) != MEMBER_NAMES:
        raise BundleError("snapshot deployment member set is invalid")
    paths = {name: _verify_member(root, members[name], name) for name in sorted(MEMBER_NAMES)}
    provenance = profile.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "model_checkpoint_sha256", "vector_bundle_sha256", "catalog_manifest_sha256",
        "dimension_bundle_sha256", "tokenizer_bundle_sha256", "category_policy_version",
        "dimension_axis_policy_version",
    }:
        raise BundleError("snapshot deployment provenance is invalid")
    if provenance["model_checkpoint_sha256"] != members["model_checkpoint"]["sha256"]:
        raise BundleError("checkpoint provenance drifted")
    if provenance["catalog_manifest_sha256"] != members["catalog_rows"]["sha256"]:
        raise BundleError("catalog provenance drifted")
    if provenance["dimension_bundle_sha256"] != members["dimension_rows"]["sha256"]:
        raise BundleError("dimension provenance drifted")

    vector_material = profile.get("vector_bundle_material")
    tokenizer_material = profile.get("tokenizer_bundle_material")
    if _sha256_bytes(canonical_json_bytes(vector_material)) != provenance["vector_bundle_sha256"]:
        raise BundleError("vector bundle provenance drifted")
    if _sha256_bytes(canonical_json_bytes(tokenizer_material)) != provenance["tokenizer_bundle_sha256"]:
        raise BundleError("tokenizer bundle provenance drifted")
    if vector_material != {
        "vectors_pc_sha256": members["vectors_pc"]["sha256"],
        "meta_pc_sha256": members["meta_pc"]["sha256"],
        "schema_sha256": members["vector_schema"]["sha256"],
        "ordered_ids_sha256": profile.get("ordered_ids_sha256"),
        "shape": [EXPECTED_ROWS, EXPECTED_DIM],
        "dtype": "float32",
    }:
        raise BundleError("vector bundle material/member mismatch")
    if tokenizer_material != {
        "tokenizer_id": TOKENIZER_ID,
        "code_sha256": members["tokenizer_code"]["sha256"],
        "bpe_sha256": members["tokenizer_bpe"]["sha256"],
        "context_length": 77,
        "safe_content_tokens": 75,
    }:
        raise BundleError("tokenizer bundle material/member mismatch")

    category_policy = _read_canonical_json(paths["category_policy"])
    dimension_policy = _read_canonical_json(paths["dimension_policy"])
    if category_policy.get("policy_version") != provenance["category_policy_version"]:
        raise BundleError("category policy provenance drifted")
    if dimension_policy.get("policy_version") != provenance["dimension_axis_policy_version"]:
        raise BundleError("dimension policy provenance drifted")
    if provenance["category_policy_version"] != CATEGORY_POLICY_VERSION or provenance["dimension_axis_policy_version"] != DIMENSION_POLICY_VERSION:
        raise BundleError("unsupported snapshot policy tuple")

    vector_schema = _read_canonical_json(paths["vector_schema"])
    if vector_schema.get("checkpoint", {}).get("sha256") != provenance["model_checkpoint_sha256"]:
        raise BundleError("vector schema/checkpoint provenance drifted")
    if vector_schema.get("text_embedding_mode") != "vanilla":
        raise BundleError("snapshot profile is not semantic vanilla")
    meta_rows = _read_canonical_jsonl(paths["meta_pc"])
    catalog_rows = _read_canonical_jsonl(paths["catalog_rows"])
    dimension_rows = _read_canonical_jsonl(paths["dimension_rows"])
    if not all(len(rows) == EXPECTED_ROWS for rows in (meta_rows, catalog_rows, dimension_rows)):
        raise BundleError("snapshot row count drifted")
    meta_ids = [row.get("id") for row in meta_rows]
    catalog_ids = [row.get("product_id") for row in catalog_rows]
    dimension_ids = [row.get("product_id") for row in dimension_rows]
    if meta_ids != catalog_ids or meta_ids != dimension_ids or len(set(meta_ids)) != EXPECTED_ROWS:
        raise BundleError("snapshot row identity join drifted")
    if [row.get("row_index") for row in catalog_rows] != list(range(EXPECTED_ROWS)) or [
        row.get("row_index") for row in dimension_rows
    ] != list(range(EXPECTED_ROWS)):
        raise BundleError("snapshot row_index order drifted")
    if _ordered_ids_sha256(meta_ids) != profile.get("ordered_ids_sha256"):
        raise BundleError("snapshot ordered product ID hash drifted")

    vectors = np.load(paths["vectors_pc"]).astype(np.float32, copy=False)
    if vectors.shape != (EXPECTED_ROWS, EXPECTED_DIM) or not np.isfinite(vectors).all():
        raise BundleError("snapshot vectors are invalid")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-3, atol=1e-3):
        raise BundleError("snapshot vectors are not L2 normalized")
    inventory = profile.get("inventory")
    if not isinstance(inventory, Mapping) or any(
        inventory.get(key) != expected for key, expected in {
            "vector_shape": [EXPECTED_ROWS, EXPECTED_DIM],
            "vector_rows": EXPECTED_ROWS,
            "catalog_rows": EXPECTED_ROWS,
            "dimension_profile_rows": EXPECTED_ROWS,
        }.items()
    ):
        raise BundleError("snapshot inventory drifted")

    identity_material = {
        "schema_version": SCHEMA_VERSION,
        "source_version": profile.get("source_version"),
        "provenance": dict(provenance),
        "ordered_ids_sha256": profile.get("ordered_ids_sha256"),
        "member_hashes": {name: value["sha256"] for name, value in sorted(members.items())},
        "retrieval_mode": profile.get("retrieval_mode"),
    }
    identity_sha = _sha256_bytes(canonical_json_bytes(identity_material))
    profile_id = profile.get("deployment_profile_id")
    if identity_sha != profile.get("deployment_identity_sha256") or profile_id != f"{SOURCE_VERSION}-{identity_sha[:16]}" or profile.get("profile_id") != profile_id:
        raise BundleError("snapshot deployment identity drifted")

    combined: list[Mapping[str, Any]] = []
    for index, (catalog, dimension) in enumerate(zip(catalog_rows, dimension_rows)):
        if any(catalog.get(key) != dimension.get(key) for key in ("row_index", "product_id", "category")):
            raise BundleError(f"catalog/dimension join drifted at row {index}")
        row = dict(catalog)
        row.update({key: value for key, value in dimension.items() if key not in {"row_index", "product_id", "category"}})
        combined.append(_deep_freeze(row))
    vectors.setflags(write=False)
    checks = {key: True for key in (
        "model_checkpoint", "vector_bundle", "catalog_manifest", "dimension_bundle",
        "tokenizer_bundle", "row_identity_join", "policy_versions",
    )}
    runtime_inventory = {
        "vector_shape": [EXPECTED_ROWS, EXPECTED_DIM],
        "vector_rows": EXPECTED_ROWS,
        "catalog_rows": EXPECTED_ROWS,
        "dimension_profile_rows": EXPECTED_ROWS,
    }
    return DeploymentProfile(
        profile_id=profile_id,
        deployment_profile_id=profile_id,
        provenance=_deep_freeze(dict(provenance)),
        checks=_deep_freeze(checks),
        inventory=_deep_freeze(runtime_inventory),
        inventory_details=_deep_freeze(dict(inventory)),
        vectors=vectors,
        catalog_rows=tuple(_deep_freeze(row) for row in catalog_rows),
        dimension_rows=tuple(_deep_freeze(row) for row in dimension_rows),
        rows=tuple(combined),
        tokenizer_id=TOKENIZER_ID,
        tokenizer_code_path=paths["tokenizer_code"],
        tokenizer_bpe_path=paths["tokenizer_bpe"],
        model_checkpoint_path=paths["model_checkpoint"],
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    path = build_snapshot_deployment(args.dataset_root, args.output_parent, args.repo_root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
