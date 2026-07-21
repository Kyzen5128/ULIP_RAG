"""Reproducibility and validation helpers for IKEA RAG training.

This module deliberately has no project-level imports so its preflight checks can
run without constructing the (large) ULIP model or downloading a retriever model.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


PROVENANCE_SCHEMA = "ikea-rag-training-provenance/1.0"
DATASET_MANIFEST_SCHEMA = "ikea-rag-dataset-manifest/1.0"


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_file(path: os.PathLike[str] | str, label: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {result}")
    return result


def inspect_rag_artifacts(
    corpus_path: os.PathLike[str] | str,
    index_path: os.PathLike[str] | str,
) -> Dict[str, Any]:
    """Validate the JSONL/FAISS pair and return content-addressed metadata."""
    corpus = _require_file(corpus_path, "RAG corpus")
    index = _require_file(index_path, "RAG FAISS index")

    row_count = 0
    with corpus.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {corpus} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, Mapping) or not str(row.get("text", "")).strip():
                raise ValueError(
                    f"RAG corpus line {line_number} must contain a non-empty 'text' field"
                )
            row_count += 1

    if row_count == 0:
        raise ValueError(f"RAG corpus is empty: {corpus}")

    try:
        import faiss  # Imported lazily so unit tests can exercise other helpers.
    except ImportError as exc:
        raise RuntimeError("faiss is required to validate the RAG index") from exc

    faiss_index = faiss.read_index(str(index))
    if int(faiss_index.ntotal) != row_count:
        raise ValueError(
            "RAG corpus/index row mismatch: "
            f"corpus={row_count}, index.ntotal={int(faiss_index.ntotal)}"
        )
    # The current retriever is all-MiniLM-L6-v2, whose output dimension is 384.
    if int(faiss_index.d) != 384:
        raise ValueError(
            "RAG index dimension is incompatible with all-MiniLM-L6-v2: "
            f"expected=384, actual={int(faiss_index.d)}"
        )

    return {
        "corpus_path": str(corpus),
        "corpus_sha256": sha256_file(corpus),
        "corpus_rows": row_count,
        "index_path": str(index),
        "index_sha256": sha256_file(index),
        "index_rows": int(faiss_index.ntotal),
        "index_dimension": int(faiss_index.d),
    }


def _file_inventory(paths: Iterable[str]) -> Dict[str, Any]:
    rows = []
    for raw_path in sorted(set(str(p) for p in paths if p)):
        path = Path(raw_path)
        try:
            stat = path.stat()
            rows.append({"path": str(path.resolve()), "bytes": int(stat.st_size)})
        except FileNotFoundError:
            rows.append({"path": str(path), "missing": True})
    return {"count": len(rows), "sha256": canonical_sha256(rows)}


def build_dataset_split_manifest(
    dataset: Any,
    *,
    split: str,
    dataset_config_path: os.PathLike[str] | str,
    train_ratio: float,
) -> Dict[str, Any]:
    """Build a deterministic manifest from ``IkeaULIP.samples``.

    JSON and pre-sampled point clouds are content-hashed. Render images are
    represented by a deterministic path/size inventory because hashing every
    training view on every launch would make preflight unnecessarily expensive.
    """
    config_path = _require_file(dataset_config_path, "IKEA dataset config")
    samples: Sequence[Mapping[str, Any]] = getattr(dataset, "samples", None)
    if samples is None:
        raise TypeError("dataset must expose IkeaULIP.samples")

    entries = []
    for position, sample in enumerate(samples):
        json_path = _require_file(sample["json_path"], "IKEA sample JSON")
        pointcloud_path = _require_file(sample["pointcloud_path"], "IKEA point cloud")
        caption = str(sample.get("caption") or sample.get("category") or "furniture")
        render_paths = list(sample.get("render_paths") or [])
        if sample.get("image_path"):
            render_paths.append(str(sample["image_path"]))
        entries.append(
            {
                "row_index": position,
                "product_id": str(sample["id"]),
                "category": str(sample.get("category") or "Unknown"),
                "caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
                "json_path": str(json_path),
                "json_sha256": sha256_file(json_path),
                "pointcloud_path": str(pointcloud_path),
                "pointcloud_sha256": sha256_file(pointcloud_path),
                "render_inventory": _file_inventory(render_paths),
            }
        )

    ordered_ids = [entry["product_id"] for entry in entries]
    return {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "split": split,
        "train_ratio": float(train_ratio),
        "dataset_config_path": str(config_path),
        "dataset_config_sha256": sha256_file(config_path),
        "sample_count": len(entries),
        "ordered_ids_sha256": canonical_sha256(ordered_ids),
        "entries_sha256": canonical_sha256(entries),
        "entries": entries,
    }


def build_dataset_bundle_manifest(
    train_manifest: Mapping[str, Any], val_manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    bundle = {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "train": dict(train_manifest),
        "val": dict(val_manifest),
    }
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    return bundle


_EXPECTED_ENHANCER_SHAPES = {
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


def extract_rag_enhancer_state(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for original_key, tensor in state_dict.items():
        key = str(original_key)
        marker = "rag_enhancer."
        if marker not in key:
            continue
        key = key.split(marker, 1)[1]
        result[key] = tensor
    return result


def validate_stage1_checkpoint(
    checkpoint_path: os.PathLike[str] | str,
    *,
    expected_rag: Optional[Mapping[str, Any]] = None,
    expected_dataset_bundle_sha256: Optional[str] = None,
    require_pipeline_provenance: bool = True,
) -> Dict[str, Any]:
    """Validate Stage 1 enhancer weights and, by default, provenance binding."""
    path = _require_file(checkpoint_path, "Stage 1 checkpoint")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to validate a Stage 1 checkpoint") from exc

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # Older supported torch releases do not accept mmap.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Stage 1 checkpoint has no valid state_dict: {path}")

    enhancer_state = extract_rag_enhancer_state(state_dict)
    missing = sorted(set(_EXPECTED_ENHANCER_SHAPES) - set(enhancer_state))
    if missing:
        raise ValueError(f"Stage 1 checkpoint is missing RAG enhancer weights: {missing}")
    shape_errors = []
    for key, expected_shape in _EXPECTED_ENHANCER_SHAPES.items():
        actual_shape = tuple(enhancer_state[key].shape)
        if actual_shape != expected_shape:
            shape_errors.append(f"{key}: expected={expected_shape}, actual={actual_shape}")
    if shape_errors:
        raise ValueError("Invalid RAG enhancer tensor shapes: " + "; ".join(shape_errors))

    provenance = checkpoint.get("provenance")
    if require_pipeline_provenance:
        if not isinstance(provenance, Mapping):
            raise ValueError(
                "Stage 1 checkpoint lacks IKEA RAG provenance; use a checkpoint "
                "created by ikea/main_ikea_rag.py"
            )
        if provenance.get("schema_version") != PROVENANCE_SCHEMA:
            raise ValueError(
                "Unsupported Stage 1 provenance schema: "
                f"{provenance.get('schema_version')!r}"
            )
        if provenance.get("training_strategy") != "stage_1":
            raise ValueError("Checkpoint provenance does not identify a Stage 1 run")

        if expected_rag is not None:
            recorded_rag = provenance.get("rag", {})
            for key in ("corpus_sha256", "index_sha256"):
                if recorded_rag.get(key) != expected_rag.get(key):
                    raise ValueError(
                        f"Stage 1/current RAG artifact mismatch for {key}: "
                        f"stage1={recorded_rag.get(key)!r}, current={expected_rag.get(key)!r}"
                    )
        if expected_dataset_bundle_sha256 is not None:
            recorded = provenance.get("dataset", {}).get("bundle_sha256")
            if recorded != expected_dataset_bundle_sha256:
                raise ValueError(
                    "Stage 1/current IKEA dataset manifest mismatch: "
                    f"stage1={recorded!r}, current={expected_dataset_bundle_sha256!r}"
                )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "provenance": provenance,
        "enhancer_state": enhancer_state,
    }


def write_json_atomic(path: os.PathLike[str] | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)

