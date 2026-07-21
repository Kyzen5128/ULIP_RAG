#!/usr/bin/env python3
"""Evaluate aligned ULIP vectors on an explicit held-out dataset split.

The vector files may have different row orders, so identity is joined only by
the product ``id`` stored in each modality's JSONL metadata.  The evaluator is
read-only unless ``--output`` is supplied and records hashes for every input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_modality(vector_dir: Path, modality: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, str]]:
    vector_path = vector_dir / f"vectors_{modality}.npy"
    meta_path = vector_dir / f"meta_{modality}.jsonl"
    vectors = np.load(vector_path, mmap_mode="r")
    metadata = read_jsonl(meta_path)
    if vectors.ndim != 2 or vectors.shape[0] != len(metadata):
        raise ValueError(
            f"{modality}: vector/meta mismatch {tuple(vectors.shape)} vs {len(metadata)} rows"
        )
    ids = [str(row.get("id") or "").strip() for row in metadata]
    if any(not product_id for product_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{modality}: product ids must be non-empty and unique")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{modality}: vectors contain non-finite values")
    return vectors, metadata, {
        "vectors_path": str(vector_path.resolve()),
        "vectors_sha256": sha256_file(vector_path),
        "metadata_path": str(meta_path.resolve()),
        "metadata_sha256": sha256_file(meta_path),
    }


def retrieval_metrics(query: np.ndarray, gallery: np.ndarray) -> dict[str, float | int]:
    scores = np.asarray(query, dtype=np.float32) @ np.asarray(gallery, dtype=np.float32).T
    order = np.argsort(-scores, axis=1, kind="stable")
    truth = np.arange(scores.shape[0])[:, None]
    ranks = np.argmax(order == truth, axis=1) + 1
    return {
        "queries": int(len(ranks)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr_at_10": float(np.mean(np.where(ranks <= 10, 1.0 / ranks, 0.0))),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--dataset-json-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    vector_dir = args.vector_dir.expanduser().resolve()
    dataset_dir = args.dataset_json_dir.expanduser().resolve()
    loaded: dict[str, tuple[np.ndarray, list[dict[str, Any]], dict[str, str]]] = {
        modality: load_modality(vector_dir, modality) for modality in ("pc", "img", "txt")
    }
    modality_maps = {
        modality: {str(row["id"]): index for index, row in enumerate(metadata)}
        for modality, (_, metadata, _) in loaded.items()
    }
    common_ids = set.intersection(*(set(mapping) for mapping in modality_maps.values()))
    selected: list[str] = []
    split_counts: dict[str, int] = {}
    dataset_hashes: dict[str, str] = {}
    for path in sorted(dataset_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        product_id = str(row.get("product_id") or row.get("id") or path.stem)
        split = str(row.get("split") or "")
        split_counts[split] = split_counts.get(split, 0) + 1
        if product_id in common_ids and (args.split == "all" or split == args.split):
            selected.append(product_id)
            dataset_hashes[product_id] = sha256_file(path)
    selected = sorted(set(selected))
    if not selected:
        raise ValueError(f"no common products found for split={args.split!r}")

    aligned: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    norm_summary: dict[str, Any] = {}
    for modality, (vectors, _, paths) in loaded.items():
        indices = [modality_maps[modality][product_id] for product_id in selected]
        values = np.asarray(vectors[indices], dtype=np.float32)
        norms = np.linalg.norm(values, axis=1)
        if np.max(np.abs(norms - 1.0)) > 1e-3:
            raise ValueError(f"{modality}: vectors are not L2 normalized")
        aligned[modality] = values
        provenance[modality] = paths
        norm_summary[modality] = {
            "min": float(norms.min()), "max": float(norms.max()), "mean": float(norms.mean())
        }

    result = {
        "schema_version": "ikea-ulip-semantic-evaluation/2.0",
        "evaluation_scope": "held_out_cross_modal_exact_product_retrieval" if args.split in {"val", "test"} else "cross_modal_exact_product_retrieval",
        "split": args.split,
        "products": len(selected),
        "dataset_split_counts": split_counts,
        "metrics": {
            "text_to_pc": retrieval_metrics(aligned["txt"], aligned["pc"]),
            "pc_to_text": retrieval_metrics(aligned["pc"], aligned["txt"]),
            "image_to_pc": retrieval_metrics(aligned["img"], aligned["pc"]),
            "pc_to_image": retrieval_metrics(aligned["pc"], aligned["img"]),
        },
        "l2_norms": norm_summary,
        "provenance": {
            "vector_dir": str(vector_dir),
            "dataset_json_dir": str(dataset_dir),
            "modalities": provenance,
            "selected_dataset_json_bundle_sha256": hashlib.sha256(
                "\n".join(f"{product_id} {dataset_hashes[product_id]}" for product_id in selected).encode("utf-8")
            ).hexdigest(),
        },
        "limitations": [
            "Exact-product retrieval is evaluated only within the selected split gallery.",
            "This does not measure V2T room-query, style preference, or placement quality.",
        ],
    }
    if args.output:
        atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
