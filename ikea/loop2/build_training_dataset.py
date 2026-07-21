#!/usr/bin/env python3
"""Build ULIP-compatible samples and a self-generated, source-backed RAG corpus."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, split_name_for_group, write_json, write_jsonl
else:
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, split_name_for_group, write_json, write_jsonl


_ULIP_TOKENIZER = None


def ulip_tokenizer():
    global _ULIP_TOKENIZER
    if _ULIP_TOKENIZER is None:
        core = Path(__file__).resolve().parents[2] / "core"
        if str(core) not in sys.path:
            sys.path.insert(0, str(core))
        from utils.tokenizer import SimpleTokenizer
        _ULIP_TOKENIZER = SimpleTokenizer()
    return _ULIP_TOKENIZER


def sentence(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    return text if text[-1] in ".!?" else text + "."


def unique_sentences(values: list[Any]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        text = sentence(value)
        key = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return output


def capped_words(text: str, maximum: int = 60) -> str:
    words = text.split()
    if len(words) <= maximum:
        return text
    return " ".join(words[:maximum]).rstrip(",;:") + "."


def capped_ulip_tokens(text: str, maximum: int = 75) -> tuple[str, bool, int]:
    """Keep complete words while enforcing ULIP's 75 content-token limit."""
    tokenizer = ulip_tokenizer()
    count = len(tokenizer.encode(text))
    if count <= maximum:
        return text, False, count
    words = text.split()
    low, high = 1, len(words)
    best = ""
    best_count = 0
    while low <= high:
        middle = (low + high) // 2
        candidate = " ".join(words[:middle]).rstrip(",;:")
        if candidate and candidate[-1] not in ".!?":
            candidate += "."
        candidate_count = len(tokenizer.encode(candidate))
        if candidate_count <= maximum:
            best, best_count = candidate, candidate_count
            low = middle + 1
        else:
            high = middle - 1
    if not best:
        raise ValueError("caption cannot be represented within the ULIP token limit")
    return best, True, best_count


def semantic_caption_details(row: dict[str, Any]) -> tuple[str, bool, int]:
    category = row["canonical_category"]
    type_name = row.get("type_name") or category
    color_text = ", ".join(row.get("colors") or [])
    structured = f"This is a {type_name} in the {category} category"
    if color_text:
        structured += f", offered in {color_text}"
    pieces = unique_sentences([structured, row.get("description"), row.get("design_text")])
    return capped_ulip_tokens(capped_words(" ".join(pieces)))


def semantic_caption(row: dict[str, Any]) -> str:
    return semantic_caption_details(row)[0]


def family_id(row: dict[str, Any]) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", str(row.get("name") or "").casefold()).strip("-")
    category = re.sub(r"[^a-z0-9]+", "-", row["canonical_category"].casefold()).strip("-")
    return f"{category}:{name or row['product_id']}"


def dimension_document(row: dict[str, Any]) -> str | None:
    dimensions = (row.get("dimension_profile") or {}).get("dimensions_mm") or {}
    known = [(axis, dimensions.get(axis)) for axis in ("width", "depth", "height") if dimensions.get(axis)]
    if not known:
        return None
    values = ", ".join(f"{axis} {value / 10.0:.1f} cm" for axis, value in known)
    return f"The {row.get('name') or row['product_id']} {row['canonical_category']} has official IKEA measurements: {values}."


def manifest_map(path: Path) -> dict[str, dict]:
    return {row["product_id"]: row for row in read_jsonl(path)} if path.exists() else {}


def publish_json_directory(staging: Path, destination: Path) -> None:
    """Atomically publish generated samples without leaving stale product JSON."""
    backup = destination.with_name(destination.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--allow-missing-renders", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--geometry-manifest", type=Path)
    parser.add_argument("--render-manifest", type=Path)
    args = parser.parse_args()
    split_ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    # Validate once even when no product passes the downstream asset gates.
    split_name_for_group("validation", *split_ratios)
    root = snapshot_root(args.version, args.root)
    catalog_path = root / "catalog" / "products.jsonl"
    rows = read_jsonl(catalog_path)
    geometry_manifest = args.geometry_manifest or root / "manifests" / "geometry_rows.jsonl"
    render_manifest = args.render_manifest or root / "manifests" / "render_rows.jsonl"
    geometry = manifest_map(geometry_manifest)
    renders = manifest_map(render_manifest)
    assets = manifest_map(root / "manifests" / "asset_rows.jsonl")
    samples = []
    corpus = []
    excluded = []
    json_dir = root / "dataset" / "json"
    staging_json_dir = root / "dataset" / ".json.staging"
    if staging_json_dir.exists():
        shutil.rmtree(staging_json_dir)
    staging_json_dir.mkdir(parents=True, exist_ok=False)
    for row in rows:
        product_id = row["product_id"]
        geometry_row = geometry.get(product_id) or {}
        render_row = renders.get(product_id) or {}
        asset_row = assets.get(product_id) or {}
        reasons = []
        if geometry_row.get("status") != "ok":
            reasons.append("GEOMETRY_NOT_APPROVED")
        if not args.allow_missing_renders and render_row.get("status") != "ok":
            reasons.append("RENDERS_NOT_APPROVED")
        if (asset_row.get("image") or {}).get("status") != "ok":
            reasons.append("PRIMARY_IMAGE_NOT_APPROVED")
        if reasons:
            excluded.append({"product_id": product_id, "reasons": reasons})
            continue
        dimensions = (row.get("dimension_profile") or {}).get("dimensions_mm") or {}
        render_paths = [item["path"] for item in render_row.get("renders") or []]
        context_images = [
            item["path"] for item in asset_row.get("supplementary_images") or []
            if item.get("status") == "ok" and item.get("asset_type") == "CONTEXT_PRODUCT_IMAGE"
        ]
        caption, caption_truncated, caption_token_count = semantic_caption_details(row)
        product_family = family_id(row)
        split = split_name_for_group(product_family, *split_ratios)
        sample = {
            "schema_version": "ikea-ulip-training-sample/2.0",
            "id": product_id,
            "product_id": product_id,
            "image": asset_row["image"]["path"],
            "pointcloud": geometry_row["pointcloud"],
            "mesh": geometry_row["canonical_glb"],
            "text": caption,
            "caption_en": caption,
            "caption_source": "ikea_official_alt_plus_structured",
            "caption_content_token_count": caption_token_count,
            "caption_truncated": caption_truncated,
            "category": row["canonical_category"],
            "split_group": product_family,
            "split": split,
            "render_images": render_paths,
            "scene_images": context_images,
            "spatial": {
                "dimensions_mm": {axis: dimensions.get(axis) for axis in ("width", "depth", "height")},
                "known_mask": {axis: dimensions.get(axis) is not None for axis in ("width", "depth", "height")},
                "footprint_bbox_mm": [dimensions.get("width"), dimensions.get("depth")],
                "front_direction": None,
                "orientation_status": (geometry_row.get("geometry_report") or {}).get("orientation_status", "unverified"),
                "pointcloud_coordinate_frame": {"x": "width", "y": "up", "z": "depth_sign_unverified", "unit": "m"},
                "clearance_requirements": None,
            },
            "meta": {
                "name": row.get("name"), "type_name": row.get("type_name"),
                "colors": row.get("colors") or [], "url": row.get("url"),
                "source": row.get("source"), "snapshot_version": args.version,
                "geometry_source": geometry_row.get("source"),
                "family_id": product_family,
            },
        }
        write_json(staging_json_dir / f"{product_id}.json", sample)
        samples.append(sample)
        corpus.append({
            "schema_version": "ikea-rag-document/2.0", "doc_id": f"{product_id}:semantic",
            "product_id": product_id, "document_type": "product_semantics",
            "category": row["canonical_category"], "text": caption,
            "split_group": product_family, "split": split,
            "source_url": row.get("url"), "source_strength": "official_product_metadata",
        })
        dimensions_text = dimension_document(row)
        if dimensions_text:
            corpus.append({
                "schema_version": "ikea-rag-document/2.0", "doc_id": f"{product_id}:dimensions",
                "product_id": product_id, "document_type": "product_dimensions",
                "category": row["canonical_category"], "text": dimensions_text,
                "split_group": product_family, "split": split,
                "source_url": row.get("url"), "source_strength": "official_product_measurements",
            })
        if row.get("contextual_image_alt"):
            corpus.append({
                "schema_version": "ikea-rag-document/2.0", "doc_id": f"{product_id}:scene",
                "product_id": product_id, "document_type": "scene_context_weak",
                "category": row["canonical_category"], "text": sentence(row["contextual_image_alt"]),
                "split_group": product_family, "split": split,
                "source_url": row.get("contextual_image_url"), "source_strength": "official_context_image_alt_weak",
            })
    publish_json_directory(staging_json_dir, json_dir)
    write_jsonl(root / "dataset" / "samples.jsonl", samples)
    write_jsonl(root / "dataset" / "excluded.jsonl", excluded)
    corpus_path = root / "corpus" / "product_corpus.jsonl"
    write_jsonl(corpus_path, corpus)
    split_corpus_paths = {}
    for split in ("train", "val", "test"):
        path = root / "corpus" / f"product_corpus_{split}.jsonl"
        write_jsonl(path, (document for document in corpus if document["split"] == split))
        split_corpus_paths[split] = path
    report = {
        "schema_version": "ikea-training-dataset-build/2.0",
        "snapshot_version": args.version,
        "catalog_rows": len(rows), "training_rows": len(samples), "excluded_rows": len(excluded),
        "corpus_documents": len(corpus),
        "training_split_rows": {split: sum(sample["split"] == split for sample in samples) for split in ("train", "val", "test")},
        "corpus_split_rows": {split: sum(document["split"] == split for document in corpus) for split in ("train", "val", "test")},
        "split_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "corpus_document_types": {kind: sum(x["document_type"] == kind for x in corpus) for kind in sorted({x["document_type"] for x in corpus})},
        "caption_content_token_max": max((sample["caption_content_token_count"] for sample in samples), default=0),
        "caption_truncated_rows": sum(sample["caption_truncated"] for sample in samples),
        "catalog_sha256": sha256_file(catalog_path),
        "asset_manifest_sha256": sha256_file(root / "manifests" / "asset_rows.jsonl"),
        "geometry_manifest_path": str(geometry_manifest),
        "geometry_manifest_sha256": sha256_file(geometry_manifest),
        "render_manifest_path": str(render_manifest),
        "render_manifest_sha256": sha256_file(render_manifest),
        "samples_sha256": sha256_file(root / "dataset" / "samples.jsonl"),
        "corpus_sha256": sha256_file(corpus_path),
        "corpus_split_sha256": {split: sha256_file(path) for split, path in split_corpus_paths.items()},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "No style/material label is inferred when IKEA does not provide one.",
            "Scene-context documents are weak evidence, not human-confirmed style labels.",
            "RAG training must use product_corpus_train.jsonl; validation/test product facts are excluded to prevent retrieval leakage.",
            "Canonical front and clearance remain unknown until separately verified.",
        ],
    }
    write_json(root / "manifests" / "training_dataset.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
