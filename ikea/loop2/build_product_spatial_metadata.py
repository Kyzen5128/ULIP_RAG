#!/usr/bin/env python3
"""Build immutable product-level spatial metadata for the V2T rule engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.build_training_dataset import family_id
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, write_json, write_jsonl
else:
    from .build_training_dataset import family_id
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, write_json, write_jsonl


def canonical_hash(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = snapshot_root(args.version, args.root)
    catalog_path = root / "catalog" / "products.jsonl"
    catalog = read_jsonl(catalog_path)
    output_rows = []
    for row in catalog:
        profile = row.get("dimension_profile") or {}
        dimensions = profile.get("dimensions_mm") or {}
        sources = profile.get("axis_source") or {}
        footprint_known = dimensions.get("width") is not None and dimensions.get("depth") is not None
        circular = footprint_known and sources.get("width") == "diameter" and sources.get("depth") == "diameter"
        footprint_kind = "circle" if circular else "rectangle_bbox" if footprint_known else "unknown"
        hard_eligible = bool(profile.get("hard_filter_eligible"))
        reasons = []
        if profile.get("status") != "three_axes":
            reasons.append("MISSING_REQUIRED_DIMENSIONS")
        if profile.get("ranges_mm"):
            reasons.append("DIMENSION_RANGE_PRESENT")
        if row.get("dimension_source_status") != "verified_product_measurements_context":
            reasons.append("DIMENSION_SOURCE_UNVERIFIED")
        if not hard_eligible and not reasons:
            reasons.append("HARD_FILTER_NOT_ELIGIBLE")
        result = {
            "schema_version": "ikea-product-spatial-metadata/2.0",
            "snapshot_version": args.version,
            "product_id": row["product_id"],
            "product_family_id": family_id(row),
            "name": row.get("name"),
            "category": row["canonical_category"],
            "dimensions_mm": {axis: dimensions.get(axis) for axis in ("width", "depth", "height")},
            "dimension_parse_status": profile.get("status"),
            "dimension_quality_status": profile.get("dimension_quality_status"),
            "dimension_source_status": row.get("dimension_source_status"),
            "dimension_axis_source": sources,
            "dimension_ranges_mm": profile.get("ranges_mm") or {},
            "hard_filter_eligible": hard_eligible,
            "height_policy": "required",
            "footprint_kind": footprint_kind,
            "footprint_bbox_mm": [dimensions.get("width"), dimensions.get("depth")] if footprint_known else None,
            "allowed_yaws_deg": [0] if circular else [0, 90, 180, 270] if footprint_known else None,
            "canonical_front": None,
            "front_status": "unverified",
            "clearance_requirements": None,
            "clearance_status": "unknown",
            "reason_codes": reasons,
            "source_url": row.get("url"),
            "source_page_sha256": row.get("page_sha256"),
            "source_document_sha256": canonical_hash(row),
        }
        output_rows.append(result)
    output_path = root / "metadata" / "product_spatial_metadata.jsonl"
    write_jsonl(output_path, output_rows)
    report = {
        "schema_version": "ikea-product-spatial-metadata-build/2.0",
        "snapshot_version": args.version,
        "rows": len(output_rows),
        "hard_filter_eligible_rows": sum(row["hard_filter_eligible"] for row in output_rows),
        "parse_status_counts": dict(sorted(Counter(row["dimension_parse_status"] for row in output_rows).items())),
        "quality_status_counts": dict(sorted(Counter(row["dimension_quality_status"] for row in output_rows).items())),
        "footprint_kind_counts": dict(sorted(Counter(row["footprint_kind"] for row in output_rows).items())),
        "front_verified_rows": 0,
        "clearance_verified_rows": 0,
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "manifests" / "product_spatial_metadata.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
