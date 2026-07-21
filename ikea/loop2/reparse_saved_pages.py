#!/usr/bin/env python3
"""Reparse byte-bound saved IKEA pages and republish the enriched catalog."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, write_json, write_jsonl
    from ikea.loop2.enrich_catalog import extract_page
else:
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, write_json, write_jsonl
    from .enrich_catalog import extract_page


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = snapshot_root(args.version, args.root)
    raw_catalog_path = root / "catalog" / "products_plp.jsonl"
    output_path = root / "catalog" / "products.jsonl"
    rows = read_jsonl(raw_catalog_path)
    source_counts = Counter()
    errors = []
    for row in rows:
        product_id = row["product_id"]
        page_path = root / "raw" / "product_pages" / f"{product_id}.html.gz"
        state_path = root / "state" / "details" / f"{product_id}.json"
        try:
            body = gzip.open(page_path, "rb").read()
            state = json.loads(state_path.read_text(encoding="utf-8"))
            actual_page_hash = hashlib.sha256(body).hexdigest()
            if actual_page_hash != state.get("page_sha256"):
                raise ValueError(
                    f"saved page hash mismatch expected={state.get('page_sha256')} actual={actual_page_hash}"
                )
            extracted = extract_page(body.decode("utf-8", errors="replace"))
            state.update(extracted)
            state["reparsed_at"] = datetime.now(timezone.utc).isoformat()
            write_json(state_path, state)
            row.update({
                "detail_status": "ok",
                "official_glb_url": (extracted.get("official_glb_urls") or [None])[0],
                "official_usdz_url": (extracted.get("official_usdz_urls") or [None])[0],
                "dimension_profile": extracted.get("dimension_profile"),
                "dimension_source": extracted.get("dimension_source"),
                "dimension_source_status": extracted.get("dimension_source_status"),
                "page_sha256": actual_page_hash,
            })
            source_counts[extracted.get("dimension_source_status") or "missing"] += 1
        except Exception as error:
            errors.append({"product_id": product_id, "error": f"{type(error).__name__}: {error}"})
    if errors:
        raise SystemExit(json.dumps({"error_count": len(errors), "errors": errors[:20]}, indent=2))
    write_jsonl(output_path, rows)
    profiles = [row.get("dimension_profile") or {} for row in rows]
    detail_manifest_path = root / "manifests" / "detail_enrichment.json"
    detail = json.loads(detail_manifest_path.read_text(encoding="utf-8"))
    detail.update({
        "input_catalog_path": str(raw_catalog_path),
        "input_catalog_sha256": sha256_file(raw_catalog_path),
        "output_catalog_path": str(output_path),
        "output_catalog_sha256": sha256_file(output_path),
        "saved_page_hash_verified_rows": len(rows),
        "dimension_source_status_counts": dict(sorted(source_counts.items())),
        "three_axis_rows": sum(profile.get("status") == "three_axes" for profile in profiles),
        "partial_dimension_rows": sum(profile.get("status") == "partial" for profile in profiles),
        "missing_dimension_rows": sum(profile.get("status") == "missing" for profile in profiles),
        "range_present_rows": sum(profile.get("dimension_quality_status") == "range_present" for profile in profiles),
        "hard_filter_eligible_rows": sum(profile.get("hard_filter_eligible") is True for profile in profiles),
        "reparsed_at": datetime.now(timezone.utc).isoformat(),
    })
    write_json(detail_manifest_path, detail)
    print(json.dumps({
        "status": "ok", "rows": len(rows),
        "dimension_source_status_counts": dict(sorted(source_counts.items())),
        "output_catalog_sha256": sha256_file(output_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
