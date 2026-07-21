#!/usr/bin/env python3
"""Fail-closed structural/category QA for an IKEA 2.0 catalog snapshot."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_TAXONOMY, DEFAULT_VERSION, load_json, read_jsonl, snapshot_root, write_json, write_jsonl
else:
    from .common import DEFAULT_ROOT, DEFAULT_TAXONOMY, DEFAULT_VERSION, load_json, read_jsonl, snapshot_root, write_json, write_jsonl


REQUIRED = ("product_id", "name", "type_name", "canonical_category", "url", "primary_image_url")


def valid_ikea_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and parsed.hostname in {"www.ikea.com", "ikea.com"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--sample-per-category", type=int, default=5)
    parser.add_argument("--require-details", action="store_true")
    args = parser.parse_args()

    root = snapshot_root(args.version, args.root)
    rows = read_jsonl(root / "catalog" / "products.jsonl")
    taxonomy = load_json(args.taxonomy)
    allowed = set(taxonomy["canonical_categories"])
    seen: set[str] = set()
    issues: list[dict] = []
    by_category: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        product_id = str(row.get("product_id") or "")
        reasons = []
        for field in REQUIRED:
            if row.get(field) in (None, "", []):
                reasons.append(f"missing:{field}")
        if product_id in seen:
            reasons.append("duplicate:product_id")
        seen.add(product_id)
        if row.get("canonical_category") not in allowed:
            reasons.append("invalid:canonical_category")
        if not valid_ikea_url(row.get("url")):
            reasons.append("invalid:product_url")
        if not valid_ikea_url(row.get("primary_image_url")):
            reasons.append("invalid:image_url")
        if args.require_details and row.get("detail_status") != "ok":
            reasons.append("missing:detail_enrichment")
        if reasons:
            issues.append({"product_id": product_id or None, "reasons": reasons, "row": row})
        else:
            by_category[row["canonical_category"]].append(row)

    samples = []
    for category in sorted(by_category):
        for row in by_category[category][:args.sample_per_category]:
            samples.append({
                "product_id": row["product_id"], "category": category,
                "name": row["name"], "type_name": row["type_name"],
                "url": row["url"], "primary_image_url": row["primary_image_url"],
                "dimensions": row.get("dimension_profile"),
            })
    write_jsonl(root / "qa" / "category_samples.jsonl", samples)
    write_jsonl(root / "qa" / "structural_issues.jsonl", issues)
    report = {
        "schema_version": "ikea-catalog-validation/2.0",
        "snapshot_version": args.version,
        "valid": not issues,
        "catalog_rows": len(rows),
        "valid_rows": len(rows) - len(issues),
        "issue_rows": len(issues),
        "category_counts": dict(sorted(Counter(row.get("canonical_category") for row in rows).items())),
        "detail_ok_rows": sum(row.get("detail_status") == "ok" for row in rows),
        "official_glb_rows": sum(bool(row.get("official_glb_url")) for row in rows),
        "three_axis_rows": sum((row.get("dimension_profile") or {}).get("axis_count") == 3 for row in rows),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "qa" / "catalog_validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

