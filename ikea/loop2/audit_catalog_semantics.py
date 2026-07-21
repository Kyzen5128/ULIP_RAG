#!/usr/bin/env python3
"""Semantic catalog audit that complements structural schema validation."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, snapshot_root, write_json, write_jsonl
else:
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, snapshot_root, write_json, write_jsonl


ACCESSORY_PHRASES = (
    "seat pad", "chair pad", "cover", "cushion", "headrest", "armrests only",
    "floor protector", "tabletop", "table top", "desk top", "underframe",
    "replacement", "spare part", "slipcover", "door only", "drawer front",
)
FURNITURE_HINTS = (
    "chair", "table", "desk", "bed", "sofa", "loveseat", "sectional", "cabinet",
    "wardrobe", "bookcase", "shelf", "bench", "stool", "ottoman", "pouf", "dresser",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = snapshot_root(args.version, args.root)
    rows = read_jsonl(root / "catalog" / "products.jsonl")
    rejected = read_jsonl(root / "catalog" / "rejected.jsonl")
    accepted_ids = {str(row["product_id"]) for row in rows}
    types: dict[str, Counter] = defaultdict(Counter)
    accepted_flags = []
    for row in rows:
        type_name = str(row.get("type_name") or "")
        types[row["canonical_category"]][type_name] += 1
        lowered = type_name.casefold()
        hits = [phrase for phrase in ACCESSORY_PHRASES if phrase in lowered]
        if hits:
            accepted_flags.append({
                "product_id": row["product_id"], "category": row["canonical_category"],
                "type_name": type_name, "hits": hits, "url": row.get("url"),
                "review_status": "manual_review_required",
            })
    rejected_furniture_like = []
    for row in rejected:
        if str(row.get("product_id") or "") in accepted_ids:
            continue
        if not any(str(reason).startswith("unrecognized_type:") for reason in row.get("reasons") or []):
            continue
        lowered = str(row.get("type_name") or "").casefold()
        hits = [word for word in FURNITURE_HINTS if word in lowered]
        if hits:
            rejected_furniture_like.append({**row, "furniture_hints": hits, "review_status": "manual_review_required"})
    type_rows = []
    for category in sorted(types):
        for type_name, count in sorted(types[category].items()):
            type_rows.append({"category": category, "type_name": type_name, "count": count})
    write_jsonl(root / "qa" / "accepted_accessory_flags.jsonl", accepted_flags)
    write_jsonl(root / "qa" / "rejected_furniture_like.jsonl", rejected_furniture_like)
    write_jsonl(root / "qa" / "type_distribution.jsonl", type_rows)
    category_counts = Counter(row["canonical_category"] for row in rows)
    report = {
        "schema_version": "ikea-semantic-catalog-audit/2.0",
        "snapshot_version": args.version,
        "catalog_rows": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "unique_product_types": len(type_rows),
        "accepted_accessory_flag_rows": len(accepted_flags),
        "rejected_furniture_like_rows": len(rejected_furniture_like),
        "low_inventory_categories": {key: value for key, value in sorted(category_counts.items()) if value < 10},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "interpretation": {
            "accepted_accessory_flags": "Review queue; a phrase can still describe a complete product (for example, chair with pad).",
            "rejected_furniture_like": "Potential false negatives; no row is automatically promoted from this queue.",
            "low_inventory_categories": "Coverage warning, not a reason to duplicate or lower the acceptance rules.",
        },
    }
    write_json(root / "qa" / "semantic_audit.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
