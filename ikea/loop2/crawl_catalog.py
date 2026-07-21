#!/usr/bin/env python3
"""Create a fresh, versioned IKEA US furniture catalog snapshot.

The filesystem JSONL snapshot is the source of truth. Mongo is an optional
derived sink and is never required to read or resume the crawl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import (DEFAULT_ROOT, DEFAULT_TAXONOMY, DEFAULT_VERSION,
                                   contains_pattern, load_json, normalize_text,
                                   sha256_file, snapshot_root, write_json, write_jsonl)
else:
    from .common import (DEFAULT_ROOT, DEFAULT_TAXONOMY, DEFAULT_VERSION,
                         contains_pattern, load_json, normalize_text,
                         sha256_file, snapshot_root, write_json, write_jsonl)


PLP_URL = "https://sik.search.blue.cdtapps.com/us/en/product-list-page"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"


def classify(product: dict[str, Any], source: dict[str, str], taxonomy: dict[str, Any]) -> tuple[str | None, list[str]]:
    type_text = normalize_text(product.get("typeName"))
    full_text = normalize_text(
        product.get("typeName"), product.get("name"), product.get("mainImageAlt"),
        product.get("pipUrl"), product.get("validDesignText"),
    )
    exact_rejects = {str(x).casefold() for x in taxonomy.get("exact_reject_types", [])}
    if type_text in exact_rejects:
        return None, [f"blocked_exact_type:{type_text}"]
    for prefix in taxonomy.get("reject_type_prefixes", []):
        if type_text.startswith(str(prefix).casefold()):
            return None, [f"blocked_type_prefix:{prefix}"]

    overrides = (taxonomy.get("source_exact_overrides") or {}).get(source["source_id"], {})
    if type_text in overrides:
        return overrides[type_text], ["source_exact_override"]

    # Reject on IKEA's product type, not marketing/color text. For example,
    # "light oak" is a valid furniture color and must not be treated as a lamp.
    rejected = [p for p in taxonomy["global_reject_patterns"] if contains_pattern(type_text, p)]
    if rejected:
        return None, [f"blocked:{p}" for p in rejected]

    for rule in taxonomy["type_rules"]:
        if any(contains_pattern(type_text, p) for p in rule["patterns"]):
            return rule["category"], []

    # The PLP source is already a curated furniture category, but a product
    # without a recognizable furniture type is fail-closed instead of using a
    # broad source-category fallback that could admit accessories.
    return None, [f"unrecognized_type:{product.get('typeName') or 'missing'}"]


def canonical_row(product: dict[str, Any], source: dict[str, str], category: str, version: str) -> dict[str, Any]:
    image_assets = []
    for item in product.get("allProductImage") or []:
        if item.get("url"):
            image_assets.append({
                "url": item["url"], "type": item.get("type") or "UNSPECIFIED",
                "alt_text": item.get("altText") or "",
            })
    images = [x["url"] for x in image_assets]
    if product.get("mainImageUrl") and product["mainImageUrl"] not in images:
        images.insert(0, product["mainImageUrl"])
    price = product.get("salesPrice") or {}
    colors = [x.get("name") for x in (product.get("colors") or []) if x.get("name")]
    variants = []
    for item in ((product.get("gprDescription") or {}).get("variants") or []):
        if item.get("id"):
            variants.append({
                "product_id": str(item["id"]),
                "url": item.get("pipUrl"),
                "image_url": item.get("mainImageUrl") or item.get("imageUrl"),
                "type_name": item.get("typeName"),
                "dimensions_raw": item.get("itemMeasureReferenceText"),
            })
    return {
        "schema_version": "ikea-canonical-product/2.0",
        "snapshot_version": version,
        "product_id": str(product.get("id") or product.get("itemNo") or ""),
        "item_no": product.get("itemNo"),
        "name": product.get("name"),
        "type_name": product.get("typeName"),
        "canonical_category": category,
        "source_category_id": source["source_id"],
        "source_category": source["source_category"],
        "url": product.get("pipUrl"),
        "description": product.get("mainImageAlt") or product.get("contextualImageAlt") or "",
        "design_text": product.get("validDesignText") or "",
        "dimensions_raw": product.get("itemMeasureReferenceText"),
        "colors": colors,
        "price": price.get("numeral") if isinstance(price, dict) else price,
        "currency": price.get("currencyCode") if isinstance(price, dict) else None,
        "images": images,
        "image_assets": image_assets,
        "primary_image_url": product.get("mainImageUrl") or (images[0] if images else None),
        "contextual_image_url": product.get("contextualImageUrl"),
        "contextual_image_alt": product.get("contextualImageAlt") or "",
        "official_glb_url": None,
        "official_usdz_url": None,
        "variants": variants,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "IKEA_US_PLP",
    }


def fetch_source(session: requests.Session, source: dict[str, str], size: int, timeout: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    response = session.get(
        PLP_URL,
        params={"category": source["source_id"], "size": size, "c": "sr", "v": "20210322"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    items = (payload.get("productListPage") or {}).get("productWindow") or []
    products = [(item.get("product") or item) for item in items if isinstance(item, dict)]
    return payload, products


def mongo_replace(rows: list[dict[str, Any]], collection_name: str) -> None:
    from ikea.mongo_conn import get_client
    client = get_client()
    collection = client["furniture_db"][collection_name]
    collection.delete_many({})
    if rows:
        collection.insert_many(rows, ordered=False)
    collection.create_index("product_id", unique=True)
    collection.create_index("canonical_category")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--only", action="append", default=[], help="source category ID; repeatable")
    parser.add_argument("--limit-per-source", type=int, default=0)
    parser.add_argument("--api-size", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--mongo-commit", action="store_true")
    parser.add_argument("--mongo-collection", default=None)
    args = parser.parse_args()

    taxonomy = load_json(args.taxonomy)
    sources = taxonomy["sources"]
    if args.only:
        wanted = set(args.only)
        sources = [x for x in sources if x["source_id"] in wanted]
        unknown = wanted - {x["source_id"] for x in sources}
        if unknown:
            parser.error(f"unknown source IDs: {sorted(unknown)}")

    root = snapshot_root(args.version, args.root)
    raw_dir = root / "raw" / "plp"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    accepted_by_id: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    source_counts: dict[str, dict[str, int]] = {}

    for index, source in enumerate(sources, 1):
        payload, products = fetch_source(session, source, args.api_size, args.timeout)
        if args.limit_per_source:
            products = products[:args.limit_per_source]
        raw_payload_path = raw_dir / f"{source['source_id']}.json"
        write_json(raw_payload_path, payload)
        accepted = 0
        for product in products:
            category, reasons = classify(product, source, taxonomy)
            product_id = str(product.get("id") or product.get("itemNo") or "")
            if not product_id or not product.get("pipUrl") or not product.get("mainImageUrl"):
                reasons.append("missing_required_identity_or_asset")
                category = None
            if category is None:
                rejected.append({
                    "product_id": product_id or None,
                    "name": product.get("name"),
                    "type_name": product.get("typeName"),
                    "url": product.get("pipUrl"),
                    "source_category_id": source["source_id"],
                    "reasons": reasons,
                })
                continue
            row = canonical_row(product, source, category, args.version)
            # Same product can appear in overlapping IKEA PLP categories. The
            # first strict classification wins and later appearances are kept
            # as provenance rather than duplicate training rows.
            existing = accepted_by_id.get(product_id)
            if existing:
                existing.setdefault("also_seen_in", []).append(source["source_id"])
            else:
                accepted_by_id[product_id] = row
                accepted += 1
        source_counts[source["source_id"]] = {
            "raw": len(products), "new_accepted": accepted,
            "rejected": sum(1 for x in rejected if x["source_category_id"] == source["source_id"]),
            "raw_payload_path": str(raw_payload_path),
            "raw_payload_sha256": sha256_file(raw_payload_path),
        }
        print(f"[{index:02d}/{len(sources)}] {source['source_id']} {source['source_category']}: "
              f"raw={len(products)} new={accepted} rejected={source_counts[source['source_id']]['rejected']}")
        if index != len(sources) and args.sleep:
            time.sleep(args.sleep)

    rows = sorted(accepted_by_id.values(), key=lambda x: (x["canonical_category"], x["product_id"]))
    raw_catalog_path = root / "catalog" / "products_plp.jsonl"
    catalog_path = root / "catalog" / "products.jsonl"
    rejected_path = root / "catalog" / "rejected.jsonl"
    write_jsonl(raw_catalog_path, rows)
    write_jsonl(catalog_path, rows)
    write_jsonl(rejected_path, rejected)
    write_json(root / "policies" / "taxonomy.json", taxonomy)

    category_counts = Counter(row["canonical_category"] for row in rows)
    manifest = {
        "schema_version": "ikea-crawl-manifest/2.0",
        "snapshot_version": args.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": PLP_URL,
        "taxonomy_policy_version": taxonomy["policy_version"],
        "taxonomy_sha256": sha256_file(args.taxonomy),
        "catalog_rows": len(rows),
        "rejected_rows": len(rejected),
        "category_counts": dict(sorted(category_counts.items())),
        "source_counts": source_counts,
        "raw_catalog_path": str(raw_catalog_path),
        "raw_catalog_sha256": sha256_file(raw_catalog_path),
        # Backward-compatible alias: crawl-stage catalog, never enriched output.
        "catalog_sha256": sha256_file(raw_catalog_path),
        "rejected_sha256": sha256_file(rejected_path),
        "root": str(root),
    }
    write_json(root / "manifests" / "crawl_manifest.json", manifest)

    if args.mongo_commit:
        collection = args.mongo_collection or f"ikea_product_{args.version.replace('-', '_')}"
        mongo_replace(rows, collection)
        manifest["mongo_collection"] = collection
        write_json(root / "manifests" / "crawl_manifest.json", manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
