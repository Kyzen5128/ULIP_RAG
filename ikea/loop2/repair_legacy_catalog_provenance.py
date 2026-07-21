#!/usr/bin/env python3
"""One-time lossless migration for snapshots enriched in place.

The migration writes ``products_plp.jsonl`` only after reconstructing the
crawl-stage rows and proving their hash equals the existing crawl manifest.
It then augments the enrichment manifest with explicit input/output hashes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, write_json, write_jsonl
else:
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, write_json, write_jsonl


ENRICHMENT_ONLY_FIELDS = {
    "detail_status", "dimension_profile", "dimension_source",
    "dimension_source_status", "page_sha256",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = snapshot_root(args.version, args.root)
    enriched_path = root / "catalog" / "products.jsonl"
    raw_path = root / "catalog" / "products_plp.jsonl"
    crawl_manifest_path = root / "manifests" / "crawl_manifest.json"
    detail_manifest_path = root / "manifests" / "detail_enrichment.json"
    crawl = json.loads(crawl_manifest_path.read_text(encoding="utf-8"))
    expected = crawl.get("raw_catalog_sha256") or crawl.get("catalog_sha256")
    if raw_path.exists():
        actual = sha256_file(raw_path)
        if actual != expected:
            raise SystemExit(f"existing raw catalog hash mismatch: expected={expected}, actual={actual}")
    else:
        reconstructed = []
        for source in read_jsonl(enriched_path):
            row = {key: value for key, value in source.items() if key not in ENRICHMENT_ONLY_FIELDS}
            row["official_glb_url"] = None
            row["official_usdz_url"] = None
            reconstructed.append(row)
        # Write to a temporary sibling and verify before accepting the result.
        temporary = raw_path.with_suffix(".jsonl.reconstructed")
        write_jsonl(temporary, reconstructed)
        actual = sha256_file(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise SystemExit(f"lossless reconstruction failed: expected={expected}, actual={actual}")
        temporary.replace(raw_path)
    crawl["raw_catalog_path"] = str(raw_path)
    crawl["raw_catalog_sha256"] = sha256_file(raw_path)
    crawl["catalog_sha256"] = crawl["raw_catalog_sha256"]
    for source_id, source in (crawl.get("source_counts") or {}).items():
        payload_path = root / "raw" / "plp" / f"{source_id}.json"
        if not payload_path.is_file():
            raise SystemExit(f"missing raw PLP payload: {payload_path}")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        product_count = int((payload.get("productListPage") or {}).get("productCount", -1))
        window_count = len((payload.get("productListPage") or {}).get("productWindow") or [])
        if product_count != window_count or int(source.get("raw", -1)) != window_count:
            raise SystemExit(
                f"raw PLP count mismatch for {source_id}: "
                f"manifest={source.get('raw')}, productCount={product_count}, window={window_count}"
            )
        source["raw_payload_path"] = str(payload_path)
        source["raw_payload_sha256"] = sha256_file(payload_path)
    write_json(crawl_manifest_path, crawl)
    detail = json.loads(detail_manifest_path.read_text(encoding="utf-8"))
    detail.update({
        "input_catalog_path": str(raw_path),
        "input_catalog_sha256": sha256_file(raw_path),
        "output_catalog_path": str(enriched_path),
        "output_catalog_sha256": sha256_file(enriched_path),
    })
    write_json(detail_manifest_path, detail)
    print(json.dumps({
        "status": "ok",
        "raw_catalog_path": str(raw_path),
        "raw_catalog_sha256": sha256_file(raw_path),
        "enriched_catalog_sha256": sha256_file(enriched_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
