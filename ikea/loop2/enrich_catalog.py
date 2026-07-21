#!/usr/bin/env python3
"""Fetch every IKEA product page and extract official 3D URLs + named dimensions."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, snapshot_root, write_json, write_jsonl
    from ikea.loop2.dimensions import parse_named_measurements
else:
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, snapshot_root, write_json, write_jsonl
    from .dimensions import parse_named_measurements


UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"
GLB_RE = re.compile(r"https://web-api\.ikea\.com/dimma/assets/[^\"'<>\\]+?\.glb(?:\?[^\"'<>\\]*)?", re.I)
USDZ_RE = re.compile(r"https://web-api\.ikea\.com/dimma/assets/[^\"'<>\\]+?\.usdz(?:\?[^\"'<>\\]*)?", re.I)
MEASUREMENTS_RE = re.compile(r'"measurements":(\[\{.*?\}\])', re.S)


def extract_page(page: str) -> dict[str, Any]:
    decoded = html.unescape(page).replace("\\u002F", "/").replace("\\/", "/")
    glbs = list(dict.fromkeys(GLB_RE.findall(decoded)))
    usdzs = list(dict.fromkeys(USDZ_RE.findall(decoded)))
    measurements: list[dict[str, Any]] = []
    dimension_source = None
    dimension_source_status = "missing"
    for match in MEASUREMENTS_RE.finditer(decoded):
        try:
            candidate = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if candidate and all(isinstance(x, dict) for x in candidate):
            names = {str(x.get("name") or "").casefold() for x in candidate}
            if names & {"width", "length", "depth", "height"}:
                measurements = candidate
                context = decoded[max(0, match.start() - 3000):match.start()]
                if '"measurementsProps"' in context:
                    dimension_source = "ikea_pip_measurementsProps"
                    dimension_source_status = "verified_product_measurements_context"
                else:
                    dimension_source = "unknown_page_measurement_array"
                    dimension_source_status = "unverified_context"
                break
    return {
        "official_glb_urls": glbs,
        "official_usdz_urls": usdzs,
        "measurements": measurements,
        "dimension_profile": parse_named_measurements(measurements),
        "dimension_source": dimension_source,
        "dimension_source_status": dimension_source_status,
    }


def fetch_one(row: dict[str, Any], state_dir: Path, html_dir: Path | None, timeout: int) -> dict[str, Any]:
    product_id = row["product_id"]
    state_path = state_dir / f"{product_id}.json"
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    try:
        response = requests.get(row["url"], headers={"User-Agent": UA}, timeout=timeout)
        response.raise_for_status()
        extracted = extract_page(response.text)
        body = response.content
        result = {
            "product_id": product_id,
            "url": row["url"],
            "status": "ok",
            "http_status": response.status_code,
            "page_sha256": hashlib.sha256(body).hexdigest(),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            **extracted,
        }
        if html_dir is not None:
            html_dir.mkdir(parents=True, exist_ok=True)
            with gzip.open(html_dir / f"{product_id}.html.gz", "wb", compresslevel=6) as handle:
                handle.write(body)
    except Exception as error:
        result = {
            "product_id": product_id,
            "url": row.get("url"),
            "status": "error",
            "error": f"{type(error).__name__}: {error}"[:500],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    write_json(state_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-html", action="store_true")
    args = parser.parse_args()

    root = snapshot_root(args.version, args.root)
    raw_catalog_path = root / "catalog" / "products_plp.jsonl"
    catalog_path = root / "catalog" / "products.jsonl"
    if not raw_catalog_path.is_file():
        raise SystemExit(
            f"missing immutable crawl-stage catalog: {raw_catalog_path}; "
            "repair a legacy snapshot before enrichment"
        )
    rows = read_jsonl(raw_catalog_path)
    targets = rows[:args.limit] if args.limit else rows
    state_dir = root / "state" / "details"
    state_dir.mkdir(parents=True, exist_ok=True)
    html_dir = root / "raw" / "product_pages" if args.save_html else None

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_one, row, state_dir, html_dir, args.timeout): row for row in targets}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results[result["product_id"]] = result
            if index % 25 == 0 or index == len(futures):
                ok = sum(x.get("status") == "ok" for x in results.values())
                print(f"details {index}/{len(futures)} ok={ok} error={index-ok}")

    all_states: dict[str, dict[str, Any]] = {}
    for path in state_dir.glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        all_states[value["product_id"]] = value
    for row in rows:
        detail = all_states.get(row["product_id"])
        if not detail or detail.get("status") != "ok":
            row["detail_status"] = detail.get("status") if detail else "pending"
            continue
        row["detail_status"] = "ok"
        row["official_glb_url"] = (detail.get("official_glb_urls") or [None])[0]
        row["official_usdz_url"] = (detail.get("official_usdz_urls") or [None])[0]
        row["dimension_profile"] = detail.get("dimension_profile")
        row["dimension_source"] = detail.get("dimension_source")
        row["dimension_source_status"] = detail.get("dimension_source_status")
        row["page_sha256"] = detail.get("page_sha256")
    write_jsonl(catalog_path, rows)

    summary = {
        "schema_version": "ikea-detail-enrichment/2.0",
        "snapshot_version": args.version,
        "input_catalog_path": str(raw_catalog_path),
        "input_catalog_sha256": hashlib.sha256(raw_catalog_path.read_bytes()).hexdigest(),
        "output_catalog_path": str(catalog_path),
        "output_catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "catalog_rows": len(rows),
        "processed_rows": len(all_states),
        "ok_rows": sum(x.get("status") == "ok" for x in all_states.values()),
        "error_rows": sum(x.get("status") == "error" for x in all_states.values()),
        "official_glb_rows": sum(bool(x.get("official_glb_urls")) for x in all_states.values()),
        "three_axis_rows": sum((x.get("dimension_profile") or {}).get("axis_count") == 3 for x in all_states.values()),
        "partial_dimension_rows": sum((x.get("dimension_profile") or {}).get("axis_count", 0) in (1, 2) for x in all_states.values()),
        "missing_dimension_rows": sum((x.get("dimension_profile") or {}).get("axis_count", 0) == 0 for x in all_states.values()),
        "range_present_rows": sum((x.get("dimension_profile") or {}).get("dimension_quality_status") == "range_present" for x in all_states.values()),
        "hard_filter_eligible_rows": sum((x.get("dimension_profile") or {}).get("hard_filter_eligible") is True for x in all_states.values()),
        "dimension_source_status_counts": {
            status: sum(x.get("dimension_source_status") == status for x in all_states.values())
            for status in sorted({str(x.get("dimension_source_status") or "missing") for x in all_states.values()})
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "manifests" / "detail_enrichment.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["error_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
