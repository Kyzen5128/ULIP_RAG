#!/usr/bin/env python3
"""Resume-safe download and byte validation for product images/official GLBs."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, snapshot_root, write_json, write_jsonl
else:
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, snapshot_root, write_json, write_jsonl


UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"


def download(url: str, path: Path, kind: str, timeout: int) -> dict[str, Any]:
    if path.exists() and path.stat().st_size > 0:
        data = path.read_bytes()
        source = "existing"
    else:
        response = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        response.raise_for_status()
        data = response.content
        source = "downloaded"
    if kind == "image":
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if min(width, height) < 128:
                raise ValueError(f"image too small: {width}x{height}")
        extra = {"width": width, "height": height}
    else:
        if len(data) < 1024 or data[:4] != b"glTF":
            raise ValueError("not a valid binary GLB")
        extra = {}
    if source == "downloaded":
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(data)
        temp.replace(path)
    return {"status": "ok", "source": source, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), **extra}


def process(row: dict[str, Any], root: Path, timeout: int) -> dict[str, Any]:
    product_id = row["product_id"]
    result: dict[str, Any] = {"product_id": product_id, "status": "ok"}
    try:
        image_path = root / "assets" / "images" / f"{product_id}.jpg"
        result["image"] = {"path": str(image_path), **download(row["primary_image_url"], image_path, "image", timeout)}
    except Exception as error:
        result["status"] = "error"
        result["image"] = {"status": "error", "error": f"{type(error).__name__}: {error}"[:500]}
    supplementary = []
    seen_urls = {row.get("primary_image_url")}
    for index, asset in enumerate(row.get("image_assets") or []):
        url = asset.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        kind = str(asset.get("type") or "image").casefold().replace("_", "-")
        path = root / "assets" / "product_images" / product_id / f"{index:02d}_{kind}.jpg"
        try:
            supplementary.append({
                "asset_type": asset.get("type"), "alt_text": asset.get("alt_text") or "",
                "url": url, "path": str(path), **download(url, path, "image", timeout),
            })
        except Exception as error:
            supplementary.append({
                "asset_type": asset.get("type"), "url": url, "path": str(path),
                "status": "error", "error": f"{type(error).__name__}: {error}"[:500],
            })
    result["supplementary_images"] = supplementary
    if row.get("official_glb_url"):
        try:
            glb_path = root / "assets" / "official_glb" / f"{product_id}.glb"
            result["official_glb"] = {"path": str(glb_path), **download(row["official_glb_url"], glb_path, "glb", timeout)}
        except Exception as error:
            result["status"] = "error"
            result["official_glb"] = {"status": "error", "error": f"{type(error).__name__}: {error}"[:500]}
    else:
        result["official_glb"] = {"status": "unavailable"}
    result["checked_at"] = datetime.now(timezone.utc).isoformat()
    write_json(root / "state" / "assets" / f"{product_id}.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    root = snapshot_root(args.version, args.root)
    rows = read_jsonl(root / "catalog" / "products.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, row, root, args.timeout) for row in rows]
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 25 == 0 or index == len(futures):
                errors = sum(x["status"] != "ok" for x in results)
                print(f"assets {index}/{len(futures)} errors={errors}")
    results.sort(key=lambda x: x["product_id"])
    write_jsonl(root / "manifests" / "asset_rows.jsonl", results)
    report = {
        "schema_version": "ikea-asset-download/2.0",
        "snapshot_version": args.version,
        "processed_rows": len(results),
        "ok_rows": sum(x["status"] == "ok" for x in results),
        "error_rows": sum(x["status"] != "ok" for x in results),
        "image_rows": sum((x.get("image") or {}).get("status") == "ok" for x in results),
        "supplementary_image_rows": sum(
            sum(image.get("status") == "ok" for image in x.get("supplementary_images") or [])
            for x in results
        ),
        "supplementary_image_errors": sum(
            sum(image.get("status") == "error" for image in x.get("supplementary_images") or [])
            for x in results
        ),
        "official_glb_rows": sum((x.get("official_glb") or {}).get("status") == "ok" for x in results),
    }
    write_json(root / "manifests" / "asset_download.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["error_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
