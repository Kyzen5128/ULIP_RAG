#!/usr/bin/env python3
"""Independent fail-closed integrity checks for every IKEA snapshot stage."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, split_name_for_group, write_json
else:
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, split_name_for_group, write_json


def issue(issues: list[dict[str, Any]], stage: str, code: str, **detail: Any) -> None:
    issues.append({"stage": stage, "code": code, **detail})


def checked_path(raw: str | None, root: Path, issues: list[dict[str, Any]], stage: str, product_id: str) -> Path | None:
    if not raw:
        issue(issues, stage, "MISSING_PATH", product_id=product_id)
        return None
    path = Path(raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        issue(issues, stage, "PATH_OUTSIDE_SNAPSHOT", product_id=product_id, path=str(path))
        return None
    if not path.is_file():
        issue(issues, stage, "FILE_MISSING", product_id=product_id, path=str(path))
        return None
    return path


def verify_assets(root: Path, catalog_ids: set[str], issues: list[dict[str, Any]]) -> dict[str, int]:
    rows = read_jsonl(root / "manifests" / "asset_rows.jsonl")
    row_ids = [str(row.get("product_id")) for row in rows]
    if set(row_ids) != catalog_ids or len(row_ids) != len(catalog_ids):
        issue(issues, "assets", "ID_SET_MISMATCH", expected=len(catalog_ids), actual=len(row_ids))
    checked_images = checked_glbs = 0
    for row in rows:
        product_id = str(row.get("product_id"))
        records = [(row.get("image") or {}, "image")]
        records.extend((record, "image") for record in row.get("supplementary_images") or [])
        if (row.get("official_glb") or {}).get("status") == "ok":
            records.append((row["official_glb"], "glb"))
        for record, kind in records:
            if record.get("status") != "ok":
                issue(issues, "assets", "NON_OK_ASSET", product_id=product_id, kind=kind, status=record.get("status"))
                continue
            path = checked_path(record.get("path"), root, issues, "assets", product_id)
            if path is None:
                continue
            expected_hash = record.get("sha256")
            actual_hash = sha256_file(path)
            if expected_hash != actual_hash:
                issue(issues, "assets", "SHA256_MISMATCH", product_id=product_id, path=str(path), expected=expected_hash, actual=actual_hash)
                continue
            try:
                if kind == "image":
                    with Image.open(path) as image:
                        image.verify()
                    checked_images += 1
                else:
                    with path.open("rb") as handle:
                        if handle.read(4) != b"glTF":
                            raise ValueError("missing glTF header")
                    checked_glbs += 1
            except Exception as error:
                issue(issues, "assets", "DECODE_FAILED", product_id=product_id, path=str(path), error=f"{type(error).__name__}: {error}")
    return {"rows": len(rows), "images_rehashed_and_decoded": checked_images, "glbs_rehashed_and_header_checked": checked_glbs}


def verify_geometry(root: Path, catalog_ids: set[str], issues: list[dict[str, Any]], manifest: Path) -> dict[str, int]:
    rows = read_jsonl(manifest)
    row_ids = [str(row.get("product_id")) for row in rows]
    if set(row_ids) != catalog_ids or len(row_ids) != len(catalog_ids):
        issue(issues, "geometry", "ID_SET_MISMATCH", expected=len(catalog_ids), actual=len(row_ids))
    counts = Counter(str(row.get("status")) for row in rows)
    for row in rows:
        if row.get("status") not in {"ok", "quarantined", "needs_visual_qa"}:
            continue
        product_id = str(row.get("product_id"))
        glb = checked_path(row.get("canonical_glb"), root, issues, "geometry", product_id)
        ply = checked_path(row.get("pointcloud"), root, issues, "geometry", product_id)
        if glb and glb.read_bytes()[:4] != b"glTF":
            issue(issues, "geometry", "INVALID_CANONICAL_GLB", product_id=product_id, path=str(glb))
        point_count = ((row.get("pointcloud_report") or {}).get("point_count"))
        if ply and point_count != 8192:
            issue(issues, "geometry", "POINT_COUNT_MISMATCH", product_id=product_id, point_count=point_count)
        if row.get("source") == "trellis" and row.get("status") == "ok":
            review = row.get("visual_review") or {}
            if not (
                review.get("decision") == "accept"
                and review.get("reviewer")
                and review.get("evidence_bundle_sha256")
            ):
                issue(issues, "geometry", "UNREVIEWED_TRELLIS_PROMOTED", product_id=product_id)
    return {"rows": len(rows), **{f"status_{key}": value for key, value in sorted(counts.items())}}


def verify_renders(root: Path, geometry_ids: set[str], issues: list[dict[str, Any]], manifest: Path) -> dict[str, int]:
    rows = read_jsonl(manifest)
    checked = 0
    for row in rows:
        product_id = str(row.get("product_id"))
        if product_id not in geometry_ids:
            issue(issues, "renders", "UNKNOWN_PRODUCT_ID", product_id=product_id)
        if row.get("status") != "ok":
            continue
        renders = row.get("renders") or []
        if len(renders) != 12:
            issue(issues, "renders", "VIEW_COUNT_MISMATCH", product_id=product_id, views=len(renders))
        for render in renders:
            path = checked_path(render.get("path"), root, issues, "renders", product_id)
            if path:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    checked += 1
                except Exception as error:
                    issue(issues, "renders", "DECODE_FAILED", product_id=product_id, path=str(path), error=f"{type(error).__name__}: {error}")
    return {"rows": len(rows), "images_decoded": checked}


def verify_dataset(root: Path, catalog_ids: set[str], issues: list[dict[str, Any]]) -> dict[str, int]:
    samples = read_jsonl(root / "dataset" / "samples.jsonl")
    corpus = read_jsonl(root / "corpus" / "product_corpus.jsonl")
    build_manifest = json.loads((root / "manifests" / "training_dataset.json").read_text(encoding="utf-8"))
    ratios = build_manifest.get("split_ratios") or {"train": 0.8, "val": 0.1, "test": 0.1}
    split_ratios = (float(ratios["train"]), float(ratios["val"]), float(ratios["test"]))
    sample_ids = {str(sample.get("product_id")) for sample in samples}
    if not sample_ids <= catalog_ids:
        issue(issues, "dataset", "UNKNOWN_SAMPLE_IDS", count=len(sample_ids - catalog_ids))
    for sample in samples:
        product_id = str(sample.get("product_id"))
        expected = split_name_for_group(str(sample.get("split_group")), *split_ratios)
        if sample.get("split") != expected:
            issue(issues, "dataset", "SAMPLE_SPLIT_MISMATCH", product_id=product_id, expected=expected, actual=sample.get("split"))
        for field in ("image", "pointcloud", "mesh"):
            checked_path(sample.get(field), root, issues, "dataset", product_id)
    for document in corpus:
        product_id = str(document.get("product_id"))
        if product_id not in sample_ids:
            issue(issues, "dataset", "CORPUS_WITHOUT_ACCEPTED_SAMPLE", product_id=product_id, doc_id=document.get("doc_id"))
        expected = split_name_for_group(str(document.get("split_group")), *split_ratios)
        if document.get("split") != expected:
            issue(issues, "dataset", "CORPUS_SPLIT_MISMATCH", doc_id=document.get("doc_id"), expected=expected, actual=document.get("split"))
    return {"samples": len(samples), "corpus_documents": len(corpus)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--through", choices=("catalog", "assets", "geometry", "renders", "dataset"), default="dataset")
    parser.add_argument("--geometry-manifest", type=Path)
    parser.add_argument("--render-manifest", type=Path)
    args = parser.parse_args()
    root = snapshot_root(args.version, args.root).resolve()
    geometry_manifest = (args.geometry_manifest or root / "manifests" / "geometry_rows.jsonl").resolve()
    render_manifest = (args.render_manifest or root / "manifests" / "render_rows.jsonl").resolve()
    stages = ("catalog", "assets", "geometry", "renders", "dataset")
    selected = stages[: stages.index(args.through) + 1]
    issues: list[dict[str, Any]] = []
    catalog = read_jsonl(root / "catalog" / "products.jsonl")
    ids = [str(row.get("product_id")) for row in catalog]
    if len(ids) != len(set(ids)):
        issue(issues, "catalog", "DUPLICATE_PRODUCT_ID", rows=len(ids), unique=len(set(ids)))
    if any(not value or value == "None" for value in ids):
        issue(issues, "catalog", "EMPTY_PRODUCT_ID")
    verified_dimension_rows = 0
    for row in catalog:
        url_tail = str(row.get("url") or "").rstrip("/").rsplit("-", 1)[-1]
        if url_tail.casefold() != str(row.get("product_id") or "").casefold():
            issue(issues, "catalog", "PRODUCT_URL_ID_MISMATCH", product_id=row.get("product_id"), url=row.get("url"))
        profile = row.get("dimension_profile") or {}
        dimensions = profile.get("dimensions_mm") or {}
        if dimensions:
            if row.get("dimension_source_status") != "verified_product_measurements_context":
                issue(issues, "catalog", "UNVERIFIED_DIMENSION_SOURCE", product_id=row.get("product_id"), status=row.get("dimension_source_status"))
            else:
                verified_dimension_rows += 1
        for axis, value in dimensions.items():
            if not isinstance(value, (int, float)) or value <= 0:
                issue(issues, "catalog", "INVALID_DIMENSION_VALUE", product_id=row.get("product_id"), axis=axis, value=value)
        if profile.get("hard_filter_eligible") is True and (
            profile.get("status") != "three_axes" or profile.get("ranges_mm")
        ):
            issue(issues, "catalog", "INVALID_HARD_FILTER_ELIGIBILITY", product_id=row.get("product_id"))
    crawl_manifest = json.loads((root / "manifests" / "crawl_manifest.json").read_text(encoding="utf-8"))
    detail_manifest = json.loads((root / "manifests" / "detail_enrichment.json").read_text(encoding="utf-8"))
    raw_catalog_path = Path(crawl_manifest.get("raw_catalog_path") or root / "catalog" / "products_plp.jsonl")
    raw_hash = sha256_file(raw_catalog_path) if raw_catalog_path.is_file() else None
    enriched_hash = sha256_file(root / "catalog" / "products.jsonl")
    if raw_hash != crawl_manifest.get("raw_catalog_sha256"):
        issue(issues, "catalog", "RAW_CATALOG_PROVENANCE_MISMATCH", expected=crawl_manifest.get("raw_catalog_sha256"), actual=raw_hash)
    if raw_hash != detail_manifest.get("input_catalog_sha256"):
        issue(issues, "catalog", "ENRICHMENT_INPUT_PROVENANCE_MISMATCH", expected=detail_manifest.get("input_catalog_sha256"), actual=raw_hash)
    if enriched_hash != detail_manifest.get("output_catalog_sha256"):
        issue(issues, "catalog", "ENRICHMENT_OUTPUT_PROVENANCE_MISMATCH", expected=detail_manifest.get("output_catalog_sha256"), actual=enriched_hash)
    raw_payloads_checked = 0
    for source_id, source in (crawl_manifest.get("source_counts") or {}).items():
        path = Path(source.get("raw_payload_path") or root / "raw" / "plp" / f"{source_id}.json")
        if not path.is_file():
            issue(issues, "catalog", "RAW_PLP_MISSING", source_id=source_id, path=str(path))
            continue
        actual = sha256_file(path)
        if actual != source.get("raw_payload_sha256"):
            issue(issues, "catalog", "RAW_PLP_SHA256_MISMATCH", source_id=source_id, expected=source.get("raw_payload_sha256"), actual=actual)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        page = payload.get("productListPage") or {}
        product_count = int(page.get("productCount", -1))
        window_count = len(page.get("productWindow") or [])
        if product_count != window_count or int(source.get("raw", -1)) != window_count:
            issue(issues, "catalog", "RAW_PLP_COUNT_MISMATCH", source_id=source_id, manifest=source.get("raw"), product_count=product_count, window_count=window_count)
            continue
        raw_payloads_checked += 1
    summary: dict[str, Any] = {"catalog": {
        "rows": len(catalog), "unique_ids": len(set(ids)),
        "raw_catalog_sha256": raw_hash, "enriched_catalog_sha256": enriched_hash,
        "raw_plp_payloads_hash_and_count_checked": raw_payloads_checked,
        "verified_dimension_rows": verified_dimension_rows,
    }}
    catalog_ids = set(ids)
    if "assets" in selected:
        summary["assets"] = verify_assets(root, catalog_ids, issues)
    if "geometry" in selected:
        summary["geometry"] = verify_geometry(root, catalog_ids, issues, geometry_manifest)
    geometry_ids = {
        str(row.get("product_id")) for row in read_jsonl(geometry_manifest)
        if row.get("status") in {"ok", "quarantined", "needs_visual_qa"}
    } if "renders" in selected else set()
    if "renders" in selected:
        summary["renders"] = verify_renders(root, geometry_ids, issues, render_manifest)
    if "dataset" in selected:
        summary["dataset"] = verify_dataset(root, catalog_ids, issues)
    report = {
        "schema_version": "ikea-snapshot-verification/2.0",
        "snapshot_version": args.version,
        "verified_through": args.through,
        "geometry_manifest": str(geometry_manifest),
        "render_manifest": str(render_manifest),
        "valid": not issues,
        "summary": summary,
        "issue_count": len(issues),
        "issues": issues[:1000],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "qa" / f"snapshot_verification_{args.through}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
