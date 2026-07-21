#!/usr/bin/env python3
"""Make source-image versus multi-view geometry sheets for review queues."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, write_jsonl
else:
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, write_jsonl


def fit(path: Path, size: int) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    image.thumbnail((size, size))
    tile = Image.new("RGB", (size, size), "white")
    tile.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return tile


def pages(values: list, rows_per_sheet: int) -> list[list]:
    if rows_per_sheet <= 0:
        raise ValueError("rows_per_sheet must be positive")
    return [values[index:index + rows_per_sheet] for index in range(0, len(values), rows_per_sheet)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--samples-per-group", type=int, default=20)
    parser.add_argument(
        "--rows-per-sheet", type=int, default=10,
        help="Paginate large review groups so every row remains visually inspectable",
    )
    parser.add_argument("--tile", type=int, default=224)
    parser.add_argument("--geometry-manifest", type=Path)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--source", choices=("all", "official", "trellis"), default="all")
    args = parser.parse_args()
    root = snapshot_root(args.version, args.root)
    catalog = {row["product_id"]: row for row in read_jsonl(root / "catalog" / "products.jsonl")}
    geometry_manifest = args.geometry_manifest or root / "manifests" / "geometry_rows.jsonl"
    geometry = read_jsonl(geometry_manifest)
    groups = defaultdict(list)
    for item in geometry:
        if args.source != "all" and item.get("source") != args.source:
            continue
        product_id = item["product_id"]
        row = catalog.get(product_id)
        if not row:
            continue
        render_dir = root / "assets" / "renders" / product_id
        views = [render_dir / f"{product_id}_r_{angle:03d}.png" for angle in (0, 90, 180, 270)]
        source = root / "assets" / "images" / f"{product_id}.jpg"
        if source.exists() and all(path.exists() for path in views):
            groups[(item["status"], row["canonical_category"])].append((row, item, source, views))
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    output_dir = root / "qa" / f"geometry_sheets{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    font = ImageFont.load_default()
    label_height = 40
    for (status, category), values in sorted(groups.items()):
        values.sort(key=lambda value: hashlib.sha256(value[0]["product_id"].encode()).hexdigest())
        selected = values[:args.samples_per_group] if args.samples_per_group > 0 else values
        safe = category.casefold().replace(" ", "_")
        for page_index, selected_page in enumerate(pages(selected, args.rows_per_sheet), 1):
            sheet = Image.new("RGB", (args.tile * 5, len(selected_page) * (args.tile + label_height)), "white")
            draw = ImageDraw.Draw(sheet)
            page_manifest = []
            for row_index, (row, geometry_row, source, views) in enumerate(selected_page):
                y = row_index * (args.tile + label_height)
                for column, path in enumerate([source, *views]):
                    sheet.paste(fit(path, args.tile), (column * args.tile, y))
                error = (geometry_row.get("geometry_report") or {}).get("relative_dimension_error")
                label = f"{row['product_id']} | {row.get('type_name')} | source={geometry_row.get('source')} | dim_error={error}"
                draw.text((4, y + args.tile + 4), label[:180], fill="black", font=font)
                page_manifest.append({
                    "status": status, "category": category, "product_id": row["product_id"],
                    "source_image": str(source), "render_images": [str(path) for path in views],
                    "dimension_error": error, "review_status": "pending_human_review" if status != "ok" else "sample_audit",
                })
            sheet_path = output_dir / f"{status}__{safe}__p{page_index:03d}.jpg"
            sheet.save(sheet_path, quality=92)
            sheet_hash = sha256_file(sheet_path)
            for item in page_manifest:
                item["sheet_path"] = str(sheet_path)
                item["sheet_sha256"] = sheet_hash
                item["sheet_page"] = page_index
            manifest.extend(page_manifest)
    write_jsonl(root / "qa" / f"geometry_sheet_manifest{suffix}.jsonl", manifest)
    print(json.dumps({"groups": len(groups), "rows": len(manifest), "geometry_manifest": str(geometry_manifest), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
