#!/usr/bin/env python3
"""Create deterministic per-category contact sheets for human visual QA."""
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
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, snapshot_root, write_jsonl
else:
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, snapshot_root, write_jsonl


def stable_order(product_id: str) -> str:
    return hashlib.sha256(product_id.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--samples-per-category", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile", type=int, default=256)
    args = parser.parse_args()
    root = snapshot_root(args.version, args.root)
    rows = read_jsonl(root / "catalog" / "products.jsonl")
    groups = defaultdict(list)
    for row in rows:
        image_path = root / "assets" / "images" / f"{row['product_id']}.jpg"
        if image_path.exists():
            groups[row["canonical_category"]].append((row, image_path))
    output_dir = root / "qa" / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    label_height = 58
    font = ImageFont.load_default()
    for category in sorted(groups):
        chosen = sorted(groups[category], key=lambda item: stable_order(item[0]["product_id"]))[:args.samples_per_category]
        rows_count = (len(chosen) + args.columns - 1) // args.columns
        sheet = Image.new("RGB", (args.columns * args.tile, rows_count * (args.tile + label_height)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (row, path) in enumerate(chosen):
            x = (index % args.columns) * args.tile
            y = (index // args.columns) * (args.tile + label_height)
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                image.thumbnail((args.tile, args.tile))
            paste_x = x + (args.tile - image.width) // 2
            paste_y = y + (args.tile - image.height) // 2
            sheet.paste(image, (paste_x, paste_y))
            label = f"{row['product_id']}\n{str(row.get('type_name') or '')[:38]}"
            draw.multiline_text((x + 4, y + args.tile + 3), label, fill="black", font=font, spacing=2)
            manifest.append({
                "category": category, "product_id": row["product_id"],
                "type_name": row.get("type_name"), "image": str(path), "url": row.get("url"),
            })
        safe_name = category.casefold().replace(" ", "_")
        sheet.save(output_dir / f"{safe_name}.jpg", quality=92)
    write_jsonl(root / "qa" / "contact_sheet_samples.jsonl", manifest)
    print(json.dumps({"categories": len(groups), "sample_rows": len(manifest), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
