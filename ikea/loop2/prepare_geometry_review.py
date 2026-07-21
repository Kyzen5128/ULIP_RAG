#!/usr/bin/env python3
"""Create a hash-bound visual geometry review queue without making decisions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, write_jsonl
else:
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=("all", "official", "trellis"), default="all")
    args = parser.parse_args()
    root = snapshot_root(args.version, args.root)
    geometry = {row["product_id"]: row for row in read_jsonl(args.geometry_manifest)}
    renders = {row["product_id"]: row for row in read_jsonl(args.render_manifest)}
    queue = []
    for product_id, row in sorted(geometry.items()):
        if row.get("status") not in {"quarantined", "needs_visual_qa"}:
            continue
        if args.source != "all" and row.get("source") != args.source:
            continue
        source_image = root / "assets" / "images" / f"{product_id}.jpg"
        render_rows = renders.get(product_id, {}).get("renders") or []
        render_paths = [Path(item["path"]) for item in render_rows]
        canonical = Path(row["canonical_glb"])
        if not source_image.is_file() or not canonical.is_file() or not render_paths or not all(path.is_file() for path in render_paths):
            raise SystemExit(f"review evidence is incomplete for {product_id}")
        queue.append({
            "schema_version": "ikea-geometry-review-decision/2.0",
            "product_id": product_id,
            "source": row.get("source"),
            "input_status": row.get("status"),
            "source_image": str(source_image),
            "source_image_sha256": sha256_file(source_image),
            "canonical_glb": str(canonical),
            "canonical_glb_sha256": sha256_file(canonical),
            "render_evidence": [{"path": str(path), "sha256": sha256_file(path)} for path in render_paths],
            "decision": "pending",
            "reviewer": None,
            "reviewer_type": None,
            "notes": None,
        })
    write_jsonl(args.output, queue)
    print(f"review_queue_rows={len(queue)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
