#!/usr/bin/env python3
"""Resume-safe multi-view rendering plus blank/corrupt image QA."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, snapshot_root, write_json, write_jsonl
else:
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, snapshot_root, write_json, write_jsonl


HERE = Path(__file__).resolve().parent


def validate_render(path: Path, resolution: int) -> dict:
    with Image.open(path) as opened:
        if opened.size != (resolution, resolution):
            raise ValueError(f"unexpected render size {opened.size}")
        if opened.mode != "RGBA":
            raise ValueError(f"render must preserve RGBA object mask, got mode={opened.mode}")
        rgba = np.asarray(opened, dtype=np.uint8)
    foreground = rgba[:, :, 3] > 8
    fraction = float(foreground.mean())
    if fraction < 0.005:
        raise ValueError(f"likely blank render, foreground_fraction={fraction:.6f}")
    if fraction > 0.95:
        raise ValueError(f"likely camera/object framing failure, foreground_fraction={fraction:.6f}")
    # ULIP's existing loader converts to RGB. Publish an explicit white
    # composite so alpha handling cannot vary by PIL/torchvision version.
    rgba_float = rgba.astype(np.float32) / 255.0
    alpha = rgba_float[:, :, 3:4]
    rgb = rgba_float[:, :, :3] * alpha + (1.0 - alpha)
    composite = Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB")
    temporary = path.with_suffix(".validated.png.tmp")
    composite.save(temporary, format="PNG")
    temporary.replace(path)
    return {
        "path": str(path), "foreground_fraction": round(fraction, 6),
        "mask_source": "render_alpha", "published_mode": "RGB_white_composite",
    }


def validate_cached_render(path: Path, resolution: int) -> None:
    """Decode a previously published RGB render before trusting its state row."""
    with Image.open(path) as opened:
        if opened.size != (resolution, resolution):
            raise ValueError(f"unexpected cached render size {opened.size}")
        if opened.mode != "RGB":
            raise ValueError(f"cached published render must be RGB, got mode={opened.mode}")
        opened.verify()


def render_one(row: dict, root: Path, args: argparse.Namespace) -> dict:
    product_id = row["product_id"]
    output_dir = root / "assets" / "renders" / product_id
    state_path = root / "state" / "renders" / f"{product_id}.json"
    expected = [output_dir / f"{product_id}_r_{round(360.0 * i / args.views):03d}.png" for i in range(args.views)]
    try:
        if not args.overwrite and state_path.is_file() and all(path.exists() for path in expected):
            cached = json.loads(state_path.read_text(encoding="utf-8"))
            if cached.get("status") == "ok" and cached.get("render_count") == args.views:
                for path in expected:
                    validate_cached_render(path, args.resolution)
                cached["cache_hit"] = True
                return cached
        command = [
            args.blender, "-b", "--python", str(HERE / "blender_render_one.py"), "--",
            "--input", row["canonical_glb"], "--output-dir", str(output_dir),
            "--product-id", product_id, "--views", str(args.views),
            "--resolution", str(args.resolution),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-2000:]
            raise RuntimeError(f"Blender render failed ({completed.returncode}): {detail}")
        checks = [validate_render(path, args.resolution) for path in expected]
        result = {
            "product_id": product_id, "status": "ok", "render_count": len(checks),
            "renders": checks, "min_foreground_fraction": min(x["foreground_fraction"] for x in checks),
            "max_foreground_fraction": max(x["foreground_fraction"] for x in checks),
            "cache_hit": False,
        }
    except Exception as error:
        result = {"product_id": product_id, "status": "error", "error": f"{type(error).__name__}: {error}"[:1000]}
    result["checked_at"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--blender", default="/snap/bin/blender")
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--geometry-manifest", type=Path)
    parser.add_argument("--output-tag", default="", help="Write separate QA manifests without replacing the canonical manifest")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent isolated Blender processes")
    args = parser.parse_args()
    root = snapshot_root(args.version, args.root)
    geometry_manifest = args.geometry_manifest or root / "manifests" / "geometry_rows.jsonl"
    geometry_rows = read_jsonl(geometry_manifest)
    # Render review-required geometry too. The training builder still requires
    # geometry status "ok", so producing QA views cannot accidentally promote
    # a quarantined or TRELLIS-generated model.
    targets = [
        row for row in geometry_rows
        if row.get("status") in {"ok", "quarantined", "needs_visual_qa"}
    ]
    if args.only_id:
        wanted = set(args.only_id)
        targets = [row for row in targets if row["product_id"] in wanted]
    if args.limit:
        targets = targets[:args.limit]
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_rows = [(row, executor.submit(render_one, row, root, args)) for row in targets]
        for index, (row, future) in enumerate(future_rows, 1):
            result = future.result()
            results.append(result)
            print(
                f"renders {index}/{len(targets)} id={row['product_id']} "
                f"status={result['status']} cache={result.get('cache_hit', False)}",
                flush=True,
            )
    all_states = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "state" / "renders").glob("*.json"))]
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    if args.output_tag:
        target_ids = {row["product_id"] for row in targets}
        manifest_states = [row for row in all_states if row.get("product_id") in target_ids]
    else:
        manifest_states = all_states
    write_jsonl(root / "manifests" / f"render_rows{suffix}.jsonl", manifest_states)
    report = {
        "schema_version": "ikea-multiview-render/2.0",
        "snapshot_version": args.version,
        "views_per_product": args.views,
        "resolution": args.resolution,
        "geometry_manifest": str(geometry_manifest),
        "state_rows": len(manifest_states),
        "ok_rows": sum(x["status"] == "ok" for x in manifest_states),
        "error_rows": sum(x["status"] == "error" for x in manifest_states),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "manifests" / f"rendering{suffix}.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["error_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
