#!/usr/bin/env python3
"""Generate fallback GLBs only for products without a validated official GLB."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, snapshot_root, write_json, write_jsonl
else:
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, snapshot_root, write_json, write_jsonl


def load_pipeline():
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    import torch
    torch.backends.cudnn.enabled = False
    from trellis.pipelines import TrellisImageTo3DPipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    return pipeline


def generate_one(pipeline, image_path: Path, output_path: Path, seed: int, texture_size: int) -> dict[str, Any]:
    from trellis.utils import postprocessing_utils
    started = time.monotonic()
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    outputs = pipeline.run(image, seed=seed)
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0], outputs["mesh"][0], simplify=0.95,
        texture_size=texture_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".glb.tmp")
    glb.export(str(temporary), file_type="glb")
    data = temporary.read_bytes()
    if len(data) < 1024 or data[:4] != b"glTF":
        temporary.unlink(missing_ok=True)
        raise ValueError("TRELLIS output is not a valid binary GLB")
    temporary.replace(output_path)
    del outputs, glb
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return {"bytes": len(data), "elapsed_seconds": round(time.monotonic() - started, 3)}


def publish_state_manifest(root: Path, version: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    all_states = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "state" / "trellis").glob("*.json"))
    ]
    write_jsonl(root / "manifests" / "trellis_rows.jsonl", all_states)
    report = {
        "schema_version": "ikea-trellis-fallback/2.0",
        "snapshot_version": version,
        "target_rows_this_run": len(results),
        "ok_rows_this_run": sum(x["status"] == "ok" for x in results),
        "error_rows_this_run": sum(x["status"] == "error" for x in results),
        "total_state_rows": len(all_states),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "manifests" / "trellis_generation.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-id", action="append", default=[])
    parser.add_argument("--sample-per-category", type=int, default=0)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--include-official-errors",
        action="store_true",
        help=(
            "Allow an explicit fallback only when an existing official GLB has "
            "already failed canonicalization with status=error. The official file "
            "is preserved and remains the provenance source of the failure."
        ),
    )
    args = parser.parse_args()
    root = snapshot_root(args.version, args.root)
    rows = read_jsonl(root / "catalog" / "products.jsonl")
    targets = []
    for row in rows:
        product_id = row["product_id"]
        official_path = root / "assets" / "official_glb" / f"{product_id}.glb"
        geometry_state_path = root / "state" / "geometry" / f"{product_id}.json"
        geometry_state = {}
        if geometry_state_path.is_file():
            geometry_state = json.loads(geometry_state_path.read_text(encoding="utf-8"))
        official_failed = (
            official_path.exists()
            and geometry_state.get("source") == "official"
            and geometry_state.get("status") == "error"
        )
        if official_path.exists() and not (args.include_official_errors and official_failed):
            continue
        image_path = root / "assets" / "images" / f"{product_id}.jpg"
        state_path = root / "state" / "trellis" / f"{product_id}.json"
        output_path = root / "assets" / "trellis_glb" / f"{product_id}.glb"
        if output_path.exists():
            continue
        if state_path.exists() and not args.retry_errors:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") == "error":
                continue
        targets.append((row, image_path, output_path, state_path, official_failed, official_path))
    if args.only_id:
        wanted = set(args.only_id)
        targets = [target for target in targets if target[0]["product_id"] in wanted]
    if args.sample_per_category:
        grouped: dict[str, list] = {}
        for target in targets:
            grouped.setdefault(target[0]["canonical_category"], []).append(target)
        targets = []
        for category in sorted(grouped):
            ordered = sorted(
                grouped[category],
                key=lambda target: hashlib.sha256(target[0]["product_id"].encode("utf-8")).hexdigest(),
            )
            targets.extend(ordered[:args.sample_per_category])
    if args.limit:
        targets = targets[:args.limit]
    if not targets:
        report = publish_state_manifest(root, args.version, [])
        print("No TRELLIS fallback targets; refreshed aggregate manifest.")
        print(json.dumps(report, indent=2))
        return 0
    missing_images = [str(target[1]) for target in targets if not target[1].exists()]
    if missing_images:
        raise SystemExit(f"{len(missing_images)} target images are missing; run download_assets.py first")

    pipeline = load_pipeline()
    results = []
    for index, (row, image_path, output_path, state_path, official_failed, official_path) in enumerate(targets, 1):
        try:
            metrics = generate_one(pipeline, image_path, output_path, args.seed, args.texture_size)
            result = {
                "product_id": row["product_id"], "status": "ok",
                "image": str(image_path), "output_glb": str(output_path),
                "seed": args.seed, "texture_size": args.texture_size, **metrics,
            }
            if official_failed:
                result["fallback_reason"] = "OFFICIAL_GLB_CANONICALIZATION_ERROR"
                result["failed_official_glb"] = str(official_path)
        except Exception as error:
            result = {
                "product_id": row["product_id"], "status": "error",
                "image": str(image_path), "error": f"{type(error).__name__}: {error}"[:1000],
            }
        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, result)
        results.append(result)
        print(f"trellis {index}/{len(targets)} id={row['product_id']} status={result['status']}")

    report = publish_state_manifest(root, args.version, results)
    print(json.dumps(report, indent=2))
    return 0 if report["error_rows_this_run"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
