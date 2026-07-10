"""
Regenerate 3D models for manifest entries with action="regen".

Strategy:
  1. Prefer the v3 image (images_v3/<asin>.jpg), fall back to the original v1
     image if v3 is absent.
  2. Run TRELLIS image-to-3D with configurable seed and texture_size.
  3. Write to a SEPARATE *_regen directory tree so the originals survive:
       /mnt/P300/data/ikea_data/glb_regen/<_id>.glb
       /mnt/P300/data/ikea_data/ply_regen/<_id>.ply
       /mnt/P300/data/ikea_data/json_regen/<_id>.json
  4. After visual QA you (or a follow-up script) can swap directories.
  5. MongoDB gets converted_3d_regen="y" plus regen metadata; the original
     converted_3d field is untouched.

Usage:
  python regen_3d_models.py                                     # dry-run
  python regen_3d_models.py --commit                            # run TRELLIS
  python regen_3d_models.py --commit --seed 7 --texture 2048    # override params
  python regen_3d_models.py --commit --limit 2                  # smoke test
  python regen_3d_models.py --commit --only-ids 67ab...,67cd... # specific items
  python regen_3d_models.py --commit --multi-seed 1,7,42        # try 3 seeds per item,
                                                                #   keep densest PLY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "furniture_db"
V1_COL = "ikea_product"

MANIFEST_PATH = Path("/home/kyzen/ULIP_RAG/ikea/docs/bad_3d_models_manifest.json")
IKEA_DATA = Path("/mnt/P300/data/ikea_data")
IMG_V1 = IKEA_DATA / "images"
IKEA_DATA_V3 = Path("/mnt/P300/data/ikea_data_V3")
IMG_V3 = IKEA_DATA_V3 / "images"

GLB_REGEN = IKEA_DATA_V3 / "glb"
PLY_REGEN = IKEA_DATA_V3 / "ply"
JSON_REGEN = IKEA_DATA_V3 / "json"

LOG_DIR = Path("/home/kyzen/ULIP_RAG/ikea/logs")


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        sys.exit(f"manifest missing: {MANIFEST_PATH}\nrun build_bad_model_manifest.py first.")
    return json.loads(MANIFEST_PATH.read_text())


def pick_image(entry: dict) -> Optional[Path]:
    v3 = Path(entry["image_v3"]) if entry.get("image_v3") else None
    v1 = Path(entry["image_v1"]) if entry.get("image_v1") else None
    if v3 and v3.exists() and v3.stat().st_size > 0:
        return v3
    if v1 and v1.exists() and v1.stat().st_size > 0:
        return v1
    return None


def read_ply_vertex_count(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
        if not head.startswith(b"ply"):
            return 0
        for line in head.split(b"\n"):
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
    except Exception:
        pass
    return 0


def init_pipeline():
    """Lazy import so dry-run doesn't require torch/TRELLIS."""
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    import torch
    torch.backends.cudnn.enabled = False
    from trellis.pipelines import TrellisImageTo3DPipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    return pipeline


def run_trellis(pipeline, image_path: Path, seed: int, texture_size: int):
    from PIL import Image
    from trellis.utils import postprocessing_utils
    image = Image.open(image_path).convert("RGB")
    outputs = pipeline.run(image, seed=seed)
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=0.95,
        texture_size=texture_size,
    )
    return outputs, glb


def regenerate_one(entry: dict, pipeline, seeds: list[int], texture_size: int, log) -> dict:
    """Runs TRELLIS (possibly multiple seeds) and keeps the densest PLY.

    Returns: {status, best_seed, point_count, image_used}
    """
    _id = entry["_id"]
    img = pick_image(entry)
    if not img:
        log(f"  [error] no image available for {_id}")
        return {"status": "no_image"}

    log(f"  image: {img}")

    best = None  # (point_count, seed, outputs, glb)
    for seed in seeds:
        t0 = time.time()
        try:
            outputs, glb = run_trellis(pipeline, img, seed=seed, texture_size=texture_size)
        except Exception as e:
            log(f"  [error] trellis failed seed={seed}: {e}")
            continue

        # Temp-save the PLY, peek vertex count, keep the best
        tmp_ply = PLY_REGEN / f"{_id}.seed{seed}.ply"
        outputs["gaussian"][0].save_ply(str(tmp_ply))
        pc = read_ply_vertex_count(tmp_ply)
        dt = time.time() - t0
        log(f"  seed={seed:>4}  points={pc:>8}  {dt:.1f}s  -> {tmp_ply.name}")

        if best is None or pc > best[0]:
            # delete previous best to save disk
            if best is not None:
                prev = PLY_REGEN / f"{_id}.seed{best[1]}.ply"
                if prev.exists():
                    prev.unlink()
            best = (pc, seed, outputs, glb)
        else:
            tmp_ply.unlink(missing_ok=True)

    if best is None:
        return {"status": "all_failed"}

    pc, seed, outputs, glb = best

    # Rename winning PLY to canonical name
    winner = PLY_REGEN / f"{_id}.seed{seed}.ply"
    final_ply = PLY_REGEN / f"{_id}.ply"
    if final_ply.exists():
        final_ply.unlink()
    winner.rename(final_ply)

    # Write GLB
    glb_path = GLB_REGEN / f"{_id}.glb"
    glb.export(str(glb_path))

    # Write JSON meta
    json_path = JSON_REGEN / f"{_id}.json"
    meta = {
        "image": str(img),
        "pointcloud": str(final_ply),
        "mesh": str(glb_path),
        "text": "",
        "category": entry.get("suggested_category") or entry.get("current_category") or "Unknown",
        "meta": {
            "source_id": _id,
            "asin": entry.get("asin"),
            "url": entry.get("url"),
            "regen_seed": seed,
            "regen_point_count": pc,
            "regen_timestamp": datetime.now().isoformat(),
        },
    }
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    return {"status": "ok", "best_seed": seed, "point_count": pc, "image_used": str(img)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="actually run TRELLIS (default: dry-run)")
    ap.add_argument("--seed", type=int, default=1, help="single seed (ignored if --multi-seed)")
    ap.add_argument("--multi-seed", type=str, default=None,
                    help="comma-separated seeds; keeps densest PLY (e.g. 1,7,42)")
    ap.add_argument("--texture", type=int, default=1024, help="texture size (default 1024)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of items")
    ap.add_argument("--only-ids", type=str, default=None,
                    help="comma-separated _ids to regen (overrides manifest filter)")
    args = ap.parse_args()

    manifest = load_manifest()
    print(f"[load] manifest entries: {len(manifest)}")

    if args.only_ids:
        want = {x.strip() for x in args.only_ids.split(",") if x.strip()}
        targets = [e for e in manifest if e["_id"] in want]
    else:
        targets = [e for e in manifest if e["action"] == "regen"]

    if args.limit:
        targets = targets[: args.limit]
    print(f"[plan] regen targets: {len(targets)}")

    seeds = [int(s) for s in args.multi_seed.split(",")] if args.multi_seed else [args.seed]
    print(f"[plan] seeds: {seeds}  texture_size: {args.texture}")
    print(f"[plan] output: {GLB_REGEN}, {PLY_REGEN}, {JSON_REGEN}")

    # Dry-run: just summarise
    if not args.commit:
        missing_img = 0
        for e in targets:
            if pick_image(e) is None:
                missing_img += 1
        print(f"[dry-run] items missing any image: {missing_img}")
        print(f"[dry-run] estimated runtime: ~{len(targets)*len(seeds)*45/60:.1f} min"
              f" (45s/seed, single GPU)")
        print("\nRe-run with --commit to regenerate.")
        return

    # Commit path — set up dirs, log, Mongo, pipeline
    for d in (GLB_REGEN, PLY_REGEN, JSON_REGEN, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"regen_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_lines: list[str] = []

    def log(msg: str):
        log_lines.append(msg)
        print(msg)

    log(f"\n=== regen_3d_models.py  at {datetime.now()}  targets={len(targets)}  seeds={seeds} ===\n")

    from pymongo import MongoClient
    from bson import ObjectId
    col = MongoClient(MONGO_URI)[DB_NAME][V1_COL]

    pipeline = init_pipeline()
    log("[trellis] pipeline ready")

    stats = {"ok": 0, "no_image": 0, "all_failed": 0}
    for i, e in enumerate(targets, 1):
        log(f"\n[{i}/{len(targets)}] {e['_id']}  reasons={e['reasons']}")
        res = regenerate_one(e, pipeline, seeds, args.texture, log)
        stats[res["status"]] = stats.get(res["status"], 0) + 1

        if res["status"] == "ok":
            col.update_one(
                {"_id": ObjectId(e["_id"])},
                {"$set": {
                    "converted_3d_regen": "y",
                    "regen_seed": res["best_seed"],
                    "regen_point_count": res["point_count"],
                    "regen_image": res["image_used"],
                    "regen_timestamp": datetime.now(),
                }},
            )

        # flush log periodically
        if i % 5 == 0:
            log_path.write_text("\n".join(log_lines))

    log("\n=== summary ===")
    for k, v in stats.items():
        log(f"  {k:<12} {v:>4}")
    log_path.write_text("\n".join(log_lines))
    print(f"\n[log] {log_path}")


if __name__ == "__main__":
    main()
