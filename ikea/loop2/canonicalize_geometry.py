#!/usr/bin/env python3
"""Prepare metric canonical GLBs and deterministic 8192-point training clouds."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, snapshot_root, write_json, write_jsonl
else:
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, snapshot_root, write_json, write_jsonl


HERE = Path(__file__).resolve().parent


def sample_pointcloud(glb_path: Path, ply_path: Path, count: int, seed: int) -> dict[str, Any]:
    loaded = trimesh.load(glb_path, force="scene", process=False)
    meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("canonical GLB contains no trimesh geometry")
    mesh = trimesh.util.concatenate(meshes)
    if not np.isfinite(mesh.vertices).all() or mesh.area <= 0:
        raise ValueError("canonical mesh is non-finite or has zero surface area")
    np.random.seed(seed)
    points, _ = trimesh.sample.sample_surface(mesh, count)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError(f"invalid sampled points: {points.shape}")
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ply_path.with_suffix(".ply.tmp")
    trimesh.PointCloud(points.astype(np.float32)).export(temporary, file_type="ply")
    temporary.replace(ply_path)
    return {
        "point_count": int(points.shape[0]),
        "bounds_m": np.stack((points.min(axis=0), points.max(axis=0))).round(6).tolist(),
        "coordinate_frame": {"x": "width", "y": "up", "z": "depth_sign_unverified", "unit": "m"},
    }


def canonicalize_one(
    row: dict[str, Any],
    root: Path,
    blender: str,
    source: str,
    input_glb: Path,
    points: int,
    seed: int,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    product_id = row["product_id"]
    output_glb = root / "assets" / "canonical_glb" / f"{product_id}.glb"
    report_path = root / "state" / "geometry_reports" / f"{product_id}.json"
    state_path = root / "state" / "geometry" / f"{product_id}.json"
    dimensions = (row.get("dimension_profile") or {}).get("dimensions_mm") or {}
    command = [
        blender, "-b", "--python", str(HERE / "blender_canonicalize.py"), "--",
        "--input", str(input_glb), "--output", str(output_glb),
        "--report", str(report_path), "--source", source,
    ]
    for name in ("width", "depth", "height"):
        if dimensions.get(name):
            command.extend((f"--{name}-mm", str(dimensions[name])))
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        output_glb.parent.mkdir(parents=True, exist_ok=True)
        completed = None
        attempt_errors = []
        for attempt in range(1, 3):
            report_path.unlink(missing_ok=True)
            completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
            if completed.returncode == 0 and report_path.is_file() and output_glb.is_file():
                break
            attempt_errors.append({
                "attempt": attempt,
                "returncode": completed.returncode,
                "report_exists": report_path.is_file(),
                "output_exists": output_glb.is_file(),
                "stdout_tail": completed.stdout[-500:],
                "stderr_tail": completed.stderr[-500:],
            })
        else:
            raise RuntimeError(f"Blender canonicalization failed after 2 attempts: {attempt_errors}")
        assert completed is not None
        report = json.loads(report_path.read_text(encoding="utf-8"))
        errors = report.get("relative_dimension_error")
        if source == "trellis":
            # TRELLIS is explicitly rescaled to catalog dimensions, so a zero
            # extent error is not independent evidence that the generated
            # shape is correct. Keep it out of training until visual QA.
            dimension_qa = "scale_applied_not_independently_verified"
            status = "needs_visual_qa"
        else:
            dimension_qa = "unavailable" if errors is None else ("pass" if max(errors) <= 0.12 else "fail")
            status = "ok" if dimension_qa != "fail" else "quarantined"
        ply_path = root / "assets" / "pointcloud_8192" / f"{product_id}.ply"
        point_report = sample_pointcloud(output_glb, ply_path, points, seed)
        result = {
            "product_id": product_id,
            "status": status,
            "source": source,
            "input_glb": str(input_glb),
            "canonical_glb": str(output_glb),
            "pointcloud": str(ply_path),
            "dimension_qa": dimension_qa,
            "geometry_report": report,
            "pointcloud_report": point_report,
            "blender_output_tail": completed.stdout[-1000:],
        }
        if fallback_reason:
            result["fallback_reason"] = fallback_reason
    except Exception as error:
        result = {
            "product_id": product_id, "status": "error", "source": source,
            "input_glb": str(input_glb), "error": f"{type(error).__name__}: {error}"[:1000],
        }
    result["checked_at"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--blender", default="/snap/bin/blender")
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-id", action="append", default=[])
    parser.add_argument("--include-trellis", action="store_true")
    parser.add_argument("--source", choices=("all", "official", "trellis"), default="all")
    parser.add_argument("--output-tag", default="", help="Write separate QA manifests without replacing the canonical manifest")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = snapshot_root(args.version, args.root)
    rows = read_jsonl(root / "catalog" / "products.jsonl")
    if args.only_id:
        wanted = set(args.only_id)
        rows = [row for row in rows if row["product_id"] in wanted]
    if args.limit:
        rows = rows[:args.limit]
    def prior_geometry_state(product_id: str) -> dict[str, Any]:
        path = root / "state" / "geometry" / f"{product_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def official_error_fallback_available(row: dict[str, Any]) -> bool:
        product_id = row["product_id"]
        official = root / "assets" / "official_glb" / f"{product_id}.glb"
        trellis = root / "assets" / "trellis_glb" / f"{product_id}.glb"
        prior = prior_geometry_state(product_id)
        return bool(
            official.is_file()
            and trellis.is_file()
            and (
                (prior.get("source") == "official" and prior.get("status") == "error")
                or (
                    prior.get("source") == "trellis"
                    and prior.get("fallback_reason") == "OFFICIAL_GLB_CANONICALIZATION_ERROR"
                )
            )
        )

    if args.source == "official":
        rows = [row for row in rows if (root / "assets" / "official_glb" / f"{row['product_id']}.glb").exists()]
    elif args.source == "trellis":
        rows = [
            row for row in rows
            if (
                (
                    not (root / "assets" / "official_glb" / f"{row['product_id']}.glb").exists()
                    and (root / "assets" / "trellis_glb" / f"{row['product_id']}.glb").exists()
                )
                or official_error_fallback_available(row)
            )
        ]
    results = []
    for index, row in enumerate(rows, 1):
        product_id = row["product_id"]
        state_path = root / "state" / "geometry" / f"{product_id}.json"
        if state_path.exists() and not args.overwrite:
            cached = json.loads(state_path.read_text(encoding="utf-8"))
            output_exists = Path(cached.get("canonical_glb") or "").is_file()
            pointcloud_exists = Path(cached.get("pointcloud") or "").is_file()
            if cached.get("status") in {"ok", "quarantined", "needs_visual_qa"} and output_exists and pointcloud_exists:
                results.append(cached)
                continue
        official = root / "assets" / "official_glb" / f"{product_id}.glb"
        trellis = root / "assets" / "trellis_glb" / f"{product_id}.glb"
        if args.include_trellis and official_error_fallback_available(row):
            result = canonicalize_one(
                row, root, args.blender, "trellis", trellis, args.points, args.seed,
                fallback_reason="OFFICIAL_GLB_CANONICALIZATION_ERROR",
            )
        elif official.exists():
            result = canonicalize_one(row, root, args.blender, "official", official, args.points, args.seed)
        elif args.include_trellis and trellis.exists():
            result = canonicalize_one(row, root, args.blender, "trellis", trellis, args.points, args.seed)
        else:
            result = {"product_id": product_id, "status": "pending_3d", "source": None}
        results.append(result)
        if index % 10 == 0 or index == len(rows):
            print(f"geometry {index}/{len(rows)} ok={sum(x['status']=='ok' for x in results)} "
                  f"quarantined={sum(x['status']=='quarantined' for x in results)} "
                  f"visual_qa={sum(x['status']=='needs_visual_qa' for x in results)} "
                  f"error={sum(x['status']=='error' for x in results)}")
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    write_jsonl(root / "manifests" / f"geometry_rows{suffix}.jsonl", results)
    report = {
        "schema_version": "ikea-geometry-preparation/2.0",
        "snapshot_version": args.version,
        "processed_rows": len(results),
        "ok_rows": sum(x["status"] == "ok" for x in results),
        "quarantined_rows": sum(x["status"] == "quarantined" for x in results),
        "needs_visual_qa_rows": sum(x["status"] == "needs_visual_qa" for x in results),
        "error_rows": sum(x["status"] == "error" for x in results),
        "pending_3d_rows": sum(x["status"] == "pending_3d" for x in results),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "manifests" / f"geometry_preparation{suffix}.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["error_rows"] == 0 and report["quarantined_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
