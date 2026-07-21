#!/usr/bin/env python3
"""Blender-side GLB decoding, helper removal and metric canonicalization.

Run through Blender, not the project Python interpreter:
  blender -b --python blender_canonicalize.py -- --input ... --output ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def args_after_separator() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--source", choices=("official", "trellis"), required=True)
    parser.add_argument("--width-mm", type=float)
    parser.add_argument("--depth-mm", type=float)
    parser.add_argument("--height-mm", type=float)
    return parser.parse_args(values)


def world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    low = Vector((math.inf, math.inf, math.inf))
    high = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            for axis in range(3):
                low[axis] = min(low[axis], point[axis])
                high[axis] = max(high[axis], point[axis])
    return low, high


def join_meshes(meshes: list[bpy.types.Object]) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.hide_set(False)
        obj.hide_render = False
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.convert(target="MESH")
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.name = "product"
    return obj


def permutation_parity(permutation: tuple[int, int, int]) -> int:
    inversions = sum(permutation[i] > permutation[j] for i in range(3) for j in range(i + 1, 3))
    return -1 if inversions % 2 else 1


def best_axis_mapping(source_extents: list[float], target_extents: list[float] | None, source: str) -> tuple[int, int, int]:
    if not target_extents:
        return (0, 1, 2)
    best = None
    # Blender's standards-compliant glTF importer has already converted Y-up
    # on disk to Z-up in the workspace. Never exchange Z with a footprint axis:
    # generated proportions are noisy enough that an unconstrained extent fit
    # can numerically prefer laying a chair or bed on its side. Front remains
    # unknown, so swapping X/Y is the only admissible ambiguity.
    for permutation in ((0, 1, 2), (1, 0, 2)):
        mapped = [source_extents[index] for index in permutation]
        error = sum(abs(math.log(max(value, 1e-9) / target)) for value, target in zip(mapped, target_extents))
        candidate = (error, permutation)
        if best is None or candidate < best:
            best = candidate
    return best[1]


def transform_vertices(obj: bpy.types.Object, permutation: tuple[int, int, int], target: list[float] | None, scale_to_target: bool) -> None:
    parity = permutation_parity(permutation)
    for vertex in obj.data.vertices:
        old = tuple(vertex.co)
        mapped = [old[permutation[0]], old[permutation[1]], old[permutation[2]]]
        if parity < 0:
            mapped[1] *= -1.0
        vertex.co = mapped
    # Vertex edits do not automatically refresh evaluated object bounds.
    # Scaling against stale pre-permutation bounds can lay the product on its
    # side and produces false dimension errors whenever permutation != identity.
    obj.data.update()
    bpy.context.view_layer.update()
    low, high = world_bounds([obj])
    extents = high - low
    if scale_to_target and target:
        obj.scale = tuple(target[i] / max(float(extents[i]), 1e-9) for i in range(3))
        bpy.context.view_layer.update()
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        low, high = world_bounds([obj])
    center = (low + high) * 0.5
    obj.location += Vector((-center.x, -center.y, -low.z))
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)


def main() -> None:
    args = args_after_separator()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.input)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("GLB contains no mesh")
    input_low, input_high = world_bounds(meshes)
    # Do not remove nodes by name/shape heuristics. A previously observed 2 m
    # Cube was proven to be Blender's default startup object, not part of the
    # DIMMA GLB. This script starts from an empty scene, so every imported mesh
    # remains product evidence unless a future source-backed rule says otherwise.
    removed: list[str] = []
    product_low, product_high = world_bounds(meshes)
    product_extents = [float(v) for v in product_high - product_low]
    target = None
    if all(value is not None and value > 0 for value in (args.width_mm, args.depth_mm, args.height_mm)):
        target = [args.width_mm / 1000.0, args.depth_mm / 1000.0, args.height_mm / 1000.0]
    permutation = best_axis_mapping(product_extents, target, args.source)
    obj = join_meshes(meshes)
    transform_vertices(obj, permutation, target, scale_to_target=args.source == "trellis")
    final_low, final_high = world_bounds([obj])
    final_extents = [float(v) for v in final_high - final_low]
    relative_error = None
    if target:
        relative_error = [abs(value - expected) / expected for value, expected in zip(final_extents, target)]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=str(output), export_format="GLB", use_selection=True,
        # glTF is Y-up on disk. Let Blender perform its normal Z-up -> Y-up
        # conversion so Blender and standards-compliant importers restore the
        # canonical product as Z-up instead of laying it on its side.
        export_apply=True, export_yup=True,
    )
    report = {
        "schema_version": "ikea-canonical-geometry/2.0",
        "source": args.source,
        "input": args.input,
        "output": str(output),
        "input_scene_extents_m": [float(v) for v in input_high - input_low],
        "product_input_extents_m": product_extents,
        "axis_permutation": list(permutation),
        "target_extents_m": target,
        "final_extents_m": final_extents,
        "relative_dimension_error": relative_error,
        "removed_helpers": removed,
        "front_direction": None,
        "orientation_status": "axis_aligned_front_unverified",
        "metric_scale_status": "official_preserved" if args.source == "official" else ("scaled_to_catalog" if target else "normalized_unknown"),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
