#!/usr/bin/env python3
"""Blender-side deterministic multi-view renderer for one canonical GLB."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--elevation-deg", type=float, default=20.0)
    return parser.parse_args(values)


def bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    low = Vector((math.inf, math.inf, math.inf))
    high = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            for axis in range(3):
                low[axis] = min(low[axis], point[axis])
                high[axis] = max(high[axis], point[axis])
    return low, high


def point_camera(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    # Preserve an independent object mask. RGB-threshold blank detection is
    # unreliable for white furniture and color-managed near-white backgrounds.
    scene.render.film_transparent = True
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.look = "AgX - Medium High Contrast"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = 0.8

    bpy.ops.import_scene.gltf(filepath=args.input)
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("canonical GLB contains no mesh")
    low, high = bounds(meshes)
    center = (low + high) * 0.5
    size = high - low
    radius = max(0.05, 0.5 * size.length)

    sun_data = bpy.data.lights.new("KeySun", "SUN")
    sun_data.energy = 2.5
    sun = bpy.data.objects.new("KeySun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(40), 0, math.radians(25))
    area_data = bpy.data.lights.new("FillArea", "AREA")
    area_data.energy = 700.0
    area_data.shape = "DISK"
    area_data.size = max(1.0, radius * 3.0)
    area = bpy.data.objects.new("FillArea", area_data)
    scene.collection.objects.link(area)
    area.location = center + Vector((-radius * 2.0, -radius * 2.0, radius * 3.0))

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.lens_unit = "FOV"
    camera_data.angle = math.radians(48.0)
    camera_data.clip_start = max(0.001, radius / 1000.0)
    camera_data.clip_end = max(100.0, radius * 100.0)
    distance = radius / math.sin(camera_data.angle / 2.0) * 1.18
    elevation = math.radians(args.elevation_deg)
    target = center + Vector((0.0, 0.0, size.z * 0.03))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.views):
        azimuth_deg = 360.0 * index / args.views
        azimuth = math.radians(azimuth_deg)
        camera.location = target + Vector((
            distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
        ))
        point_camera(camera, target)
        scene.render.filepath = str(output_dir / f"{args.product_id}_r_{round(azimuth_deg):03d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
