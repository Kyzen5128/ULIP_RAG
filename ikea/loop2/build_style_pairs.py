#!/usr/bin/env python3
"""Build leakage-safe weak room/product style pairs from official IKEA images."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.build_training_dataset import family_id
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, split_name_for_group, write_json, write_jsonl
else:
    from .build_training_dataset import family_id
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, split_name_for_group, write_json, write_jsonl


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            if a > b:
                a, b = b, a
            self.parent[b] = a


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    split_name_for_group("validation", *ratios)
    root = snapshot_root(args.version, args.root)
    catalog_path = root / "catalog" / "products.jsonl"
    asset_path = root / "manifests" / "asset_rows.jsonl"
    catalog = {row["product_id"]: row for row in read_jsonl(catalog_path)}
    assets = {row["product_id"]: row for row in read_jsonl(asset_path)}

    raw_pairs: list[dict[str, Any]] = []
    union = UnionFind()
    seen: set[tuple[str, str]] = set()
    for product_id, row in catalog.items():
        asset = assets.get(product_id) or {}
        primary = asset.get("image") or {}
        if primary.get("status") != "ok":
            continue
        product_family = family_id(row)
        for context in asset.get("supplementary_images") or []:
            if context.get("status") != "ok" or context.get("asset_type") not in {"CONTEXT_PRODUCT_IMAGE", "INSPIRATIONAL_IMAGE"}:
                continue
            scene_hash = str(context.get("sha256") or "")
            if not scene_hash or (scene_hash, product_id) in seen:
                continue
            seen.add((scene_hash, product_id))
            scene_node = f"scene:{scene_hash}"
            family_node = f"family:{product_family}"
            union.union(scene_node, family_node)
            raw_pairs.append({
                "schema_version": "ikea-style-pair/2.0",
                "scene_id": scene_hash,
                "scene_image": context["path"],
                "scene_image_sha256": scene_hash,
                "scene_alt_text": context.get("alt_text") or "",
                "product_id": product_id,
                "product_family_id": product_family,
                "product_image": primary["path"],
                "product_image_sha256": primary["sha256"],
                "category": row["canonical_category"],
                "label": "weak_positive",
                "label_strength": "weak",
                "evidence": "official_ikea_context_image_linked_to_product",
            })

    component_members: dict[str, set[str]] = defaultdict(set)
    for node in union.parent:
        component_members[union.find(node)].add(node)
    component_keys = {
        root_node: "|".join(sorted(members)) for root_node, members in component_members.items()
    }
    scene_products: dict[str, set[str]] = defaultdict(set)
    for pair in raw_pairs:
        scene_products[pair["scene_id"]].add(pair["product_id"])
    pairs = []
    for pair in raw_pairs:
        component = union.find(f"scene:{pair['scene_id']}")
        group_key = component_keys[component]
        pair["split_group_sha256"] = hashlib.sha256(group_key.encode("utf-8")).hexdigest()
        pair["split"] = split_name_for_group(group_key, *ratios)
        pair["scene_positive_product_ids"] = sorted(scene_products[pair["scene_id"]])
        pairs.append(pair)
    pairs.sort(key=lambda row: (row["split"], row["scene_id"], row["product_id"]))
    output = root / "style" / "weak_pairs.jsonl"
    write_jsonl(output, pairs)
    report = {
        "schema_version": "ikea-style-pair-build/2.0",
        "snapshot_version": args.version,
        "pair_rows": len(pairs),
        "unique_scenes": len(scene_products),
        "unique_products": len({pair["product_id"] for pair in pairs}),
        "connected_components": len(component_members),
        "multi_positive_scenes": sum(len(products) > 1 for products in scene_products.values()),
        "split_rows": dict(sorted(Counter(pair["split"] for pair in pairs).items())),
        "category_rows": dict(sorted(Counter(pair["category"] for pair in pairs).items())),
        "split_unique_scenes": {
            split: len({pair["scene_id"] for pair in pairs if pair["split"] == split})
            for split in ("train", "val", "test")
        },
        "catalog_sha256": sha256_file(catalog_path),
        "asset_manifest_sha256": sha256_file(asset_path),
        "pairs_sha256": sha256_file(output),
        "label_policy": {
            "weak_positive": "Product is linked to an official IKEA context/inspirational image; this is not a human style-preference judgment.",
            "negative_sampling": "No fixed negatives are asserted; training must treat all products linked to the same scene as positives.",
            "split": "Connected components of scene hashes and product families are indivisible across train/val/test.",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "manifests" / "style_pairs.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
