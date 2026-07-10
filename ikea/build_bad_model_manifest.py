"""
Consolidate every quality signal for the 733 existing 3D models into a single
manifest JSON. Tomorrow's meeting only needs to edit the `action` field.

Inputs:
  - MongoDB furniture_db.ikea_product  (v1, source of truth for glb items)
  - /tmp/ikea_delisted_ids.txt         (158 delisted _id, one per line)
  - /tmp/ikea_36_mislabel.csv          (36 mislabel rows with url_guess)
  - /mnt/P300/data/ikea_data/ply/*.ply (point counts for sparse detection)

Output:
  /home/kyzen/ULIP_RAG/ikea/docs/bad_3d_models_manifest.json

Manifest entry schema:
{
  "_id": "...",
  "asin": "...",
  "name": "...",
  "url": "...",
  "current_category": "sofa",
  "clip_confidence": 0.78,
  "glb_path": "/mnt/.../glb/<_id>.glb",
  "ply_path": "/mnt/.../ply/<_id>.ply",
  "json_path": "/mnt/.../json/<_id>.json",
  "image_v1": "/mnt/.../images/<_id>.jpg",
  "image_v3": "/mnt/.../images_v3/<asin>.jpg",    # may not exist yet
  "point_count": 123456,
  "reasons": ["duplicate_asin", "mislabeled_category"],
  "action": "delete",                              # delete | regen | fix_label | keep
  "suggested_category": "Nightstand",              # only if mislabeled
  "dedup_keep": false                               # for duplicate_asin groups only
}

Thresholds (see docs/3d_model_quality_audit.md):
  SPARSE_PC_THRESHOLD = 50_000  points
  NOT_FURNITURE_KEYWORDS = ["cover", "underframe", "tabletop", "table-top",
                            "deco-strip", "strip"]

Usage:
  python build_bad_model_manifest.py                    # build manifest
  python build_bad_model_manifest.py --print-summary    # also print breakdown
"""

import argparse
import csv
import json
import os
import struct
from collections import defaultdict
from pathlib import Path

from pymongo import MongoClient


from mongo_conn import get_mongo_uri  # 2026-07-10 統一連線(mongo 已啟用 --auth,舊無認證 URI 已失效)
MONGO_URI = get_mongo_uri()
DB_NAME = "furniture_db"
V1_COL = "ikea_product"

IKEA_DATA = Path("/mnt/P300/data/ikea_data")
GLB_DIR = IKEA_DATA / "glb"
PLY_DIR = IKEA_DATA / "ply"
JSON_DIR = IKEA_DATA / "json"
IMG_V1 = IKEA_DATA / "images"
IMG_V3 = Path("/mnt/P300/data/ikea_data_V3/images")

DELISTED_TXT = Path("/tmp/ikea_delisted_ids.txt")
MISLABEL_CSV = Path("/tmp/ikea_36_mislabel.csv")
OUT_MANIFEST = Path("/home/kyzen/ULIP_RAG/ikea/docs/bad_3d_models_manifest.json")

SPARSE_PC_THRESHOLD = 50_000
NOT_FURNITURE_KEYWORDS = ["cover", "underframe", "tabletop", "table-top",
                          "deco-strip", "strip", "drawer-front", "leg-"]


def read_ply_vertex_count(path: Path) -> int:
    """Parse the PLY header to get vertex count. Returns 0 if unreadable."""
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


def load_delisted_ids() -> set[str]:
    if not DELISTED_TXT.exists():
        print(f"[warn] {DELISTED_TXT} missing — delisted check disabled")
        return set()
    return {ln.strip() for ln in DELISTED_TXT.read_text().splitlines() if ln.strip()}


def load_mislabel_map() -> dict[str, dict]:
    """Returns {_id: {suggested_category, url_guess, ikea_url, conf}}."""
    if not MISLABEL_CSV.exists():
        print(f"[warn] {MISLABEL_CSV} missing — mislabel check disabled")
        return {}
    m = {}
    with open(MISLABEL_CSV, newline="") as f:
        for row in csv.DictReader(f):
            m[row["id"]] = {
                "suggested_category": row["url_guess"],
                "product_slug": row["product_slug"],
                "ikea_url": row["ikea_url"],
                "conf": float(row["conf"]),
            }
    return m


def is_not_furniture(slug: str) -> bool:
    s = (slug or "").lower()
    return any(kw in s for kw in NOT_FURNITURE_KEYWORDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-summary", action="store_true")
    args = ap.parse_args()

    col = MongoClient(MONGO_URI)[DB_NAME][V1_COL]
    docs = list(col.find(
        {"converted_3d": "y"},
        {"_id": 1, "asin": 1, "name": 1, "url": 1, "main_type": 1,
         "clip_confidence": 1, "images": 1, "image": 1}
    ))
    print(f"[load] {len(docs)} v1 glb items")

    delisted = load_delisted_ids()
    mislabel = load_mislabel_map()
    print(f"[load] delisted={len(delisted)}  mislabel={len(mislabel)}")

    # Group by asin for duplicate detection
    asin_groups: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        a = d.get("asin")
        if a:
            asin_groups[a].append(d)

    # Pick best-conf doc per duplicate group
    dedup_keepers: set[str] = set()
    dup_losers: set[str] = set()
    for a, grp in asin_groups.items():
        if len(grp) <= 1:
            continue
        grp_sorted = sorted(grp, key=lambda x: x.get("clip_confidence") or 0.0, reverse=True)
        dedup_keepers.add(str(grp_sorted[0]["_id"]))
        for loser in grp_sorted[1:]:
            dup_losers.add(str(loser["_id"]))

    manifest = []
    for d in docs:
        _id = str(d["_id"])
        asin = d.get("asin")
        reasons: list[str] = []

        ply_path = PLY_DIR / f"{_id}.ply"
        point_count = read_ply_vertex_count(ply_path) if ply_path.exists() else 0

        # ----- reason detection (order = display priority) -----
        if _id in delisted:
            reasons.append("delisted")
        if _id in mislabel and is_not_furniture(mislabel[_id]["product_slug"]):
            reasons.append("not_furniture")
        if _id in dup_losers:
            reasons.append("duplicate_asin_loser")
        if _id in dedup_keepers:
            reasons.append("duplicate_asin_keeper")
        if 0 < point_count < SPARSE_PC_THRESHOLD:
            reasons.append("sparse_pointcloud")
        if _id in mislabel and "not_furniture" not in reasons:
            reasons.append("mislabeled_category")

        # ----- action decision (priority: delete > fix_label > regen > keep) -----
        if "not_furniture" in reasons or "delisted" in reasons or "duplicate_asin_loser" in reasons:
            action = "delete"
        elif "sparse_pointcloud" in reasons:
            action = "regen"
        elif "mislabeled_category" in reasons:
            action = "fix_label"
        else:
            action = "keep"

        entry = {
            "_id": _id,
            "asin": asin,
            "name": d.get("name"),
            "url": d.get("url"),
            "current_category": d.get("main_type"),
            "clip_confidence": d.get("clip_confidence"),
            "glb_path": str(GLB_DIR / f"{_id}.glb"),
            "ply_path": str(ply_path),
            "json_path": str(JSON_DIR / f"{_id}.json"),
            "image_v1": str(IMG_V1 / f"{_id}.jpg"),
            "image_v3": str(IMG_V3 / f"{asin}.jpg") if asin else None,
            "point_count": point_count,
            "reasons": reasons or ["ok"],
            "action": action,
        }
        if _id in mislabel:
            entry["suggested_category"] = mislabel[_id]["suggested_category"]
            entry["product_slug"] = mislabel[_id]["product_slug"]
        if _id in dedup_keepers:
            entry["dedup_keep"] = True
        if _id in dup_losers:
            entry["dedup_keep"] = False

        manifest.append(entry)

    # Write manifest
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[write] {OUT_MANIFEST}  ({len(manifest)} entries)")

    # Summary
    action_count = defaultdict(int)
    reason_count = defaultdict(int)
    for e in manifest:
        action_count[e["action"]] += 1
        for r in e["reasons"]:
            reason_count[r] += 1

    print("\n=== Action breakdown ===")
    for a, c in sorted(action_count.items(), key=lambda x: -x[1]):
        print(f"  {a:<12} {c:>4}")

    print("\n=== Reason breakdown ===")
    for r, c in sorted(reason_count.items(), key=lambda x: -x[1]):
        print(f"  {r:<28} {c:>4}")

    if args.print_summary:
        print("\n=== First 5 delete entries ===")
        for e in [x for x in manifest if x["action"] == "delete"][:5]:
            print(f"  {e['_id']}  reasons={e['reasons']}  url={e['url']}")
        print("\n=== First 5 fix_label entries ===")
        for e in [x for x in manifest if x["action"] == "fix_label"][:5]:
            print(f"  {e['_id']}  {e['current_category']} -> {e.get('suggested_category')}")


if __name__ == "__main__":
    main()
