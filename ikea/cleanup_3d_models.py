"""
Apply manifest actions to the 733 existing 3D models.

Reads:  /home/kyzen/ULIP_RAG/ikea/docs/bad_3d_models_manifest.json
Edits:
  - filesystem: /mnt/P300/data/ikea_data/{glb,ply,json,rendered_images}/
  - MongoDB:    furniture_db.ikea_product (converted_3d / main_type)

Action semantics:
  delete     -> remove glb/ply/json + rendered_images/<_id>/ ; mark converted_3d="deleted"
                (for delisted items additionally set status="delisted")
  fix_label  -> update main_type in Mongo + category in per-model JSON; no file deletion
  regen      -> NO-OP here (handled by regen_3d_models.py)
  keep       -> NO-OP

Safety:
  * Default is dry-run; pass --commit to actually modify.
  * --action-filter lets you run a single action class at a time.
  * Every delete/update is logged to ./logs/cleanup_YYYYMMDD_HHMMSS.log

Usage:
  python cleanup_3d_models.py                                 # dry-run all actions
  python cleanup_3d_models.py --action-filter fix_label       # dry-run only fix_label
  python cleanup_3d_models.py --action-filter delete --commit # really delete
  python cleanup_3d_models.py --commit                        # run every action
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pymongo import MongoClient
from bson import ObjectId


from mongo_conn import get_mongo_uri  # 2026-07-10 統一連線(mongo 已啟用 --auth,舊無認證 URI 已失效)
MONGO_URI = get_mongo_uri()
DB_NAME = "furniture_db"
V1_COL = "ikea_product"

MANIFEST_PATH = Path("/home/kyzen/ULIP_RAG/ikea/docs/bad_3d_models_manifest.json")
IKEA_DATA = Path("/mnt/P300/data/ikea_data")
RENDERED_DIR = IKEA_DATA / "rendered_images"
LOG_DIR = Path("/home/kyzen/ULIP_RAG/ikea/logs")

VALID_ACTIONS = {"delete", "fix_label", "regen", "keep"}


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        sys.exit(f"manifest missing: {MANIFEST_PATH}\nrun build_bad_model_manifest.py first.")
    return json.loads(MANIFEST_PATH.read_text())


def do_delete(entry: dict, commit: bool, log) -> tuple[int, int]:
    """Returns (files_removed, bytes_freed)."""
    paths = [Path(entry["glb_path"]), Path(entry["ply_path"]), Path(entry["json_path"])]
    rendered = RENDERED_DIR / entry["_id"]
    removed = 0
    freed = 0
    for p in paths:
        if p.exists() and p.is_file():
            freed += p.stat().st_size
            removed += 1
            log(f"  rm file {p}")
            if commit:
                p.unlink()
    if rendered.exists() and rendered.is_dir():
        dir_bytes = sum(f.stat().st_size for f in rendered.rglob("*") if f.is_file())
        dir_files = sum(1 for _ in rendered.rglob("*") if _.is_file())
        freed += dir_bytes
        removed += dir_files
        log(f"  rm dir  {rendered}  ({dir_files} files, {dir_bytes/1e6:.1f}MB)")
        if commit:
            shutil.rmtree(rendered)
    return removed, freed


def do_fix_label(entry: dict, commit: bool, col, log) -> bool:
    new = entry.get("suggested_category")
    if not new:
        log(f"  [skip] no suggested_category for {entry['_id']}")
        return False

    old = entry.get("current_category")
    log(f"  mongo {entry['_id']}  main_type: {old!r} -> {new!r}")

    # update per-model JSON (keep file, just rewrite category)
    jp = Path(entry["json_path"])
    if jp.exists():
        try:
            obj = json.loads(jp.read_text())
            obj["category"] = new
            log(f"  json  {jp}  category -> {new!r}")
            if commit:
                jp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        except Exception as e:
            log(f"  [warn] failed to rewrite {jp}: {e}")

    if commit:
        col.update_one({"_id": ObjectId(entry["_id"])}, {"$set": {"main_type": new}})
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="apply changes (default: dry-run)")
    ap.add_argument("--action-filter", choices=sorted(VALID_ACTIONS),
                    help="only process entries with this action")
    args = ap.parse_args()

    manifest = load_manifest()
    print(f"[load] manifest entries: {len(manifest)}")

    if args.action_filter:
        manifest = [e for e in manifest if e["action"] == args.action_filter]
        print(f"[filter] action == {args.action_filter}  -> {len(manifest)} entries")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"cleanup_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_lines: list[str] = []

    def log(msg: str):
        log_lines.append(msg)
        print(msg)

    log(f"\n=== cleanup_3d_models.py  commit={args.commit}  at {datetime.now()} ===\n")

    col = MongoClient(MONGO_URI)[DB_NAME][V1_COL]

    stats = defaultdict(int)
    bytes_freed = 0

    delisted_ids: list[str] = []

    for e in manifest:
        action = e["action"]
        stats[f"action_{action}"] += 1

        if action == "delete":
            log(f"[delete] {e['_id']}  reasons={e['reasons']}")
            removed, freed = do_delete(e, args.commit, log)
            stats["files_removed"] += removed
            bytes_freed += freed
            if "delisted" in e["reasons"]:
                delisted_ids.append(e["_id"])
            if args.commit:
                col.update_one(
                    {"_id": ObjectId(e["_id"])},
                    {"$set": {"converted_3d": "deleted"}}
                )
        elif action == "fix_label":
            log(f"[fix_label] {e['_id']}  {e.get('current_category')!r} -> {e.get('suggested_category')!r}")
            if do_fix_label(e, args.commit, col, log):
                stats["labels_fixed"] += 1
        elif action in {"regen", "keep"}:
            stats[f"skipped_{action}"] += 1

    # Bulk-mark delisted products
    if delisted_ids and args.commit:
        col.update_many(
            {"_id": {"$in": [ObjectId(x) for x in delisted_ids]}},
            {"$set": {"status": "delisted"}},
        )
        log(f"\n[mongo] marked {len(delisted_ids)} docs with status='delisted'")
    elif delisted_ids:
        log(f"\n[mongo dry-run] would mark {len(delisted_ids)} docs with status='delisted'")

    log("\n=== summary ===")
    for k, v in sorted(stats.items()):
        log(f"  {k:<22} {v:>6}")
    log(f"  {'bytes_freed':<22} {bytes_freed/1e6:>6.1f} MB")

    log_path.write_text("\n".join(log_lines))
    print(f"\n[log] {log_path}")

    if not args.commit:
        print("\n[dry-run] no changes made. Re-run with --commit to apply.")


if __name__ == "__main__":
    main()
