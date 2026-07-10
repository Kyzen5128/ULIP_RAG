"""
Download primary image (images[0]) for every v3 product into
/mnt/P300/data/ikea_data/images_v3/{asin}.jpg

Source:       furniture_db.ikea_product_v3_fresh_2026q2
Destination:  /mnt/P300/data/ikea_data/images_v3/

Resume-safe: skips files that already exist locally.

Usage:
    python download_v3_images.py                  # dry-run (prints plan, no writes)
    python download_v3_images.py --commit         # actually download
    python download_v3_images.py --commit -j 8    # 8 parallel workers
    python download_v3_images.py --only-3d        # only asins that have a v1 glb
    python download_v3_images.py --limit 20       # cap total jobs (smoke)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from pymongo import MongoClient
from tqdm import tqdm


MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "furniture_db"
V3_COL = "ikea_product_v3_fresh_2026q2"
V1_COL = "ikea_product"
OUT_DIR = "/mnt/P300/data/ikea_data_V3/images"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

NON_FURNITURE_RE = (
    r"^Box$|^Box,|^Box with|^Box set"
    r"|^Cover |cover$|^Chair cover|^Stool cover|^Mattress cover|^Headboard with cover"
    r"|^Underframe|^Tabletop|^Cable box|^Acoustic plant box"
    r"|^Organizer|^Adjustable organizer"
)


def load_targets(only_3d: bool, limit: Optional[int]):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    v3 = db[V3_COL]

    query = {
        "images": {"$exists": True, "$ne": []},
        "asin": {"$exists": True, "$ne": None},
        "type": {"$not": {"$regex": NON_FURNITURE_RE, "$options": "i"}},
    }
    projection = {"_id": 0, "asin": 1, "images": 1}

    if only_3d:
        glb_asins = {d["asin"] for d in db[V1_COL].find(
            {"converted_3d": "y", "asin": {"$exists": True, "$ne": None}},
            {"_id": 0, "asin": 1},
        )}
        query["asin"] = {"$in": list(glb_asins)}

    raw = list(v3.find(query, projection))
    docs = [d for d in raw
            if isinstance(d.get("images"), list) and d["images"] and d["images"][0]]
    skipped = len(raw) - len(docs)
    if skipped:
        print(f"[load] skipped {skipped} docs with None/empty images")
    if limit:
        docs = docs[:limit]
    return docs


def download_one(asin: str, url: str, out_dir: str, timeout: int = 20):
    """Returns (asin, status, error). status in {ok, skipped, failed}."""
    path = os.path.join(out_dir, f"{asin}.jpg")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return asin, "skipped", None
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
        r.raise_for_status()
        if len(r.content) < 1000:
            return asin, "failed", f"too small ({len(r.content)} bytes)"
        with open(path, "wb") as f:
            f.write(r.content)
        return asin, "ok", None
    except Exception as e:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        return asin, "failed", str(e)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="actually download (default: dry-run)")
    ap.add_argument("--only-3d", action="store_true", help="only asins that have a v1 glb")
    ap.add_argument("--limit", type=int, default=None, help="cap number of downloads")
    ap.add_argument("-j", "--jobs", type=int, default=8, help="parallel workers (default 8)")
    args = ap.parse_args()

    docs = load_targets(args.only_3d, args.limit)
    total = len(docs)
    existing = 0
    if os.path.isdir(OUT_DIR):
        existing = sum(1 for a in (d["asin"] for d in docs)
                       if os.path.exists(os.path.join(OUT_DIR, f"{a}.jpg")))
    todo = total - existing

    print(f"[plan] mongo docs w/ images     : {total}")
    print(f"[plan] already downloaded       : {existing}")
    print(f"[plan] new downloads needed     : {todo}")
    print(f"[plan] output dir               : {OUT_DIR}")
    print(f"[plan] parallel workers         : {args.jobs}")

    if not args.commit:
        print("\n[dry-run] No files written. Re-run with --commit to download.")
        return

    if todo == 0:
        print("\nNothing to do.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    ok = skipped = failed = 0
    errors = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(download_one, d["asin"], d["images"][0], OUT_DIR): d["asin"]
            for d in docs
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="download"):
            asin, status, err = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                if len(errors) < 20:
                    errors.append((asin, err))

    dt = time.time() - t0
    print(f"\n[done] ok={ok} skipped={skipped} failed={failed} in {dt:.1f}s")
    if errors:
        print("\n[first 20 failures]")
        for asin, err in errors:
            print(f"  {asin}  {err}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
