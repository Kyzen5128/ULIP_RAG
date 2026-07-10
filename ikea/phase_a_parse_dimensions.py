"""
Phase A: Parse size_options strings → meta.dimensions_mm

Source (read-only):  furniture_db.ikea_product
Target (write):      furniture_db.ikea_product_v2_2026q2

Safety:
- Default is DRY-RUN (prints parse results, NO DB write).
- Pass --commit to actually update the target collection.
- Never touches the source collection.
- Only adds `meta.dimensions_mm` + `meta.dimensions_source` to target docs.

Usage:
    python phase_a_parse_dimensions.py                # dry-run, report only
    python phase_a_parse_dimensions.py --sample 20    # dry-run on 20 docs
    python phase_a_parse_dimensions.py --commit       # WRITE to v2 collection
"""

import argparse
import re
from collections import defaultdict
from datetime import datetime

from pymongo import MongoClient, UpdateOne


from mongo_conn import get_mongo_uri  # 2026-07-10 統一連線(mongo 已啟用 --auth,舊無認證 URI 已失效)
MONGO_URI = get_mongo_uri()
DB_NAME = "furniture_db"
SRC = "ikea_product"                 # read-only
DST = "ikea_product_v2_2026q2"       # write target

# Unicode 分數對照表（取自 fetch_dimensions.py）
UNICODE_FRACS = {
    '¼': 0.25, '½': 0.5, '¾': 0.75,
    '⅛': 0.125, '⅜': 0.375, '⅝': 0.625, '⅞': 0.875,
    '⅓': 1/3,  '⅔': 2/3,
}


def parse_inch_to_mm(s: str):
    """英吋字串 → mm（float）。失敗回 None。"""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    for uc, val in UNICODE_FRACS.items():
        s = s.replace(uc, f' {val}')
    s = re.sub(r'["\u201d\u2033]', '', s).strip()

    m = re.match(r'^(\d+)\s+(\d+)/(\d+)$', s)
    if m:
        inches = int(m.group(1)) + int(m.group(2)) / int(m.group(3))
        return round(inches * 25.4, 1)

    m = re.match(r'^(\d+)\s+([\d.]+)$', s)
    if m:
        inches = int(m.group(1)) + float(m.group(2))
        return round(inches * 25.4, 1)

    m = re.match(r'^[\d.]+$', s)
    if m:
        return round(float(s) * 25.4, 1)

    return None


# 優先順序：key 越靠前越優先
LENGTH_KEYS = ["Width", "Length"]
WIDTH_KEYS  = ["Depth", "Seat depth"]
HEIGHT_KEYS = ["Height", "Height including back cushions", "Backrest height"]


def pick_dim(size_options: dict, candidates: list):
    """依優先順序挑欄位並解析。回傳 (mm_value, used_key) 或 (None, None)。"""
    if not isinstance(size_options, dict):
        return None, None
    for key in candidates:
        if key in size_options:
            v = parse_inch_to_mm(size_options[key])
            if v is not None and v > 50:  # 過濾雜訊（< 5 cm 視為無效）
                return v, key
    return None, None


# 2 維字串第二個數字的語意分組
DEPTH_TYPES = {
    "Sofa", "Recliner", "Storage Ottoman",
    "Dining Chair", "Bar Stool", "Bench",
    "Office Desk", "Dining Table", "Coffee Table",
    "Vanity Table", "Kitchen Island", "Nightstand",
    "Sideboard", "TV Stand", "Shoe Rack",
}
HEIGHT_TYPES = {"Wardrobe", "Bookshelf", "Filing Cabinet"}
LENGTH_TYPES = {"Bed", "Bunk Bed"}


def _sanity_check(dims: dict, main_type: str, is_2d_inferred: bool):
    """品質過濾，回傳 None (pass) 或 fail reason。
    is_2d_inferred=True 才執行類別特定的 sanity（因為 2D 第二軸是推斷的）。
    3D 完整資料相信原始順序，不做 cabinet/bed 啟發式攔截。
    """
    vals = [v for v in dims.values() if v is not None]
    if any(v > 5000 for v in vals):
        return "value_too_large"
    if all(v < 300 for v in vals):
        return "all_dims_too_small"  # 全部 < 30cm 視為零件
    if not is_2d_inferred:
        return None
    if main_type in LENGTH_TYPES:
        L = dims.get("length", 0)
        if L and L < 1500:
            return "bed_length_too_short"
    if main_type in HEIGHT_TYPES:
        H = dims.get("height", 0)
        if H and H < 500:  # 放鬆到 50cm，矮型 Filing Cabinet 也收
            return "cabinet_height_too_short"
    return None


def parse_str_format(s: str, main_type: str):
    """
    解析字串格式 '31 1/2x11x16 7/8 "' 或 '74 3/4x35 3/8 "'。
    3 維 → length/width/height
    2 維 → 依 main_type 判斷第二軸是 D / H / L
    回傳 (dims_dict, format_tag) 或 (None, reason)
    """
    if not isinstance(s, str):
        return None, "not_str"
    s = s.replace('"', '').replace('\u201d', '').replace('\u2033', '').strip()
    if not any(ch.isdigit() for ch in s):
        return None, f"no_digits({s[:20]})"

    parts = [p.strip() for p in s.split('x') if p.strip()]
    cleaned = []
    for p in parts:
        # 區間 '87 1/4-137 3/4' 取第一個（保守取下限）
        if '-' in p:
            p = p.split('-')[0].strip()
        m = re.match(r'^(\d+)/(\d+)$', p)  # 伸縮規格 '59/78' 取第一個
        if m:
            p = m.group(1)
        cleaned.append(p)

    mms = [parse_inch_to_mm(p) for p in cleaned]
    mms = [v for v in mms if v is not None and v > 50]

    # 3 維
    if len(mms) >= 3:
        dims = {"length": mms[0], "width": mms[1], "height": mms[2]}
        reason = _sanity_check(dims, main_type, is_2d_inferred=False)
        if reason:
            return None, f"sanity_{reason}"
        return dims, "str_WxDxH"

    # 2 維：依類別分派
    if len(mms) == 2:
        W, X = mms

        if main_type in DEPTH_TYPES:
            dims = {"length": W, "width": X}
            tag = "str_WxD_depth"
        elif main_type in HEIGHT_TYPES:
            dims = {"length": W, "height": X}
            tag = "str_WxH_height"
        elif main_type in LENGTH_TYPES:
            # 床：原始 W × L，輸出時長的放 length
            longer, shorter = (X, W) if X > W else (W, X)
            dims = {"length": longer, "width": shorter}
            tag = "str_LxW_bed"
        else:
            return None, f"2d_unknown_maintype({main_type})"

        reason = _sanity_check(dims, main_type, is_2d_inferred=True)
        if reason:
            return None, f"sanity_{reason}"
        return dims, tag

    return None, f"only_{len(mms)}_dims"


def extract_dimensions(doc: dict):
    """從單一 doc 抽出 dimensions_mm。回傳 (parsed_dict, status)。"""
    size_options = doc.get("size_options")
    if not size_options:
        return None, "no_size_options"

    if isinstance(size_options, dict):
        length, k1 = pick_dim(size_options, LENGTH_KEYS)
        width,  k2 = pick_dim(size_options, WIDTH_KEYS)
        height, k3 = pick_dim(size_options, HEIGHT_KEYS)

        hits = sum(x is not None for x in (length, width, height))
        if hits < 2:
            return None, f"dict_only_{hits}_dims"

        dims = {}
        if length is not None: dims["length"] = length
        if width  is not None: dims["width"]  = width
        if height is not None: dims["height"] = height

        return {
            "dimensions_mm": dims,
            "source_keys":   {"length": k1, "width": k2, "height": k3},
            "source_format": "dict",
            "confidence":    "high" if hits == 3 else "medium",
        }, "ok"

    if isinstance(size_options, str):
        mt = doc.get("main_type") or ""
        dims, tag = parse_str_format(size_options, mt)
        if dims is None:
            return None, f"str_{tag}"
        confidence = "high" if tag == "str_WxDxH" else "medium"
        return {
            "dimensions_mm": dims,
            "source_keys":   {"raw": size_options},
            "source_format": tag,
            "confidence":    confidence,
        }, "ok"

    return None, f"unknown_type_{type(size_options).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true",
                    help="Actually write to MongoDB (default is dry-run)")
    ap.add_argument("--sample", type=int, default=0,
                    help="Only process first N docs (0 = all)")
    ap.add_argument("--show", type=int, default=10,
                    help="Print first N parsed examples")
    args = ap.parse_args()

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    src = db[SRC]
    dst = db[DST]

    # 前置檢查
    src_count = src.estimated_document_count()
    dst_count = dst.estimated_document_count()
    print(f"=== Phase A: size_options → meta.dimensions_mm ===")
    print(f"Source     : {DB_NAME}.{SRC}  ({src_count} docs, read-only)")
    print(f"Target     : {DB_NAME}.{DST}  ({dst_count} docs)")
    print(f"Mode       : {'COMMIT (will write)' if args.commit else 'DRY-RUN (no write)'}")
    print(f"Sample     : {args.sample if args.sample else 'ALL'}")
    print()

    if dst_count == 0:
        print(f"[ABORT] Target {DST} is empty. Confirm v2 collection exists first.")
        return

    cursor = src.find({}, {"_id": 1, "size_options": 1, "main_type": 1, "name": 1})
    if args.sample:
        cursor = cursor.limit(args.sample)

    ok = 0
    fail_reasons = defaultdict(int)
    by_maintype = defaultdict(lambda: {"ok": 0, "fail": 0})
    shown = 0
    updates = []

    for doc in cursor:
        _id = doc["_id"]
        main_type = doc.get("main_type") or "UNKNOWN"

        parsed, status = extract_dimensions(doc)
        if parsed is None:
            fail_reasons[status] += 1
            by_maintype[main_type]["fail"] += 1
            continue

        ok += 1
        by_maintype[main_type]["ok"] += 1

        if shown < args.show:
            dims = parsed["dimensions_mm"]
            keys = parsed["source_keys"]
            print(f"  [{_id}] {doc.get('name','')[:40]}")
            print(f"    dims : {dims}")
            print(f"    from : {keys}")
            shown += 1

        updates.append(UpdateOne(
            {"_id": _id},
            {"$set": {
                "meta.dimensions_mm":     parsed["dimensions_mm"],
                "meta.dimensions_source": {
                    "method":        "offline_parse_size_options",
                    "source_format": parsed["source_format"],
                    "confidence":    parsed["confidence"],
                    "keys":          parsed["source_keys"],
                    "parsed_at":     datetime.utcnow(),
                },
            }},
        ))

    total = ok + sum(fail_reasons.values())
    print()
    print(f"--- Summary ---")
    print(f"Total processed : {total}")
    print(f"OK              : {ok}  ({ok/total*100:.1f}%)")
    print(f"Failed          : {sum(fail_reasons.values())}")
    for reason, n in sorted(fail_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason:20s}: {n}")

    print()
    print(f"--- By main_type (ok / total) ---")
    for mt, s in sorted(by_maintype.items(), key=lambda x: -x[1]["ok"]):
        tot = s["ok"] + s["fail"]
        rate = s["ok"] / tot * 100 if tot else 0
        print(f"  {mt:25s}: {s['ok']:4d} / {tot:4d}  ({rate:.1f}%)")

    print()
    if args.commit:
        if not updates:
            print("No updates to apply.")
            return
        print(f"[COMMIT] Writing {len(updates)} docs → {DB_NAME}.{DST} ...")
        res = dst.bulk_write(updates, ordered=False)
        print(f"  matched  : {res.matched_count}")
        print(f"  modified : {res.modified_count}")
        print(f"  upserted : {res.upserted_count}")
    else:
        print(f"[DRY-RUN] Would update {len(updates)} docs in {DB_NAME}.{DST}.")
        print(f"Pass --commit to actually write.")


if __name__ == "__main__":
    main()
