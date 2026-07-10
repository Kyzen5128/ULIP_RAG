"""
Fresh IKEA scrape with dimensions parsed at ingest time (v3).

Source of truth:  IKEA (live catalog via ikea_api + Playwright fallback)
Write target:     furniture_db.ikea_product_v3_fresh_2026q2
Error log:        furniture_db.ikea_fetch_log_v3
v1 table untouched: furniture_db.ikea_product

Improvements vs ikea/get_data.py:
  1. Parse size_options → meta.dimensions_mm at scrape time
  2. No-auth Mongo URI (aligned with current mongod)
  3. fetched_at timestamp
  4. asin unique index (dedupe)
  5. --limit N per keyword (smoke test)
  6. Playwright only when ikea_api fields missing (same as v1)
  7. Capture <model-viewer> glb / usdz URL (no download)
  8. Errors → ikea_fetch_log_v3
  9. Expanded keyword list (28 keywords → 20 main_types)
 10. Skip already-scraped asin (resume-safe)

Usage:
    python fetch_v3_products.py --limit 5               # dry-run, 5/keyword
    python fetch_v3_products.py --limit 50 --commit     # smoke w/ DB write
    python fetch_v3_products.py --commit                # full run
    python fetch_v3_products.py --commit --only Sofas   # single keyword
"""

import argparse
import asyncio
import random
import traceback
from collections import defaultdict
from datetime import datetime

import ikea_api
import requests as _requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from tqdm import tqdm

from phase_a_parse_dimensions import extract_dimensions


MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "furniture_db"
DST = "ikea_product_v3_fresh_2026q2"
LOG = "ikea_fetch_log_v3"

# ── 資料源 1（主力）：product-list-page API ──
# 用 category ID 而非 keyword 搜尋，原因：search API 每次最多回傳 120 筆，
# PLP API 可一次拿到整個分類的完整清單（~500 筆），且分類邊界更精確。
CATEGORIES = {
    "fu003":  "Sofa",            # Sofas & sectionals
    "10670":  "Sofa",            # Sectionals
    "10663":  "Sofa",            # Sleeper sofas
    "fu006":  "Recliner",        # Armchairs & accent chairs
    "20926":  "Storage Ottoman",  # Ottomans & footstools
    "25219":  "Dining Chair",    # Dining chairs
    "20864":  "Bar Stool",       # Bar stools
    "25220":  "Bench",           # Benches
    "20649":  "Office Desk",     # Desks
    "21825":  "Dining Table",    # Dining tables
    "10705":  "Coffee Table",    # Coffee tables
    "20657":  "Vanity Table",    # Dressing tables
    "20656":  "Nightstand",      # Nightstands
    "10412":  "Sideboard",       # Sideboards
    "10475":  "TV Stand",        # TV units
    "19053":  "Wardrobe",        # Wardrobes
    "10382":  "Bookshelf",       # Bookcases
    "10550":  "Bookshelf",       # Shelving units
    "20652":  "Filing Cabinet",  # Filing cabinets
    "bm003":  "Bed",             # Beds
    "18723":  "Bunk Bed",        # Bunk beds
}

# ── 資料源 2（補漏）：search API（上限 120/keyword）──
# 這三個分類 IKEA 沒有對應的 category ID，只能透過關鍵字搜尋補齊。
SEARCH_KEYWORDS = {
    "Kitchen Islands": "Kitchen Island",
    "Shoe Cabinets":   "Shoe Rack",
    "Consoles":        "Sideboard",
}

# ikea_api 物件欄位對映（與 v1 一致）
# 注意：images 不在此表 —— 由 build_doc() 從 allProductImage[].url 組裝
FIELD_MAPPING = {
    "image":         "mainImageUrl",
    "name":          "name",
    "url":           "pipUrl",
    "description":   "description",
    "price":         "salesPrice",
    "color_options": "colors",
    "view_url":      "pipUrl",
    "size_options":  "itemMeasureReferenceText",
    "asin":          "id",
    "star":          "ratingValue",
    "brand":         "brand",
    "starNum":       "ratingCount",
    "type":          "typeName",
    "category":      "category",
}


constants = ikea_api.Constants(country="us", language="en")
search = ikea_api.Search(constants)

PLP_BASE = "https://sik.search.blue.cdtapps.com/us/en/product-list-page"


def fetch_category_products(cat_id: str) -> list:
    """product-list-page API：無 120 上限，一次最多 ~500 筆。
    v=20210322 是 IKEA CDN API 目前唯一接受的 stable version tag，
    換版本號會 400；size=500 是實測上限，超過仍只回傳 ~500。
    """
    params = {
        "category": cat_id,
        "size": 500,
        "c": "sr",
        "v": "20210322",
    }
    r = _requests.get(PLP_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    plp = data.get("productListPage", {})
    items = plp.get("productWindow", [])
    return [{"product": item} for item in items] if items else []


async def fetch_missing_fields(context, url: str, need_size: bool = True, max_retries: int = 3):
    """Playwright fallback —— 抓 description / images / size_options (dict) / 3D URL。

    只在 API 漏欄位時才呼叫，因為 Playwright 速度慢且對 IKEA 有 rate-limit 風險。
    context 為共用 BrowserContext（而非每次重新 launch），省掉 Chromium 啟動開銷。
    need_size=False 用於已有 size_options 的情況，跳過等待 measurements selector，
    明顯縮短 per-page 耗時。
    """
    for attempt in range(max_retries):
        try:
            page = await context.new_page()
            try:
                await page.goto(url, timeout=60000)
                if need_size:
                    try:
                        await page.wait_for_selector(
                            "ul.pipf-measurements-modal__measurements-container",
                            timeout=3500,
                        )
                    except Exception:
                        # 頁面可能根本沒尺寸 modal,放行繼續解析其他欄位
                        pass
                # 不論是否等到 selector,都給短暫渲染時間給 description / images
                await page.wait_for_timeout(800)
                html = await page.content()
            finally:
                await page.close()

            soup = BeautifulSoup(html, "html.parser")
            result = {}

            # IKEA 2026 前端改版：所有 PIP 元件 class prefix 從 pip- 改為 pipf-，
            # 舊 selector pip-product-summary__description 已失效。
            desc_el = soup.find("p", class_="pipf-product-summary__description")
            result["description"] = desc_el.get_text(strip=True) if desc_el else ""

            rating = soup.find("span", class_="pipf-highlight-reviews__header")
            result["star"] = rating.get_text(strip=True) if rating else ""

            reviews = soup.find("div", class_="pipf-highlight-reviews__card-total-reviews")
            result["starNum"] = reviews.get_text(strip=True) if reviews else ""

            breadcrumbs = soup.find("ol", class_="bc-breadcrumb__list")
            if breadcrumbs and len(breadcrumbs.find_all("li")) > 1:
                span = breadcrumbs.find_all("li")[1].find("span")
                result["category"] = span.get_text(strip=True) if span else ""
            else:
                result["category"] = ""

            # images: 從 pipf- gallery thumbnails 抓（API 已有 allProductImage，此為 Playwright fallback）
            gallery = soup.find("div", class_="pipf-product-gallery__thumbnails")
            result["images"] = [
                img.get("src") for img in gallery.find_all("img") if img.get("src")
            ] if gallery else []

            # 尺寸 dict 格式,鍵名如 Width / Depth / Height
            # IKEA 2026 新 DOM: ul.pipf-measurements-modal__measurements-container
            #                 > li.pipf-measurements-modal__product-measurement-wrapper
            #                   > span.pipf-measurements-modal__product-measurement-name (後接 text node 為值)
            size_container = soup.find(
                "ul", class_="pipf-measurements-modal__measurements-container"
            )
            dimensions = {}
            if size_container:
                for m in size_container.find_all(
                    "li", class_="pipf-measurements-modal__product-measurement-wrapper"
                ):
                    name_span = m.find(
                        "span", class_="pipf-measurements-modal__product-measurement-name"
                    )
                    if not name_span:
                        continue
                    key = name_span.get_text(strip=True).replace(":", "")
                    value_node = name_span.next_sibling
                    if value_node:
                        dimensions[key] = value_node.strip()
            result["size_options"] = dimensions or ""

            # 官方 3D 模型只存 URL，不下載：
            # IKEA 的 GLB 是低面數展示模型，品質遠不如 TRELLIS 重建，
            # 留著備查，但 regen pipeline 直接用產品圖重建。
            mv = soup.find("model-viewer")
            if mv:
                result["model_glb_url"] = mv.get("src")
                result["model_usdz_url"] = mv.get("ios-src")

            result["view_url"] = url
            result.update({"source": "IKEA", "price_string": "USD", "brand": "IKEA"})
            return result

        except Exception as e:
            print(f"  [playwright {attempt + 1}/{max_retries}] {e}")
            await asyncio.sleep(random.uniform(2, 5))
    return {}


async def fetch_keyword_products(keyword: str):
    """呼叫 ikea_api search,回傳 product list(最多 2000)。"""
    endpoint = search.search(keyword, limit=1)
    res = await ikea_api.run_async(endpoint)
    if not isinstance(res, dict) or "searchResultPage" not in res:
        return []

    max_products = res["searchResultPage"]["products"]["main"].get("max", 0)
    if max_products == 0:
        return []

    endpoint = search.search(keyword, limit=min(max_products, 2000))
    res = await ikea_api.run_async(endpoint)
    if not isinstance(res, dict) or "searchResultPage" not in res:
        return []

    products = res["searchResultPage"]["products"]["main"].get("items", [])
    return products if isinstance(products, list) else []


def build_doc(item: dict, main_type: str, keyword: str):
    """從 ikea_api product item 抽出欄位。"""
    product = item.get("product", {}) or {}
    doc = {k: product.get(v) for k, v in FIELD_MAPPING.items()}

    price_info = doc.get("price")
    if isinstance(price_info, dict):
        doc["price"] = price_info.get("numeral")
        doc["price_string"] = price_info.get("currencyCode", "USD")

    variants = product.get("gprDescription", {}).get("variants", []) or []
    doc["variant_urls"] = [v.get("pipUrl", "") for v in variants if v.get("pipUrl")]

    # IKEA 2026 API 異動：category PLP 回應不再有頂層 images 欄位，
    # 改成 allProductImage[{url, type}] 陣列；search API 仍有舊格式，兩邊統一到此。
    all_imgs = product.get("allProductImage") or []
    doc["images"] = [img["url"] for img in all_imgs if img.get("url")] or None

    doc["main_type"] = main_type
    doc["search_keyword"] = keyword
    return doc


def parse_and_attach_dimensions(doc: dict) -> str:
    """用 Phase A 的 extract_dimensions 將 size_options 轉 meta.dimensions_mm。"""
    parsed, status = extract_dimensions(doc)
    if parsed is None:
        return status
    doc.setdefault("meta", {})
    doc["meta"]["dimensions_mm"] = parsed["dimensions_mm"]
    doc["meta"]["dimensions_source"] = {
        "method":        "inline_parse_at_scrape",
        "source_format": parsed["source_format"],
        "confidence":    parsed["confidence"],
        "keys":          parsed["source_keys"],
        "parsed_at":     datetime.utcnow(),
    }
    return "ok"


async def process_item(item, main_type, keyword, dst, log, args, counters, context):
    if not isinstance(item, dict):
        counters["skip_bad_item"] += 1
        return

    doc = build_doc(item, main_type, keyword)
    asin = doc.get("asin")
    if not asin:
        counters["skip_no_asin"] += 1
        return

    # Resume-safe：整個爬蟲可以中斷後重跑，已入庫的 asin 直接跳過，
    # 避免重複呼叫 Playwright 且不觸發 MongoDB unique index 衝突。
    if dst.find_one({"asin": asin}, {"_id": 1}):
        counters["skip_exists"] += 1
        return

    # 只在 API 有漏欄位時才啟 Playwright，以減少觸發 IKEA rate-limit 的次數。
    # 實測約 30~40% 的產品 API 已有完整欄位，可完全跳過 Playwright。
    missing = [k for k, v in doc.items() if not v and k not in ("meta",)]
    if missing and doc.get("url"):
        need_size = not doc.get("size_options")
        extra = await fetch_missing_fields(context, doc["url"], need_size=need_size)
        for f in missing:
            if extra.get(f):
                doc[f] = extra[f]
        for k in ("model_glb_url", "model_usdz_url"):
            if extra.get(k):
                doc[k] = extra[k]

    dim_status = parse_and_attach_dimensions(doc)
    doc["fetched_at"] = datetime.utcnow()
    doc["dim_parse_status"] = dim_status
    counters[f"dim_{dim_status}"] += 1

    if args.commit:
        try:
            dst.insert_one(doc)
            counters["inserted"] += 1
        except DuplicateKeyError:
            counters["skip_exists"] += 1
        except Exception as e:
            log.insert_one({
                "asin":      asin,
                "keyword":   keyword,
                "phase":     "insert",
                "error":     str(e),
                "traceback": traceback.format_exc(),
                "at":        datetime.utcnow(),
            })
            counters["error_insert"] += 1
    else:
        counters["dry_run_ok"] += 1

    # 0.3~0.8s 隨機間隔：共用 BrowserContext 省去啟動開銷後速度大幅提升，
    # 若不降速 IKEA 會在約 200 req 後開始 429，此範圍是實測安全區間。
    await asyncio.sleep(random.uniform(0.3, 0.8))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true",
                    help="Actually write to MongoDB (default: dry-run)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max items per keyword (0 = no limit, cap 2000)")
    ap.add_argument("--only", type=str, default=None,
                    help="Only scrape one source key (category ID or search keyword)")
    ap.add_argument("--search-only", action="store_true",
                    help="Only run search API sources (skip category API)")
    ap.add_argument("--category-only", action="store_true",
                    help="Only run category API sources (skip search API)")
    args = ap.parse_args()

    all_keys = {**{k: v for k, v in CATEGORIES.items()}, **SEARCH_KEYWORDS}
    if args.only and args.only not in all_keys:
        print(f"[ABORT] Unknown key '{args.only}'.")
        print(f"Categories: {', '.join(CATEGORIES.keys())}")
        print(f"Search kw:  {', '.join(SEARCH_KEYWORDS.keys())}")
        return

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    dst = db[DST]
    log = db[LOG]

    # sparse=True：允許 asin 為 null 的 doc 共存（不計入 unique 約束），
    # 避免 Playwright fallback 失敗導致 asin=None 的 doc 彼此衝突。
    dst.create_index("asin", unique=True, sparse=True)

    # 建立工作清單：(source_key, main_type, source_type)
    jobs = []
    if not args.search_only:
        for cat_id, mt in CATEGORIES.items():
            if args.only and args.only != cat_id:
                continue
            jobs.append((cat_id, mt, "category"))
    if not args.category_only:
        for kw, mt in SEARCH_KEYWORDS.items():
            if args.only and args.only != kw:
                continue
            jobs.append((kw, mt, "search"))

    print(f"=== IKEA v3 fresh scrape (Phase A at-ingest) ===")
    print(f"Target     : {DB_NAME}.{DST}  (existing: {dst.estimated_document_count()} docs)")
    print(f"Log        : {DB_NAME}.{LOG}")
    print(f"Mode       : {'COMMIT (writing)' if args.commit else 'DRY-RUN (no write)'}")
    print(f"Per-src lim: {args.limit or 'all'}")
    print(f"Only       : {args.only or 'all sources'}")
    print(f"Sources    : {len(jobs)} ({sum(1 for _,_,t in jobs if t=='category')} category + {sum(1 for _,_,t in jobs if t=='search')} search)")
    print()

    counters = defaultdict(int)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0")
        try:
            for source_key, main_type, source_type in tqdm(jobs, desc="sources"):
                try:
                    if source_type == "category":
                        products = fetch_category_products(source_key)
                    else:
                        products = await fetch_keyword_products(source_key)
                except Exception as e:
                    log.insert_one({
                        "source_key": source_key, "source_type": source_type,
                        "phase": "fetch_list",
                        "error": str(e), "traceback": traceback.format_exc(),
                        "at": datetime.utcnow(),
                    })
                    print(f"[{source_key}] fetch failed: {e}")
                    continue

                if args.limit:
                    products = products[: args.limit]

                print(f"[{source_key} → {main_type}] {len(products)} products ({source_type})")

                for item in tqdm(products, desc=source_key, leave=False):
                    await process_item(item, main_type, source_key, dst, log, args, counters, context)

                await asyncio.sleep(random.uniform(1, 3))
        finally:
            await browser.close()

    print()
    print("--- Summary ---")
    for k in sorted(counters.keys()):
        print(f"  {k:28s}: {counters[k]}")
    print()
    if not args.commit:
        print("[DRY-RUN] Nothing was written. Pass --commit to persist.")


if __name__ == "__main__":
    asyncio.run(main())
