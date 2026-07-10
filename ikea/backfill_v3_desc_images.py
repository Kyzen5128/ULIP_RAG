"""
Backfill: 補 ikea_product_v3_fresh_2026q2 中缺漏的 description 與 images

問題根源：
  - images  → IKEA 前端改版，allProductImage[].url 才是正確來源（舊 FIELD_MAPPING 錯誤）
  - description → selector 從 pip- 前綴改為 pipf- 前綴

策略：
  Phase A (fast, no Playwright)：
    重新呼叫所有 category/search API，從 allProductImage 組出 images list，
    按 asin 批量寫入 MongoDB。不啟 Playwright，幾分鐘完成。

  Phase B (Playwright parallel)：
    用 CONCURRENCY 個同時頁面抓 description（以及補漏的 images）。
    約 3073 筆 × 2.5s / CONCURRENCY，預計 ~20-40 分鐘。

Usage:
    python backfill_v3_desc_images.py                    # Phase A only（dry-run）
    python backfill_v3_desc_images.py --commit           # Phase A commit
    python backfill_v3_desc_images.py --phase b          # Phase B dry-run（印樣本）
    python backfill_v3_desc_images.py --phase b --commit # Phase B commit（~30 min）
    python backfill_v3_desc_images.py --phase all --commit  # 兩個 Phase 都跑
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
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm

from fetch_v3_products import CATEGORIES, SEARCH_KEYWORDS, fetch_category_products, fetch_keyword_products

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME   = "furniture_db"
DST       = "ikea_product_v3_fresh_2026q2"
CONCURRENCY = 8     # Playwright 並行數

# ─────────────────────── Phase A：從 API 補 images ───────────────────────

async def _phase_a_fetch_images_async():
    """重跑所有 category + search source，收集 asin → images mapping。"""
    asin_images: dict[str, list[str]] = {}

    def _extract(items):
        for it in items:
            product = it.get("product", {}) or {}
            asin = product.get("id")
            if not asin:
                continue
            all_imgs = product.get("allProductImage") or []
            urls = [img["url"] for img in all_imgs if img.get("url")]
            if urls and asin not in asin_images:
                asin_images[asin] = urls

    # Category sources（同步 HTTP，不需 async）
    print(f"[Phase A] 掃 {len(CATEGORIES)} 個 category API...")
    for cat_id in tqdm(CATEGORIES, desc="categories"):
        try:
            _extract(fetch_category_products(cat_id))
        except Exception as e:
            print(f"  [warn] category {cat_id}: {e}")

    # Search fallback sources（ikea_api 需 async）
    print(f"[Phase A] 掃 {len(SEARCH_KEYWORDS)} 個 search API...")
    for keyword in SEARCH_KEYWORDS:
        try:
            items = await fetch_keyword_products(keyword)
            _extract(items)
        except Exception as e:
            print(f"  [warn] keyword '{keyword}': {e}")

    print(f"[Phase A] API 掃完，取得 {len(asin_images)} 筆 asin→images mapping")
    return asin_images


def phase_a_fetch_images():
    return asyncio.run(_phase_a_fetch_images_async())


def phase_a_apply(asin_images: dict, commit: bool):
    """將 asin_images mapping 寫進 MongoDB。"""
    db  = MongoClient(MONGO_URI)[DB_NAME]
    col = db[DST]

    # 找 v3 中還是 null images 的 doc
    need_fix = list(col.find({"images": None}, {"_id": 1, "asin": 1}))
    total = len(need_fix)
    print(f"[Phase A] v3 中 images=null 的筆數: {total}")

    updates = []
    hit = 0
    miss = 0
    for doc in need_fix:
        asin = doc["asin"]
        urls = asin_images.get(asin)
        if urls:
            updates.append(UpdateOne({"_id": doc["_id"]}, {"$set": {"images": urls}}))
            hit += 1
        else:
            miss += 1

    print(f"[Phase A] 有 mapping: {hit}  無 mapping（需 Phase B）: {miss}")

    if not commit:
        print(f"[Phase A] DRY-RUN — 跳過寫入（加 --commit 才執行）")
        return hit, miss

    if updates:
        res = col.bulk_write(updates, ordered=False)
        print(f"[Phase A] 寫入完成 → modified: {res.modified_count}")
    else:
        print("[Phase A] 無需更新")

    return hit, miss


# ─────────────────────── Phase B：Playwright 補 description ───────────────────────

async def fetch_desc_images_playwright(context, url: str, max_retries: int = 3):
    """抓 description（必要）+ images（補漏），回傳 dict 或 {}。"""
    for attempt in range(max_retries):
        page = None
        try:
            page = await context.new_page()
            await page.goto(url, timeout=60000)
            # 等 description 元素出現（通常 < 2s），否則超時繼續
            try:
                await page.wait_for_selector(
                    "p.pipf-product-summary__description", timeout=5000
                )
            except Exception:
                await page.wait_for_timeout(2000)
            html = await page.content()
            await page.close()
            page = None

            soup = BeautifulSoup(html, "html.parser")
            result = {}

            # description（新 selector）
            desc_el = soup.find("p", class_="pipf-product-summary__description")
            result["description"] = desc_el.get_text(strip=True) if desc_el else None

            # images fallback（如果 Phase A 沒補到）
            gallery = soup.find("div", class_="pipf-product-gallery__thumbnails")
            imgs = [img.get("src") for img in gallery.find_all("img") if img.get("src")] if gallery else []
            result["images_playwright"] = imgs or None

            return result

        except Exception as e:
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
            if attempt < max_retries - 1:
                await asyncio.sleep(random.uniform(2, 5))
            else:
                return {}
    return {}


async def phase_b_worker(queue, context, col, commit, counters, pbar):
    """單一 worker：共用 browser context，從 queue 取 doc，抓 description，寫回 MongoDB。"""
    while True:
        try:
            doc = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        url = doc.get("url") or doc.get("view_url")
        if not url:
            counters["no_url"] += 1
            pbar.update(1)
            continue

        result = await fetch_desc_images_playwright(context, url)

        set_fields = {}
        desc = result.get("description")
        if desc:
            set_fields["description"] = desc
            counters["desc_ok"] += 1
        else:
            counters["desc_fail"] += 1

        # 補 images（如果當前還是 null）
        if doc.get("images") is None:
            imgs = result.get("images_playwright")
            if imgs:
                set_fields["images"] = imgs
                counters["images_playwright_ok"] += 1

        if set_fields and commit:
            try:
                col.update_one({"_id": doc["_id"]}, {"$set": set_fields})
                counters["updated"] += 1
            except Exception as e:
                counters["error"] += 1

        pbar.update(1)
        await asyncio.sleep(random.uniform(0.2, 0.6))


async def phase_b_run(commit: bool, desc_sample: int = 0):
    """並行抓 description，用 CONCURRENCY 個 browser worker。"""
    db  = MongoClient(MONGO_URI)[DB_NAME]
    col = db[DST]

    # 找需要補 description 的 doc
    query = {"description": None}
    total = col.count_documents(query)
    print(f"[Phase B] description=null 筆數: {total}")

    if desc_sample:
        docs = list(col.find(query, {"_id": 1, "asin": 1, "url": 1, "view_url": 1, "images": 1}).limit(desc_sample))
        print(f"[Phase B] --sample {desc_sample}：只跑前 {len(docs)} 筆")
    else:
        docs = list(col.find(query, {"_id": 1, "asin": 1, "url": 1, "view_url": 1, "images": 1}))

    if not docs:
        print("[Phase B] 沒有需要補的 doc，跳過")
        return

    if not commit:
        # dry-run：只跑前 3 筆看看
        sample = docs[:3]
        print(f"[Phase B] DRY-RUN — 抓前 {len(sample)} 筆樣本:")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context()
            for d in sample:
                url = d.get("url") or d.get("view_url", "")
                res = await fetch_desc_images_playwright(ctx, url)
                print(f"  [{d['asin']}] desc={str(res.get('description',''))[:80]}")
                print(f"           imgs={len(res.get('images_playwright') or [])} 張")
            await ctx.close()
            await browser.close()
        print("[Phase B] DRY-RUN 完成，加 --commit 才批量寫入")
        return

    queue    = asyncio.Queue()
    for d in docs:
        await queue.put(d)

    counters = defaultdict(int)
    pbar     = tqdm(total=len(docs), desc="Phase B desc")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        workers = [
            phase_b_worker(queue, context, col, commit, counters, pbar)
            for _ in range(CONCURRENCY)
        ]
        await asyncio.gather(*workers)
        await context.close()
        await browser.close()

    pbar.close()

    print(f"\n[Phase B] 完成")
    print(f"  desc 成功: {counters['desc_ok']}")
    print(f"  desc 失敗: {counters['desc_fail']}")
    print(f"  images 從 Playwright 補回: {counters['images_playwright_ok']}")
    print(f"  MongoDB updated: {counters['updated']}")
    print(f"  error: {counters['error']}")


# ─────────────────────── main ───────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true",
                    help="實際寫入 MongoDB（預設 dry-run）")
    ap.add_argument("--phase", choices=["a", "b", "all"], default="a",
                    help="執行哪個 phase：a=images only, b=description only, all=兩個都跑（預設 a）")
    ap.add_argument("--sample", type=int, default=0,
                    help="Phase B：只跑前 N 筆（0=全跑）")
    args = ap.parse_args()

    db  = MongoClient(MONGO_URI)[DB_NAME]
    col = db[DST]
    total = col.estimated_document_count()
    null_imgs = col.count_documents({"images": None})
    null_desc = col.count_documents({"description": None})
    print(f"=== Backfill v3 desc + images ===")
    print(f"Collection : {DB_NAME}.{DST}  ({total} docs)")
    print(f"images=null: {null_imgs}")
    print(f"desc=null  : {null_desc}")
    print(f"Mode       : {'COMMIT' if args.commit else 'DRY-RUN'}")
    print(f"Phase      : {args.phase}")
    print()

    run_a = args.phase in ("a", "all")
    run_b = args.phase in ("b", "all")

    if run_a:
        print("── Phase A: 從 API 補 images ──")
        asin_images = phase_a_fetch_images()
        phase_a_apply(asin_images, args.commit)
        print()

    if run_b:
        print("── Phase B: Playwright 補 description ──")
        asyncio.run(phase_b_run(args.commit, desc_sample=args.sample))


if __name__ == "__main__":
    main()
