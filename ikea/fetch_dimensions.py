"""
從 IKEA 產品頁面抓尺寸，補進 JSON 的 meta.dimensions_cm。

用法：
    cd /home/kyzen/ULIP_RAG
    python ikea/fetch_dimensions.py
"""

import json, os, time, re, glob
import requests
from bs4 import BeautifulSoup

JSON_DIR = "./storage/ikea_data/json"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
DELAY = 1.5

# Unicode 分數對照表
UNICODE_FRACS = {
    '¼': 0.25, '½': 0.5, '¾': 0.75,
    '⅛': 0.125, '⅜': 0.375, '⅝': 0.625, '⅞': 0.875,
    '⅓': 1/3,  '⅔': 2/3,
}

def _parse_inch_str(s: str) -> float | None:
    """
    解析各種英吋格式：
      '31 3/4 "', '31 3/4"', '77 ½"', '29 7/8 "', '31.75"'
    回傳 cm（×2.54），失敗回傳 None
    """
    s = s.strip()
    # 替換 Unicode 分數為小數
    for uc, val in UNICODE_FRACS.items():
        s = s.replace(uc, f' {val}')
    # 去掉引號
    s = re.sub(r'["\u201d\u2033]', '', s).strip()

    # 格式 1：整數 + 分數  "31 3/4" 或 "31 0.25"
    m = re.match(r'^(\d+)\s+(\d+)/(\d+)$', s)
    if m:
        inches = int(m.group(1)) + int(m.group(2)) / int(m.group(3))
        return round(inches * 2.54, 1)

    m = re.match(r'^(\d+)\s+([\d.]+)$', s)
    if m:
        inches = int(m.group(1)) + float(m.group(2))
        return round(inches * 2.54, 1)

    # 格式 2：純數字或小數
    m = re.match(r'^[\d.]+$', s)
    if m:
        return round(float(s) * 2.54, 1)

    return None


def scrape_ikea_dimensions(url: str) -> dict | None:
    """
    回傳 {"width_cm": float, "depth_cm": float, "height_cm": float}
    或 None（抓不到）
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"  [GET FAIL] {e}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    full_text = soup.get_text(" ", strip=True)

    # 抽出 "Measurements ... Packaging" 之間的文字
    m = re.search(r'Measurements\s+(.*?)\s+Packag', full_text, re.DOTALL | re.IGNORECASE)
    if not m:
        # fallback：從 "Measurements" 抓後 400 字元
        m2 = re.search(r'Measurements\s+(.{0,400})', full_text, re.DOTALL | re.IGNORECASE)
        meas_text = m2.group(1) if m2 else full_text
    else:
        meas_text = m.group(1)

    dims = {}

    # 在 Measurements 區塊裡找 Depth / Height / Width
    dim_pattern = re.compile(
        r'\b(Width|Depth|Height)\s*:\s*'         # 標籤
        r'([\d]+(?:\s+[\d]+/[\d]+)?'             # 整數 + 可選分數
        r'|[\d]+(?:\s+[¼½¾⅛⅜⅝⅞⅓⅔])?'          # 整數 + unicode 分數
        r'|[\d.]+)'                               # 純小數
        r'\s*["\u201d\u2033]?',                  # 可選引號
        re.IGNORECASE
    )

    for hit in dim_pattern.finditer(meas_text):
        key = hit.group(1).lower() + "_cm"
        if key in dims:
            continue  # 只取第一個（避免 "Seat depth" 等覆蓋）
        v = _parse_inch_str(hit.group(2))
        if v and v > 5:  # 過濾掉明顯錯誤的小值
            dims[key] = v

    return dims if len(dims) >= 2 else None


def main():
    paths = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
    print(f"Total JSON: {len(paths)}")

    ok = skip = fail = 0

    for i, path in enumerate(paths, 1):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # 已有尺寸就跳過
        if obj.get("meta", {}).get("dimensions_cm"):
            skip += 1
            continue

        url = obj.get("meta", {}).get("url")
        if not url:
            fail += 1
            continue

        fname = os.path.basename(path)
        print(f"[{i}/{len(paths)}] {fname[:30]} → ", end="", flush=True)

        dims = scrape_ikea_dimensions(url)
        if dims:
            obj.setdefault("meta", {})["dimensions_cm"] = dims
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            print(f"OK {dims}")
            ok += 1
        else:
            print("FAIL (no dims)")
            fail += 1

        time.sleep(DELAY)

    print(f"\nDone. ok={ok}  skip={skip}  fail={fail}")


if __name__ == "__main__":
    main()
