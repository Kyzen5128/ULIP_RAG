# IKEA v3 爬蟲執行結果報告

> 執行日期：2026-04-17
> 腳本：[ikea/fetch_v3_products.py](../fetch_v3_products.py)
> 目標：修正 v1→v3 資料量落差，切換至 Category API + Phase A 尺寸解析整合

---

## 背景：為何重爬

| 版本 | Collection | 筆數 | 問題 |
|---|---|---|---|
| v1（原始） | `ikea_product` | 2,527 | 使用 Search API，每 keyword 上限 120 筆 |
| v2 | `ikea_product_v2_2026q2` | 2,112 | Phase A 過濾後（尺寸解析成功者）|
| v3（舊） | `ikea_product_v3_fresh_2026q2` | 1,707 | 同樣用 Search API，資料更少 |
| **v3（新）** | `ikea_product_v3_fresh_2026q2` | **3,073** | 切換至 Category API，無上限 |

**根本原因**：IKEA `search-result-page` API 有 **120 筆 hard cap**，不論 `size` 設多大都回傳最多 120 筆。
**解決方案**：改用 `product-list-page` API（依 category ID 查詢），`size=500` 可完整取得所有商品。

---

## 技術架構：雙來源策略

### Primary：Category API（21 個分類 ID）

```
GET https://sik.search.blue.cdtapps.com/us/en/product-list-page
Params: category=<ID>, size=500, c=sr, v=20210322
回應結構: productListPage.productWindow[]
```

| Category ID | main_type | 商品數 |
|---|---|---|
| fu003, 10670, 10663 | Sofa | 204+... |
| fu006 | Recliner | — |
| 20926 | Storage Ottoman | — |
| 25219 | Dining Chair | — |
| 20864 | Bar Stool | — |
| 25220 | Bench | — |
| 20649 | Office Desk | — |
| 21825 | Dining Table | — |
| 10705 | Coffee Table | — |
| 20657 | Vanity Table | — |
| 20656 | Nightstand | — |
| 10412 | Sideboard | — |
| 10475 | TV Stand | — |
| 19053 | Wardrobe | — |
| 10382, 10550 | Bookshelf | — |
| 20652 | Filing Cabinet | — |
| bm003 | Bed | 196 |
| 18723 | Bunk Bed | 48 |

### Fallback：Search API（3 個無 Category ID 的分類）

| Keyword | main_type |
|---|---|
| Kitchen Islands | Kitchen Island |
| Shoe Cabinets | Shoe Rack |
| Consoles | Sideboard |

> 已在前次搜尋取得（105 + 76 + 99 筆），本次 `--category-only` 跑法不重跑。

---

## 執行細節

| 項目 | 數值 |
|---|---|
| 執行模式 | `--category-only --commit` |
| 來源數量 | 21 個 Category |
| 執行時間 | ~2 小時 22 分鐘 |
| 新插入筆數 | **1,236** |
| 已存在跳過 | 1,200（deduplicate by `asin`） |
| 尺寸解析成功（此批） | 918 |
| Log 檔案 | `ikea/logs/category_run_20260417.log` |

---

## 最終結果

### Collection 總覽

| Collection | 筆數 | 說明 |
|---|---|---|
| `ikea_product`（原表） | 2,527 | v1，唯讀，永不修改 |
| `ikea_product_v2_2026q2` | 2,112 | Phase A 過濾後，已有 dimensions |
| `ikea_product_v3_fresh_2026q2` | **3,073** | 最新主力 collection |

### 各類別分布（v3）

| main_type | 筆數 |
|---|---|
| Wardrobe | 451 |
| Bookshelf | 393 |
| Sofa | 338 |
| Office Desk | 303 |
| Dining Chair | 162 |
| Bed | 161 |
| Recliner | 148 |
| Bar Stool | 129 |
| Sideboard | 110 |
| Filing Cabinet | 108 |
| Kitchen Island | 105 |
| TV Stand | 100 |
| Coffee Table | 98 |
| Nightstand | 96 |
| Bench | 87 |
| Dining Table | 85 |
| Storage Ottoman | 78 |
| Shoe Rack | 76 |
| Bunk Bed | 27 |
| Vanity Table | 18 |
| **合計** | **3,073** |

### 尺寸覆蓋率（v3）

| 指標 | 數值 |
|---|---|
| 有 `meta.dimensions_mm` 的筆數 | **2,334** |
| 無尺寸（需後續處理） | 739 |
| 整體覆蓋率 | **76.0%** |

#### 各類別覆蓋率

| main_type | 總數 | 有尺寸 | 覆蓋率 |
|---|---|---|---|
| TV Stand | 100 | 100 | 100.0% |
| Sideboard | 110 | 104 | 94.5% |
| Wardrobe | 451 | 418 | 92.7% |
| Shoe Rack | 76 | 70 | 92.1% |
| Filing Cabinet | 108 | 99 | 91.7% |
| Sofa | 338 | 301 | 89.1% |
| Bench | 87 | 77 | 88.5% |
| Office Desk | 303 | 265 | 87.5% |
| Kitchen Island | 105 | 82 | 78.1% |
| Dining Chair | 162 | 126 | 77.8% |
| Vanity Table | 18 | 14 | 77.8% |
| Nightstand | 96 | 71 | 74.0% |
| Recliner | 148 | 107 | 72.3% |
| Bar Stool | 129 | 79 | 61.2% |
| Bookshelf | 393 | 260 | 66.2% |
| Storage Ottoman | 78 | 42 | 53.8% |
| Coffee Table | 98 | 52 | 53.1% |
| Dining Table | 85 | 43 | 50.6% |
| Bunk Bed | 27 | 6 | 22.2% |
| Bed | 161 | 18 | 11.2% |

> **Bed（11.2%）與 Bunk Bed（22.2%）覆蓋率低** 的主因：IKEA 床品大量使用文字尺碼（Twin / Queen / King / Full）而非數字，無法解析。

#### 尺寸來源格式分布

| source_format | 筆數 | 說明 |
|---|---|---|
| str_WxDxH | 1,025 | 3 維字串（高信度） |
| dict | 681 | 結構化 dict（高信度） |
| str_WxD_depth | 514 | 2 維推斷—深度（中信度） |
| str_WxH_height | 104 | 2 維推斷—高度（中信度） |
| str_LxW_bed | 10 | 2 維推斷—床長（中信度） |
| 無尺寸 | 739 | — |

#### 信度分布

| confidence | 筆數 |
|---|---|
| high | 1,559 |
| medium | 775 |

---

## 解析失敗原因（本批 category 跑的部分）

| 原因 | 筆數 | 說明 |
|---|---|---|
| dim_str_only_1_dims | 58 | 只有 1 個數值（如座椅高） |
| dim_no_size_options | 58 | IKEA 頁面未填尺寸 |
| dim_str_no_digits (Twin/Queen/King...) | ~41 | 床尺碼文字描述 |
| dim_str_sanity_all_dims_too_small | 62 | 全部 < 30 cm（配件） |
| dim_str_sanity_cabinet_height_too_short | 37 | 2D 推斷高度過矮 |
| dim_str_sanity_bed_length_too_short | 15 | 床長 < 150 cm |
| dim_dict_only_1_dims | 26 | dict 中只有 1 個維度 |
| dim_dict_only_0_dims | 8 | dict 中無維度 key |
| dim_str_only_0_dims | 11 | 純文字 |
| dim_str_sanity_value_too_large | 1 | > 500 cm |

---

## 已知問題 / 後續改進

### 1. Bed 覆蓋率偏低（11.2%）
- 原因：IKEA 床品以「Twin / Full / Queen / King」命名，無具體 cm 數值
- 可行方案：建立 size_name → mm 對照表（Twin≈38"×75"，Full≈54"×75"，Queen≈60"×80"，King≈76"×80"）

### 2. Category API 噪音
- Wardrobe 類別有少量 LED 照明等非家具商品混入
- 建議：後續加入商品類型過濾（檢查 `type_name` 欄位）

### 3. 目前僅跑 `--category-only`
- Kitchen Island / Shoe Rack / Sideboard 的 Search 來源已在前次跑過
- 若需完整更新 Search 來源：`python fetch_v3_products.py --search-only --commit`

---

## 安全確認

| 資源 | 狀態 |
|---|---|
| `ikea_product`（原表） | ✅ 未動（2,527 筆） |
| `ikea_product_v2_2026q2` | ✅ 未動（2,112 筆） |
| `/mnt/P300/data/ikea_data/json/`（733 筆 ULIP JSON） | ✅ 未動 |
| `/mnt/P300/data/ikea_data/ply/`（26 GB 點雲） | ✅ 未動 |
| ULIP 架構 / 權重 | ✅ 未動 |

---

## 相關檔案

| 檔案 | 說明 |
|---|---|
| [ikea/fetch_v3_products.py](../fetch_v3_products.py) | v3 爬蟲主腳本（Category + Search 雙來源） |
| [ikea/logs/category_run_20260417.log](../logs/category_run_20260417.log) | 本次執行完整 log |
| [ikea/docs/phase_a_dimensions_result.md](./phase_a_dimensions_result.md) | Phase A 尺寸解析說明 |
| [ikea/docs/ulip_rag_phase_plan.md](./ulip_rag_phase_plan.md) | Phase A/B/C 整體規劃 |

---

## 下一步：Phase B（Rule Engine）

v3 資料已備齊（3,073 筆，76% 有尺寸），可進入 Phase B：

1. 在 `app_ikea_retrieval.py` 新增 `filter_by_dimensions(top_n, max_length, max_width, max_height)`
2. ULIP Top-N → Rule Engine 尺寸過濾 → Top-K
3. `confidence == 'medium'` 資料可加 ±5% 容差
4. 無 `dimensions_mm` 的 doc → 跳過尺寸過濾，保留在結果中（safe default）
