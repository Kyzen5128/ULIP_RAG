# Phase A 執行結果報告：補齊家具尺寸

> 執行日期：2026-04-16
> 腳本：[ikea/phase_a_parse_dimensions.py](../phase_a_parse_dimensions.py)
> 目標：把 IKEA `size_options` 字串 / dict 解析成結構化 `meta.dimensions_mm`

---

## 🎯 目的

讓 ULIP_RAG 檢索系統能用「家具實際尺寸」做 hard filter（例：「找放得進 200×90 cm 的沙發」），是後續 Rule Engine（Phase B）的資料基礎。

---

## ✅ 最終結果

| 指標 | 數值 |
|---|---|
| 處理總筆數 | 2527 |
| 成功解析並寫入 | **2112** (83.6%) |
| 解析失敗（已刪除） | 415 |
| 寫入位置 | `furniture_db.ikea_product_v2_2026q2` |
| 原表是否動到 | ❌ **完全未動** |
| ply / glb / json / 模型動到 | ❌ **完全未動** |

### 各類別成功率

| main_type | 成功 / 總數 | 成功率 |
|---|---|---|
| Vanity Table | 54 / 54 | 100.0% |
| Nightstand | 15 / 15 | 100.0% |
| Shoe Rack | 11 / 11 | 100.0% |
| Sideboard | 131 / 133 | 98.5% |
| Wardrobe | 719 / 756 | 95.1% |
| Bookshelf | 169 / 179 | 94.4% |
| Filing Cabinet | 166 / 177 | 93.8% |
| TV Stand | 40 / 45 | 88.9% |
| Office Desk | 159 / 180 | 88.3% |
| Sofa | 269 / 309 | 87.1% |
| Kitchen Island | 20 / 25 | 80.0% |
| Recliner | 30 / 40 | 75.0% |
| Dining Chair | 133 / 182 | 73.1% |
| Bench | 38 / 56 | 67.9% |
| Storage Ottoman | 37 / 57 | 64.9% |
| Bunk Bed | 3 / 6 | 50.0% |
| Dining Table | 87 / 187 | 46.5% |
| Coffee Table | 2 / 5 | 40.0% |
| Bed | 15 / 50 | 30.0% |
| Bar Stool | 14 / 60 | 23.3% |

---

## 🧠 解析策略

### Step 1：依 `size_options` 型別分流

| 型別 | 筆數 | 處理方式 |
|---|---|---|
| `dict` | 682 | 抽 Width / Depth / Height key |
| `str` | 1845 | 用 `x` 分隔，依長度 + main_type 推斷 |

### Step 2：dict 處理（傳統 IKEA 多選項表格）

優先順序：
- `length` ← `Width` → `Length`
- `width` ← `Depth` → `Seat depth`
- `height` ← `Height` → `Height including back cushions` → `Backrest height`

### Step 3：str 處理（壓縮字串如 `74 3/4x35 3/8 "`）

#### 3 維字串（W × D × H）— 1067 筆，high confidence
直接對應 length / width / height。

#### 2 維字串（W × ?）— 526 筆，medium confidence
依 `main_type` 推斷第二軸：

| 群組 | 涉及 main_type | 第二軸 | 寫入欄位 |
|---|---|---|---|
| **DEPTH 派** | Sofa, Recliner, Storage Ottoman, Dining Chair, Bar Stool, Bench, Office Desk, Dining Table, Coffee Table, Vanity Table, Kitchen Island, Nightstand, Sideboard, TV Stand, Shoe Rack | Depth | `{length, width}` |
| **HEIGHT 派** | Wardrobe, Bookshelf, Filing Cabinet | Height | `{length, height}` |
| **LENGTH 派** | Bed, Bunk Bed | Length | `{length, width}`（長軸放 length） |

#### 單位轉換
- 英吋 → 公釐：`× 25.4`
- 支援分數格式：`3/4`, `½`, `¾`, `⅛`, `⅜`, `⅝`, `⅞`, `⅓`, `⅔`
- 區間如 `87 1/4-137 3/4` → 取下限
- 伸縮如 `59/78` → 取第一個

### Step 4：Sanity Check（防呆）

| 檢查 | 動作 |
|---|---|
| 任一值 > 500 cm | 拒絕（單位錯誤） |
| 全部值 < 30 cm | 拒絕（疑似零件） |
| **2D 推斷** Bed length < 150 cm | 拒絕（不像床） |
| **2D 推斷** 櫃類 height < 50 cm | 拒絕（可能 W×D 被誤判） |
| 3D 完整資料 | 不啟動類別 sanity（信任原始順序） |

---

## 📦 寫入欄位結構

每筆成功解析的 doc 在 `ikea_product_v2_2026q2` 中新增：

```jsonc
{
  "meta": {
    "dimensions_mm": {
      "length": 2279.7,    // float, 單位 mm
      "width":  949.3,
      "height": 828.7      // 部分 doc 可能缺
    },
    "dimensions_source": {
      "method":        "offline_parse_size_options",
      "source_format": "dict" | "str_WxDxH" |
                       "str_WxD_depth" | "str_WxH_height" | "str_LxW_bed",
      "confidence":    "high" | "medium",
      "keys":          {...},   // dict 模式記錄使用的 key
      "raw":           "...",   // str 模式記錄原始字串
      "parsed_at":     ISODate("2026-04-16T06:24:42Z")
    }
  }
}
```

### 信度分布

| confidence | 筆數 | 來源 |
|---|---|---|
| high | 1499 | 3 維資料（`dict` 完整三軸 + `str_WxDxH`） |
| medium | 613 | 2 維推斷（依 main_type 規則） |

### Format 分布

| source_format | 筆數 |
|---|---|
| str_WxDxH | 1067 |
| dict | 519 |
| str_WxD_depth | 328 |
| str_WxH_height | 196 |
| str_LxW_bed | 2 |

---

## 🔍 樣本驗證（人工抽查）

| 商品 | 類別 | Format | 解析結果 (cm) | 對不對 |
|---|---|---|---|---|
| KIVIK | Sofa | dict | 228 × 95 × 83 | ✅ IKEA 經典 3 人沙發 |
| BOAXEL | Bookshelf | str_WxDxH | 62 × 40 × 200 | ✅ 直立柱式書櫃 |
| BRIMNES | Wardrobe | str_WxH_height | 117 × _ × 190 | ✅ 衣櫃高度合理 |
| NYHAMN | Bed | str_LxW_bed | 200 × 140 | ✅ Full size 床面 |
| LINANÄS | Sofa | dict | 197 × 81 × 76 | ✅ |
| FRIHETEN | Sofa | dict | 230 × 151 × 86 | ✅ L 型沙發 |
| HAVSTEN | Wardrobe | (擋掉) | — | ✅ 應該擋（沙發被誤分類） |
| VIPPÄRT | Bed | (擋掉) | — | ✅ 應該擋（38×38×8 是配件） |

---

## 🗑️ 失敗原因與後處理

### 415 筆失敗主因

| 原因 | 筆數 | 說明 |
|---|---|---|
| `only_1_dims` (str) | 177 | 只有 1 個數字（座椅高之類） |
| `no_size_options` | 128 | IKEA 頁面就沒填 |
| `only_0_dims` (str) | 47 | 純文字描述 |
| `only_1_dims` (dict) | 23 | dict 中只有 1 個維度 key |
| `cabinet_height_too_short` | 15 | 2D 推 H 過矮 |
| `only_0_dims` (dict) | 12 | dict 中無維度 key |
| `bed_length_too_short` | 9 | 床長 < 150 cm |
| `all_dims_too_small` | 2 | 全部 < 30 cm |
| 床尺寸名稱 (`Twin`) | 1 | 無數字 |
| `value_too_large` | 1 | > 500 cm |

### 失敗筆數的去向

✅ 已從 `ikea_product_v2_2026q2` **刪除**（415 → 0）
✅ 原表 `ikea_product` **仍保留** 完整 2527 筆，未來想救可從原表 re-aggregate

---

## 🛡️ 安全狀態最終確認

| 項目 | 狀態 |
|---|---|
| `ikea_product` 原表 | 2527 筆，0 個欄位變動 |
| `ikea_product_v2_2026q2` | 2112 筆，全部有 `meta.dimensions_mm` |
| `/mnt/P300/data/ikea_data/json/` (733 筆 ULIP 訓練 JSON) | 0 動 |
| `/mnt/P300/data/ikea_data/ply/` (26 GB 點雲) | 0 動 |
| `/mnt/P300/data/ikea_data/glb/` (mesh) | 0 動 |
| `/mnt/P300/data/ikea_data/rendered_images/` (3.2 GB) | 0 動 |
| ULIP 模型架構 | 0 動 |
| ULIP 訓練權重 | 0 動 |
| TRELLIS / LLaVA pipeline | 0 動 |

---

## 📁 相關檔案

| 檔案 | 說明 |
|---|---|
| [ikea/phase_a_parse_dimensions.py](../phase_a_parse_dimensions.py) | 主腳本（dry-run 預設，需 `--commit` 才寫入） |
| [ikea/fetch_dimensions.py](../fetch_dimensions.py) | 提供 `_parse_inch_str()` 的英吋分數解析邏輯 |
| [ikea/docs/ulip_rag_phase_plan.md](./ulip_rag_phase_plan.md) | Phase A / B / C 規劃 |
| [ikea/docs/data_architecture_and_dimension_plan.md](./data_architecture_and_dimension_plan.md) | 資料架構與決策依據 |

---

## 🚀 下一步：Phase B（Rule Engine）

讓 `app_ikea_retrieval.py` 能依 `meta.dimensions_mm` 做 hard filter：

1. 新增 `filter_by_dimensions(top_n, max_length, max_width, max_height)` 函數
2. 整合到檢索 pipeline：ULIP Top-N → Rule Engine 過濾 → Top-K
3. 對於 `confidence == 'medium'` 的資料可選擇較寬鬆的容差（例 ±5%）
4. 對於沒 `dimensions_mm` 的 doc（v2 已無，但其他 collection 可能有）：跳過尺寸過濾，保留在結果中（safe default）

---

## 📌 重要慣例（給未來參考）

1. **單位**：`meta.dimensions_mm` 一律用 **mm（公釐）** 浮點數
2. **欄位語意**：`length` = 最長水平軸；`width` = 次長水平軸（深度）；`height` = 垂直軸
3. **可缺欄位**：`height` 或 `width` 可能 `None`（依資料來源）
4. **修改範圍**：所有後續資料修改一律寫到 v2 系列 collection，原表只讀
