# ULIP_RAG 後續方向規劃：Phase A / B / C

> 目標：釐清 ULIP_RAG 系統該往哪走、每個 Phase 做什麼、動到什麼、不動什麼。

---

## 🎯 最終系統要做出的東西

一個**家具檢索系統**，使用者輸入像這樣：

> 「幫我找**現代簡約風**的**皮革沙發**，放在客廳**200×90 cm** 的位置，**適合放牆角**的那種」

系統要能回傳：**符合外型 + 符合尺寸 + 符合擺放語意**的 Top-K 家具。

對應 CAMERA 架構圖：Knowledge Base → Dense Retriever → ULIP Contrastive → Rule Engine → Top-K。

---

## 🧠 核心觀念：兩個概念要分清楚

| 概念 | 是什麼 | 住在哪 | 例子 |
|---|---|---|---|
| **尺寸（dimensions）** | 具體數字 | **Metadata**（MongoDB）+ Rule Engine | 224×92×90 cm |
| **空間語意（placement semantics）** | 描述性語言 | **ULIP text encoder**（透過 caption） | 「適合放牆角」「靠牆」「小巧」 |

👉 **尺寸永遠不進模型。空間語意才進模型（透過 caption）。**

---

## 🔧 模型會動到的程度（全局原則）

| 項目 | 狀態 |
|---|---|
| ULIP **架構**（網路層、encoder、loss） | ❌ **永遠不動** |
| ULIP **權重** | 🔄 最多重訓 **1 次**（Phase C 才需要） |
| 現有 733 筆 ply / glb / render | ❌ **永遠不重跑** |
| MongoDB 原表 `ikea_product` | ❌ **不覆寫**（用備份 `ikea_product_v2_2026q2`） |

---

## 📦 Phase A：補尺寸到 MongoDB

### 要幹嘛
把 IKEA 已經有的 `size_options` 字串（例：`"88 1/4 ""`）解析成結構化的 `meta.dimensions_mm`。

### 為什麼要做
- 目前 MongoDB 沒有結構化尺寸欄位，檢索系統無法做「200×90 cm 放得下」這類過濾
- 這是整個 Rule Engine 的**基礎資料**

### 做什麼事
1. 讀 `ikea_product.size_options` 字串
2. 用 `fetch_dimensions.py` 既有的 `_parse_inch_str()` 做單位轉換（英吋→mm，支援 Unicode ¼½¾⅛⅜⅝⅞⅓⅔）
3. 映射：`length = Width`, `width = Depth`, `height = Height`
4. 寫入 `ikea_product_v2_2026q2.meta.dimensions_mm = {length, width, height}`（float, 單位 mm）
5. 印出各 `main_type` 的解析成功率

### 動到什麼
| 檔案 / 資源 | 動作 |
|---|---|
| `ikea_product_v2_2026q2` (MongoDB) | ✏️ 新增 `meta.dimensions_mm` 欄位 |
| `ikea_product` (MongoDB 原表) | ❌ 不動 |
| 檔案系統（ply/glb/json） | ❌ 不動 |
| ULIP 架構 | ❌ 不動 |
| ULIP 權重 | ❌ 不動（完全不重訓） |

### 成本
- **時間**：半天
- **空間**：幾乎零（欄位級增補）

### 完成後能做什麼
- ✅ 每筆家具都有真實 mm 尺寸
- ✅ 未來前端可以顯示尺寸
- ✅ 為 Phase B 的 Rule Engine 準備好資料

---

## 🧰 Phase B：Rule Engine（幾何過濾層）

### 要幹嘛
在檢索 pipeline 的**後處理階段**加入尺寸過濾，把「外型像但放不下」的候選淘汰。

### 為什麼要做
- ULIP 只看外型，不懂尺寸
- 沒有 Rule Engine 時，「找 200×90 cm 的沙發」語意會被當作 caption 的一部分模糊匹配，不精準
- Rule Engine 用 **hard constraint** 做精準過濾

### 做什麼事
1. 在 `app_ikea_retrieval.py` 加 `filter_by_dimensions()` 函數
2. ULIP 先拿 Top-N（例：Top-50）候選
3. 讀每筆候選的 `meta.dimensions_mm`，比對使用者指定的最大長寬高
4. 淘汰放不進去的 → Top-K（例：Top-10）

### 動到什麼
| 檔案 / 資源 | 動作 |
|---|---|
| `app_ikea_retrieval.py` | ✏️ 新增 filter 函數與 pipeline 串接 |
| MongoDB | ❌ 不動（只讀） |
| ULIP 架構 | ❌ 不動 |
| ULIP 權重 | ❌ 不動（完全不重訓） |
| 檔案系統 | ❌ 不動 |

### 成本
- **時間**：1 天
- **空間**：零

### 完成後能做什麼
- ✅ 能精準過濾「放得進 200×90 空間」類查詢
- ✅ 能做 AABB 檢查（家具 bounding box 能否塞進房間指定位置）
- ✅ Phase A + B 完成 = **MVP 可上線**

---

## 🧠 Phase C（選配）：ULIP 學會空間語意

### 要幹嘛
讓 ULIP 的 **text encoder** 理解「適合放牆角」「靠牆」「小巧」「寬大」這類**語意**，不是數字。

### 為什麼做（什麼時候做）
- **不是一開始就做**
- 等 Phase A + B 上線，觀察使用者查詢，如果發現「小巧的沙發」「靠牆的櫃子」這類**語意查詢**效果不好，才做
- 如果使用者都用精確數字查詢（「200×90」），那 Phase C 可以跳過

### 做什麼事
1. 寫腳本掃過 733 份 JSON
2. 根據 `category` + `shape_features` + `dimensions` 特徵，**增強** `llava_caption_en`：
   - L 型沙發 → 加 `"suitable for corner placement"`
   - 長櫃 → 加 `"wall-mounted style"`
   - 小尺寸 → 加 `"compact"`
   - 大尺寸 → 加 `"oversized"`
3. 不動 ply / glb / render（素材零改動）
4. 用升級後的 JSON **重訓 ULIP 一次**
5. 評估檢索準確率變化

### 動到什麼
| 檔案 / 資源 | 動作 |
|---|---|
| 733 份 JSON（`/mnt/P300/data/ikea_data/json/`） | ✏️ 增強 `llava_caption_en` 字串 |
| `build_3d.py` | ✏️ 未來新進資料的 caption 模板也升級 |
| ULIP **架構** | ❌ **不動**（完全不改網路結構） |
| ULIP **權重** | 🔄 **重訓 1 次** |
| ply / glb / render | ❌ **不動** |
| MongoDB | ❌ 不動 |
| TRELLIS | ❌ 不重跑 |

### 成本
- **時間**：2 天（caption 升級半天 + ULIP 重訓 1~1.5 天）
- **空間**：~5 GB（新 checkpoint）
- **GPU**：重訓一次

### 完成後能做什麼
- ✅ 查詢「適合放牆角的沙發」→ 命中 L 型沙發
- ✅ 查詢「小巧的桌子」→ 命中小尺寸桌子
- ✅ 查詢「靠牆的長櫃」→ 命中長型櫃子

---

## 🚫 明確不做的事

| 項目 | 為什麼不做 |
|---|---|
| 全部重來（重爬 + 重跑 TRELLIS + 重訓） | TRELLIS 架構上不產生真實尺度，重跑 2~3 週拿到一樣的東西 |
| 改 ULIP 網路架構 | 現有架構已經能做對比學習，尺寸該放 metadata 不該塞進網路 |
| 把 dimensions 數字當 feature 送進 ULIP | 模糊了「形狀 encoder」與「尺寸 rule」的解耦 |
| 覆寫原始 `ikea_product` | 保留原表做對照，修改只發生在 `ikea_product_v2_2026q2` |
| 重跑 TRELLIS | 同輸入同輸出，浪費 GPU |

---

## 🗺️ 推薦執行順序

```
Day 1 上午 ─── Phase A parser（半天）
Day 1 下午 ─── Phase A 驗證（抽 50 筆人工檢查）
Day 2     ─── Phase B Rule Engine + 整合 app_ikea_retrieval
Day 3     ─── 整合測試、上線 MVP

──────── 以上完成即可交付 MVP ────────

Day 4~5   ─── 收集使用者查詢資料
              觀察「語意查詢」效果
              決定要不要做 Phase C

若決定做 Phase C：
Day 6     ─── 掃 733 JSON + caption 增強
Day 7~8   ─── ULIP 重訓 + 評估
Day 9     ─── 上線升級版模型
```

---

## 📊 三個 Phase 總覽對照

| 項目 | Phase A | Phase B | Phase C |
|---|---|---|---|
| 要解決的問題 | 尺寸資料不存在 | 尺寸過濾能力 | 空間語意理解 |
| 動 MongoDB | ➕ 新欄位 | ❌ 只讀 | ❌ 不動 |
| 動 `app_ikea_retrieval.py` | ❌ | ✏️ 加 filter | ✏️ 小幅微調 |
| 動 JSON 檔 | ❌ | ❌ | ✏️ 增強 caption |
| 動 `build_3d.py` | ❌ | ❌ | ✏️ caption 模板 |
| 動 ULIP 架構 | ❌ | ❌ | ❌ |
| 重訓 ULIP | ❌ | ❌ | 🔄 1 次 |
| 重跑 TRELLIS | ❌ | ❌ | ❌ |
| 時間成本 | 半天 | 1 天 | 2 天 |
| 空間成本 | ~0 | 0 | ~5 GB |
| 何時做 | 現在 | A 做完 | 選配 / 看需求 |

---

## 🎯 一句話總結

> **尺寸住 metadata（Phase A + B）。空間語意住 caption（Phase C）。模型架構永遠不動。**
