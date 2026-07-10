# 733 個 3D 模型品質體檢報告

> 體檢日期：2026-04-20
> 資料來源：`/mnt/P300/data/ikea_data/`（v1 爬蟲時代生成的 TRELLIS 3D 模型）
> 目的：盤點哪些 3D 模型「不能用」、哪些「可以修」、哪些「直接刪」

---

## 🎯 TL;DR

| 狀態 | 筆數 | 比例 |
|---|---|---|
| 🟢 **完全可用**（可直接進 Phase B RAG） | **445** | **60.7%** |
| 🟡 有任一問題（需處理） | 288 | 39.3% |
| &nbsp;&nbsp;&nbsp;&nbsp;└ 產品已下架（IKEA 2026 清單掃過） | 158 | 21.6% |
| &nbsp;&nbsp;&nbsp;&nbsp;└ v2 解析不出尺寸（無法做尺寸 filter） | 158 | 21.6% |
| &nbsp;&nbsp;&nbsp;&nbsp;└ Category 標籤錯誤（CLIP conf < 0.5） | 36 | 4.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;└ 重複 asin（同商品存多份） | 81 | 11.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;└ 點雲過稀疏（< 50k 點） | 8 | 1.1% |

> 上述狀態互有重疊，綜合後「至少有一個問題」共 288 筆，「一個問題都沒有」只有 445 筆。

---

## 1️⃣ 產品已下架 → 158 筆（21.6%）

### 為什麼不能用
- IKEA 官網 URL 已回 404，**無法連到真實產品頁**
- 價格、庫存、SKU 全部失效
- RAG 結果給使用者時，點擊連結會跑到無商品頁

### 怎麼量到的
對全部 733 筆用 `curl -IL` 做 HEAD 檢查，`--max-time 12` 秒，30 併發：
- 200 存活：575 筆
- **404 下架：158 筆**
- 這 158 個 `_id` 清單已存在 `/tmp/ikea_delisted_ids.txt`

### 哪些類別掉最慘

| main_type | 下架筆數 / 總數 | 下架率 |
|---|---|---|
| Dining Table | 26 / 50 | **52%** |
| Bar Stool | 17 / 50 | 34% |
| Dining Chair | 13 / 50 | 26% |
| Recliner | 10 / 40 | 25% |
| Bookshelf | 12 / 50 | 24% |
| TV Stand | 10 / 45 | 22% |
| Bench | 10 / 50 | 20% |
| Office Desk | 10 / 50 | 20% |
| Filing Cabinet | 2 / 49 | 4% |

### 處置建議
- **不重爬**（產品真的沒了，重爬也沒用）
- RAG 仍保留 3D 資產，但在 MongoDB 加 `"status": "delisted"` 標記
- 檢索結果中這類項目**不顯示「購買連結」按鈕**，只顯示 3D 預覽作為家具形狀參考

---

## 2️⃣ 無尺寸（Phase B Rule Engine 用不了）→ 158 筆（21.6%）

### 為什麼不能用
- 點雲檔案本身已被 TRELLIS **歸一化到 bbox ∈ [0, 1]**，**3D 檔裡沒真實公釐資訊**
- 實際尺寸得查 MongoDB 的 `meta.dimensions_mm`
- 這 158 筆在 v2 解析失敗 → 無法做「放得進 200×90 cm 空間嗎？」這種尺寸過濾

### 樣本驗證（單位立方體）
抽 15 筆讀 PLY 點雲 bbox，結果都在 [0, 1]：
```
Wardrobe        bbox = (0.60, 0.99, 0.29)
Bar Stool       bbox = (0.18, 0.99, 0.14)
Recliner        bbox = (0.71, 0.99, 0.92)
```
最大軸永遠 ~1.00 → 證明 **3D 檔是「形狀」不是「實尺寸」**。

### 解析失敗原因（Phase A 留下的診斷）

| 原因 | 說明 |
|---|---|
| `only_1_dims` | 只有 1 個數字（例：只寫座椅高 45 cm） |
| `cabinet_height_too_short` | 2D 推斷出的高度 < 50 cm，被 sanity check 擋 |
| `bed_length_too_short` | 床長 < 150 cm，不像床 |
| 床尺碼文字（Twin/Queen/King） | IKEA 床品用文字描述，無數字 |
| `no_size_options` | IKEA 頁面就沒填尺寸 |

### 處置建議
- **短期（save 50+ 筆）**：建立床尺碼對照表（Twin ≈ 38″×75″、Queen ≈ 60″×80″），可救回 Bed/Bunk Bed 類
- **中期**：放寬 sanity 閾值，改為 warn 而非 reject
- **無法救的**：3D 仍可進視覺檢索，但不進尺寸 filter

---

## 3️⃣ Category 標籤錯誤 → 36 筆（4.9%）

### 為什麼不能用
- JSON 寫 `"category": "TV Stand"`，但 3D 實際是夜櫃 / 書架 / 衣櫃
- 使用者按類別搜「電視架」會得到床頭櫃、沙發布套等無關結果
- **v1 爬蟲時 IKEA `category` API 就標錯**，不是 TRELLIS 或 ULIP 的鍋

### 分類

| 錯誤類型 | 筆數 | 能否救 |
|---|---|---|
| 類別完全標錯（家具 → 別種家具） | 27 | ✅ 可用 URL slug 自動修 |
| 根本不是家具（布套、桌板、底座） | 6 | ❌ 直接刪 |
| URL 無法判斷（需人工） | 3 | 👀 視覺檢查 |

### 最荒謬的 10 個案例

| JSON 寫 | 實際是 | Slug |
|---|---|---|
| TV Stand | **夜櫃** | `knarrevik-nightstand-bright-yellow` |
| TV Stand | **夜櫃** | `setskog-nightstand-black` |
| TV Stand | **衣櫃** | `nordkisa-open-wardrobe-with-sliding-door-bamboo` |
| TV Stand | **電競桌** | `fredde-gaming-desk-black` |
| TV Stand | **筆電架** | `vittsjoe-laptop-stand-black-brown-glass` |
| Bed | **書架** | `kleppstad-shelf-white` |
| Bed | **戶外單椅** | `havsten-armchair-outdoor-beige` |
| Bed | **桌板**（非家具） | `lagkapten-tabletop-white` |
| Bed | **沙發布套**（非家具） | `soederhamn-cover-for-sofa-section` |
| TV Stand | **裝飾條**（非家具） | `havstorp-rounded-deco-strip` |

### 處置建議（3 階段）
1. **27 筆自動修 label**：用 URL slug 關鍵字對應表（`nightstand` → Nightstand、`desk` → Office Desk 等），更新 JSON 的 `category`
2. **6 筆刪除**：`cover-for-*`, `underframe`, `tabletop`, `deco-strip` 這類根本不是家具
3. **3 筆人工處理**：`enhet-wall-fr-w-shelves`（上牆框架）、2 筆 `aengsjoen-backsjoen-bathroom-vanity`（浴室盥洗台）

完整清單：`/tmp/ikea_36_mislabel.csv`

---

## 4️⃣ 重複 asin → 80 組 / 81 筆浪費（約 11%）

### 為什麼不能用
- 同一商品（同 asin、同 URL）**被存了 2 次以上**
- TRELLIS 跑了 2 次 → 2 組 ply/glb/圖片
- ULIP 算了 2 個向量
- 檢索時 Top-K 會被同商品重複占據，多樣性差

### 實例

| asin | name | 重複次數 | URL slug |
|---|---|---|---|
| s39575884 | MORABO | 2 | `morabo-sofa-with-chaise-gunnared-dark-gray-wood-s39575884` |
| s79442458 | UPPLAND | 2 | `uppland-sectional-4-seat-corner-kilanda-dark-blue-s79442458` |
| 50512258 | SLATORP | 2 | `slatorp-sofa-with-chaise-tallmyra-white-black-50512258` |

從 `_id` 時間戳看：
- `67d10a...` 開頭（3/12 早批爬蟲）
- `67d173...` 開頭（3/12 晚批爬蟲）
- 同一商品跨兩批爬蟲都進來了，**去重 key 用 `_id` 而非 `asin`**，所以漏掉

### 處置建議
- **dedup by asin**，每組保留品質最好的那筆：
  1. 第 1 優先：點雲點數多
  2. 第 2 優先：CLIP confidence 高
  3. 第 3 優先：產品 URL 還活著（非 404）
- **可省空間**：約 2.9 GB（81 × 26GB/733）

---

## 5️⃣ 點雲過稀疏 → 8 筆（1.1%）

### 為什麼不能用
- TRELLIS 正常輸出至少 8192 點
- 這 8 筆 < 50k 點，遠低於 median 41.5 萬點
- ULIP 編碼可能不穩定，檢索準確度掉

### 最嚴重的 4 筆（< 8192 點）

| 點數 | main_type | CLIP conf |
|---|---|---|
| 2,400 | Bed | 0.84 |
| 3,968 | Bed | 0.60 |
| 3,968 | Bed | 0.60 |
| 3,968 | Bed | 0.60 |

推測原因：IKEA 床的照片是整體大圖，TRELLIS 可能只生了床墊或床頭板，模型破損。

### 處置建議
- **直接刪除** 或 **重跑 TRELLIS**（但若原圖品質差，重跑意義不大）

---

## ⚠️ 733 筆共同限制：點雲是歸一化的

**不是 bug，但下游要注意**：
- 所有 733 筆的 PLY 點雲 bbox = 單位立方體 [0, 1]
- **「3D 形狀」和「實際尺寸」是分離的兩件事**
- Phase B Rule Engine 必須：
  1. 讀 `meta.dimensions_mm` 取真實公釐
  2. 把點雲 bbox × 真實尺寸 才能還原實體
- 若下游假設 3D 本身帶公釐，結果會全錯

---

## 💾 磁碟空間概況（ikea_data 總計 30 GB）

| 資料夾 | 大小 | 佔比 | 可省空間（dedup 後）|
|---|---|---|---|
| `ply/`（點雲） | **26 GB** | 87% | ~2.9 GB |
| `rendered_images/` | 3.2 GB | 11% | ~0.35 GB |
| `glb/`（mesh） | 1.1 GB | 3.7% | ~0.12 GB |
| `images/`（原圖） | 101 MB | — | — |
| `vectors/`（ULIP 512d） | 5.2 MB | — | — |
| `json/` | 4.1 MB | — | — |

**Phase B 檢索實際只需要**：`vectors/` + `json/` + `rendered_images/` ≈ **3.3 GB**
其他（ply/glb）只在「看 3D 預覽」時需要，可移冷儲存。

---

## 🔧 處置建議優先序

### Priority 1：馬上做（1 小時內）
- ✂️ **dedup**：80 組重複 asin → 保留最佳者（節省 2.9 GB、消除 11% 影子重複）
- 🗑️ **刪除 6 筆非家具**：布套、桌板、底座、裝飾條
- 🔧 **修 27 筆錯 label**：用 URL slug 自動重判

### Priority 2：可選（救尺寸）
- 📏 **建床尺碼對照表**：Twin / Full / Queen / King → mm，可救 20+ 筆 Bed 類
- 📏 **放寬 sanity 閾值**：把 reject 改 warn，可救 10+ 筆櫃類

### Priority 3：長期
- 🆕 **補新 3D 模型**：v3 現有 3,073 筆商品中，只有 324 筆有 3D（10.5%）→ 若要提升檢索品質，可跑第二批 TRELLIS pipeline

### 不建議做
- ❌ **重跑全部 733 筆 TRELLIS**：成本高、解決不了 category 錯標（那是 v1 爬蟲的鍋）、解決不了下架問題
- ❌ **刪除所有下架產品的 3D**：3D 資產仍有視覺檢索價值，留著只加個標記即可

---

## 📊 處置後預估可用量

| 處置步驟 | 動作 | 剩餘筆數 |
|---|---|---|
| 起點 | 原始 | 733 |
| Step 1 | dedup | 653（-80） |
| Step 2 | 刪 6 筆非家具 | 647 |
| Step 3 | 刪 8 筆點雲過稀疏 | 639 |
| Step 4 | 修 27 筆 label | 639（不減少） |
| Step 5 | 標 150 筆下架但保留 | 639 |
| **最終可用** | | **639 筆獨立商品** |
| 其中 **可做尺寸 filter** | v2 有 `dimensions_mm` | **~500 筆** |

---

## 📁 相關檔案

| 檔案 | 說明 |
|---|---|
| `/mnt/P300/data/ikea_data/json/{id}.json` | 每筆 3D 模型的 metadata |
| `/mnt/P300/data/ikea_data/ply/{id}.ply` | 點雲（歸一化到 [0,1]） |
| `/mnt/P300/data/ikea_data/glb/{id}.glb` | Mesh |
| `/mnt/P300/data/ikea_data/rendered_images/{id}/*.png` | 多角度渲染圖 |
| `/mnt/P300/data/ikea_data/vectors/` | ULIP 512d 向量（pc + img + txt） |
| `/tmp/ikea_delisted_ids.txt` | 158 個 404 下架的 `_id` 清單 |
| `/tmp/ikea_36_mislabel.csv` | 36 筆 CLIP conf < 0.5 錯標清單（Excel 可開） |
| `ikea/docs/phase_a_dimensions_result.md` | Phase A 尺寸解析說明 |
| `ikea/docs/phase_v3_scrape_result.md` | v3 爬蟲結果（3,073 筆） |
