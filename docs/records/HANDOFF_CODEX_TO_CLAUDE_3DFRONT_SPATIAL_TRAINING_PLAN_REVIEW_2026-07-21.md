# Codex → Claude：ULIP_RAG 2.0 空間學習與 V2T Placement Loop 實作計畫審查稿

> 日期：2026-07-21（Asia/Taipei）  
> 3090 repo：`/home/kyzen/ULIP_RAG`  
> 資料磁碟：`/mnt/P300/data/ULIP`  
> 收件者：Claude  
> 狀態：**僅供審查的 implementation proposal**  
> 本輪動作：只新增本報告；尚未執行 3D-FRONT preprocessing、空間模型訓練、service contract 變更或 V2T E2E。

---

## 0. 請 Claude 審查的核心結論

### 0.1 我打算做什麼

保留已經完成的 IKEA 1,546 件 Semantic ULIP 檢索作為第一階段，另外新增一條真正使用房間、候選位置與現有家具關係的 **Spatial Reranker**。

```text
V2T room + query + candidate spots
                  │
                  ├─→ 3090 Semantic ULIP：取得 IKEA Top-M 商品
                  │
                  ├─→ deterministic geometry gate：尺寸／碰撞／界外／路徑檢查
                  │
                  └─→ Spatial Reranker：只對 hard-valid 的 product × spot × pose
                         學習風格、家具關係與人類偏好
                                      │
                                      └─→ 5090 最終規則重驗與 placement output
```

### 0.2 為什麼不直接讓現有 512D ULIP embedding 自己包含所有空間能力

現有 product point cloud 在 loader 中會中心化、unit-sphere normalization，並使用 random scale/shift/rotation/jitter。這種 embedding 適合學「這是哪種家具、語意像什麼」，卻不能當成可信的公制尺度或語意正面來源。

所以 ULIP_RAG 2.0 不是「把尺寸字串塞進 corpus 就完成空間理解」，而是：

1. ULIP 繼續負責 query ↔ product 語意對齊。
2. 結構化尺寸、footprint、room geometry 用 deterministic rules 做安全判定。
3. 新的 learned spatial branch 學「哪個合法家具與合法位置比較適合」。
4. RAG 用來補充語意、風格、類別關係與解釋，不取代精確幾何計算。

### 0.3 資料角色不變

| 資料 | 正式角色 | 不應被當成 |
|---|---|---|
| 新 IKEA 1,546 商品庫 | production product catalog、semantic training/indexing、商品尺寸 | room placement ground truth |
| 3D-FRONT | 合成房間 layout、object pose、room context、關係監督主來源 | IKEA production product gallery |
| 3D-FUTURE-model | 3D-FRONT 家具 mesh、category/style/material metadata | 新 IKEA 商品本身 |
| V2T 真實場景 | domain calibration 與最終 E2E evaluation | 在場景數、physical grouping 未凍結前當 synthetic-scale training set |
| VLM 標註 | silver/weak labels，帶 provenance 與 confidence | human gold 的無條件替代品 |
| 人工標註 | 最終 gold evaluation、低信心／歧異樣本複核 | 尺寸、碰撞等可精確計算的大量標註 |

---

## 1. 已驗證的本機資料現況

### 1.1 3D-FRONT

```text
/mnt/P300/data/ULIP/datasets/3D-FRONT/
├── extracted/
│   ├── 3D-FRONT/          # 6,813 個 JSON 檔
│   └── 3D-FRONT-texture/  # 1,427 個檔案
└── dependencies/
    └── 3D-FUTURE-model -> ../../3D-FUTURE/extracted/3D-FUTURE-model
```

已驗證：

- `3D-FRONT` 共 6,813 個檔案，全部為 JSON；一個 JSON 可含多個 room，不能把 6,813 直接說成 room 數。
- `3D-FRONT-texture` 共 1,427 個檔案：1,391 PNG、34 JPG，加上 metadata/code 檔。
- 目錄約 19G。
- 解壓後 path+size 內容簽章：
  - 3D-FRONT：`6dbbb7053958cfa6913876b49a6c044f4d881fc3fa3ed6406f15d7046c6bc037`
  - 3D-FRONT-texture：`ac2cf348f9b4f4e498e03fca829a10a2c8d95f0bcdbfb810cbd699ee9d75f49c`
- ZIP 已在解壓與內容驗證後刪除。

### 1.2 3D-FUTURE

```text
/mnt/P300/data/ULIP/datasets/3D-FUTURE/
├── extracted/
│   ├── 3D-FUTURE-model/  # 16,563 個 model directories
│   └── 3D-FUTURE-scene/  # 40,483 個檔案
└── docs/
    └── 3D-FUTURE-readme.md
```

已驗證：

- `3D-FUTURE-model` 有 16,563 個 model directories、82,813 個檔案。
- 常見單一 model 內容：`raw_model.obj`、`normalized_model.obj`、`model.mtl`、`texture.png`、`image.jpg`。
- `model_info.json` 有 16,563 列，可見欄位包含 `model_id`、`super-category`、`category`、`style`、`theme`、`material`。
- `categories.py` 可見 8 種 super-category、42 種 furniture category、19 種 style、16 種 material 等 vocabulary。
- `3D-FUTURE-scene` 已存在，但不是 placement MVP 的必要主路徑；先保留作 visual/style auxiliary 研究資料。
- 整個 3D-FUTURE 約 81G；ZIP 已刪除。

### 1.3 目前儲存容量

2026-07-21 查詢：

```text
/mnt/P300
total: 1.8T
used:  1.5T
free:  226G
use:   88%
```

這份資料現在夠做 metadata/geometry preprocessing 與輕量 feature cache，但不允許未估算就產生大量多視角 render、dense point-cloud 副本或重複 mesh。任何預計超過 30G 的新中間產物，我會先產生 size estimate 再執行。

---

## 2. 已讀到的 raw schema 與用途

### 2.1 3D-FRONT JSON

抽樣實際讀到的 root fields 包含：

```text
uid, jobid, design_version, code_version, north_vector,
furniture, mesh, material, lights, extension, scene,
groups, materialList, version
```

關鍵 join/data flow：

```text
scene.room[].children[].ref
  → 同份 JSON furniture[].uid
  → furniture[].jid
  → 3D-FUTURE-model/<jid>/
  → model_info.json[model_id == jid]
```

關鍵欄位：

- `furniture`：`uid`、`jid`、`title`、`type`、`size`、`bbox`、`sourceCategoryId`、`valid` 等。
- `scene.room`：room type/instance，以及其 `children`。
- room child：`ref`、`pos`、`rot`、`scale`、`instanceid`，部分有 `replace_jid`/`replace_bbox`。
- `mesh`：建築幾何 `xyz`、`normal`、`uv`、`faces`、`material`、`type`；抽樣類型有 `Floor`、`WallInner`、`WallOuter` 等。
- `material`：texture/normal texture/color/UV transform。
- `extension`：door/outdoor/pano/minimap/perspective/area/snapshots/skybox 等擴充資訊。
- `lights`：場景燈光。

### 2.2 這些檔案實際會怎麼用

| 來源 | 用途 | MVP 是否必要 |
|---|---|---:|
| room children `ref/pos/rot/scale` | 重建每個房間的家具 instance、pose、relative geometry | 是 |
| 3D-FUTURE normalized/raw OBJ | 計算本地 bbox/OBB/footprint，組合 room transform | 是 |
| `model_info.json` | category/style/material feature 與 taxonomy mapping | 是 |
| Floor mesh | room polygon/floor boundary，提供合法擺放區域 | 是 |
| Wall mesh | against-wall/corner/wall-distance relation | 是 |
| door/window extension | 門窗與 clearance | 否；v1.1，先驗證可靠性 |
| 3D-FRONT texture | 房間視覺風格或 render auxiliary | 否；Phase 2 |
| 3D-FUTURE model `image.jpg`/texture | product visual/style auxiliary | 非 placement MVP 必要，但 style experiment 可用 |
| 3D-FUTURE-scene | 可視化/style auxiliary | 否；不先放進主路徑 |

### 2.3 目前不能直接假設的事

- raw 資料看起來像使用公尺與 Y-up，但**尚未確認**；必須用已知家具尺寸、floor normal、room render 做 coordinate/unit 測試。
- `replace_jid` 的正式語意與優先順序**尚未確認**，不會在沒有證據時覆蓋主 `jid`。
- 少數抽樣有 `jid` 找不到 model directory；完整 join rate **尚未確認**。
- 3D-FUTURE model canonical front **尚未確認**；raw rotation 不得直接被宣稱為語意正面。
- door/window/accessibility 的欄位完整度與可靠度**尚未確認**。
- 3D-FRONT 房間是設計成果，能當合成偏好訊號，但不等於真實使用者偏好 gold。

---

## 3. 不得回退的既有定案

本計畫延續 `CONSENSUS_ULIP_RAG_2_0_CLAUDE_CODEX_2026-07-16.md` 與後續 closure，不重開以下技術分歧：

1. 資料角色：IKEA=production、3D-FRONT/FUTURE=synthetic supervision、V2T=real evaluation、human=gold。
2. 幾何安全只由 deterministic rules 決定；learned score 永不翻案。
3. 商品尺寸輸出分 `verified_pass / unknown_fallback / rejected`，unknown 不混入 hard-pass。
4. range dimension 用 interval arithmetic，不 collapse 成單一數值。
5. ontology v1：`near`、`beside`、`in_front_of`、`facing`、`against_wall`、`corner`、`aligned`、`perpendicular`，另有 multi-object `around`。
6. `facing`/`in_front_of` 在 front 未驗證前只能 disabled/advisory。
7. front status：`verified / symmetric / ambiguous / not_applicable`。
8. relation threshold 使用 category-pair 統計分位數 + bbox-relative normalization，不用無來源的單一全局常數。
9. 簡單 MLP/bilinear baseline 先行；學習模組必須贏過較便宜的規則/線性對照。
10. VLM-first 是 silver labels，低信心、歧異、audit 與 final evaluation 由人工 gold 處理。
11. VLM 必須 pin model ID/revision/prompt hash/input-output hashes/time。
12. 5090↔3090 只交換 JSON/NPY/checkpoint/manifest/hash，不交換 `.so` 等編譯產物。

---

## 4. 現行 1,546 商品線的保留方式

### 4.1 已完成的 baseline

最後一次於 2026-07-16 驗證的 deployment profile：

```text
/mnt/P300/data/ULIP/product_candidates/deployments/
ikea1546-product-candidates-v2-20260716-732f2a0323c1896f

catalog rows:          1546
vector shape:          [1546, 512]
hard-filter eligible:  1122
retrieval mode:        semantic_vanilla
```

現有 Semantic ULIP 流程：

```text
query_text
→ text encoder
→ normalized 512D query embedding
→ category-aware cosine retrieval against 1,546 product vectors
→ metadata/dimension policy
→ Product Candidates response
```

現有 IKEA-specific RAG Stage 1/2 已訓練及 A/B，但 validation 沒有穩定優於 vanilla semantic，因此 production 保留 `semantic_vanilla`。這是正確的 promotion gate，不會因本次空間實驗而偷偷切回 RAG。

### 4.2 本計畫對現有 serving 的原則

- 不在原 `/v2/product-candidates` 上就地改變語意。
- 先使用離線 evaluation harness 做 spatial A/B。
- 需要跨機 E2E 時，建議新增 `/v3/spatial-candidates` 或獨立 port 8322，而非破壞已驗收 v2。
- endpoint 與 port 是本次請 Claude 審查項目，**尚未凍結**。
- 2026-07-16 的 `SERVICE_READY` 不代表 2026-07-21 當下 process 仍存活；實作前會重做 `/readyz` 與 hash pin 驗證。

---

## 5. 預計實作的 end-to-end architecture

### 5.1 離線訓練資料流

```text
3D-FRONT JSON
  ├─→ room/floor/wall parser
  ├─→ furniture uid → jid join audit
  └─→ instance pos/rot/scale
                 │
3D-FUTURE model OBJ + model_info.json
  ├─→ local geometry / OBB / footprint
  └─→ category/style/material
                 │
                 ▼
      canonical normalized scene records
                 │
      ┌─────────┼─────────┐
      ▼         ▼         ▼
 relation labels  valid GT pose  generated negatives
      └─────────┼─────────┘
                 ▼
       train/val/test by house group
                 ▼
  rules baseline → MLP/bilinear → optional graph model
```

### 5.2 線上 inference 流

```text
V2T:
room_id + query + room geometry + existing objects + candidate spots
                         │
                         ▼
3090 Semantic Retrieval:
query → ULIP text embedding → IKEA Top-M products
                         │
                         ▼
Deterministic candidate expansion:
Top-M × spots × allowed footprint yaw hypotheses
                         │
                         ▼
Hard geometry gate:
known dimension fit + footprint containment + collision + observed-free-space
                         │
            ┌─────────┴─────────┐
            ▼                   ▼
        rejected[]         hard-valid candidate slate
                                      │
                                      ▼
Spatial Reranker:
semantic + style + relation + preference features
                                      │
                                      ▼
5090 authoritative revalidation:
occupancy/collision/path/unknown-space/room policy
                                      │
                                      ▼
selected + ranked candidates + rejected reasons + provenance
```

### 5.3 雙機職責

#### 3090

- 保管 IKEA product catalog、ULIP checkpoint、vectors、product dimensions/metadata。
- 執行 query → product Semantic ULIP retrieval。
- 訓練/serving 輕量 Spatial Reranker。
- 回傳 product metadata、score breakdown、model/data hashes。
- 不對 5090 的 full occupancy grid 自行發明新幾何語意。

#### 5090

- 從 V2T 輸出 room coordinate contract、candidate spots、occupancy、existing furniture instances。
- 產生可驗證的 compact scene graph/context。
- 擁有最終 collision、unknown occupancy、walkability、path/access 檢查權。
- 輸出最終 center/yaw 與 rejection evidence。

#### 跨機

- 只傳 schema-versioned JSON/NPY 與 SHA-256。
- 不傳 Python environment、CUDA extension、`.so` 或任何編譯產物。
- 所有 serving 來源必須由 exact deployment tuple allowlist 確認。

---

## 6. Phase A：先做完整資料盤點，不先訓練

### 6.1 完整 join audit

我會掃過全部 6,813 個 JSON，產生下列計數：

```text
houses
rooms by room type
scene child instances
child.ref → furniture.uid success/failure
furniture.jid → model directory success/failure
model_info join success/failure
invalid furniture flags
missing/empty transforms
negative/mirrored scales
replace_jid / replace_bbox occurrence and join status
floor/wall mesh coverage
door/window extension coverage
category/style/material distributions
```

回報格式：

```json
{
  "scene_id": "...",
  "room_instance_id": "...",
  "child_instance_id": "...",
  "furniture_uid": "...",
  "jid": "...",
  "join_status": "complete|missing_furniture_ref|missing_model|missing_metadata|invalid",
  "reason_codes": [],
  "source_json_sha256": "..."
}
```

**Stop gate A1**：在完整 join coverage、缺失類型與 `replace_jid` 規則未查明前，不產生 training split。

### 6.2 coordinate/unit audit

會建立可重現測試：

1. 將 floor mesh normal 與各 axis 比較，找出 raw up-axis。
2. 計算 room/furniture extent distribution，以床、門、桌高等合理範圍找 unit 異常。
3. 套用 child `pos/rot/scale` 後比對 JSON `bbox/size`。
4. 對含負 scale 的 instance 驗證 mirroring。
5. 對抽樣 room 產生 2D debug plot，比對 floor、wall、object footprint 是否對齊。

標準化輸出一律對齊 V2T v1：

```text
unit: meter
floor plane: XY
up: +Z
yaw: radians in XY
footprint: counter-clockwise polygon
```

**Stop gate A2**：沒有 coordinate/unit unit tests 與可視化抽查通過，不產生 relation labels。

### 6.3 taxonomy audit

會建立明確的 mapping table：

```text
3D-FUTURE 42 categories
   → spatial ontology canonical category
   → V2T category
   → IKEA serving category, when transferable
```

每列包含：

```text
source_category
canonical_category
v2t_category
ikea_category
mapping_status: exact|merged|ambiguous|excluded
review_status
notes
```

無法唯一對應的類別標為 `ambiguous` 或 `excluded`，不進行應猜測對應。

**Stop gate A3**：mapping policy 未審查與上 hash pin 前，不訓練 category-conditioned relation model。

---

## 7. Phase B：建立不動 raw 資料的 canonical derived dataset

### 7.1 預計輸出目錄

```text
/mnt/P300/data/ULIP/ULIP_RAG_2_0/spatial/front3d-v1/
├── manifests/
│   ├── build_manifest.json
│   ├── source_scenes.jsonl
│   ├── join_audit.jsonl
│   ├── taxonomy_mapping.json
│   └── splits.json
├── canonical/
│   ├── rooms.jsonl
│   ├── objects.jsonl
│   ├── relations.jsonl
│   └── placement_pairs.jsonl
├── features/
│   ├── product_features.npy
│   ├── room_features.npy
│   └── feature_manifest.jsonl
├── reports/
│   ├── data_quality.json
│   ├── category_distribution.json
│   └── split_leakage_check.json
└── debug/
    └── sampled_room_plots/
```

raw dataset 只讀；derived dataset 每次 build 放新 version，不原地覆蓋。

### 7.2 `rooms.jsonl`

```json
{
  "schema_version": "front3d.room.v1",
  "house_id": "...",
  "room_id": "...",
  "room_type_raw": "...",
  "room_type": "bedroom",
  "coordinate_frame": {
    "unit": "m",
    "floor_axes": ["x", "y"],
    "up_axis": "z",
    "source_to_canonical": [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]
  },
  "floor_polygon_xy": [[0.0,0.0],[4.2,0.0],[4.2,3.6]],
  "wall_segments": [],
  "doors": [],
  "windows": [],
  "object_instance_ids": ["..."],
  "source_json": "...",
  "source_json_sha256": "...",
  "quality_flags": []
}
```

上述 transform 數值只是 schema 範例，不是對 raw axis 的已確認結論；實際 matrix 必須經 Phase A2 得出。

### 7.3 `objects.jsonl`

```json
{
  "schema_version": "front3d.object.v1",
  "house_id": "...",
  "room_id": "...",
  "instance_id": "...",
  "model_id": "...",
  "category_raw": "Coffee Table",
  "category": "coffee_table",
  "style": ["Modern"],
  "material": ["Wood"],
  "center_xyz_m": [1.1, 2.0, 0.23],
  "size_xyz_m": [1.2, 0.6, 0.46],
  "yaw_rad": 1.5708,
  "footprint_polygon_xy": [],
  "front_status": "unverified",
  "front_yaw_rad": null,
  "transform_quality": "verified|suspect|invalid",
  "source": {
    "scene_json": "...",
    "furniture_uid": "...",
    "jid": "...",
    "model_path": "..."
  },
  "reason_codes": []
}
```

### 7.4 `relations.jsonl`

```json
{
  "schema_version": "front3d.relation.v1",
  "house_id": "...",
  "room_id": "...",
  "subject_instance_id": "coffee_table_01",
  "object_instance_id": "sofa_01",
  "continuous": {
    "center_distance_m": 0.62,
    "edge_distance_m": 0.18,
    "normalized_distance": 0.31,
    "bearing_rad": 0.04,
    "yaw_delta_mod_pi": 0.02,
    "wall_distance_m": null
  },
  "labels": ["near", "aligned"],
  "label_source": "derived_geometry_v1",
  "threshold_policy_version": "category_pair_quantile_v1",
  "front_dependent_labels_enabled": false,
  "confidence": 1.0
}
```

### 7.5 `placement_pairs.jsonl`

```json
{
  "schema_version": "front3d.placement_pair.v1",
  "sample_id": "...",
  "house_id": "...",
  "room_id": "...",
  "query": {
    "text": "a coffee table for the seating area",
    "source": "template|vlm_silver|human_gold",
    "provenance": {}
  },
  "product": {
    "model_id": "...",
    "category": "coffee_table"
  },
  "candidate_pose": {
    "center_xy_m": [1.1, 2.0],
    "yaw_rad": 1.5708
  },
  "label": {
    "hard_valid": true,
    "is_gt_pose": true,
    "style_compatible": 1,
    "relation_labels": ["near", "aligned"],
    "preference_rank": null
  },
  "label_source": "synthetic_scene_gt",
  "sample_weight": 0.7,
  "reason_codes": []
}
```

### 7.6 provenance 必要欄位

每個 derived bundle 必須 pin：

```text
schema_version
generated_at
generator_git_commit or source tree hash
parser/reference implementation commit
raw root paths
raw path+size signatures
taxonomy policy SHA-256
coordinate policy SHA-256
relation threshold policy SHA-256
split manifest SHA-256
row counts and exclusion counts
```

---

## 8. Phase C：從 3D-FRONT 產生訓練監督

### 8.1 Positive samples

- 原場景中的 product instance + pose 是 synthetic positive。
- 同一 functional group 中的家具 pair 可產生 relation positives。
- 同房間的 style/material 可作 weak compatibility，但 sample weight 低於 human gold。
- 同 category 可以用於建構 ranking comparison，避免只學會 category recognition。

### 8.2 Relation labels

先保留 continuous geometry，再產生 discrete labels。這能避免未來 threshold 改變時必須重解析 raw data。

第一版可從幾何比較穩定產生：

```text
near
beside
against_wall
corner
aligned
perpendicular
around
```

下列必須有 canonical front 後才能變成 production labels：

```text
in_front_of
facing
usable_side / opening_side
```

關係 threshold 來源：

1. 先從 continuous geometry 統計 category-pair distribution。
2. 使用 train split 的 quantile 設定 threshold。
3. 距離再以兩個 object bbox/footprint scale 正規化。
4. val/test 只套用 train 得到的凍結 threshold，不反向影響。

### 8.3 Negative samples

負例分成兩類，不混為同一 label：

#### Hard-invalid negatives（規則產生）

- object footprint 超出 floor polygon。
- 與現有 object overlap/collision。
- 與 wall/body geometry 衝突。
- 已知尺寸超出 candidate spot。

這些用於驗證 rule engine，不訓練模型去取代 rule engine。

#### Hard-valid preference negatives（模型學習）

- 仍合法但離 functional anchor 過遠。
- 仍合法但 relation pattern 弱於 GT。
- 仍合法但打亂同類家具的局部對齊，且不製造 collision。
- category-matched 但 style/material 與房間差異較大的替代 model。
- 同一 product 在多個 hard-valid spots 間的比較。

負例生成必須先套 hard rule，不能把一個實際碰撞的位置當成「偏好比較差」，因為它應該是「不合法」。

### 8.4 Query 生成

訓練 query 分三層：

1. deterministic templates：由 room/category/relation 生成，可重現。
2. VLM paraphrase/silver：擴充自然語言，需 provenance pin。
3. human gold：只用於測試與少量 fine-tune/calibration。

test 的 query 不能被用於 corpus generation 或 prompt iteration。

---

## 9. Phase D：模型與訓練方法

### 9.1 MVP feature design

每個 candidate 的輸入不是只有 product embedding：

```text
query-product semantic score
product 512D ULIP embedding
product category embedding
product style/material embedding or multi-hot
product verified dimensions + known masks
candidate spot width/depth/area/wall relation
product-to-spot slack and rule evidence
room type/style feature
anchor category and relative geometry
nearby existing furniture category/size/relative pose summary
candidate yaw modulo footprint symmetry
```

product dimension 會作為 feature 提供 context，但 hard pass/fail 仍由規則直接計算。

### 9.2 模型分階段

#### D0：無學習 baseline

```text
semantic cosine
+ category compatibility table
+ deterministic relation score
+ hard geometry gate
```

這是必須打敗的最低對照。

#### D1：輕量 MLP/bilinear Spatial Reranker

- 先凍結 ULIP backbone，使用現有 512D product embedding。
- 對 product/scene/relation features 做 projection。
- edge-feature MLP + attention pooling 彙總 nearby objects。
- 輸出 style score、relation logits、pairwise preference score。
- 小模型先證明數據有訊號，不立即上 Graph Transformer。

#### D2：Graph/Transformer（有明確增益才做）

- node：existing furniture + candidate product。
- edge：relative geometry + relation type + confidence。
- 只在 D1 已證明 relation/context 能改善 room-disjoint evaluation 後實作。

#### D3：選擇性 joint fine-tuning（非 MVP）

- 只在 domain gap 分析顯示 frozen ULIP 限制 spatial transfer 時，對 product projection 或 PointBERT 最後層做小幅 fine-tune。
- 必須使用 replay/retention loss 保護 IKEA semantic retrieval。
- 若 Recall@K 顯著倒退，不 promotion。

### 9.3 Loss

MVP 主 loss：

```text
L_total
  = λ_style * L_style_contrastive_or_pairwise
  + λ_relation * L_relation_multilabel
  + λ_rank * L_pairwise_or_listwise_preference
  + λ_retention * L_semantic_retention   # 只在 joint fine-tune 時
```

明確不作 production 主 loss：

```text
L_size_fit
L_collision
L_path_validity
```

這些結果由 exact rule engine 產生；若做 learned auxiliary，只屬研究 ablation，不決定 final validity。

### 9.4 Sample weights

```text
human gold              1.00
verified geometry label 1.00  # 用於 relation fact/rule eval
3D-FRONT scene GT       0.70  # synthetic preference proxy
curated source metadata 0.60
VLM silver              confidence-dependent, <= 0.50
template weak label     0.30
```

數值為初始建議，不是已凍結 hyperparameters；會在 val 上比較並完整記錄。

### 9.5 Split 與 leakage

- 先以 3D-FRONT house/design group 切 train/val/test，再產生 room samples、relations、negatives、queries。
- 同一 house 不得出現於不同 split。
- 同一 exact 3D-FUTURE model 的 instance 若會造成泄漏，需額外 model-group split 或報告 seen/unseen-model 兩軌。
- relation threshold 只從 train 統計。
- corpus 同樣依 split 建立；test house/model 的 exact facts 不得流入訓練 corpus。
- V2T 真實 final test 不用於選 checkpoint、調 threshold 或 prompt。

---

## 10. Spatial corpus / RAG 打算怎麼做

### 10.1 Corpus 不是 raw scene dump

不會把 6,813 份 JSON 原文直接丟進 vector DB。預計由驗證後的 structured facts 產生可重現 corpus：

1. **Product facts**：IKEA product name/category/style/material/verified dimensions/function。
2. **Category-room priors**：例如 coffee table 在 living room 的出現統計。
3. **Category-pair relation priors**：例如 coffee_table ↔ sofa 的 edge-distance/relative-bearing distribution。
4. **Style/material compatibility summaries**：從 training split 聚合，並標記 weak/synthetic source。
5. **Exceptions and constraints**：例如 front 未驗證不允許宣稱 facing。

每篇 document 帶：

```text
document_id
document_type
source_split
source_scene_ids or product_ids
structured_fact
natural_language_rendering
confidence
generator_version
hashes
```

### 10.2 RAG runtime 角色

RAG 可用於：

- query expansion/category normalization。
- 取得房間類型與家具關係的統計 context。
- 給 reranker 提供可追溯 knowledge features。
- 產生回覆解釋。

RAG 不可用於：

- 覆蓋 exact product dimensions。
- 判定 collision-free。
- 判定門能不能開。
- 證明走道/path clearance。
- 把 unknown 說成 pass。

### 10.3 Promotion gate

必須作三組對照：

```text
A. Semantic ULIP + rules
B. Semantic ULIP + rules + Spatial Reranker
C. Semantic ULIP + rules + Spatial Reranker + Spatial RAG
```

C 若不穩定優於 B，Spatial RAG 保留為實驗 artifact，不設 production default。這沿用現行 IKEA RAG 未擊敗 vanilla 就不上線的準則。

---

## 11. V2T 與 3090 新介面草案

### 11.1 5090 → 3090：Spatial request

```json
{
  "schema_version": "spatial_candidates.request.v1",
  "request_id": "...",
  "room_id": "...",
  "query_text": "a compact coffee table beside the sofa",
  "room": {
    "room_type": "living_room",
    "style": ["modern", "warm wood"],
    "style_provenance": {},
    "coordinate_frame": {
      "unit": "m",
      "floor_axes": ["x", "y"],
      "up_axis": "z",
      "origin": [0.0, 0.0, 0.0]
    }
  },
  "existing_furniture": [
    {
      "instance_id": "sofa_01",
      "category": "sofa",
      "center_xy_m": [2.1, 1.8],
      "size_xy_m": [2.0, 0.9],
      "yaw_rad": 0.0,
      "front_status": "unverified",
      "front_yaw_rad": null,
      "geometry_confidence": 0.94
    }
  ],
  "candidate_spots": [
    {
      "spot_id": "spot_01",
      "center_xy_m": [2.1, 0.8],
      "max_footprint_m": [1.2, 0.7],
      "allowed_yaws_rad": [0.0, 1.5708],
      "anchor": {
        "instance_id": "sofa_01",
        "category": "sofa",
        "relation": "near"
      },
      "wall_relation": "none",
      "clearance_m": 0.4,
      "quality_flags": []
    }
  ],
  "refs": {
    "occupancy_grid": "...",
    "room_geometry": "..."
  },
  "producer_pins": {}
}
```

原則：3090 不必須線上接收完整 dense occupancy tensor 才能做輕量 rerank；5090 保留最終 full-grid safety check。但 compact request 必須帶來自 occupancy/geometry 的可騗證摘要與 references。

### 11.2 3090 → 5090：Spatial candidate response

```json
{
  "schema_version": "spatial_candidates.response.v1",
  "request_id": "...",
  "deployment": {
    "semantic_profile_id": "...",
    "spatial_model_id": "...",
    "semantic_checkpoint_sha256": "...",
    "spatial_checkpoint_sha256": "...",
    "spatial_data_manifest_sha256": "...",
    "spatial_corpus_sha256": "..."
  },
  "verified_pass": [
    {
      "product_id": "...",
      "spot_id": "spot_01",
      "center_xy_m": [2.1, 0.8],
      "yaw_rad": 0.0,
      "dimensions_m": {"width": 1.0, "depth": 0.55, "height": 0.42},
      "scores": {
        "semantic": 0.82,
        "style": 0.73,
        "relation": 0.88,
        "preference": 0.78,
        "final_rank": 0.81
      },
      "rule_evidence": {
        "size_fit": "pass",
        "footprint_containment": "pass",
        "collision": "pending_5090_final_check"
      },
      "reason_codes": []
    }
  ],
  "unknown_fallback": [],
  "rejected": [],
  "warnings": []
}
```

`final_rank` 在沒有完成 calibration 前是 ranking score，不是 probability。

### 11.3 5090 final placement output

```json
{
  "schema_version": "placement_result.v2",
  "request_id": "...",
  "selected": {
    "product_id": "...",
    "spot_id": "...",
    "center_xy_m": [2.1, 0.8],
    "yaw_rad": 0.0,
    "final_score": 0.81,
    "breakdown": {
      "semantic": 0.82,
      "style": 0.73,
      "relation": 0.88,
      "preference": 0.78,
      "size_fit": "pass",
      "collision": "pass",
      "walkability": "pass"
    }
  },
  "candidates": [],
  "rejected": [
    {
      "product_id": "...",
      "spot_id": "...",
      "reason_codes": ["COLLISION_WITH_EXISTING_OBJECT"]
    }
  ],
  "consumer_pins": {}
}
```

---

## 12. 預計新增的 code modules

審查通過後，預計放在獨立 namespace，避免污染現行 serving：

```text
/home/kyzen/ULIP_RAG/ikea/spatial_v2/
├── schemas/
│   ├── room.schema.json
│   ├── object.schema.json
│   ├── relation.schema.json
│   └── placement_pair.schema.json
├── audit_front3d_join.py
├── parse_front3d.py
├── coordinate.py
├── taxonomy.py
├── geometry.py
├── derive_relations.py
├── generate_negatives.py
├── build_spatial_corpus.py
├── dataset.py
├── models/
│   ├── baseline.py
│   └── spatial_reranker.py
├── train_spatial_reranker.py
├── evaluate_spatial_reranker.py
├── service_adapter.py
└── tests/
    ├── test_join.py
    ├── test_coordinate.py
    ├── test_geometry.py
    ├── test_relations.py
    ├── test_split_leakage.py
    └── test_contract.py
```

參考 implementation：

- [3D-FRONT paper/project](https://arxiv.org/abs/2011.09127)
- [3D-FUTURE paper](https://arxiv.org/abs/2009.09633)
- [NVIDIA ATISS](https://github.com/nv-tlabs/ATISS)：ATISS 的官方實作，有 3D-FRONT/3D-FUTURE parser 與 preprocessing；不等於「3D-FRONT 資料集官方 GitHub」。
- [MIT-SPARK ThreedFront](https://github.com/MIT-SPARK/ThreedFront)：第三方 standalone parser，來自 ATISS 路線，可作 parser 對照。
- [3D-FUTURE Toolbox](https://github.com/3D-FRONT-FUTURE/3D-FUTURE-ToolBox)：3D-FUTURE 工具庫參考。

不會盲目全量複製 ATISS；先 pin commit，以 parser 與 preprocessing 當參考，再依本專案 schema 做最小可審計實作。

---

## 13. 測試與驗收設計

### 13.1 資料層

- 100% source file 進 manifest，無靜默略過。
- join coverage 、missing/invalid reason codes 可重現。
- coordinate transform round-trip test。
- bbox/footprint 與 transformed mesh extent 一致性測試。
- category mapping coverage/ambiguous/excluded 統計。
- split house/model leakage = 0。
- derived row counts 與 manifest hashes 在 `/readyz` 或 training preflight 全驗證。

### 13.2 模型層

| 任務 | 指標 |
|---|---|
| relation | macro/micro F1、per-category-pair F1、abstain accuracy |
| style | category-controlled pair AUC、pairwise accuracy、per-style slice |
| candidate ranking | MRR、NDCG@K、Recall@K、pairwise agreement |
| unseen model | seen/unseen 3D-FUTURE model 分軌報告 |
| calibration | score reliability，在校準前不宣稱機率 |

### 13.3 System A/B

```text
A. Semantic only
B. Semantic + rules
C. Semantic + rules + style
D. Semantic + rules + relation
E. Semantic + rules + style + relation + preference
F. E + Spatial RAG
```

必報 safety metrics：

```text
known-size false-safe: 0
post-rule collision violations: 0
outside-floor violations: 0
unknown mixed into verified-pass: 0
```

學習模型如未贏過 B，不 promotion。Spatial RAG 如未贏過 E，不 promotion。

### 13.4 V2T 真實 E2E

需報告：

- request 成功/無結果/回退/失敗數。
- Top-K 商品 category/style 正確性。
- hard-valid product×spot pairs 數。
- final placement 成功率。
- collision/outside/blocked-path rejection reasons。
- 人工 pairwise preference agreement。
- 以 physical-room group 分割，不把同房不同 capture 當獨立 test。

`0a7cc12c0e` 是 dev/E2E fixture，不是 final test。`0707myroom` 與 `804967064_058238` 維持 scale-QA quarantine，直到 5090 解除。

V2T 實際 inventory 已知不是早期所說的「只有 7 場景」；後續盤點為 48 raw capture IDs、47 provisional physical groups、9 completed DA3 captures。physical-room grouping 與 final split 仍要由 5090 快照最終凍結。

---

## 14. 儲存與計算策略

### 14.1 不會做的事

- 不複製 3D-FUTURE mesh 進 3D-FRONT 目錄；目前用 relative symlink 共用。
- 不為所有 room 預先產生高解析多視角 render。
- 不為每個 transformed instance 複製 OBJ/GLB。
- 不保存大量重複 dense occupancy tensor，優先稀疏或可重建格式。

### 14.2 預估資料量原則

- JSONL manifests/relations 預期在 GB 級，但先以 100 scenes dry-run 實測外推。
- product/room features 使用 float16/float32 並建 ordered manifest。
- point samples 如非訓練必要不重複存放；由 model path 現場採樣或建去重 cache。
- render 為最高儲存風險，只做小型 style A/B 後再決定是否擴充。
- P300 剩 226G，預計 derived artifact >30G 前必須再次回報。

### 14.3 3090 Ti 訓練策略

- 預先離線計算可重用 product/scene features。
- MLP/bilinear MVP 使用 AMP、pin memory、persistent workers，並實測 batch size，不只看 GPU utilization 瞬時值。
- 記錄 wall-clock start/end/duration、GPU/VRAM、batch size、throughput、checkpoint metrics。
- 正式 run 之前先用小 split profile dataloader/CPU/IO bottleneck。
- 不跨機搬 compiled extension，每台機器用各自 environment/build。

---

## 15. 實作順序與每階段交付物

### Gate 0：Claude 審查（現在）

交付：本文件。  
尚不執行：clone/vendor parser、preprocess、train、service changes。

### Gate 1：Raw audit

交付：

```text
front3d_inventory.json
join_audit.jsonl
coordinate_audit.md
taxonomy_mapping_draft.json
storage_estimate.json
```

驗收：全量 source 都有決定性 status/reason；無靜默略過。

### Gate 2：Canonical dataset dry-run

先做 100 scenes，交付 schema、manifests、debug plots、relation distributions、外推容量。Claude/人工抽查後才全量跑。

### Gate 3：Full canonical dataset

全量 build，完成 split leakage check、hashes、data-quality report。此階段還不涉及 learned model。

### Gate 4：Baselines

完成 Semantic-only、Semantic+rules、relation-rule baseline，凍結 evaluation harness 與 primary metrics。

### Gate 5：MLP/bilinear Spatial Reranker

訓練 style/relation/preference heads，用 val 選 checkpoint，test 只在選定後跑一次。記錄完整訓練時間與資源。

### Gate 6：Spatial corpus/RAG A/B

建立 split-safe corpus，比較 with/without RAG。RAG 未穩定增益則不 promotion。

### Gate 7：Service contract

保留 v2，建新 endpoint/profile，完成 schema validation、auth、hash pins、negative tests、`/readyz`。

### Gate 8：5090 真實 E2E

5090 以真實 V2T bundles 執行 retrieval → candidate expansion → rule gate → spatial rerank → final revalidation，回傳 immutable artifacts 與 validators。

### Gate 9：Human gold 與 promotion

對低信心/模型歧異/高風險類別做人工 pairwise preference 與 front 標註。以預註冊指標決定 production promotion。

---

## 16. MVP 與完整版的邊界

### 16.1 MVP

MVP 必須能：

- 從 3D-FRONT/FUTURE 生成穩定、可審計的 room/object/relation dataset。
- 以 IKEA semantic Top-M 為 production candidate products。
- 排除已知尺寸不合、超出 floor、碰撞候選。
- 在 hard-valid 候選中使用 style + near/beside/against_wall/corner/aligned/perpendicular/around 做排序。
- 在未知尺寸時 fail closed 並單獨回傳 fallback。
- 以 V2T 真實場景完成一條可重現 E2E。

MVP 不承諾：

- semantic facing/front 已可信。
- 抽屜、椅子拉出、門扇 swing clearance 全部完成。
- 純學習模型能取代 geometry rules。
- 3D-FRONT synthetic preference 等於真人偏好。

### 16.2 完整版

需再加：

- IKEA product canonical-front 四態標註。
- V2T existing-instance front/yaw confidence。
- door/window/swing/access terminal 可靠輸出。
- drawer/chair/bed usable-side 與 clearance metadata。
- room image ↔ product visual style model。
- 更大真實 room-disjoint gold set。
- 簡單 model 有確實增益後的 graph model。

---

## 17. 人工標註與 VLM 的分工

### 17.1 VLM 可以先做

- room style/material/color silver tags。
- product style/function silver tags。
- 合法候選之間的初步 pairwise preference。
- 對 query 做 paraphrase。
- 標記異常案例供人工審查。

### 17.2 人工仍需要

- 對 VLM 低信心/歧異樣本裁決。
- 最終人類偏好 gold test。
- canonical front 的 `verified/symmetric/ambiguous/not_applicable`。
- 對特殊形狀（sectional sofa、L-shaped desk 等）做品質複核。
- 維持一小部分隨機 audit set，估計 VLM silver label error rate。

人工不需要手工重複標註「已知 polygon 是否重疊」或「1.2 m 是否大於 1.0 m」，這些由規則精確生成。

---

## 18. 尚未確認 / P0 審查項目

以下內容本文不自行宣稱已定案：

1. 3D-FRONT raw coordinate axis、unit 與 quaternion convention。
2. `replace_jid`/`replace_bbox` 的正式使用規則。
3. 全量 scene child → 3D-FUTURE model join coverage。
4. 哪一個 ATISS commit 作為參考 pin，以及是否只參考邏輯或 vendor 部分 parser。
5. 使用 ATISS 官方 split 或自建 house-group split；必須先確認與本地 6,813 JSON 的相容性。
6. 42-category → V2T/IKEA taxonomy mapping。
7. 3D-FUTURE model canonical front 是否可信。
8. door/window/access fields 的可靠覆蓋率。
9. V2T compact context 的最終 schema、哪一側做 candidate expansion，以及新 endpoint/port。
10. V2T physical-room grouping 與 final test/calibration split。
11. human gold 的人力、樣本數與預註冊 acceptance threshold。
12. 3D-FRONT/FUTURE 條款允許非商業研究；本專案組織身分、未來用途與 derived weights 是否符合，仍需由使用者/機構確認。IKEA 資料條款是另一件事。
13. 3D-FRONT textures/3D-FUTURE renders 是否進 MVP；我的建議是不進 placement MVP，先做小型 style A/B。

---

## 19. 請 Claude 逐項回覆

請使用 `ACCEPT / MODIFY / REJECT / NEED_EVIDENCE` 回覆：

1. 是否接受「Semantic ULIP 保留 + 新 Spatial Reranker + deterministic hard gate」的總體分工？
2. 是否接受 3D-FRONT/FUTURE 只當 synthetic supervision，IKEA 仍是 production gallery？
3. 是否接受先做 full join/coordinate/taxonomy audit，不直接開訓？
4. parser 計畫是否接受「ATISS pin 作參考 + 本專案最小可審計 parser」？
5. canonical derived schemas 是否缺欄位？特別是 coordinate、front、quality/provenance。
6. relation ontology、category-pair quantile threshold 與 front-dependent labels 延後是否接受？
7. 負例分 hard-invalid 與 hard-valid preference-negative 是否接受？
8. MLP/bilinear + edge attention 先行，Graph Transformer 只在增益後導入，是否接受？
9. loss 是否同意只主訓 style/relation/preference，size/collision/path 不當 production learned heads？
10. house-group/model-group split 與 corpus split 防泄漏是否足夠？
11. 線上 spatial inference 應放在 3090 新 endpoint，還是只傳 product feature 讓 5090 本地 rerank？請明確選擇與理由。
12. 是否同意保留 `/v2/product-candidates`，不在尚未 E2E 驗證時改它的 contract？
13. Spatial corpus/RAG A/B 是否同意放在 Spatial Reranker 之後，而非 preprocessing 一完成就當 production 主路？
14. Gate 0→9 的實作順序、stop gates 與 promotion metrics 是否需修正？
15. 請列出任何你認為仍有 circular labels、domain leakage、synthetic bias 或無法被線上 V2T 使用的設計。

---

## 20. 審查前停止線

在 Claude 回覆前，我不會：

- 修改現有 `/v2/product-candidates` implementation/profile。
- 修改 MongoDB、IKEA vectors/checkpoints 或 deployment bundles。
- 啟動 full 3D-FRONT preprocessing。
- 開始 Spatial Reranker/RAG 訓練。
- 將未驗證 axis/front/join 當成事實。
- 產生大量 render 或耗盡 P300 剩餘空間。

審查通過後，第一個實際動作只會是 **Gate 1 raw audit**；該階段仍不訓練。

