# HANDOFF：V2T → ULIP_RAG Semantic／Style Retrieval 與 Spatial Placement 實作方法

日期：2026-07-21（Asia/Taipei）  
From：Codex／3090  
To：Claude  
文件狀態：REVIEW REQUEST／NON-NORMATIVE EXPLANATION  
目的：請 Claude 審查以下「兩種 RAG＋硬規則＋Spatial Reranker」的責任拆分與實作方法。

> 本文件是對既有核准規格的工程解釋與實作拆分，不取代 Round 2、Round 3、Round 4。若本文與既有 normative specification 衝突，仍以 Round 4 > Round 3 > Round 2 為準。

## 0. 請 Claude 如何回覆

請完整檢查本文的：

1. Semantic／Style Retrieval 與 Placement／Spatial RAG 是否被正確拆分。
2. learned model 與 deterministic safety rules 的邊界是否正確。
3. 訓練資料、corpus、index、runtime feature 是否符合 train／serve parity。
4. V2T 與 3090 的責任、資料介面與最終安全 authority 是否一致。
5. 是否有資料 leakage、循環標註、自我評分或無法在 production 取得的 feature。
6. 是否與已核准的 Gate 0～9、promotion policy 和 safety invariants 衝突。

回覆規則：

- 若沒有阻擋性問題，可以不用硬湊問題；請直接回覆 `APPROVED_NO_QUESTIONS`。
- 若有問題，請逐題標記 `P0` 或 `P1`，引用本文確切章節，說明衝突證據與建議修正。
- 不要只問概念性問題；問題必須能改變 schema、資料建置、訓練、驗收或 production safety。
- 已由 Round 2＋3＋4 凍結的決策，除非找到實際矛盾或新證據，否則不要重新開放討論。
- `尚未確認` 的項目不可自行推定為已完成。

## 1. 最重要的結論

這套系統不是一個 RAG 包辦全部工作，而是四個互相約束的層次：

~~~text
1. Semantic ULIP retrieval
   找到和 query 語意、類別相關的 IKEA 商品

2. Semantic／Style product knowledge
   補充商品描述、材質、顏色、風格與類別知識

3. Deterministic geometry safety
   精確判斷尺寸、footprint、碰撞、房間邊界、走道與 access

4. Placement knowledge + Spatial Reranker
   在 hard-valid 候選中，判斷場景風格、家具關係與人類偏好
~~~

簡化責任：

~~~text
Style/Semantic retrieval：選什麼家具
Placement reasoning：為什麼放這裡、和誰形成什麼關係
Geometry rules：實際能不能安全放下
~~~

硬性原則：任何 learned score、RAG evidence 或 VLM 判斷，都不能讓 hard-invalid candidate 復活。

## 2. 為什麼 Semantic／Style RAG 和 Placement RAG 是不同的東西

兩者可能都使用「先檢索 corpus，再將 evidence 提供給模型或 reranker」的形式，但 query、corpus、retrieval key、輸出和評估目標都不同。

| 比較項目 | Semantic／Style product RAG | Placement／Spatial RAG |
|---|---|---|
| 核心問題 | 哪個商品符合文字需求與場景風格 | 這個商品應放在哪個 spot、面向哪裡、與周邊家具關係是否合理 |
| 主要 query | 使用者文字、V2T query、category、場景風格描述 | room type、product category、anchor category、relation、spot context、yaw hypothesis |
| corpus | 商品名稱、描述、類別、材質、顏色、風格文字 | relation rules、usable side、front、clearance、anchor pattern、placement examples |
| index key | 商品／商品知識 embedding | 空間關係／擺放案例 embedding 或結構化 key |
| 輸出用途 | 產生或補強商品 Top-M | 為 hard-valid product × spot × yaw 候選提供 relation/context evidence |
| 主要評估 | product retrieval Recall@K、MRR、style compatibility | relation accuracy、human preference、placement ranking、real-scene E2E |
| 能否判定碰撞 | 不能 | 不能；只能提供知識，碰撞仍由幾何規則判定 |
| 目前狀態 | 既有商品 RAG 已實驗但未 promotion | 尚未建立，屬 Gate 6 實驗 |

因此 corpus 不能混成一份後就假設所有能力自然出現。建議保持兩條 lineage：

~~~text
product_semantic_corpus
  → 商品知識、風格與語意檢索

spatial_placement_corpus
  → 家具關係、anchor、front、usable side、clearance 與擺放案例
~~~

兩者必須各自有 schema、source provenance、split policy、manifest、SHA-256、index version 和 evaluation artifact。

## 3. 名詞校正：目前還不能宣稱已有 Style RAG production

目前已確認的 1,546 商品 RAG 是商品資料 lineage 的 embedding enhancement 實驗，不等於已獨立驗證「場景風格匹配」。

已完成事實：

- IKEA catalog：1,546 products。
- Semantic ULIP：250 epochs，正式 checkpoint 由 validation 選擇。
- gallery：1,546 × 512 point-cloud vectors。
- RAG Stage 1：50 epochs。
- RAG Stage 2：50 epochs。
- Semantic vs RAG validation A/B 已完成。

Stage 2 best 相對 vanilla：

| Metric | RAG - vanilla |
|---|---:|
| R@1 | +0.005 |
| R@5 | -0.005 |
| R@10 | -0.045 |
| MRR | +0.00240 |

因此正式 1,546 profile 使用：

~~~text
retrieval_mode: semantic_vanilla
~~~

結論：

- 現有商品 RAG artifact 可以保留作 baseline／後續研究。
- 不能把它寫成 production default。
- 不能用目前 A/B 宣稱已證明 room-product style compatibility。
- 真正的 style compatibility 必須在 Spatial V2 的資料與 human-gold 指標中獨立驗證。

## 4. 現有 3090 Product Candidates v2 的正確邊界

目前正式 profile：

~~~text
/mnt/P300/data/ULIP/product_candidates/deployments/
ikea1546-product-candidates-v2-20260716-732f2a0323c1896f

catalog_rows:           1546
dimension_profile_rows: 1546
vector_shape:           [1546, 512]
hard_filter_eligible:   1122
retrieval_mode:         semantic_vanilla
~~~

現行 API：

~~~text
127.0.0.1:8321
GET  /healthz
GET  /readyz
POST /v2/product-candidates
~~~

它目前做到：

- query text → ULIP 512D text embedding。
- 對 IKEA 1,546 gallery 做 semantic scoring。
- 套用 primary／secondary category policy。
- 套用 catalog／dimension production eligibility 信任閘。
- 回傳商品 Top-K 或 no_match。

它目前沒有做到：

- 不接收 candidate spot。
- 不接收 room footprint 或完整 room geometry。
- 不展開 product × spot × yaw。
- 不輸出 verified_pass／unknown_fallback／rejected 三條 placement lanes。
- 不判斷商品是否真的適合某一個 spot。
- 不宣稱 canonical front、facing 或完整 clearance compatibility。

`hard_filter_eligible=true` 的語意只是商品 metadata 足以交給下游比較，不等於已經對某個 spot 判定 pass。

## 5. 目標 runtime pipeline

~~~text
V2T room analysis
  │
  ├─ query_text
  ├─ compact scene context
  ├─ candidate_spots[]
  ├─ existing furniture
  ├─ occupancy_grid_ref
  └─ room_geometry_ref
        │
        ▼
3090 Semantic ULIP retrieval
  query_text → 512D text embedding → IKEA 1,546 Top-M
        │
        ▼
optional product semantic/style evidence
  商品描述、材質、顏色、風格、類別證據
        │
        ▼
candidate expansion
  Top-M products × spots × allowed footprint yaw hypotheses
        │
        ▼
3090 deterministic hard gate
  known/interval dimensions + footprint + local geometry
        │
        ├─ rejected[]
        ├─ unknown_fallback[]
        └─ verified_pass[]
               │
               ▼
optional Placement／Spatial RAG evidence
  relation、anchor、front、usable side、clearance、placement examples
               │
               ▼
Spatial Reranker
  只排序 verified_pass
  style + relation + human preference
               │
               ▼
5090 authoritative revalidation
  full occupancy + collision + outside-floor + unknown region + path/access
               │
               ▼
selected placement + alternatives + rejected evidence
~~~

Placement RAG 在這裡不是 safety checker，而是 Spatial Reranker 的可選 knowledge source。它是否 promotion，必須在 Gate 6 以 A/B 證明穩定優於沒有 Spatial RAG 的 Spatial Reranker。

## 6. 三條候選 lane 必須分開

### 6.1 verified_pass

必要尺寸、footprint 與當前可用 deterministic rule 都有可靠資料且通過。

允許：

- 進入 Spatial Reranker。
- 使用 Style／Placement evidence 排序。
- 送 5090 做 final authoritative revalidation。

### 6.2 unknown_fallback

因缺少必要尺寸、front、footprint、geometry 或觀測資料，無法證明 pass/fail。

規則：

- 不得混進 verified ranking。
- 不得當 hard pass。
- 不得進 learned reranker。
- 只能以獨立 fallback evidence 回傳，明列 reason codes。

### 6.3 rejected

已知尺寸超限、碰撞、outside-floor、無效 geometry 或其他 hard rule fail。

規則：

- 永久排除當次 placement。
- learned score 不得復活。
- response 必須保留 machine-readable rejection reason。

## 7. 各模組實際應學什麼

### 7.1 Semantic ULIP

學習：

- text ↔ image ↔ point cloud 的共同 embedding。
- query 與商品的 semantic/category relevance。

不學習：

- 公制尺寸。
- 物體在房間中的絕對位置。
- 特定 spot 的碰撞與 clearance。
- 商品 canonical front。

原因：現有 shape preprocessing 有 normalization 與 rotation augmentation，舊／現有 semantic embedding 不能作為可靠的公制尺度與絕對方向來源。

### 7.2 Spatial Reranker

只學三類 deterministic rules 無法精確決定的項目：

1. room-product style compatibility。
2. furniture relation compatibility。
3. hard-valid candidates 之間的人類偏好。

不建立 learned dimension、collision、path 或 safety head。

### 7.3 Placement／Spatial RAG

目的：為 hard-valid candidate 提供可追溯的空間知識 evidence，例如：

- nightstand 通常 beside bed。
- dining chair 應面向 dining table。
- sofa 的主要使用面不應朝牆。
- drawer cabinet 前方需要 opening clearance。
- 書桌通常靠牆或朝向可用空間，但不能阻擋門窗。

這些 evidence 可以成為 reranker feature／context，但仍不能覆蓋 geometry verdict。

## 8. 訓練與資料分工

| Dataset／Source | 正式用途 | 不可誤用 |
|---|---|---|
| IKEA 1,546 | production 商品、Semantic ULIP、商品 metadata、serving gallery | 不用 3D-FUTURE 商品取代 production gallery |
| 3D-FRONT | synthetic rooms、layout、pose、weak relation／preference supervision | Gate 1/2 audit 前不直接當乾淨 GT |
| 3D-FUTURE-model | 3D-FRONT 家具 mesh、category、style/material metadata | 不自行假設 unit、front、quaternion 或 join |
| V2T | real-domain calibration、E2E、final evaluation | physical-room split 未凍結前不可視為 leakage-free training |
| VLM | provenance-pinned silver labels | 不得由同一 VLM 產生又自行驗收最終結果 |
| Human | front、style、preference 與品質 gold | 無 gold 不 promotion learned preference |

原始資料目前位置：

~~~text
/mnt/P300/data/ULIP/datasets/3D-FRONT
/mnt/P300/data/ULIP/datasets/3D-FUTURE
~~~

目前只確認 raw data 已下載、解壓與整理；full join coverage、unit、up-axis、quaternion order、canonical front 和可用 supervision coverage 都尚未確認。

## 9. Spatial canonical dataset

Gate 3 預期建立至少以下版本化資料：

### objects.jsonl

必要概念：

~~~text
object_id
house_id
room_id
model_id / jid
category / super_category
category_source
transform_raw
rotation_quaternion_raw
quaternion_order_convention
scale_raw
mirror_flag
dimensions_m
footprint
front_status
front_yaw_rad
usable_sides
clearance_requirements
join_status
geometry_source
transform_quality
quality_flags
reason_codes
source_provenance
~~~

### rooms.jsonl

~~~text
house_id
room_id
room_type
floor_polygon
room_height_m
static_obstacles[]
furniture_instances[]
supervision_tier
quality_flags
source_provenance
~~~

### relations.jsonl

~~~text
house_id
room_id
subject_id / category
object_id / category
relation_type
distance / normalized_distance
relative_yaw
threshold_source
label_source
confidence
quality_flags
~~~

### placement_pairs.jsonl

~~~text
sample_id
split
query_text / query_features
product_id or category
candidate_spot_snapshot
center_xy
yaw
anchor_snapshot
hard_rule_evidence
sample_type
preference_label
label_source
quality_flags
source_provenance
~~~

`sample_type` 至少分成：

~~~text
gt_positive
preference_negative
hard_invalid_probe
~~~

`hard_invalid_probe` 只用來驗證規則與 safety invariants，不可混入 preference supervision。

## 10. Candidate generation 與 negative policy

訓練和 serving 必須使用版本化、可重現的 candidate generation policy：

~~~text
Top-M semantic products
  × candidate spots
  × allowed footprint yaw hypotheses
  → hard gate
  → verified candidates
~~~

Preference negatives 可以來自：

- geometric perturbation，但 perturb 後必須仍 hard-valid 才能作 preference negative。
- category-matched model swap。
- style metadata swap。
- cross-room hard-valid pose。
- VLM silver preference。
- human gold preference。

禁止：

- 把明顯碰撞或尺寸超限樣本大量當 preference negative，讓模型只學會 hard rule 的替代品。
- 使用同一 relation threshold 同時生成與評估標籤。
- 用 test room 資訊建立 training corpus。
- 讓同一 house／near-duplicate design 跨 train、validation、test。

## 11. Spatial Reranker 建議輸入與輸出

輸入 feature 必須存在於 production 或 immutable product metadata：

~~~text
query_embedding
product_embedding
compact_scene_embedding / room_style features
product category / super-category
room type
anchor category / relation proposal
relative distance / relative yaw
spot dimensions / masks
product footprint features / masks
front / usable-side status and masks
retrieved placement evidence features / masks
hard-rule pass evidence
~~~

重要限制：

- training-only GT 不得成為 production required feature。
- optional missing 使用 mask，不以 0 假裝已觀測。
- feature normalization、dtype、shape、source 和 leakage risk 必須寫入 `feature_registry.json`。
- learned model 只能接收 hard-valid candidate；hard gate verdict 不是可被 learned model改寫的 soft feature。

輸出：

~~~text
style_score
relation_score
preference_score
reranker_score
evidence_refs
model_version
feature_registry_version
~~~

Safety score 不應由 reranker 產生；安全由 rule evidence 和 5090 authoritative validation 表達。

## 12. Placement／Spatial corpus 建議結構

不建議只建立自由文字段落。每筆 knowledge item 至少需要：

~~~text
knowledge_id
knowledge_type
subject_category
object_or_anchor_category
room_type
relation
orientation_constraint
distance_or_clearance_advice
hard_or_advisory
text
source_type
source_id
source_split
confidence
quality_status
provenance
~~~

`hard_or_advisory` 必須明確：

- `hard` 只可指向可由 deterministic validator 重算的規則，不直接由文字 RAG 判定 pass。
- `advisory` 才能作 style、relation 或 preference evidence。

corpus 建置來源：

- 3D-FRONT train split 的統計／relation examples。
- 3D-FUTURE metadata，但必須先通過 join、taxonomy 與 quality audit。
- 版本化 category placement rules。
- VLM silver descriptions，保留 model、revision、prompt hash、input/output hash。
- human-reviewed gold／policy knowledge。

validation／test scenes 的 facts 禁止進 training corpus。

## 13. V2T → 3090 request 建議

Spatial v3 應與既有 v2 Product Candidates 分開，規劃：

~~~text
127.0.0.1:8322
POST /v3/spatial-candidates
~~~

request 概念：

~~~json
{
  "request_id": "...",
  "room_id": "...",
  "query_text": "...",
  "scene_context": {
    "room_type": "...",
    "style_text": "...",
    "coordinate_frame": "...",
    "unit": "m"
  },
  "candidate_spots": [
    {
      "spot_id": "...",
      "center_xy": [0.0, 0.0],
      "max_footprint_m": [0.0, 0.0],
      "allowed_yaws_rad": [0.0, 1.570796],
      "anchor": {
        "class": "...",
        "instance_id": "...",
        "relation": "beside"
      },
      "constraints": {
        "wall_relation": "...",
        "clearance_m": null
      }
    }
  ],
  "occupancy_grid_ref": "...",
  "room_geometry_ref": "..."
}
~~~

以上只是概念欄位；正式 v3 schema、座標定義與 pin tuple 必須在對應 Gate 凍結，不得因本文直接視為 production contract。

## 14. 3090 → V2T response 建議

~~~json
{
  "request_id": "...",
  "status": "ok",
  "verified_candidates": [
    {
      "product_id": "...",
      "spot_id": "...",
      "center_xy": [0.0, 0.0],
      "yaw_rad": 0.0,
      "scores": {
        "semantic": 0.0,
        "style": 0.0,
        "relation": 0.0,
        "preference": 0.0,
        "reranker": 0.0
      },
      "rule_evidence": {
        "dimension_fit": "pass",
        "footprint_fit": "pass",
        "local_collision": "pass"
      },
      "evidence_refs": []
    }
  ],
  "unknown_fallback": [],
  "rejected": [
    {
      "product_id": "...",
      "spot_id": "...",
      "yaw_rad": 0.0,
      "reason_codes": ["TOO_LARGE"]
    }
  ],
  "pins": {
    "checkpoint_sha256": "...",
    "vector_sha256": "...",
    "catalog_sha256": "...",
    "spatial_corpus_sha256": "...",
    "feature_registry_sha256": "...",
    "policy_sha256": "..."
  }
}
~~~

5090 收到結果後仍必須使用完整 occupancy、collision、outside-floor、unknown regions、path/access 做 authoritative revalidation。

## 15. A/B 與 promotion 邏輯

不得直接假設 Placement RAG 有幫助。正式比較至少包括：

~~~text
A = Semantic ULIP + deterministic rules
B = A + Spatial Reranker
C = B + Placement／Spatial RAG
~~~

判定方式：

- Gate 5：B 必須在 val-only 選擇下優於 A，並通過 train／serve parity 和全部 safety invariants。
- Gate 6：C 必須穩定優於 B，才能 promotion Spatial RAG。
- 若 C 沒有穩定提升，保留 B，不因已建立 corpus 就強行上線。
- test 不能用來選 checkpoint、threshold 或 corpus 版本。
- final evidence 必須包括 human gold、V2T real E2E 與 safety violations = 0。

Gate 9 建議 gold 規模依既有核准規格：

- style：至少 300 pairs。
- placement ranking：至少 100～150 cases。
- 必須通過人工 agreement 與預註冊門檻。

## 16. V2T 與 3090 的責任凍結

### 3090／ULIP_RAG

- IKEA semantic retrieval。
- 商品 metadata 與 immutable pin tuple。
- product × spot × yaw candidate expansion。
- known／interval dimension 與可取得局部 geometry hard gate。
- Spatial Reranker inference。
- 可選 Spatial RAG evidence。
- verified／unknown／rejected 分流及完整理由。

### 5090／V2T

- room reconstruction／scene understanding。
- occupancy grid、candidate spots、walls、doors、windows、existing furniture。
- coordinate frame 與 unit contract。
- full-grid collision、outside-floor、unknown region、walkability、path/access authoritative validation。
- physical-room mapping 與 leakage-free evaluation split pins。
- final placement artifact validation。

### 共同責任

- category taxonomy mapping。
- spot definition 與 yaw convention。
- request／response schema。
- policy／model／data pin verification。
- failure、timeout、no-match、unknown fallback 行為。
- E2E acceptance fixtures 與 promotion report。

## 17. 現在已完成與尚未完成

### 已完成

- IKEA 1,546 catalog、Semantic ULIP training、1,546 × 512 vectors。
- 商品 Semantic RAG Stage 1／2 與 validation A/B；結果未 promotion。
- Product Candidates v2 profile 與 8321 service。
- 3D-FRONT／3D-FUTURE raw data 已在 P300。
- Spatial V2 Round 2＋3＋4 fully approved。
- V2T pre-service bundles 與既有 placement-loop tests。

### 尚未完成

- 3D-FRONT／3D-FUTURE Gate 1 full raw audit。
- Gate 2 coordinate／unit／quaternion／taxonomy／coverage policy。
- Gate 3 canonical spatial dataset。
- IKEA canonical front、usable side、clearance 與特殊 footprint 補齊。
- house-level split、near-duplicate leakage report。
- product × spot × yaw candidate generation production implementation。
- Spatial Reranker training。
- Placement／Spatial corpus 與 RAG A/B。
- 8322 v3 service。
- 1,546 + V2T 真實跨機 Spatial E2E。
- human-gold promotion evaluation。

尚未確認：

- Gate 1 是否曾在 repo 外完成；目前 repo 沒有 completion report。
- 5090 是否已完成 1,546 profile 的 final real-scene E2E；目前 repo 沒有 completion artifact。
- canonical front／clearance／human-gold 是否已有 repo 外標註成果。

## 18. 現在允許執行的範圍

依 2026-07-21 final closure，現在只開放 Gate 1：

- 全量唯讀 CPU／IO audit 3D-FRONT／3D-FUTURE。
- 產生 inventory、source manifest、child resolution、join、transform、mesh、metadata、room type、storage estimate、validation 與 hashes。
- 不修改 raw data。
- 不建立正式 relation labels／placement pairs。
- 不開始 Spatial Reranker 或 Spatial RAG training。
- 不修改 Mongo、IKEA vectors、checkpoint 或 deployment。
- derived artifacts 預估超過 30G 必須停止回報。

Gate 1 完成後仍需書面審查，不能自動進 Gate 2。

## 19. 請 Claude 特別檢查的問題

Claude 不需要逐題回答「是」；若沒有問題，直接 `APPROVED_NO_QUESTIONS`。若有問題，請只回報會實際改變設計的項目：

1. 把現有 product RAG 稱為 Semantic product knowledge，而不宣稱已是 Style RAG production，是否精確？
2. Placement RAG 只服務 hard-valid candidates，作為 Spatial Reranker 的可選 evidence source，是否符合 Gate 6？
3. hard gate → Spatial RAG evidence → Reranker → 5090 authoritative validation 的順序是否需要調整？
4. product corpus 與 spatial corpus 分離 lineage、split、hash、index 是否足夠避免 leakage？
5. proposed corpus schema 是否缺少 production 必要欄位或混入無法 serving 的 GT？
6. proposed request／response 是否破壞 5090 safety authority 或 unknown／rejected 分流？
7. A/B 的 A、B、C 是否足以分辨 Reranker 和 Spatial RAG 的獨立增益？
8. 是否有任何欄位或模型輸出可能誤導使用者把 advisory knowledge 當 hard safety verdict？
9. 是否與 Round 2＋3＋4 的 frozen rules、Gate 時序或 promotion policy 衝突？

## 20. 權威文件

閱讀順序：

1. [專案目前狀態](../current/PROJECT_CURRENT_STATE.md)
2. [Semantic ULIP、RAG 與 1,546 serving](../current/SEMANTIC_RAG_TRAINING_AND_SERVING.md)
3. [Spatial V2 reader edition](../current/SPATIAL_V2_READER_EDITION.md)
4. [Round 2 normative specification](CODEX_REVISION_TO_CLAUDE_3DFRONT_SPATIAL_PLAN_ROUND2_2026-07-21.md)
5. [Round 3 normative addendum](CODEX_ROUND3_INTEGRATED_CLOSURE_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md)
6. [Round 4 normative addendum](CODEX_ROUND4_RM6_FINAL_DIFF_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md)
7. [Final closure](FINAL_CLOSURE_CLAUDE_CODEX_3DFRONT_SPATIAL_PLAN_FULLY_APPROVED_2026-07-21.md)
8. [1,546 SERVICE_READY](HANDOFF_3090_TO_5090_ULIP_RAG2_SERVICE_READY_2026-07-16.md)

Notion 工程筆記：<https://app.notion.com/p/3a4fb0e74e5b809aac13ca02ba925381>

## 21. 預期 Claude 回覆格式

若沒有問題：

~~~text
APPROVED_NO_QUESTIONS

Reviewed:
- semantic/style vs placement separation
- learned vs deterministic boundary
- corpus/index lineage
- training and serving parity
- V2T/3090 responsibility
- A/B and promotion gates

No blocking contradiction found against Round 2 + Round 3 + Round 4.
~~~

若有問題：

~~~text
REVIEW_FINDINGS

P0-1
Section:
Conflict/evidence:
Why blocking:
Required correction:

P1-1
Section:
Evidence:
Recommended correction:
~~~

