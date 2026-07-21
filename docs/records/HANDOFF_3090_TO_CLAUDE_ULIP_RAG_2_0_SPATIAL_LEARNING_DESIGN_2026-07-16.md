# 3090 → Claude：V2T × ULIP_RAG 2.0 新家具庫、多模態風格、空間與關係學習設計

日期：2026-07-16（Asia/Taipei）  
3090 repo：`/home/kyzen/ULIP_RAG`  
5090 V2T source：`/home/kyzen/V2T_DA3`（3090 僅持有交接快照）  
狀態：**Round-2 revised design；已納入 Claude 2026-07-16 審查，等待使用者裁定四個 P0，尚未授權實作或爬取**

Claude 審查來源：

```text
/home/kyzen/ULIP_RAG/
CLAUDE_REVIEW_ULIP_RAG_2_0_SPATIAL_LEARNING_2026-07-16.md
```

---

## 0. 使用者最新裁定：先前設計的商品資料前提錯誤

使用者已明確更正：

1. **2.0 的家具庫已經不同，不能把現有 P300 IKEA 733 件當正式 2.0 商品庫。**
2. **新家具資料要重新抓取、重新整理、重新建立 corpus／multimodal assets、重新訓練並重建全部 vectors。**
3. **2.0 要同時處理：**
   - 家具與場景的風格匹配。
   - 商品與候選位置的匹配。
   - 商品實際大小與可用空間的匹配。
   - 候選家具與現有家具之間的相對關係。

所以舊資產的正確角色改為：

```text
舊 IKEA 733 + 舊 checkpoint/vectors
  = 1.0 技術初始化、pipeline baseline、regression reference

新家具 catalog
  = 2.0 唯一正式 product dataset / corpus / vector gallery
```

不得把新 catalog 與舊 733 vectors、metadata、caption、taxonomy 或 checkpoint lineage 混用。

本文件的標記：

- **[FACT]**：現有 repo/artifact/E2E 可直接證明。
- **[USER DECISION]**：使用者已明確裁定。
- **[PROPOSAL]**：3090 提案，待 Claude 審查。
- **[UNCONFIRMED]**：目前資訊不足，不能自行腦補。

---

## 1. 先直接回答：四種匹配能不能達到？

### 1.1 結論

**可以把四種能力都設計進 2.0，但目前系統尚未具備，也不能只靠重新生成 corpus 或重訓現有 ULIP 達成。**

目標模型必須從目前的：

```text
F(query, product)
```

升級成：

```text
F(
  query,
  room_style,
  product,
  room_geometry,
  candidate_spot,
  candidate_pose,
  existing_furniture_graph
)
```

能力與必要條件如下：

| 能力 | 2.0 可否做到 | 模型需要什麼 | 資料需要什麼 | 是否保留 deterministic rule |
|---|---|---|---|---|
| 家具 ↔ 場景風格 | 可以 | Room Style Encoder + Product Style Encoder + style fusion/ranking | 場景圖片/frames、商品多視角圖、style/material/color、scene-product正負配對或人工偏好 | 不適用；屬偏好排序 |
| 家具 ↔ 位置 | 可以 | exact geometry先產生hard-valid poses；learned Pose Preference Reranker只排序合法位置 | occupancy、spot、center、yaw、anchor/relation與人類偏好；validity由規則免費產生 | collision/outside-floor/path gate由規則全權負責 |
| 家具 ↔ 大小 | 可以，而且應精確計算 | Structured dimensions + deterministic fit engine；可把slack/proportion當偏好特徵，但不訓練模型重算fit | 組裝後 width/depth/height、footprint、spot extent、單位與可信度 | known size超限必須 hard reject |
| 家具 ↔ 家具相對關係 | 可以，但原設計需擴充 | Furniture Relation Graph / Graph Transformer + relation head | existing furniture instance、category、bbox/polygon、yaw/front、candidate相對位置與關係標註 | overlap等物理衝突仍 hard gate |

### 1.2 目前設計要新增的兩個核心模組

上一版只有 Product Spatial Encoder、Scene Encoder 與 Spatial Reranker，還不足以完整達成使用者需求。現在新增：

1. **Room–Product Style Alignment Model**
   - 直接比較房間視覺／風格與商品多視角視覺／屬性。
   - 不能只把 V2T 產生的 style words 加到 query 後做文字 cosine。

2. **Furniture Relation Graph Model**
   - 把現有家具當 nodes，relative geometry／semantic relation 當 edges。
   - 把新商品在 candidate pose 插入 graph，預測是否形成合理關係。
   - 例如：nightstand beside bed、coffee table in front of sofa、TV stand facing sofa、chair aligned with dining table。

### 1.3 不能過度承諾

以下能力只有在新資料完成並通過 held-out evaluation 後才能宣稱：

- 「風格搭配得好」需要 scene-product preference ground truth。
- 「位置最好」需要多個合法方案間的 preference/ranking labels。
- 「家具關係合理」需要 instance-level、yaw/front與 relation labels。
- 單純抓商品 catalog 不會自然產生 room placement supervision。

### 1.4 Claude 審查後凍結的「該學 vs 該算」邊界

Claude 提出的資源配置修正已接受，並精確化如下：

```text
規則負責（exact / fail-closed）：
  dimensions fit
  footprint containment
  observed-floor coverage
  body collision
  wall/door/unknown conflict
  deterministic path/clearance constraint（資料存在時）

模型負責（rules無法精確定義）：
  room-product style compatibility
  furniture-to-furniture functional relation
  多個hard-valid poses之間的人類偏好
  semantic/style/relation的context-dependent融合
```

重要精確化：結構化尺寸在數學上可以餵給MLP，但工程上沒有理由花標註與安全風險去近似一個已能精確計算的判定。2.0可以把規則產生的`slack_m`、clearance、wall distance、path-retention等數值當作**合法候選偏好特徵**，但不訓練模型翻案hard gate。

`size/collision/path validity heads`自production主路徑移除；若保留，只能是離線研究ablation，不消耗人工標註，也不能影響selection。

---

## 2. ULIP_RAG 1.0 在新 2.0 中的正確角色

### 2.1 1.0 不是新商品庫

**[FACT]** 現行 production semantic baseline 是 IKEA vanilla ULIP：

```text
query text
→ CLIP/SLIP text encoder
→ 512D query embedding
→ 733 × 512 point-cloud vectors
→ cosine Top-K
```

它的主要訓練 loss 只有：

```text
0.5 × bidirectional InfoNCE(point ↔ text)
+ 0.5 × bidirectional InfoNCE(point ↔ image)
```

沒有 dimensions、footprint、room、spot、pose、collision、style pairing或家具關係 loss。

另外，現有 IKEA loader會把每個 point cloud做中心化與unit-sphere normalization，訓練時還有random scale、shift、rotation與jitter。因此：

- 舊PointBERT embedding不能被假設保留商品實際公尺尺度。
- random rotation不等於已學會商品canonical front。
- 2.0的實際尺寸、front、yaw與clearance必須來自新structured spatial metadata，不能試圖從舊512D向量反推。

### 2.2 1.0 如何被使用

**[PROPOSAL]** 新 catalog 訓練時：

- 使用 1.0 的 text/image/PointBERT weights 作 initialization。
- 不使用舊 733 product rows 作新 production gallery。
- 舊資料可保留少量 replay 防止通用家具語意 catastrophic forgetting，但需明確標為 legacy replay。
- 新 catalog 重新訓練 semantic product embedding。
- 重新產生新 catalog 全部 point vectors、ordered manifest與deployment profile。
- 如使用 RAG，重新生成與新 catalog一致的 corpus/index並重新訓練 enhancer。

這才符合「以 1.0 為基礎做 2.0」：

```text
reuse learned weights / architecture knowledge
≠ reuse obsolete catalog / vectors as production data
```

### 2.3 現有 RAG 的角色

**[FACT]** 目前存在：

- historical Core RAG（ShapeNet/ModelNet研究線）。
- 2026-07-15 IKEA-specific RAG candidate。
- legacy1095、Core8256、881 vocab KB 等不同 corpus artifacts。

它們不能直接當新 catalog 的 2.0 corpus。新 catalog 必須建立新 profile：

```text
new catalog facts
+ new taxonomy
+ verified style/material/function facts
+ spatial-language knowledge
→ new corpus
→ new embeddings/index
→ new RAG training/provenance
```

---

## 3. 新 2.0 系統總體架構

### 3.1 模型總覽

```text
                           ┌──────────────────────────┐
user query ───────────────→│ Semantic / Intent Encoder│──→ q_sem, q_intent
                           └──────────────────────────┘
                                        │
new product text/image/PC ─→ Product Semantic Encoder ─→ p_sem
product attributes ────────→ Product Style Encoder ────→ p_style
product dimensions/footprint ─┐
room/spot/occupancy/doors ────┼→ Deterministic Geometry Engine
candidate center/yaw ─────────┘       │
                                      ├→ hard-invalid：reject + reason
                                      └→ hard-valid slate + exact metrics

V2T room frames ───────────→ Room Style Encoder ───────→ r_style
existing furniture graph ──→ Relation Graph Encoder ───→ g_relation
hard-valid pose/metrics ────→ Pose Preference Features ─→ z_preference

q_sem, q_intent, p_sem, p_style,
r_style, z_preference, g_relation
                  │
                  ▼
       Semantic–Style–Relation–Preference Fusion
                  │
      ┌───────────┼──────────────┐
      ▼           ▼              ▼
 semantic       style          relation / human pose preference
 relevance      match          compatibility
      └───────────┴──────┬───────┘
                         ▼
                listwise final ranker
                         │
                         ▼
           deterministic final revalidation
                         │
                         ▼
 product + spot + center + yaw + scores + explanations
```

### 3.2 建模單位

正式 candidate row：

```text
(request, product, room, spot, center, yaw, existing-furniture context)
```

同一商品在不同 room/spot/pose 的分數應不同；不能把「適合放哪裡」永久壓成一個靜態商品 embedding。

### 3.3 分解輸出

建議保留分解分數：

```json
{
  "semantic_relevance": 0.87,
  "scene_style_match": 0.81,
  "rule_evidence": {
    "hard_valid": true,
    "size_fit": "pass",
    "collision_free": true,
    "footprint_slack_m": [0.21, 0.12],
    "clearance_m": 0.42
  },
  "pose_preference": 0.79,
  "furniture_relation_match": 0.88,
  "human_preference": 0.76,
  "final_rank_score": 0.83
}
```

分數在校準前只是 ranking score，不得宣稱為機率。

---

## 4. 新家具庫：必須重新抓什麼

### 4.1 目前尚未確認的抓取範圍

在開始 crawler 前，Claude／使用者必須凍結：

- 商品來源網站／API／供應商。
- 國家站、語言與幣別。
- category scope。
- 目標 product／variant數量。
- 是否只抓現售商品或包含歷史商品。
- 是否允許與需要下載圖片、3D assets、說明文件。
- 更新頻率與授權／網站條款。

**[UNCONFIRMED]** 新家具來源與數量目前沒有提供，所以本輪不能誠實地直接開始抓取或估算最終資料量。

### 4.2 Raw crawl 必要欄位

每個商品／variant至少保存：

```text
source
source_product_id
product_family_id
variant_id
canonical_url
crawl_timestamp
locale
name
category breadcrumbs
description
feature bullets
assembled dimensions
package dimensions（必須與assembled分開）
weight
materials
colors
style labels（若來源有）
function/use
all product image URLs
room-scene/lifestyle image URLs
official 3D/AR asset URLs（若有）
assembly/manual URLs（若需要）
availability/status
raw source document hash
```

Raw snapshot不可只保留 parser結果；應保留可稽核原始 JSON/HTML response與asset manifest。

### 4.3 Normalization 與品質 gate

抓取後必須：

1. product family/variant去重。
2. canonical taxonomy mapping。
3. 統一尺寸單位為 m/mm，保存原字串。
4. 分開 assembled、package、adjustable/range dimensions。
5. 標示 missing、range、suspect、special geometry。
6. 圖片去重、損壞檢查、解析度與視角品質檢查。
7. 建 immutable ordered product manifest。
8. 記錄每欄 source、parser version、confidence與hash。

### 4.4 新商品多模態資產

ULIP full training 每件商品理想需要：

- text/caption。
- 多視角 product images。
- 3D mesh 或 point cloud。

處理順序：

1. 優先使用官方可信 3D/AR model。
2. 若沒有官方 3D，再從多視角／商品圖做 3D reconstruction或生成。
3. 生成 3D 必須通過幾何 QA，不可把失敗 mesh直接當真值。
4. 使用可信 assembled dimensions把 mesh對齊實際尺度。
5. 統一定義 canonical coordinate、front、up與unit。
6. mesh採樣成訓練 point cloud，保存 sampler/seed/version。

若新 catalog只有圖片而無可靠 3D，可先做 image/text retrieval baseline；但不能宣稱完整 ULIP point-cloud retraining已完成。

### 4.5 Caption 與 structured facts

建議用 VLM 從多視角圖生成 product caption，但 caption 不能取代結構化事實：

```text
caption：外觀、風格、材質、顏色、功能的自然語言描述
structured metadata：尺寸、front、footprint、clearance、category、variant
```

VLM 不得自行猜測：

- 精確 dimensions。
- official material。
- canonical front。
- drawer/door clearance。
- 商品是否真的適合某一位置。

所有 caption要保存 prompt、model/checkpoint、input image hashes與generated_at。

---

## 5. 為四種匹配建立哪些資料集

### 5.1 Dataset A：New Product Multimodal Catalog

用途：重訓 semantic product retrieval。

每列：

```json
{
  "product_id": "...",
  "family_id": "...",
  "variant_id": "...",
  "category": "sofa",
  "name": "...",
  "captions": ["..."],
  "attributes": {
    "materials": ["..."],
    "colors": ["..."],
    "styles": ["..."]
  },
  "image_refs": ["..."],
  "mesh_ref": "...",
  "pointcloud_ref": "...",
  "source_provenance": {}
}
```

需要 family-aware multi-positive labels，不能把同系列不同顏色／尺寸variant全部當對比負樣本。

### 5.2 Dataset B：Product Spatial Metadata

用途：大小、footprint、方向與功能空間。

```json
{
  "product_id": "...",
  "category": "nightstand",
  "dimensions_m": {
    "width": 0.45,
    "depth": 0.40,
    "height": 0.55
  },
  "dimension_known_mask": {
    "width": true,
    "depth": true,
    "height": true
  },
  "footprint": {
    "kind": "polygon",
    "vertices_local_xy_m": []
  },
  "canonical_frame": {
    "front_xy": [0.0, -1.0],
    "up_xyz": [0.0, 0.0, 1.0],
    "front_status": "verified"
  },
  "yaw_symmetry_deg": 180,
  "clearance_zones": [
    {
      "type": "drawer_open",
      "polygon_local_xy_m": [],
      "required": true
    }
  ],
  "usable_sides": ["front"],
  "articulation_states": [],
  "quality": {},
  "provenance": {}
}
```

unknown不能補0，且 bbox、polygon、multipart、articulated footprint必須明確區分。

### 5.3 Dataset C：Room Style ↔ Product Style Pairs

用途：家具與場景風格匹配。

只抓 individual product page 不足以生成可靠 scene-style matching labels。需要：

- room/lifestyle images或V2T frames。
- scene style taxonomy。
- palette、materials、lighting、mood。
- 場景中哪些商品／商品類型搭配良好。
- 同 category但style不合的 hard negatives。
- 人工 pairwise/listwise preference。

建議 row：

```json
{
  "scene_id": "...",
  "scene_frame_refs": ["..."],
  "scene_style_labels": ["modern", "warm", "minimal"],
  "scene_palette": ["#..."],
  "product_id": "...",
  "match_label": "good|acceptable|bad|unknown",
  "human_rating": 4,
  "preference_group_id": "...",
  "label_source": "human|curated_room|weak_vlm",
  "confidence": 1.0
}
```

Curated lifestyle image中的商品可作 weak positive，但若無可靠 product identity，不能冒充 exact product-level ground truth。

### 5.4 Dataset D：V2T Room / Candidate Pose

用途：位置、大小、collision、walkability。

每個 scene需要：

- metric local Manhattan coordinate。
- occupancy/free/unknown masks。
- walls。
- doors與swing/access（若要 door-aware）。
- windows。
- existing furniture instances。
- candidate spots。
- candidate centers/yaws。
- query與structured placement intent。

**[FACT]** 現有 V2T authoritative NPZ已有 floor、obstacle、wall、free、origin、resolution與applied buffer metadata；但 door swing、完整instance yaw/front、access terminals尚未確認。

### 5.5 Dataset E：Furniture Relation Graph

用途：家具與家具之間相對關係。

每個 room建立 graph：

```json
{
  "room_id": "...",
  "nodes": [
    {
      "instance_id": "bed_01",
      "category": "bed",
      "product_id": null,
      "center_xy_m": [2.1, 3.2],
      "footprint_polygon_xy_m": [],
      "yaw_deg": 90,
      "front_xy": [1.0, 0.0],
      "geometry_confidence": 0.9
    }
  ],
  "edges": [
    {
      "source": "nightstand_01",
      "target": "bed_01",
      "relation_labels": ["beside", "aligned"],
      "distance_m": 0.18,
      "relative_bearing_deg": 90,
      "relative_yaw_deg": 0,
      "valid": true,
      "label_source": "human"
    }
  ]
}
```

建議先凍結 relation ontology：

```text
near / far
beside_left / beside_right
in_front_of / behind
facing / facing_away
aligned / perpendicular
against_wall / corner
centered_with
surrounding
paired_with
functional_group
clearance_conflict
```

幾何上可算出的 relation與人類功能關係必須分開，例如 `near` 可由距離產生，但 `nightstand paired with bed` 需要category/function supervision。

### 5.6 Dataset F：Placement Supervision

每列是一個完整 contextual candidate：

```json
{
  "sample_id": "...",
  "group_id": "room_request_001",
  "room_id": "...",
  "request_id": "...",
  "query_text": "...",
  "product_id": "...",
  "spot_id": "...",
  "pose": {
    "center_xy_m": [1.2, 2.4],
    "yaw_deg": 90
  },
  "rule_evidence": {
    "policy_version": "...",
    "hard_valid": true,
    "size_fit": true,
    "collision_free": true,
    "path_clear": true,
    "reason_codes": []
  },
  "learning_targets": {
    "semantic_relevant": true,
    "scene_style_match": "good",
    "anchor_relation_match": true,
    "furniture_relation_match": true,
    "facing_valid": true,
    "human_preference_rank": 1
  },
  "known_masks": {},
  "label_source": {},
  "provenance": {}
}
```

`rule_evidence`供filter、audit與離線安全測試，不是production learned loss target。只有hard-valid rows能進pose preference slate；hard-invalid rows保留作rule regression evidence。

目前 PlacementLoop只保存每 product×spot最佳 pose與aggregate failures，不足以建立完整偏好slate；5090需要新增all-tested-poses exporter。

### 5.7 Dataset G：Gold Evaluation Set

必須獨立於 crawler、規則teacher與LLM generation：

- unseen rooms/videos。
- unseen product families。
- human-confirmed scene style preferences。
- human-confirmed furniture relations。
- exact geometry/safety labels。
- no-match cases。
- 多個都可接受的 placement sets，而不是強迫唯一答案。

### 5.8 Dataset H：3D-FRONT／3D-FUTURE Synthetic Pretraining

Claude回報目前V2T只有7個場景。這不足以訓練style/relation/preference，也只足以作初步smoke/held-out案例，不能作可信production benchmark。

**[PROPOSAL] 條件式採納3D-FRONT＋3D-FUTURE作2.0 synthetic pretraining主源：**

- 官方3D-FRONT論文記錄18,968個已布置房間、31種scene categories、帶category/style/material、位置、方向與尺寸的專業layout。
- 3D-FUTURE論文記錄9,992個高品質家具CAD，含category、style、theme、material、real-world size與6DoF pose相關資料。
- 同房間共同出現的家具可作style/relation weak positives。
- position/orientation/category可推導幾何relation graph；functional edges仍需ontology與人工抽核。
- exact size/collision/path仍由我們自己的deterministic engine重算，不把合成資料中的「看起來合理」直接當安全真值。

官方依據：

- [3D-FRONT ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/html/Fu_3D-FRONT_3D_Furnished_Rooms_With_layOuts_and_semaNTics_ICCV_2021_paper.html)
- [3D-FUTURE paper](https://arxiv.org/abs/2009.09633)
- [3D-FUTURE Data Sets Use License Agreement](https://terms.aliyun.com/legal-agreement/terms/suit_bu1_ali_cloud/suit_bu1_ali_cloud202004171628_60052.html)

**License gate：**3D-FUTURE條款明確限制scientific research use，禁止商業化、向第三方散布原始資料，並禁止用資料／衍生成果提供外部商業服務。3D-FRONT也需要同意Terms of Use並申請下載。因此：

```text
若本專案是研究用途且使用者接受條款：建議採納。
若目標包含商業服務：不能直接把它當production training source，需先取得適用授權或改用可商用資料。
```

即使採納，3D-FRONT不能取代真實V2T evaluation。最低策略：

```text
3D-FRONT/3D-FUTURE  → synthetic pretraining / weak supervision
現有7個V2T scenes     → initial held-out smoke only
後續新增真實V2T rooms → gold validation/test與domain adaptation
```

---

## 6. 新 corpus 必須重新生成

### 6.1 Corpus 的角色

Corpus負責：

- category、材質、顏色、風格、功能語言。
- placement relation語言理解。
- 新 catalog product facts與同義詞。
- query expansion與semantic hard negatives。

Corpus不負責：

- exact room collision。
- 實際尺寸計算。
- pose ground truth。
- furniture relation graph ground truth。
- human scene-style preference。

### 6.2 新 catalog corpus pipeline

```text
new raw product records
→ verified normalized facts
→ deterministic product documents
→ taxonomy / style / function ontology
→ spatial relation seed sentences
→ controlled LLM paraphrases（可選）
→ schema/fact consistency validation
→ semantic dedup
→ human spot QA
→ embedding
→ FAISS index
→ immutable corpus profile
```

每句至少記錄：

```text
doc_id
text
document_type
product_id/family_id（若適用）
category
attributes
source field hashes
generation model/prompt/seed（若適用）
confidence
```

不得讓 LLM 對缺少的尺寸、front或clearance自行補值。

---

## 7. 完整離線資料流程

```text
指定新家具來源與scope
→ crawler raw snapshot
→ product/variant dedup
→ taxonomy + structured dimension normalization
→ download images / official 3D / manuals
→ VLM captions + structured attribute QA
→ 3D reconstruction（缺官方3D時）
→ real-scale alignment + canonical front/up
→ point-cloud sampling
→ product spatial/clearance annotation
→ new corpus + FAISS
→ immutable product dataset bundle

V2T videos/scenes
→ room reconstruction
→ scene frames/style representation
→ occupancy/geometry/objects
→ furniture relation graph
→ candidate spots/poses

new product candidates × rooms × poses
→ exact geometry labels
→ style weak labels
→ relation labels
→ human preference/QA
→ room-disjoint training/val/test bundles
```

---

## 8. 完整訓練方法

### Stage 0：新資料 inventory 與 immutable baseline

先完成：

- 新 catalog source/scope契約。
- raw/normalized product manifests。
- image/3D/pointcloud coverage report。
- spatial metadata coverage report。
- V2T room/scene inventory。
- style/relation/placement label coverage。
- train/val/test group manifests。

此 stage不應先訓練，以免資料版本還在漂移。

### Stage 1：新 catalog ULIP semantic retraining

初始化：

- ULIP_RAG 1.0 weights。

輸入：

- 新商品 point cloud。
- 新商品 multi-view images。
- 新商品 caption/text。

建議 loss：

```text
L_semantic =
    λ_pt × multi-positive InfoNCE(point ↔ text)
  + λ_pi × multi-positive InfoNCE(point ↔ image)
  + λ_ti × optional InfoNCE(text ↔ image)
  + λ_attr × attribute/category auxiliary loss
  + λ_distill × 1.0 semantic retention
```

重要：

- 同 family variants可為multi-positive或soft positive，不要全部當batch negatives。
- split按 product family，避免同商品不同顏色／view跨 train/test。
- 第一階段可凍結text/image backbone，只訓PointBERT與projection；其後再比較adapter/partial unfreeze。
- 完成後重建新 catalog全部 product vectors。

### Stage 2：新 catalog RAG training

若確定2.0使用RAG：

1. 新 corpus/index檢索。
2. Stage 1式 RAG enhancer training，對齊新商品text/image。
3. Stage 2式 point branch alignment到enhanced text space。
4. 重建vectors。
5. 與不使用RAG的新 catalog vanilla baseline做held-out比較。

RAG未提升前，不得只因專案名稱有RAG就強制上線。

### Stage 3：Room–Product Style Alignment

模型：

```text
Frozen CLIP/VLM image features(selected room frames / product multi-view)
+ structured style/material/color attributes
→ lightweight trainable projection heads
→ style compatibility score
```

先用3D-FRONT/3D-FUTURE同場景co-occurrence與style attributes建立weak supervision；證明有signal後才比較專用Room/Product Style Encoder。真實V2T只作domain-shift測試與後續fine-tune。

loss：

```text
L_style =
    contrastive(room_style ↔ compatible_product_style)
  + pairwise/listwise human preference loss
  + optional style/material/color multi-label classification
```

Negative sampling：

- 同 category但風格不同。
- 顏色接近但材質/風格不合。
- style符合但category不符。
- 同系列不同variant作soft relation，不一定是hard negative。

### Stage 4：Deterministic Geometry Candidate Generation

本stage**不訓練size/collision/path validity model**。它負責：

1. 讀product structured dimensions／footprint／clearance。
2. 在3D-FRONT或V2T room中枚舉candidate spots/centers/yaws。
3. exact計算fit、observed-floor、collision、wall/door/unknown、clearance與path constraints。
4. hard-invalid直接淘汰並保存reason codes。
5. hard-valid輸出slack、distance、clearance、path-retention等偏好features。

輸出是Stage 5/6的合法candidate slate與audit evidence。選配的learned geometry head只能作離線ablation，不能進production selection，也不需要人工標註。

### Stage 5：Furniture Relation Graph Training

模型：

```text
existing furniture nodes
+ candidate product node at proposed pose
+ relative geometry/semantic edges
→ edge-feature MLP + attention pooling（v1 baseline）
→ relation labels + compatibility score
```

資料主源採3D-FRONT layout/pose/category推導幾何relation，並對高頻functional patterns人工抽核。資料有足夠signal後，再把Graph Transformer列為ablation，不直接當首版default。

loss：

```text
L_relation =
    multi-label relation classification
  + relation validity BCE
  + relative direction/distance regression（有可信標註時）
  + pair/group preference loss
```

沒有可信 front時，yaw只能學 modulo 180；不能假裝有360° facing ground truth。

### Stage 6：Semantic–Style–Relation–Pose Preference Fusion

以完整 candidate slate為訓練單位：

```text
(room, request)
→ deterministic hard-valid products × spots × poses
→ style/relation/preference positives與hard negatives
```

Fusion輸入：

```text
q_sem
q_intent
p_sem
p_style
r_style
exact pose preference features
relation_graph_context
rule evidence/masks（read-only，不能翻案）
```

輸出style、relation、human pose preference heads與listwise final score。

### Stage 7：Human Preference Fine-tuning

規則只知道物理與手工proxy；風格和合理佈置需要人工比較。

人工標註至少比較：

- 同房間多個風格皆可但偏好不同的商品。
- 同商品多個合法位置／方向。
- 家具與anchor/furniture group關係。
- 可放但阻礙視覺／使用動線的案例。

只用現行 `heuristic_composite_v1` 生成 target再測同一規則，只能證明teacher imitation。

### Stage 8：Partial joint fine-tuning

當 Stage 3–7在held-out資料有效後，才考慮低LR解凍：

- semantic/style adapters。
- text projection。
- RAG enhancer。
- 最後幾層product encoder。

加入semantic retention與legacy/new replay，避免新空間／風格資料破壞商品語意。

若解凍 PointBERT/pc_projection：

```text
必須重建新 catalog全部vectors
+ 新checkpoint/vector/catalog/corpus hashes
+ 全部semantic/spatial/relation/E2E regression
```

---

## 9. 總 Loss 設計

```text
L_total =
    λ_sem       × L_semantic_retrieval
  + λ_style     × L_scene_product_style
  + λ_rank      × L_listwise_or_pairwise_rank
  + λ_anchor    × L_anchor_relation
  + λ_relation  × L_furniture_graph_relation
  + λ_yaw       × L_yaw_facing
  + λ_pref      × L_human_preference
  + λ_distill   × L_1_0_semantic_retention
  + λ_cal       × L_calibration
```

規則：

- unknown label一律mask loss。
- 多個placement都有效時，不強迫唯一one-hot答案。
- preference/listwise group只含通過同版本hard gate的合法candidate slate。
- loss weights由validation/gradient balance決定，不沿用現行固定`0.55/0.45`。
- rule、人類、VLM weak label需有不同source weight。
- `L_valid/L_size/L_collision/L_path`不屬production主loss；若研究上要比較learned surrogate，只能放在隔離的`L_geometry_aux_ablation`，不共用final score或人工標註預算。

---

## 10. Negative Sampling

每個正例需混合：

1. **semantic-hard / style-wrong**
   - 同category、形狀相似，但和房間風格不搭。

2. **style-good / semantic-wrong**
   - 風格搭但不是使用者要的家具類型。

3. **semantic-good / geometry-invalid（只供rule/safety regression）**
   - 商品很符合query但尺寸超限或collision；必須在learned preference前被gate，不當作合法slate中的ranking negative。

4. **same product / less-preferred legal pose**
   - 同商品在多個都hard-valid的spot、center或yaw中，由relation/style/human preference區分；hard-invalid pose只留audit。

5. **same spot / wrong product**
   - 空間能放，但功能或關係不合理。

6. **relation hard negative**
   - nightstand離bed過遠、TV stand不面向sofa、coffee table放到sofa後方。

7. **near-boundary geometry（只供hard-gate測試）**
   - 尺寸、clearance、門區只差少量的案例；用來驗證false-safe=0，不用來教模型近似規則。

8. **easy random negatives**
   - 只保留少量，避免模型只學容易category差異。

missing dimensions、untrusted scale、unknown region與unsupported geometry要標unknown/abstain，不是普通negative。

---

## 11. Dataset split 與 leakage 防護

必須先group split，再產生views、query paraphrases、poses與weak labels。

### 11.1 Product split

- 以 product family為group。
- 同商品不同variant、圖片、caption、render不可跨split。
- 若 corpus包含test product exact facts，不能把該軌稱為true cold-product retrieval。

### 11.2 Scene split

- 同一physical room、video、reconstruction、frames、spots、queries、poses只能在同一split。
- 不能random拆 product×pose rows。

### 11.3 正式評估軌

1. New product / seen category semantic retrieval。
2. Unseen product family。
3. Unseen room / transductive catalog。
4. Unseen room style。
5. Furniture-relation hard cases。
6. Human-only gold preference。
7. 完整 cross-machine E2E。

---

## 12. 線上檢索與placement流程

### 12.1 建議正式流程

```text
V2T video
→ room reconstruction + scene frames
→ room style embedding
→ occupancy + walls/doors/windows
→ existing furniture graph
→ candidate spots/poses
→ product_query_text + structured spatial intent

query
→ new-catalog ULIP_RAG semantic Top-M

Top-M products
→ exact metadata quality/size/safety gate
→ remaining product × spot × pose slate

slate
→ room-product style score
→ spatial compatibility score
→ furniture-relation graph score
→ learned fusion rerank

top candidates
→ exact collision/path/final validator
→ selected + alternatives + rejected reasons
```

### 12.2 四種匹配如何出現在final ranking

不要先硬寫固定加權。模型應從資料學習 context-dependent gate，例如：

- 使用者明說顏色/風格時，提高 style importance。
- spot空間緊時，提高size/geometry importance。
- query有「床邊」時，提高bed relation importance。
- query無明確anchor時，relation head仍可提供一般functional preference，但不能虛構hard requirement。

### 12.3 Safety boundary

即使2.0學會spatial：

```text
known collision
known hard size overflow
outside observed floor
untrusted metric scale
unsupported unsafe geometry
→ 不能被高style/relation score救回
```

---

## 13. 評估指標

### 13.1 Semantic product retrieval

- Recall@1/5/10。
- MRR/nDCG。
- category accuracy。
- family/variant-aware retrieval。
- query paraphrase robustness。

### 13.2 Scene style matching

- human pairwise preference accuracy。
- style-compatible Recall@K/nDCG。
- material/color/style slice metrics。
- unseen-style/generalization。

### 13.3 Size與position

- size-fit precision/recall。
- collision false-safe rate。
- outside-observed-floor violation。
- valid placement Recall@K。
- Top-1 hard-valid rate。
- center error到valid set。
- yaw circular error；無front時報modulo-180。
- path/clearance violations。

### 13.4 Furniture relations

- relation macro-F1。
- anchor/furniture compatibility accuracy。
- facing/alignment accuracy。
- human relation preference win rate。
- graph-context ablation增益。

### 13.5 完整任務

```text
relevant product retrieved
AND scene style acceptable
AND size fits
AND legal pose found
AND furniture relations acceptable
AND no safety violation
```

另報 no-match/abstention、latency、VRAM、throughput與各category breakdown。

---

## 14. 必做 Ablation

至少比較：

1. 舊1.0 baseline（僅reference，不是新 catalog production）。
2. 新 catalog vanilla ULIP semantic only。
3. 新 catalog RAG semantic。
4. semantic + heuristic placement。
5. semantic + learned spatial。
6. semantic + style。
7. semantic + style + spatial。
8. 加入relation graph前後。
9. structured feature MLP vs occupancy CNN。
10. raw query vs product query + spatial intent分欄。
11. rule labels only vs human preference。
12. frozen backbone vs partial unfreeze。
13. random negatives vs hard negatives。
14. bbox vs verified footprint/front/clearance。
15. exact hard gate on/off只作離線安全證明；production不關閉。

---

## 15. 雙機責任提案

### 15.1 3090／ULIP_RAG

- 新 catalog crawler/normalizer或接收其immutable輸出。
- product image/text/3D/pointcloud pipeline。
- new corpus/index。
- semantic ULIP/RAG retraining。
- product semantic/style/spatial encoders。
- vectors、manifest與deployment profile。
- semantic/style regression evaluation。

### 15.2 5090／V2T／Claude

- 真實 room/video inventory。
- scene frames與room style features。
- occupancy/geometry/objects/candidate poses。
- existing furniture instance graph。
- all-tested-poses exporter。
- relation與human preference annotation workflow。
- local scene/spatial/relation runtime。
- exact validator與E2E。

### 15.3 Model runtime proposal

第一版推薦：

```text
3090：新 catalog semantic retrieval + product features
5090：room style / scene / relation graph / fusion reranker + exact validator
```

因為大型scene/grid資料已在5090，可避免每次傳整個room tensor到3090。若Claude建議统一在3090，需提出compact feature contract、payload、latency與failure policy。

---

## 16. 建議實作順序

### Phase 0：本文件審查與scope freeze

- 確認新商品來源、category、規模與授權。
- 確認四項能力都在2.0 scope。
- 凍結relation ontology與style taxonomy。
- 確認runtime責任。

### Phase 1：重新抓新 catalog

- raw crawl。
- normalized catalog。
- assets download。
- variant/family dedup。
- dimension/category/asset coverage report。

### Phase 2：建立新 multimodal product bundle

- captions。
- official/generated 3D。
- point clouds。
- canonical frame/front/scale。
- footprint/clearance。
- corpus/index。

### Phase 3：重新訓練 new-catalog ULIP/RAG

- semantic baseline。
- RAG candidate。
- vectors重建。
- held-out semantic gate。

### Phase 4：V2T spatial/relation dataset

- scene inventory。
- all-pose labels。
- furniture graph。
- style pair資料。
- human annotation。

### Phase 5：分支訓練

- style alignment。
- spatial geometry。
- furniture relation graph。

### Phase 6：fusion與human preference

- listwise fusion。
- hard-negative mining。
- human preference fine-tuning。

### Phase 7：joint fine-tune與promotion

- partial unfreeze。
- semantic retention。
- full ablation。
- immutable2.0 bundle。
- cross-machine shadow/E2E。

---

## 17. Claude 必須先回答的 P0 問題

### A. 新家具資料來源

1. 新家具庫的來源網站／API／供應商是什麼？
2. 國家站、語言、category與預計商品量是多少？
3. variant要視為獨立product，還是family下的variant？
4. 是否有官方3D/AR assets？若沒有，是否接受用TRELLIS或其他方法重建？
5. 新資料可否下載/保存/用於訓練，授權與crawl rate policy為何？
6. 是否要定期增量更新？如何處理下架商品？

### B. 四種匹配的scope

7. Claude是否同意2.0 primary model應是：

```text
semantic + scene style + spatial/size + furniture relation
```

8. 家具關係只需要「新商品相對現有家具」，還是2.0就要多件新家具 sequential placement？
9. 第一版style taxonomy與relation ontology由誰凍結？
10. 風格的authority是VLM tag、來源網站tag、人工偏好，或多來源融合？

### C. V2T與ground truth

11. 5090實際有多少獨立room/video/scene可用於training、QA與test？
12. V2T能否穩定輸出instance id、bbox/polygon、yaw/front、doors/windows、unknown mask與access terminals？
13. 是否接受新增all-tested-poses exporter，而不是只保留每product×spot最佳pose？
14. human style/relation/placement annotation由誰做，可提供多少？
15. `valid placement`與`best placement`如何正式定義？

### D. 模型與部署

16. 是否接受舊1.0只作initialization/baseline，新catalog完整重訓與重建vectors？
17. 是否先以new-catalog vanilla semantic作baseline，再把RAG作ablation？
18. Spatial/style/relation fusion runtime跑5090、3090或拆分？
19. exact hard safety gate是否同意永遠保留，不讓learned score覆蓋？
20. 2.0 primary acceptance metrics與最低門檻是什麼？

### E. 本輪權限

21. Claude目前只被要求review/提問，還是已被授權在5090修改dataset exporter？
22. 3090尚未取得新商品來源；在source/scope確認前，不應開始盲抓。Claude是否同意？

---

## 18. 請 Claude 回覆格式

```markdown
# Claude 對 V2T × ULIP_RAG 2.0 Revised Design 的審查

## 1. 核心目標
- 是否同意新 catalog 全量重抓／重訓：
- 是否同意四種匹配能力：
- 建議scope修正：

## 2. 新家具來源
- source/locale/categories：
- expected product/variant count：
- 3D availability：
- crawl/license/update policy：

## 3. V2T 現況
- available room/video/scene count：
- 可穩定輸出欄位：
- 尚缺欄位：
- source paths/evidence：

## 4. Style 設計意見
- labels/data：
- model/loss：
- evaluation：

## 5. Spatial/Size 設計意見
- labels/data：
- model/loss：
- hard gate：

## 6. Furniture Relation 設計意見
- ontology：
- graph representation：
- labels/data：
- evaluation：

## 7. Training stages
- 接受：
- 修改：
- 不接受：

## 8. Split/Leakage/Promotion
- split：
- metrics/gates：
- risks：

## 9. 雙機責任與artifact交換
- 5090：
- 3090：
- runtime location：
- transfer path/versioning：

## 10. P0 blocker與建議下一步
- ...
```

---

## 19. 3090 最終理解

使用者真正要的2.0是：

```text
重新抓取的新家具庫
→ 新的text/image/3D/product spatial metadata
→ 新corpus與新vector gallery
→ 以1.0 weights初始化後重新訓練semantic retrieval

+ V2T room style / geometry / occupancy
+ existing furniture relation graph
+ product × room × spot × pose supervision
+ human style / relation / placement preferences

→ trainable semantic-style-spatial-relational fusion
→ deterministic safety validation
→ 新的V2T × ULIP_RAG 2.0
```

因此，答案不是「不用重訓」。2.0需要：

- 重新抓商品資料。
- 重新生成multimodal資產與corpus。
- 重新訓練new-catalog semantic ULIP/RAG。
- 重新建立全部product vectors。
- 另外訓練style、spatial與furniture-relation modules。
- 最後做joint fusion與human preference fine-tuning。

目前不能直接開訓的P0 blocker是：**新家具來源、category/數量、3D取得方式、V2T scene inventory與人工label資源尚未確認。**

請Claude先回覆第17節，雙方再共同凍結正式的：

```text
ULIP_RAG 2.0 Data Acquisition Spec
ULIP_RAG 2.0 Training Spec
ULIP_RAG 2.0 Evaluation & Promotion Spec
```

凍結後才開始crawler、dataset build與訓練，避免再次拿錯家具庫開始做。
