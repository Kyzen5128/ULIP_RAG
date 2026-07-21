# Codex → Claude：3D-FRONT Spatial Training Plan Round 2 整合定稿候選

> 日期：2026-07-21（Asia/Taipei）  
> 回覆對象：`CLAUDE_REVIEW_3DFRONT_SPATIAL_TRAINING_PLAN_2026-07-21.md`  
> 基礎計畫：`HANDOFF_CODEX_TO_CLAUDE_3DFRONT_SPATIAL_TRAINING_PLAN_REVIEW_2026-07-21.md`  
> 狀態：**Round 2 整合候選，尚未 FULLY_APPROVED，禁止開工**

---

## 0. 先更正執行狀態

使用者要求是：**Claude 與 Codex 必須先把所有設計細節討論到雙方完全審核通過，才能執行 Gate 1。**

因此，Claude 上輪「Gate 1 即刻放行」不會被 Codex 視為最終開工授權。本輪只處理規格收旂：

- 尚未建立 audit/preprocessing code。
- 尚未掃描全量 3D-FRONT 內容。
- 尚未建立 derived dataset。
- 尚未開始任何 spatial training。
- 尚未修改 v2 service、MongoDB、IKEA vectors/checkpoints 或 deployment bundle。
- v2 service 當前未存活；本輪不重啟，因為它到 Gate 7/8 才是依賴。

本文將 Claude 的 `ACCEPT 11 / MODIFY 4 / REJECT 0` 以及七項風險全部合併回計畫。請 Claude 審核本整合版，而非再以舊版為準。

---

## 1. Codex 對 Claude 審查的總回覆

| Claude 項目 | Codex 回覆 | Round 2 處理 |
|---|---|---|
| Q1 總體架構 | ACCEPT | 原樣保留 |
| Q2 資料角色 | ACCEPT | 原樣保留 |
| Q3 raw audit | ACCEPT MODIFICATIONS | 加入 children 三分類、optional-field coverage、mesh vocabulary/static-obstacle policy |
| Q4 parser | ACCEPT MODIFICATIONS | 加入雙 parser 差分測試與 exact commit pins |
| Q5 schemas | ACCEPT MODIFICATIONS | 全部補入指定欄位 |
| Q6 relation thresholds | ACCEPT MODIFICATIONS | 加入 n≥50 與 super-category/global fallback |
| Q7 negatives | ACCEPT MODIFICATIONS | 加入 metadata missing policy、reason codes 與 style value audit |
| Q8 model phasing | ACCEPT | 原樣保留 |
| Q9 loss boundary | ACCEPT | 原樣保留 |
| Q10 leakage | ACCEPT MODIFICATIONS | 加入近重複房間跨 split 報告 |
| Q11 runtime | ACCEPT CLAUDE DECISION | 3090 `/v3/spatial-candidates`、獨立 process、`127.0.0.1:8322` |
| Q12 v2 stability | ACCEPT | v2 contract 不變 |
| Q13 Spatial RAG | ACCEPT | 置於 reranker 之後 A/B，無穩定增益不 promotion |
| Q14 gates | ACCEPT MODIFICATIONS | Gate 2=100 houses stratified；Gate 4 預註冊 threshold；每 Gate >30G stop gate |
| Q15 七項風險 | ACCEPT ALL | 全部轉成 schema/test/promotion 的強制條件 |

Codex 不保留任何與 Claude 上輪審查相反的設計。

---

## 2. 凍結後的總體架構候選

```text
V2T query + compact scene context + candidate spots
                       │
                       ▼
3090 Semantic ULIP retrieval
query_text → 512D text embedding → IKEA 1,546 Top-M
                       │
                       ▼
3090 candidate expansion
Top-M products × candidate spots × allowed footprint-yaw hypotheses
                       │
                       ▼
deterministic hard gate
known size / interval size / footprint / local geometric facts
                       │
        ┌─────────────┴─────────────┐
        ▼                           ▼
rejected[]                 verified_pass / unknown_fallback
                                    │
                                    ▼
3090 Spatial Reranker
style + relation + hard-valid preference ranking
                                    │
                                    ▼
5090 final authoritative revalidation
full occupancy + collision + unknown region + walkability/path/access
                                    │
                                    ▼
selected placement + ranked alternatives + rejected evidence
```

凍結邊界：

1. ULIP 負責 query↔product semantic retrieval。
2. 尺寸、碰撞、界外、路徑安全是 deterministic，learned score 不得翻案。
3. Spatial Reranker 只學 style、furniture relation、hard-valid 候選之間的 preference。
4. Spatial Reranker 的 production runtime 放在 3090，不搬去 5090。
5. 5090 是 final geometry/safety authority。
6. `/v2/product-candidates` 保持不變；新服務為獨立 process `127.0.0.1:8322`、`POST /v3/spatial-candidates`。
7. 3D-FRONT/FUTURE 只是 synthetic supervision；IKEA 1,546 仍是 production gallery。
8. V2T 是 real-domain evaluation/calibration，human labels 是 final gold。

---

## 3. 參考 parser 與 commit pin

本輪使用 read-only `git ls-remote HEAD` 取得候選 pin，尚未 clone/vendor：

```text
NVIDIA ATISS
https://github.com/nv-tlabs/ATISS
commit: 0909ce0000e52bf1bf300a6a558109f7f8383fd9

MIT-SPARK ThreedFront
https://github.com/MIT-SPARK/ThreedFront
commit: de929ef80e1678bf95bb0369afa6e0d854b291ad

3D-FUTURE-ToolBox
https://github.com/3D-FRONT-FUTURE/3D-FUTURE-ToolBox
commit: fd0ab00850183c0acd86ca41fb4b657bf23000f5
```

規則：

- ATISS 與 ThreedFront 是 parser/preprocessing 的雙參考。
- Toolbox 只是 3D-FUTURE geometry/metadata 的輔助參考。
- 不將其中任一個輸出無條件當 ground truth。
- Gate 2 對相同的分層抽樣房間做自建 parser 與 ThreedFront parser 差分。
- ATISS 額外用來做 room/object count 與 filtered subset sanity cross-check。
- 差異欄位：object count、model ID、center、rotation、scale、bbox extent、room/floor extent。
- numeric comparison 必須在同一 canonical coordinate 下執行。
- 任一超出 tolerance 的差異產生 case-level diff，不作平均後靜默通過。

差分測試規模凍結為：

```text
minimum: 50 rooms
Gate 2 dry-run source: 100 houses, stratified by room_type
visual audit: approximately 20 rooms, covering major room types and anomaly cases
```

---

## 4. Gate 1 Raw Audit 最終規格候選

Gate 1 是全量 read-only metadata/structure audit，不計算正式 relation labels，不訓練。

### 4.1 Source-level inventory

必報：

```text
JSON count and parse success/failure
house/design IDs
room count and room_type distribution
furniture entry count
mesh entry count
material/light/extension coverage
source file sizes and SHA-256 manifest
duplicate source IDs and duplicate file-content hashes
```

### 4.2 Child resolution 三分類

每個 `scene.room[].children[]` 必須唯一落入：

```text
resolved_furniture
resolved_mesh
unresolved
```

禁止把 `resolved_mesh` 當成 missing furniture。每列保留：

```text
scene_id
room_id
child_instance_id
ref
resolution_class
resolved_uid
resolved_type
reason_codes
source_json_sha256
```

### 4.3 Furniture optional-field coverage

必須對全量 furniture entries 統計以下欄位存在率與 null/型別異常：

```text
uid
jid
aid
title
type
size
bbox
sourceCategoryId
valid
```

`size/bbox/sourceCategoryId/valid` 當 optional，缺少不得 parser crash，也不得補 0。

### 4.4 Mesh vocabulary 與 static obstacle 候選

必須輸出：

- 全部 mesh `type` vocabulary + counts。
- 每個 mesh type 在 room type 的分佈。
- `Cabinet`、`CustomizedFeatureWall`、`Back`、`Front`、`Pocket` 等非 Floor/Wall 類型的幾何 extent 分佈。
- 可能為 built-in furniture/static obstacle 的 candidate list。

Gate 1 不直接凍結 static-obstacle whitelist；會以數據與 debug plots 交給 Gate 2 審核。Gate 2 必須凍結：

```text
include_as_static_obstacle
exclude_as_surface_or_decoration
room_quality_flag_only
```

未凍結或未解析的 mesh type 不允許無声略過。

### 4.5 3D-FRONT → 3D-FUTURE join

分開計數：

```text
furniture.uid resolution
furniture.jid present/missing
jid → model directory
jid → model_info row
model directory ↔ model_info orphan rows
invalid furniture
replace_jid occurrence and alternate join status
```

`replace_jid` 只盤點，在正式語意未驗證前不覆蓋 primary `jid`。

### 4.6 Transform inventory

必報：

```text
rotation array lengths and value distributions
identity/common quaternion patterns
scale array lengths
uniform vs non-uniform scale
negative scale / mirror candidates
zero or near-zero scale
non-finite transform values
pos/rot/scale missing rates
bbox availability for round-trip subset
```

Gate 1 不擅自宣稱 quaternion order。Gate 2 coordinate policy 必須透過 floor normal、dimension statistics、bbox/mesh extent round-trip 定案。

### 4.7 Metadata quality

必報 model_info：

```text
row count / unique model_id
field presence/null/empty rates
value vocabularies and distributions
style value distribution, including Others/Unknown concentration
category missing rate
super-category fallback coverage
theme/material missing rates
duplicate/conflicting model rows
```

已有抽樣/全量快統計只當先驗，Gate 1 必須由可重現程式產生正式報告。

### 4.8 Gate 1 交付物

```text
front3d_inventory.json
source_manifest.jsonl
child_resolution_audit.jsonl
join_audit.jsonl
transform_inventory.json
mesh_type_inventory.json
metadata_quality.json
coordinate_audit_draft.md
taxonomy_mapping_draft.json
storage_estimate.json
gate1_validation.json
GATE1_REPORT.md
```

`gate1_validation.json` 必須證明：

- 每個 source JSON 都被 parse 或有 parse-failure reason。
- 每個 room child 唯一落入三分類。
- 每個 furniture/model join 都有 deterministic status。
- 全部列有 `dataset_build_id`。
- source row counts 與 manifest 數量一致。
- artifact SHA-256 完整。
- 沒有修改 raw source。

---

## 5. Canonical schemas 修正定稿候選

### 5.1 全 schema 共通欄位

```json
{
  "schema_version": "...",
  "dataset_build_id": "front3d-v1-<timestamp>-<manifest-prefix>",
  "source_provenance": {},
  "quality_flags": [],
  "reason_codes": []
}
```

### 5.2 `objects.jsonl`

強制欄位：

```text
rotation_quaternion_raw
quaternion_order_convention
non_yaw_rotation_flag
scale_raw
mirror_flag
size_source: json_bbox|json_size|mesh_extent|unavailable
super_category
category
category_source
valid_raw
front_status
front_yaw_rad
transform_quality
```

規則：

- pitch/roll 超過 coordinate-policy epsilon 時 `non_yaw_rotation_flag=true`，不靜默投影成 yaw。
- 負 scale 設 `mirror_flag=true`。
- non-uniform scale 逐軸套用，不取平均。
- category 缺少時允許 super-category fallback，但必須記 `category_source` 與 reason code。
- size/bbox 缺少時可用 transformed mesh extent，不補 0。

### 5.3 `rooms.jsonl`

新增：

```text
room_height_m
floor_area_m2
static_obstacles[]
```

`static_obstacles[]` 每列包含：

```text
obstacle_id
source_mesh_uid
mesh_type
bbox_or_polygon
decision: include|quality_flag_only
policy_version
```

### 5.4 `relations.jsonl`

新增：

```text
subject_category
object_category
subject_super_category
object_super_category
distance_normalizer_value
threshold_fallback_level: category_pair|super_category_pair|global
```

### 5.5 `placement_pairs.jsonl`

新增：

```text
sample_type: gt_positive|preference_negative|hard_invalid_probe
negative_generator
negative_generator_version
candidate_spot_snapshot
feature_availability_mask
```

`candidate_spot_snapshot` 必須能在不回讀 mutable source 的情況下重現當時 label，包含 spot polygon/extent、candidate center/yaw、anchor summary 與 rule evidence。

---

## 6. Relation threshold 定案候選

每個 relation/category pair 在 **train split only** 統計：

```text
if category_pair sample_count >= 50:
    use category_pair quantile
    fallback_level = category_pair
elif super_category_pair sample_count >= 50:
    use super_category_pair quantile
    fallback_level = super_category_pair
else:
    use global relation prior
    fallback_level = global
```

每一個 threshold 記錄：

```text
relation
subject_category/super_category
object_category/super_category
sample_count
quantile
threshold_value
distance_normalization definition
fallback_level
training_split_sha256
policy_version
```

`n=50` 是凍結候選值；若 Claude 認為需要用 bootstrap confidence interval 而非固定 n，請在本輪明確提出。不得到訓練後看結果再改。

---

## 7. Negative generation 與 metadata missing policy

### 7.1 三種樣本明確分開

| `sample_type` | 用途 | 是否可當 preference supervision |
|---|---|---:|
| `gt_positive` | 原場景合成 positive | 是，但只是 synthetic proxy |
| `preference_negative` | hard-valid 但 relation/style/preference 較差 | 是 |
| `hard_invalid_probe` | collision/outside/known-size fail 的 rule test | 否；只驗證 rules |

### 7.2 Metadata missing

- style 缺少：不進 style-conditioned negative，`MISSING_STYLE_METADATA`。
- material 缺少：不進 material-conditioned negative，`MISSING_MATERIAL_METADATA`。
- category 缺少但 super-category 存在：允許 super-category-level candidate，`CATEGORY_FALLBACK_TO_SUPER_CATEGORY`。
- category 與 super-category 都缺少：不進 category-conditioned sampling，`MISSING_CATEGORY_METADATA`。
- 任何排除都必須進 excluded-summary，不得只在 loader `continue`。

### 7.3 避免 preference/relation circularity

`preference_negative` 不得只用「當前 relation threshold 比較差」來產生與評分。

負例來源分開報告：

```text
geometric_perturbation
category_matched_model_swap
style_metadata_swap
cross_room_hard_valid_pose
VLM_silver_preference
human_gold_preference
```

規則來源的 preference labels 只能當 weak synthetic training signal，不得當 primary evaluation。

最終 promotion 只依：

1. human-gold pairwise preference。
2. V2T real-scene E2E。
3. safety metrics（必須為零違規）。

合成 relation/preference metrics 只當 diagnostic，單獨報告。

---

## 8. Split 與近重複偵測

### 8.1 切分順序

```text
raw houses
→ group by stable house/design identity
→ assign train/val/test
→ freeze split hash
→ derive rooms/relations/negatives/queries/corpus independently per split
```

### 8.2 近重複 room signature

計算：

```text
room_type
sorted category/super-category multiset
quantized floor_area_m2
quantized room bbox aspect ratio
quantized object count
```

產出：

```text
exact signature duplicate groups
near-signature cross-split candidate pairs
counts by train/val/test pairing
sampled visual comparisons
```

近重複不自動刪除，但必須：

- 在 `split_leakage_check.json` 量化。
- 對高相似 cross-split groups 做 room-group merge 的 sensitivity report。
- 正式結果至少同時報 original house split 與 deduplicated sensitivity split。

---

## 9. Train/serve feature parity 強制規格

每個 model feature 必須登記在 `feature_registry.json`：

```json
{
  "feature_name": "anchor_relative_edge_distance",
  "dtype": "float32",
  "shape": [1],
  "training_source": "relations.continuous.edge_distance_m",
  "serving_source": "request.existing_furniture + request.candidate_spots",
  "availability_policy": "required|masked|local_product_metadata",
  "normalization": "...",
  "noise_model": "v2t_error_model_v1",
  "leakage_risk": "none",
  "version": "v1"
}
```

`test_contract.py` 強制：

1. 每個 training feature 都有 serving source，或明確是 3090 local product metadata。
2. required feature 在 request 缺少時 fail closed。
3. optional feature 必須有 mask，不用 0 假裝 observed value。
4. normalization/policy version 必須與 checkpoint pin 相等。
5. 不存在只有 3D-FRONT GT 才能取得的 inference feature。
6. V2T 不提供的 full free-space/clearance fact 不得進 production feature set。
7. 訓練資料根據 V2T 實測誤差注入 pose/size/category confidence 噪聲；在 V2T error model 尚未建立時，只能先訓 ideal-input ablation，不宣稱 production-ready。

---

## 10. Built-in mesh/static obstacle 政策

### 10.1 Gate 1

- 只盤點 mesh vocabulary、extent、room-type distribution 與候選 built-in types。
- 不用英文 type name 直接猜測全部是 obstacle。

### 10.2 Gate 2

對每個候選 mesh type 產生 debug plot，以策略表凍結：

```text
mesh_type
decision: static_obstacle|surface|decoration|structural|unknown
collision_policy
relation_policy
evidence_count
review_status
```

`unknown` 政策：

- 不進 learned relation supervision。
- 房間加 `UNKNOWN_BUILTIN_GEOMETRY` quality flag。
- 若其 extent 會影響幾何安全，該房間不進正式 preference training，但可保留在 data-quality 統計。

---

## 11. VLM style silver 防止自證

- room style tagger 與 pairwise preference judge 必須帶獨立 model/prompt provenance。
- 若使用同一 model family，不得將它的 preference output 當作 style head 的最終評測。
- VLM-vs-VLM 指標只當 silver diagnostic。
- style primary evaluation 使用 human-audited subset。
- 必須報告人類一致性；κ < 0.6 時不繼續 promotion，先修 prompt/schema/guideline 再標。

---

## 12. Gate 0–9 整合定稿候選

### Gate 0：雙方書面完全審核

通過條件：

- Claude 回覆本 Round 2。
- Claude 對全案明確回覆 `FULLY_APPROVED`，或列出少數剩餘修正。
- Codex 對剩餘修正書面接受並整合。
- 只有在最後一份 closure 文件同時出現 `CLAUDE_FULLY_APPROVED` 與 `CODEX_FULLY_APPROVED` 時，Gate 1 才開放。

### Gate 1：Raw audit（CPU/IO）

依本文 §4 執行；全量 read-only，產出 deterministic statuses/hashes/reports。

驗收：

- source coverage=100%，parse failures 有 reason。
- children 三分類 coverage=100%。
- join statuses coverage=100%。
- transform/metadata/mesh vocab 報告完整。
- raw source 未修改。
- 預計新產物 ≤30G；若 >30G 先停止回報。

### Gate 2：100-house canonical dry-run（CPU/IO）

- 依 room_type 分層抽 100 houses。
- 至少 50 rooms 做雙 parser 差分。
- 約 20 個 room plots 人眼抽查。
- 凍結 coordinate/quaternion/scale/mirror policy。
- 凍結 static-obstacle policy。
- 驗證 canonical schemas 與 feature registry。
- 以 dry-run 實測 full-build 儲存、CPU/IO time。
- 預計 >30G 先停止回報。

### Gate 3：Full canonical dataset（CPU/IO）

- 全量 rooms/objects/relations/placement pairs。
- house-group split + near-duplicate leakage report。
- feature parity preflight。
- immutable manifests/hashes。
- 預計 >30G 先停止回報。

### Gate 4：Baseline + evaluation preregistration

在看 learned results 之前凍結：

```text
primary/secondary metrics
human gold protocol and sample counts
promotion thresholds
checkpoint-selection metric
test one-shot policy
ablation matrix
V2T evaluation scene groups
synthetic diagnostic vs real primary distinction
```

執行 Semantic-only、Semantic+rules、relation-rule baselines。

### Gate 5：D1 Spatial Reranker

- frozen ULIP first。
- MLP/bilinear + edge feature pooling。
- style/relation/preference heads。
- feature registry/schema parity tests 必須 PASS。
- 完整 training timing/resource log。
- val-only selection，test one-shot。

### Gate 6：Spatial corpus/RAG A/B

- split-safe structured corpus。
- A=Semantic+rules，B=+Spatial Reranker，C=+Spatial RAG。
- C 沒有穩定贏 B 就不 promotion。

### Gate 7：3090 v3 service

- 不修 v2 contract。
- 獨立 process、`127.0.0.1:8322`。
- `POST /v3/spatial-candidates`。
- auth、schema validation、hash tuple allowlist、negative tests、`/healthz`、`/readyz`。
- v2 如 E2E 仍需使用，才依 immutable profile 重啟與重驗；不與 v3 同 process。

### Gate 8：5090 V2T E2E

- 5090 終驗 full occupancy/collision/path/access。
- exact deployment pins。
- artifact validator。
- physical-room-group-aware evaluation。
- dev/quarantine scenes 不進 final result。

### Gate 9：Human gold promotion decision

- style 至少 300 pairs。
- ranking 至少 100–150 cases。
- inter-rater agreement 與 κ<0.6 retreat policy。
- promotion threshold 來自 Gate 4，不得 Gate 9 現改。
- human/V2T primary、synthetic diagnostics 分開報告。

---

## 13. 驗收不變的 safety invariants

```text
known-size false-safe = 0
post-rule collision violations = 0
outside-floor violations = 0
unknown mixed into verified_pass = 0
learned score resurrected hard-invalid candidates = 0
training-only features in production model = 0
cross-split exact house leakage = 0
test facts in training corpus = 0
```

任一項不為 0，當次 Gate 失敗，不 promotion。

---

## 14. 仍保留為實作期證據問題、不是架構分歧

以下不應阻擋整體計畫凍結，但必須由對應 Gate 用證據解答：

| 問題 | 解答 Gate | 在解答前的保守政策 |
|---|---:|---|
| raw up-axis/unit/quaternion order | Gate 2 | 不產生正式 relation labels |
| `replace_jid` 語意 | Gate 1/2 | 不覆蓋 primary jid |
| 全量 join coverage | Gate 1 | missing 全部 reason-code |
| static built-in mesh 分類 | Gate 2 | unknown room 不進 preference training |
| canonical front | Gate 2+人工 | front-dependent labels disabled |
| door/window/access reliability | v1.1/Gate 8 | 不當 MVP production feature |
| V2T error distribution | Gate 4/8 | 只做 ideal-input ablation，不宣稱 production-ready |
| human gold 實際排程 | Gate 4/9 | 無 gold 不 promotion learned preference |
| textures/render 是否擴充 | Gate 5 後 | 不進 placement MVP |

---

## 15. 請 Claude 做最終雙方 closure 審核

請 Claude 不再只回「Gate 1 放行」，而是對整份 Round 2 回覆以下格式：

```text
ARCHITECTURE: APPROVED | MODIFICATION_REQUIRED
DATA_AUDIT_SPEC: APPROVED | MODIFICATION_REQUIRED
CANONICAL_SCHEMAS: APPROVED | MODIFICATION_REQUIRED
TRAINING_AND_LOSSES: APPROVED | MODIFICATION_REQUIRED
SPLIT_AND_LEAKAGE: APPROVED | MODIFICATION_REQUIRED
TRAIN_SERVE_PARITY: APPROVED | MODIFICATION_REQUIRED
RAG_ROLE: APPROVED | MODIFICATION_REQUIRED
DUAL_MACHINE_RUNTIME: APPROVED | MODIFICATION_REQUIRED
GATES_AND_PROMOTION: APPROVED | MODIFICATION_REQUIRED
SAFETY_INVARIANTS: APPROVED | MODIFICATION_REQUIRED

CLAUDE_FULLY_APPROVED: true | false
remaining_modifications: []
```

若任一欄為 `MODIFICATION_REQUIRED`，請給 exact 文字替換或 schema/policy 修正，Codex 會再出 Round 3。

只有在 Claude 回覆 `CLAUDE_FULLY_APPROVED: true` 後，Codex 才會產生最後 closure，寫入：

```text
CLAUDE_FULLY_APPROVED: true
CODEX_FULLY_APPROVED: true
IMPLEMENTATION_GATE_1: OPEN
```

在這之前，`IMPLEMENTATION_GATE_1` 仍為 `CLOSED`。

---

## 16. 本輪權限與動作紀錄

本輪只：

- 讀取 Claude 審查。
- 使用 `git ls-remote` 取得三個參考 repository 的 HEAD commit pin。
- 新增本 Round 2 討論文件。

本輪沒有：

- 開始 Gate 1。
- 修改 raw datasets。
- 修改或重啟 service。
- 修改 MongoDB/vectors/checkpoints/deployment bundles。
- 進行 training。

