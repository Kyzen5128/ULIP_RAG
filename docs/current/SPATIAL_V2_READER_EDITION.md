# ULIP_RAG 2.0 Spatial Placement 計畫：核准規格閱讀版

> NON-NORMATIVE READER EDITION  
> 本文只為降低 Round 2＋3＋4 疊加閱讀成本，不取代原始核准文件。  
> 正式規格仍是 Round 2＋Round 3＋Round 4，衝突優先序為 Round 4 > Round 3 > Round 2。

狀態基準：2026-07-21  
核准狀態：Claude fully approved、Codex fully approved  
目前 gate：Gate 1 OPEN；final closure 成文時 Gate 1 NOT_STARTED

## 1. Specification / approval chain

| 層次 | 文件 | SHA-256 |
|---|---|---|
| REFERENCE | [Base rationale／原始計畫](../records/HANDOFF_CODEX_TO_CLAUDE_3DFRONT_SPATIAL_TRAINING_PLAN_REVIEW_2026-07-21.md) | b9f951de6e596e447122bab618cb7e6934a9203b5a874a0ce7e731bb694edf31 |
| NORMATIVE Round 2 | [主規格](../records/CODEX_REVISION_TO_CLAUDE_3DFRONT_SPATIAL_PLAN_ROUND2_2026-07-21.md) | fd2269d79501ad0d6d697386c58d93281c104b04a7014a23d8bc917513c54dc4 |
| NORMATIVE Round 3 | [RM-1～RM-5 addendum](../records/CODEX_ROUND3_INTEGRATED_CLOSURE_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md) | ddab7ddc5b32e500871813dc042143d4aa8d0c98de0fa0439cb023716a9052b1 |
| NORMATIVE Round 4 | [RM-6 addendum](../records/CODEX_ROUND4_RM6_FINAL_DIFF_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md) | dc7f89308188375000f8af5cd8b72a1039a86cbe45406c8d9748380fc29ca92b |
| APPROVAL EVIDENCE | [Claude RM-6 confirmation](../records/CLAUDE_RM6_CONFIRM_FULLY_APPROVED_3DFRONT_SPATIAL_2026-07-21.md) | db08754e4bc2fe864b7854825b83d10a19a2e1390ce299b0ef785760588cd992 |
| CLOSURE EVIDENCE | [Final closure](../records/FINAL_CLOSURE_CLAUDE_CODEX_3DFRONT_SPATIAL_PLAN_FULLY_APPROVED_2026-07-21.md) | 634742354c51036e0b207aafef97d9afb128c3daad41fecbe64aca6ea67611d2 |

任何原始 normative file 內容變更都需要新 review、版本與 closure；不能沿用上表核准狀態。

## 2. 系統邊界

~~~text
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
known/interval size + footprint + local geometry
        ┌──────────┼──────────────┐
        ▼          ▼              ▼
   rejected[]  unknown_fallback[] verified_pass[]
                   │              │
                   │              ▼
                   │    3090 Spatial Reranker
                   │    style + relation + hard-valid preference
                   │              │
                   │              ▼
                   │    5090 authoritative revalidation
                   │    full occupancy + collision + unknown regions + path/access
                   │              │
                   ▼              ▼
          separate fallback   selected placement
          evidence only       + alternatives + rejection evidence
~~~

凍結責任：

- ULIP：query ↔ product semantic retrieval。
- 3090 hard gate：已知／區間尺寸與可取得的局部幾何規則。
- Spatial Reranker：只排序 hard-valid candidates。
- unknown_fallback 保持獨立，不進 learned reranker，也不能混入 verified ranking。
- 5090：final geometry 與 safety authority。
- v2 Product Candidates contract 保持不變。
- Spatial v3 另開獨立 process：127.0.0.1:8322、POST /v3/spatial-candidates。

## 3. 什麼要學、什麼不能學

Learned：

- room-product style compatibility
- furniture relation compatibility
- hard-valid candidates 之間的 human preference

Deterministic：

- known／interval dimension fit
- footprint containment
- collision
- outside-floor
- observed-space validity
- 資料存在時的 walkability、path、access

硬性不變量：learned score 不得讓 hard-invalid candidate 復活。

## 4. 資料分工

| dataset/source | 角色 | 禁止誤用 |
|---|---|---|
| IKEA 1,546 | production products、semantic retrieval、商品 metadata | 不用 3D-FUTURE 商品取代 production gallery |
| 3D-FRONT | synthetic rooms、layout、pose、relation supervision | 未經 audit 不直接視為乾淨 GT |
| 3D-FUTURE-model | furniture mesh、category、style/material metadata | canonical front、unit、join 不可自行假設 |
| V2T | real-domain E2E／calibration | physical-room split 未凍結前不當無 leakage training |
| VLM | provenance-pinned silver labels | 不得用同一 judge 自證最終品質 |
| Human | final gold preference/front/quality audit | 無 gold 不 promotion learned preference |

## 5. Parser 與可重現性

候選 reference pins：

| project | commit |
|---|---|
| NVIDIA ATISS | 0909ce0000e52bf1bf300a6a558109f7f8383fd9 |
| MIT-SPARK ThreedFront | de929ef80e1678bf95bb0369afa6e0d854b291ad |
| 3D-FUTURE-ToolBox | fd0ab00850183c0acd86ca41fb4b657bf23000f5 |

- ATISS 與 ThreedFront 是雙參考 parser。
- Toolbox 是 geometry／metadata 輔助參考。
- Gate 2 至少 50 rooms 做雙 parser differential test。
- Gate 2 dry-run 使用按 room_type 分層的 100 houses。
- 約 20 rooms 做人工 visual audit。
- 比較 object count、model ID、center、rotation、scale、bbox extent、room/floor extent。
- 任何 tolerance 外差異產生 case-level diff，不得平均後靜默通過。

## 6. Gate 1：全量 raw audit

Gate 1 只做 read-only、CPU/IO audit，不建立正式 relation labels，也不訓練。

### Source 與 child resolution

每個 source JSON 必須成功 parse，或留下 parse-failure reason。每個 room child 唯一分類為：

~~~text
resolved_furniture
resolved_mesh
unresolved
~~~

resolved_mesh 不能被誤算為 missing furniture。

### Join 與 transform

必須分開盤點：

- furniture.uid resolution
- jid present／missing
- jid → model directory
- jid → model_info
- directory／metadata orphan
- invalid furniture
- replace_jid occurrence 與 alternate join status
- quaternion array/value distribution
- uniform／non-uniform／negative／zero scale
- missing／non-finite pos、rot、scale
- bbox/mesh round-trip subset

Gate 1 不自行決定 quaternion order，也不以 replace_jid 覆蓋 primary jid。

### Optional metadata

統計 uid、jid、aid、title、type、size、bbox、sourceCategoryId、valid 的 coverage、null 與型別異常。optional 欄位缺失不得 crash，也不得補 0。

### Mesh 與 room types

- 列出完整 mesh type vocabulary、counts、extent 與 room-type distribution。
- built-in/static obstacle 只列候選，不在 Gate 1 依英文名稱直接定案。
- 產生完整 raw room type vocabulary、room/house/children counts、join completeness 與 floor-area distribution。

### Gate 1 必要 artifacts

~~~text
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
per_room_join_completeness.jsonl
join_completeness_histogram.json
unjoinable_geometry_crosstab.json
unjoinable_jid_pattern_analysis.json
room_type_vocabulary.json
~~~

所有 artifacts 必須有 schema/version/provenance/hash。預計新產物超過 30G 必須停止回報，不能繼續。

## 7. Canonical dataset contracts

全部 schemas 共用：

~~~text
schema_version
dataset_build_id
source_provenance
quality_flags
reason_codes
~~~

### objects.jsonl

重要欄位：

~~~text
rotation_quaternion_raw
quaternion_order_convention
non_yaw_rotation_flag
scale_raw
mirror_flag
size_source
super_category
category
category_source
valid_raw
front_status
front_yaw_rad
transform_quality
join_status
geometry_source
~~~

join_status：

~~~text
model_joined | jid_missing_model | no_jid
~~~

geometry_source：

~~~text
model_mesh | bbox_only | size_only | unavailable
~~~

規則：

- non-uniform scale 逐軸套用，不取平均。
- negative scale 標 mirror。
- pitch/roll 超過 policy epsilon，不得靜默投影為 yaw。
- category 缺失可使用 super-category fallback，但需記來源。
- size/bbox 缺失可用 transformed mesh extent，不補 0。
- unjoinable 但有 bbox/size 的 instance 只作 obstacle-grade geometry。
- obstacle-grade 不得作 relation/style/preference subject/object。
- unavailable 不以 zero-size 代替。

### rooms.jsonl

包含：

~~~text
room_height_m
floor_area_m2
static_obstacles[]
supervision_tier
~~~

supervision tiers：

| tier | 條件 | 允許監督 |
|---|---|---|
| tier_full | 所有 furniture children join | relation／preference |
| tier_partial | 未 join 者都有 bbox/size geometry | joined subset 可產監督 |
| tier_context_only | 存在 unavailable geometry | 只入統計，不產 preference |

### relations.jsonl

包含 subject/object category、super-category、distance normalizer 與 threshold fallback level。

### placement_pairs.jsonl

sample_type：

~~~text
gt_positive
preference_negative
hard_invalid_probe
~~~

hard_invalid_probe 只驗證規則，不能當 preference supervision。candidate_spot_snapshot 必須足以離線重現 spot、pose、anchor 與 rule evidence。

## 8. Relation 與 negative policy

Relation threshold 只使用 train split：

~~~text
category pair n >= 50
  → category-pair quantile
else super-category pair n >= 50
  → super-category-pair quantile
else
  → global prior
~~~

n=50 已凍結；threshold 只建立 weak relation labels，不參與 geometry safety。

Preference negatives 可來自：

- geometric perturbation
- category-matched model swap
- style metadata swap
- cross-room hard-valid pose
- VLM silver preference
- human gold preference

不能只用當前 relation threshold 同時產生與評估 preference label。primary promotion evidence 是 human gold、V2T real E2E 與零 safety violation。

## 9. Split 與 leakage

~~~text
raw houses
  → stable house/design grouping
  → train/val/test assignment
  → freeze split hash
  → 各 split 獨立衍生 rooms/relations/negatives/queries/corpus
~~~

Near-duplicate signature 使用 room_type、category multiset、quantized floor area、aspect ratio、object count。所有 bins、fold policy、metric 與 threshold 必須寫入 near_duplicate_policy.json 並 hash；不得事後改 bins 讓 cross-split duplicate 消失。

正式報告同時提供 original house split 與 deduplicated sensitivity split。

## 10. Train/serve parity

所有 model features 登記於 feature_registry.json：

- training source
- serving source
- dtype／shape
- normalization
- required／masked policy
- noise model
- leakage risk
- version

強制規則：

1. 每個 training feature 都必須有 production serving source或本地 immutable product metadata。
2. required missing 時 fail closed。
3. optional 缺值使用 mask，不以 0 假裝觀測值。
4. normalization 與 policy version 必須和 checkpoint pin 相符。
5. 不能使用只有 3D-FRONT GT 才能取得的 inference feature。
6. V2T 不提供的 full free-space／clearance fact 不能進 production feature。
7. ideal-input 模型若通過 Gate 8 預註冊門檻，可直接進 Gate 9；若失敗且歸因 input noise，才依 V2T error model 重訓。
8. spot generation 必須 version/hash；Gate 8 比較 training spots 與 V2T proposals 分布，偏移時依 Gate 4 預註冊方法重校準。

必要 artifacts：

~~~text
feature_registry.json
spot_generation_policy.json
v2t_error_model.json
spot_distribution_comparison.json
input_noise_policy.json
~~~

## 11. Gate 2 的關鍵 stop condition

Gate 2 依 Gate 1 全量資料決定：

- coordinate/quaternion/scale/mirror policy
- static obstacle policy
- room_type_policy
- near_duplicate_policy
- supervision tiers 與 eligible coverage
- feature registry draft
- full-build storage estimate

room_type eligibility 候選不能直接當真值；必須依全量 vocabulary、geometry、join tiers 與 V2T 目標共同審核。

Round 4 的 RM-6 規定：

- supervision_coverage_floor.json 必須在 Gate 2 建立。
- 它可使用 Gate 1 audit，但必須盲於任何 model/baseline 結果。
- model_results_observed 必須為 false。
- Gate 2 以實測 tier/coverage 對照預先 hash 的 floor，成文做 proceed/no-proceed。
- Gate 4 只能重申或收緊，不得放寬。
- 若收緊後 Gate 3 dataset 不達標，停止 learned training。

## 12. Gate 0～9

| Gate | 內容 | 進下一關前的核心條件 |
|---:|---|---|
| 0 | 雙方書面審核 | Claude/Codex fully approved |
| 1 | 全量 raw audit | 100% source/child/join status、有 hashes、raw 未改 |
| 2 | 100-house canonical dry-run | dual-parser、coordinate、mesh、room type、coverage floor 與書面 proceed |
| 3 | full canonical dataset | immutable schemas/manifests、house split、leakage report、feature parity |
| 4 | baselines 與 evaluation preregistration | 指標、gold、promotion、test one-shot、V2T 選組、shift/calibration 全先凍結 |
| 5 | D1 Spatial Reranker | frozen ULIP、MLP/bilinear baseline、val-only selection、parity PASS |
| 6 | Spatial corpus/RAG A/B | C=Spatial RAG 未穩定勝 B=Spatial Reranker 就不 promotion |
| 7 | 3090 v3 service | 8322 獨立 process、auth/schema/pins/negative tests/readyz |
| 8 | 5090 V2T E2E | full-grid safety、exact pins、artifact validator、physical-room split |
| 9 | human-gold promotion | style ≥300 pairs、ranking ≥100–150 cases、agreement 與預註冊門檻 |

每一 Gate 都是 stop gate；不能因總計畫已核准就自動連跑。

## 13. V2T final evaluation pins

V2T evaluation scene groups 只能使用 5090 凍結並提供 hash 的 physical-room mapping/split：

~~~text
v2t_physical_room_mapping_sha256
v2t_split_snapshot_sha256
v2t_scene_policy_sha256
v2t_split_selection_rule_sha256
~~~

在快照未提供前：

- 3090 只能預註冊排除 dev fixture 0a7cc12c0e、兩個 scale-QA quarantine，以及 room-type stratification 規則。
- 不得先點名 final scene IDs。
- 可以做 synthetic validation/baseline。
- 不可宣稱 V2T final-test split 已凍結，也不可用 final results promotion。

## 14. Safety invariants

全部必須等於 0：

~~~text
known-size false-safe
post-rule collision violations
outside-floor violations
unknown mixed into verified_pass
learned score resurrected hard-invalid candidates
training-only features in production model
cross-split exact house leakage
test facts in training corpus
unjoinable instances silently dropped
unavailable geometry replaced by zero-size
obstacle-grade instances used as semantic supervision
rooms with unresolved furniture geometry used for preference training
unregistered training features
unversioned spot-generation policies
unversioned near-duplicate parameters
ineligible room types producing placement_pairs
Gate 2 proceed without a model-blind pre-pinned coverage floor
~~~

最後一項 validator 另外核對：

- policy exists、hash valid。
- Gate 1 hashes 相符。
- model_results_observed=false。
- Gate 2 decision 引用 exact policy hash。
- Gate 2 decision timestamp 早於 Gate 3 build。
- Gate 4 policy 等於或嚴於 Gate 2。

任一 invariant 不為 0，對應 Gate 失敗，不得 promotion。

## 15. 目前尚未確認、不可腦補

- raw up-axis、unit、quaternion order：Gate 2 才定案。
- replace_jid 語意：Gate 1/2 驗證前不覆蓋 primary jid。
- full join coverage：Gate 1 才有正式全量數字。
- static built-in mesh 類別：Gate 2 才凍結。
- canonical front：需 Gate 2 與人工標註。
- door/window/access reliability：v1.1／Gate 8。
- V2T error distribution：Gate 4/8。
- human-gold 排程：Gate 4/9。
- textures/render 是否擴充：不屬於 placement MVP。
- Gate 1 是否已在 repo 外執行：尚未確認。

## 16. 實作開始點

目前唯一由 final closure 開放的工作是 Gate 1 raw audit。Gate 2～9 仍未獲得跳關授權。Gate 1 完成後必須提交 artifacts、validation、hashes 與 GATE1_REPORT.md，再做下一輪書面審核。
