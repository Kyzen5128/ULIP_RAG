# Claude × Codex：3D-FRONT Spatial Training Plan 最終雙方 Closure

> 日期：2026-07-21（Asia/Taipei）  
> 專案：ULIP_RAG 2.0 × V2T Spatial Placement Loop  
> 性質：雙方設計審核最終收旂與 implementation gate 狀態  
> 本文不是 Gate 1 執行報告，也不代表 audit 已開始。

---

## 1. 最終核准狀態

```text
RM_1_DIFF: CONFIRMED
RM_2_DIFF: CONFIRMED
RM_3_DIFF: CONFIRMED
RM_4_DIFF: CONFIRMED
RM_5_DIFF: CONFIRMED
RM_6_DIFF: CONFIRMED

ARCHITECTURE: APPROVED
DATA_AUDIT_SPEC: APPROVED
CANONICAL_SCHEMAS: APPROVED
TRAINING_AND_LOSSES: APPROVED
SPLIT_AND_LEAKAGE: APPROVED
TRAIN_SERVE_PARITY: APPROVED
RAG_ROLE: APPROVED
DUAL_MACHINE_RUNTIME: APPROVED
GATES_AND_PROMOTION: APPROVED
SAFETY_INVARIANTS: APPROVED

CLAUDE_FULLY_APPROVED: true
CODEX_FULLY_APPROVED: true
remaining_modifications: []

IMPLEMENTATION_GATE_1: OPEN
GATE_1_EXECUTION_STATUS: NOT_STARTED
```

`OPEN` 表示規格審核已完成、允許之後依規格執行；`NOT_STARTED` 表示本 closure 建立時尚未開始 Gate 1 raw audit。

---

## 2. Normative specification

正式實作規格由以下三份文件組成：

1. `CODEX_REVISION_TO_CLAUDE_3DFRONT_SPATIAL_PLAN_ROUND2_2026-07-21.md`
2. `CODEX_ROUND3_INTEGRATED_CLOSURE_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md`
3. `CODEX_ROUND4_RM6_FINAL_DIFF_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md`

衝突優先序：

```text
Round 4 > Round 3 > Round 2
```

`HANDOFF_CODEX_TO_CLAUDE_3DFRONT_SPATIAL_TRAINING_PLAN_REVIEW_2026-07-21.md` 保留作為設計 rationale/reference，不凌駕 Round 2～4 normative spec。

### 2.1 文件簽章

```text
Base rationale
b9f951de6e596e447122bab618cb7e6934a9203b5a874a0ce7e731bb694edf31
HANDOFF_CODEX_TO_CLAUDE_3DFRONT_SPATIAL_TRAINING_PLAN_REVIEW_2026-07-21.md

Round 2
fd2269d79501ad0d6d697386c58d93281c104b04a7014a23d8bc917513c54dc4
CODEX_REVISION_TO_CLAUDE_3DFRONT_SPATIAL_PLAN_ROUND2_2026-07-21.md

Round 3
ddab7ddc5b32e500871813dc042143d4aa8d0c98de0fa0439cb023716a9052b1
CODEX_ROUND3_INTEGRATED_CLOSURE_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md

Round 4
dc7f89308188375000f8af5cd8b72a1039a86cbe45406c8d9748380fc29ca92b
CODEX_ROUND4_RM6_FINAL_DIFF_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md

Claude final approval
db08754e4bc2fe864b7854825b83d10a19a2e1390ce299b0ef785760588cd992
CLAUDE_RM6_CONFIRM_FULLY_APPROVED_3DFRONT_SPATIAL_2026-07-21.md
```

任一 normative file 內容改動都會改變 SHA-256，需新的 review/version，不得沿用本 closure 聲稱仍獲核准。

---

## 3. 已凍結架構

```text
V2T query + compact scene context + candidate spots
  → 3090 Semantic ULIP retrieval against IKEA 1,546 gallery
  → 3090 candidate expansion: Top-M × spots × allowed footprint yaws
  → deterministic hard gate
  → 3090 Spatial Reranker on hard-valid candidates only
  → 5090 authoritative full-grid/collision/path/access revalidation
  → final placement + alternatives + rejection evidence
```

模型學習：

```text
room-product style compatibility
furniture relation compatibility
human preference among hard-valid candidates
```

規則決定：

```text
known/interval dimension fit
footprint containment
collision
outside-floor
observed-space validity
walkability/path/access when data exists
```

learned score 永不得復活 hard-invalid candidate。

---

## 4. 已凍結資料角色

| 資料 | 角色 |
|---|---|
| IKEA 1,546 | production products、semantic retrieval/indexing、product metadata |
| 3D-FRONT | synthetic room/layout/pose/relation supervision |
| 3D-FUTURE-model | 3D-FRONT furniture geometry/category/style/material metadata |
| V2T | real-domain E2E evaluation/calibration |
| VLM | provenance-pinned silver/weak labels |
| Human | final gold preference/front/quality audit |

3D-FRONT/FUTURE 不取代 IKEA production gallery；V2T 不在 final physical-room split 凍結前被當成無泄漏的 training set。

---

## 5. Gate 1 允許執行的範圍

Gate 1 只是全量、read-only、CPU/IO raw audit。

允許：

- 讀取 3D-FRONT JSON、3D-FUTURE model directories 與 metadata。
- 計算 source manifests/hashes。
- 建立 children 三分類、join status、optional-field coverage、transform inventory。
- 建立 mesh vocabulary、unjoinable geometry cross-tab、room-type vocabulary。
- 建立 audit code、validators、reports 與新的 versioned derived audit directory。

禁止：

- 修改 raw 3D-FRONT/3D-FUTURE。
- 產生正式 relation labels/placement training pairs。
- 開始 spatial model/RAG training。
- 修改 MongoDB、IKEA vectors/checkpoints/deployment bundles。
- 修改或重啟 v2/v3 service。
- 將 quaternion/unit/front/replace_jid 的未驗證推論當成事實。
- 預估新產物超過 30G 仍繼續；必須先停止回報。

---

## 6. Gate 1 必要交付物

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

per_room_join_completeness.jsonl
join_completeness_histogram.json
unjoinable_geometry_crosstab.json
unjoinable_jid_pattern_analysis.json
room_type_vocabulary.json
```

每份 artifact 必須有 schema/version/provenance/hash；每個 source JSON、room child 與 join row 都必須有 deterministic status 或 reason code，禁止靜默略過。

---

## 7. Gate 1 後仍然必須停下審核

Gate 1 完成不代表可自動進 Gate 2。Gate 1 報告需先提交雙方審核。

Gate 2 才處理：

- 100-house stratified canonical dry-run。
- coordinate/quaternion/scale/mirror policy。
- dual-parser differential test。
- static-obstacle policy。
- room-type policy。
- supervision tiers。
- near-duplicate policy。
- model-blind `supervision_coverage_floor.json` 與 hash。
- Gate 2 proceed/no-proceed 的雙方書面裁決。

---

## 8. Final safety invariants

```text
known-size false-safe = 0
post-rule collision violations = 0
outside-floor violations = 0
unknown mixed into verified_pass = 0
learned score resurrected hard-invalid candidates = 0
training-only features in production model = 0
cross-split exact house leakage = 0
test facts in training corpus = 0
unjoinable instances silently dropped = 0
unavailable geometry replaced by zero-size = 0
obstacle-grade instances used as semantic supervision = 0
rooms with unresolved furniture geometry used for preference training = 0
unregistered training features = 0
unversioned spot-generation policies = 0
unversioned near-duplicate parameters = 0
ineligible room types producing placement_pairs = 0
Gate 2 proceed decisions made without a model-blind, pre-pinned supervision-coverage floor = 0
```

任一 invariant 不為 0，對應 Gate 失敗，不得 promotion。

---

## 9. Closure 結論

Claude 與 Codex 已完成全案審查，RM-1～RM-6 已全數收旂，沒有剩餘修改或技術分歧。

```text
DESIGN_REVIEW: CLOSED
CLAUDE_FULLY_APPROVED: true
CODEX_FULLY_APPROVED: true
remaining_modifications: []
IMPLEMENTATION_GATE_1: OPEN
GATE_1_EXECUTION_STATUS: NOT_STARTED
```

本 closure 只開放 Gate 1，不開放 Gate 2～9。後續每一 Gate 仍必須依 normative spec 完成交付、驗證與書面審核。

