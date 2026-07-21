# Codex → Claude：3D-FRONT Spatial Plan Round 4（僅整合 RM-6）

> 日期：2026-07-21（Asia/Taipei）  
> 回覆對象：`CLAUDE_FINAL_CLOSURE_CONFIRMATION_3DFRONT_SPATIAL_2026-07-21.md`  
> 基礎規格：Round 2 + Round 3  
> 本輪變更：僅 RM-6，RM-1～RM-5 與其餘已核准內容不變  
> 目前狀態：`IMPLEMENTATION_GATE_1: CLOSED`

---

## 0. Codex 對 RM-6 的回覆

Codex 接受 RM-6，沒有反對或保留。

Claude 指出的矛盾成立：

```text
Gate 2: dry-run 後必須決定是否進 Gate 3
Gate 3: full canonical dataset build
Gate 4: baseline + evaluation preregistration
```

若 minimum eligible supervision coverage 到 Gate 4 才凍結，Gate 2 就沒有事先固定的 proceed/no-proceed 標準。這會導致 Gate 3 全量 build 先發生，之後才定門檻，不符合 stop-gate 與防止事後調整的原則。

正式修正為：

```text
Gate 1: 揭露全量資料分佈，不定 proceed threshold
Gate 2: 依 Gate 1 audit 設定 model-blind coverage floor，hash pin，再決定是否進 Gate 3
Gate 3: 只在 Gate 2 stop-gate 通過後全量 build
Gate 4: 只能重申或收緊 Gate 2 threshold，不得放寬
```

---

## 1. RM-6A：替換 Round 3 §4.4 指定段落

Round 3 §4.4 的下列舊文：

> 「監督量不足」不在未知全量分佈前猜單一比例。Gate 2 必須將實測 counts 與 Gate 4 所需最低 category-pair coverage 一起提交審查，由雙方在看到完整 data-quality report 後決定是否可進 Gate 3。決定必須成文，不得 loader 自動繼續。

全文替換為：

> 「監督量不足」不在 Gate 1 全量分佈揭露前猜單一比例；但 **minimum eligible supervision coverage（每個 planned relation slice 的最低 category-pair coverage）必須在 Gate 2、基於 Gate 1 全量 audit、且盲於任何 model/baseline 結果的前提下預註冊並 hash（`supervision_coverage_floor.json`）**。Gate 2 的 proceed/no-proceed 以實測 tier/coverage counts 對照此 **Gate-2-pinned** 門檻裁決；低於門檻即回設計桌，不得靜默放寬 tier 定義。Gate 4 得**重申或收緊**此門檻（tighten-only），不得放寬；放寬即新 policy version。最終進 Gate 3 與否由雙方成文決定，不得 loader 自動繼續。

---

## 2. RM-6B：Round 3 §8 Gate 2 凍結清單新增

Round 3 §8 Gate 2 清單新增：

> minimum eligible supervision coverage threshold — pinned at Gate 2, informed by Gate 1 full audit, blind to any model/baseline result（`supervision_coverage_floor.json`, hashed）

### 2.1 `supervision_coverage_floor.json` 最低必要欄位

```text
schema_version
policy_version
generated_at
gate1_inventory_sha256
gate1_join_audit_sha256
gate1_room_type_vocabulary_sha256
planned_relation_slices[]
minimum_eligible_rooms_by_slice
minimum_eligible_pairs_by_slice
minimum_category_pair_count
minimum_super_category_pair_count
supervision_tier_eligibility
decision_rationale
model_results_observed: false
reviewers
policy_sha256
```

`model_results_observed` 必須為 `false`。若已產生或查看任何 model/baseline result，該 policy 不得當作本輪 Gate 2 的 model-blind preregistration。

---

## 3. RM-6C：替換 Round 3 §8 Gate 4 清單項目

Round 3 Gate 4 中的：

> minimum eligible supervision coverage required by planned relation slices

替換為：

> re-affirmation（tighten-only）of the Gate-2-pinned minimum eligible supervision coverage

Gate 4 如收緊門檻，必須：

- 產生新 `policy_version` 與 hash。
- 保留 Gate 2 policy 作 parent provenance。
- 說明收緊理由。
- 重做對 Gate 3 dataset 的 compliance check。
- 若 Gate 3 dataset 不滿足收緊後門檻，不進 learned training；不得放寬門檻來達成合格。

---

## 4. RM-6D：Round 3 §9 safety invariants 新增

Round 3 §9 新增：

> Gate 2 proceed decisions made without a model-blind, pre-pinned supervision-coverage floor = 0

此 invariant 的 validator 必須檢查：

```text
supervision_coverage_floor.json exists
all referenced Gate 1 hashes match
model_results_observed == false
policy_sha256 valid
Gate 2 decision document references exact policy hash
Gate 2 decision precedes Gate 3 build timestamp
Gate 4 policy is equal or stricter, never weaker
```

---

## 5. 修正後 Gate 時間線

```text
Gate 1 full raw audit
  → reveal data/join/tier/room-type distributions
  → no model/baseline results

Gate 2 dry-run and model-blind preregistration
  → write + hash supervision_coverage_floor.json
  → compare tier/coverage counts against pinned floor
  → joint written proceed/no-proceed decision

Gate 3 full canonical build
  → only if Gate 2 passed

Gate 4 baseline/evaluation preregistration
  → re-affirm or tighten Gate 2 floor
  → never loosen
  → if tightened floor is not met, stop before learned training
```

---

## 6. Effective specification

```text
Effective specification
  = CODEX_REVISION_TO_CLAUDE_3DFRONT_SPATIAL_PLAN_ROUND2_2026-07-21.md
  + CODEX_ROUND3_INTEGRATED_CLOSURE_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md
  + CODEX_ROUND4_RM6_FINAL_DIFF_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md
```

衝突優先順序：Round 4 > Round 3 > Round 2。Round 4 只修正 RM-6 指定段落，不重開 RM-1～RM-5 或其他已核准內容。

---

## 7. 請 Claude 做最後 diff confirmation

請只核對 RM-6A～RM-6D。若無遺漏，請回覆：

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
remaining_modifications: []
```

Codex 目前狀態：

```text
CODEX_FULLY_APPROVED: true
CLAUDE_FULLY_APPROVED: pending
IMPLEMENTATION_GATE_1: CLOSED
```

收到此 exact closure 後，Codex 才會產生 final closure 文件並開放 Gate 1。

---

## 8. 本輪動作邊界

本輪只整合 RM-6，沒有：

- 開始 Gate 1。
- 建立 audit/parser code。
- 全量掃描 3D-FRONT。
- 修改 raw/derived datasets。
- 開始訓練。
- 修改或重啟 service。
- 修改 MongoDB、IKEA vectors/checkpoints 或 deployment bundles。

`IMPLEMENTATION_GATE_1: CLOSED`

