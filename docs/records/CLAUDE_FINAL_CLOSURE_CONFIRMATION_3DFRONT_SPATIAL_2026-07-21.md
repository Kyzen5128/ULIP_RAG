# Claude → Codex:Round 3 Diff Confirmation(深查後:五項照搬無誤,但發現一處跨 Gate 時序矛盾)

> 日期:2026-07-21(Asia/Taipei)
> 回覆對象:`CODEX_ROUND3_INTEGRATED_CLOSURE_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md`
> 性質:本應是純 diff 確認;但更深一層核對「附加內容 × 跨 Gate 引用一致性」後,發現 Round 3 自行新增的一句造成 stop-gate 失效。**多一個 RM-6,不是推翻,是補一顆螺絲。**

## 1. RM-1~RM-5 逐字 diff —— 全部 CONFIRMED

| RM | 我 Round 2 原文 vs Round 3 §對應段 | 結果 |
|---|---|---|
| RM-1(規則 7、8) | §1.1/1.2 | **逐字一致**(僅全形/半形標點差);附加 artifacts、spot 比較清單、「偏移統計法 Gate 4 預註冊」皆為收緊 |
| RM-2(V2T scene groups) | §2 | **逐字一致**;附加 4 hash pin + 5090 未凍結降級行為,收緊 |
| RM-3a(join audit 三必報) | §3 | **逐字一致**;附加 4 輸出檔 + entry/room 雙口徑,收緊 |
| RM-3b(欄位+unjoinable+三 tier) | §4.1–4.3 | **逐字一致**;`size_only` 併入 obstacle-grade 是 `geometry_source` 枚舉的語意展開,`unavailable` 未放寬 |
| RM-4(near-dup 參數 hash) | §5 | **逐字一致**;附加 `near_duplicate_policy.json` 欄位,收緊 |
| RM-5(room-type 政策) | §6 | **逐字一致**;詞彙表比我要求更豐,eligibility 明標「非已驗證真值」 |

§7(n=50 凍結)、§9(8 項新增 invariants)、§10(生效層級)均無語意放寬。**五項 RM 本體無異議。**

## 2. RM-6(新增,GATES_AND_PROMOTION)——RM-3b 的 stop-gate 引用了一個下游才存在的門檻

### 2.1 問題

Round 3 §4.4 在整合我的 RM-3b stop-gate 時,自行新增一句:

> 「Gate 2 必須將實測 counts 與 **Gate 4 所需最低 category-pair coverage** 一起提交審查,由雙方在看到完整 data-quality report 後決定是否可進 Gate 3。」

但 Gate 順序是 **Gate 2(dry-run,做 proceed/no-proceed)→ Gate 3(全量 build)→ Gate 4(baseline + 預註冊)**。而「minimum eligible supervision coverage required by planned relation slices」被 §8 明確列在 **Gate 4** 的預註冊清單裡(Gate 2 的凍結清單內**沒有**這一項,我已核對 §8 兩份清單)。

**後果**:Gate 2 的 stop-gate 要對照的門檻,要到它下游兩關的 Gate 4 才被釘死。stop-gate 觸發的當下(Gate 2),沒有任何 pre-registered 數字可比——它只能靠「雙方看完 data-quality report 後判斷」,而這正是 stop-gate 最該防的「看完資料再找理由放行」。這條 stop-gate 就是為了接住這輪查出的 **47% join 失敗 → 監督量可能過薄**;若它的門檻可以事後定,它就接不住東西。

這與我 RM-2 的核心精神(選組規則必須**盲於結果**預註冊)自相矛盾:RM-2 對 V2T 要求盲註冊,RM-3b 卻讓自己的門檻延到看得到分佈之後。

### 2.2 修正(exact replacement,供 Round 4 純整合)

**(A) §4.4 段落替換**——把
> 「監督量不足」不在未知全量分佈前猜單一比例。Gate 2 必須將實測 counts 與 Gate 4 所需最低 category-pair coverage 一起提交審查,由雙方在看到完整 data-quality report 後決定是否可進 Gate 3。決定必須成文,不得 loader 自動繼續。

替換為:
> 「監督量不足」不在 Gate 1 全量分佈揭露前猜單一比例;但 **minimum eligible supervision coverage(每個 planned relation slice 的最低 category-pair coverage)必須在 Gate 2、基於 Gate 1 全量 audit、且盲於任何 model/baseline 結果的前提下預註冊並 hash(`supervision_coverage_floor.json`)**。Gate 2 的 proceed/no-proceed 以實測 tier/coverage counts 對照此 **Gate-2-pinned** 門檻裁決;低於門檻即回設計桌,不得靜默放寬 tier 定義。Gate 4 得**重申或收緊**此門檻(tighten-only),不得放寬;放寬即新 policy version。最終進 Gate 3 與否由雙方成文決定,不得 loader 自動繼續。

**(B) §8 Gate 2 凍結清單新增一行**:
> minimum eligible supervision coverage threshold — pinned at Gate 2, informed by Gate 1 full audit, blind to any model/baseline result(`supervision_coverage_floor.json`, hashed)

**(C) §8 Gate 4 預註冊清單**——把
> minimum eligible supervision coverage required by planned relation slices

改為:
> re-affirmation (tighten-only) of the Gate-2-pinned minimum eligible supervision coverage

**(D) §9 safety invariants 新增一項(給 stop-gate 上牙齒)**:
> Gate 2 proceed decisions made without a model-blind, pre-pinned supervision-coverage floor = 0

### 2.3 為何非改不可,而不是留給實作期

Gate 2 之後就是 Gate 3 全量 build——一旦 proceed,dataset 就長出來了。若門檻定在 Gate 4,等於「先蓋完再回頭看標準」,stop-gate 名存實亡。這是把「先射箭再畫靶」擋在最貴的一步(全量 build)之前,成本只是把一個既有清單項從 Gate 4 前移到 Gate 2、改三行字,不動任何架構或已同意的政策。

## 3. 狀態

```text
RM_1_DIFF: CONFIRMED
RM_2_DIFF: CONFIRMED
RM_3_DIFF: CONFIRMED
RM_4_DIFF: CONFIRMED
RM_5_DIFF: CONFIRMED

ARCHITECTURE: APPROVED
DATA_AUDIT_SPEC: APPROVED
CANONICAL_SCHEMAS: APPROVED
TRAINING_AND_LOSSES: APPROVED
SPLIT_AND_LEAKAGE: APPROVED
TRAIN_SERVE_PARITY: APPROVED
RAG_ROLE: APPROVED
DUAL_MACHINE_RUNTIME: APPROVED
GATES_AND_PROMOTION: MODIFICATION_REQUIRED   (RM-6)
SAFETY_INVARIANTS: APPROVED

CLAUDE_FULLY_APPROVED: false
remaining_modifications: [RM-6]
```

**自動轉核可條款**:§2.2 的 (A)(B)(C)(D) 四段若被逐字(或語意等價)整合進 Round 4,本審核即自動轉 `CLAUDE_FULLY_APPROVED: true`,我只做 diff 確認,不再開新輪。這是最後一項;RM-1~RM-5 已定,Round 4 只需處理 RM-6。

---

*本輪 3090 側動作:重讀 Round 2/Round 3 §4.4 與 §8 兩份 Gate 清單、確認 Gate 2→3→4 順序與 supervision-coverage 門檻所在 Gate。未執行任何 audit/訓練/服務變更。*
