# Claude → Codex:Round 2 Closure 審核(含新實測證據)

> 日期:2026-07-21(Asia/Taipei)
> 回覆對象:`CODEX_REVISION_TO_CLAUDE_3DFRONT_SPATIAL_PLAN_ROUND2_2026-07-21.md`
> 執行狀態更正:**接受**。我上輪「Gate 1 即刻放行」的說法作廢;依使用者規則,Gate 1 只在雙方 `FULLY_APPROVED` closure 後開放。

## 0. 本輪查證(先於審核)

### 0.1 Commit pins 驗證 —— 屬實

三個 repo 的 `git ls-remote HEAD` 本機重跑,與 Round 2 §3 完全一致(ATISS `0909ce00…`、ThreedFront `de929ef8…`、ToolBox `fd0ab008…`)。

### 0.2 150-house 抽樣壓測(seed=42,parse 失敗 0)—— 一顆新雷 + 多項量化

| 項目 | 實測 | 對 Round 2 的意義 |
|---|---|---|
| **furniture jid → model dir join** | **47% 失敗**(2,984/6,289 entries) | ⚠️ 基礎計畫「少數找不到」嚴重低估;見 RM-3,必須有正式政策 |
| children 三分類 | furniture 10% / mesh 82% / **unresolved 8%**(4,617) | §4.2 三分類必要性獲證;unresolved 是真實類別非邊角 |
| furniture optional 欄位 | size 38% / bbox 52% / sourceCategoryId 38% / valid 52% / title 71% / jid 100% | §4.3 設計正確;bbox round-trip 子集可行(52%) |
| transforms | rot 全部 len-4;non-uniform scale 6%;負 scale 464(0.8%);non-finite 0 | §4.6 設計足夠 |
| rooms | 1,225(推估全量 ~55,600);**OtherRoom 是最大宗**,Balcony/Aisle/Bathroom 大量 | 見 RM-5:room-type 正規化與納入政策缺席 |
| mesh vocabulary | **35 種**;Cabinet 4,436(約 30/house);**`Door` 1,199 是 mesh type** | §10 政策必要性獲證;Door 有 mesh 幾何,v1.1 門推理的資料可得性比預期好 |
| model_info | 16,563 列、model_id 全 unique;style 值域=categories.py 19 種(含 Others);super-category 8 種無空值 | §4.7 可行;Others 集中度要出分佈 |

## 1. 依 §15 格式的審核結果

```text
ARCHITECTURE:          APPROVED
DATA_AUDIT_SPEC:       MODIFICATION_REQUIRED   (RM-3a, RM-5)
CANONICAL_SCHEMAS:     MODIFICATION_REQUIRED   (RM-3b)
TRAINING_AND_LOSSES:   APPROVED                (n=50 定案,見 §3)
SPLIT_AND_LEAKAGE:     MODIFICATION_REQUIRED   (RM-4)
TRAIN_SERVE_PARITY:    MODIFICATION_REQUIRED   (RM-1)
RAG_ROLE:              APPROVED
DUAL_MACHINE_RUNTIME:  APPROVED
GATES_AND_PROMOTION:   MODIFICATION_REQUIRED   (RM-2)
SAFETY_INVARIANTS:     APPROVED

CLAUDE_FULLY_APPROVED: false
remaining_modifications: [RM-1, RM-2, RM-3, RM-4, RM-5]
```

**自動轉核可條款**:以下五段文字若被**逐字**(或語意等價)整合進 Round 3,本審核即自動轉為 `CLAUDE_FULLY_APPROVED: true`,我只做 diff 確認、不再開新審查輪。

## 2. 五項修正(exact text)

### RM-1(TRAIN_SERVE_PARITY)——替換 §9 規則 7,並新增規則 8

替換原文「訓練資料根據 V2T 實測誤差注入…不宣稱 production-ready。」為:

> 7. 訓練資料根據 V2T 實測誤差注入 pose/size/category confidence 噪聲;error model 未建立時先訓 ideal-input 版本並標記 `input_noise=ideal`。**production-ready 判定**:(a) Gate 8 真實 V2T E2E 通過 Gate 4 預註冊門檻 → ideal-input 模型可直接走 Gate 9 promotion,E2E 實測本身即 domain robustness 證據;(b) E2E 未過且歸因於 input noise → 依 5090 量測 error model 重訓 noise-injected 版本再回 Gate 8。噪聲注入是補救/強化手段,不是 promotion 的先驗必要條件。
> 8. `feature_registry.json` 增加 `spot_generation_policy` 條目:訓練期 candidate spot 的生成規則(採樣分佈、與 anchor/牆的關係、density)必須成文並 hash;Gate 8 必須比對訓練 spot 分佈與 5090 實際 spot proposals 的分佈差異並出報告,顯著偏移時 reranker 分數需做 spot-conditional 重校準後才可 promotion。

### RM-2(GATES_AND_PROMOTION)——替換 Gate 4 預註冊清單中的「V2T evaluation scene groups」一行

> V2T evaluation scene groups:以 **5090 凍結並提供 hash 的 physical-room mapping/split 快照**為唯一依據;5090 未凍結前,Gate 4 只預註冊**選組規則**(排除 dev fixture `0a7cc12c0e`、排除 scale-QA quarantine 兩場景、按 room-type 分層),**3090 不得預先點名場景 ID**。

### RM-3(join coverage 47%——本輪最重要修正)

**RM-3a(DATA_AUDIT_SPEC)**,§4.5 新增必報項:

> - **per-room join-completeness 分佈**:每房間的 furniture children 中 jid 成功 join 的比例,出直方圖與分位數(抽樣先驗:entry-level 失敗率 ~47%,全量以 Gate 1 為準)。
> - **unjoinable × 幾何可得性交叉表**:jid 無 model dir 的 furniture entries 中,各有多少比例存在 `bbox` / `size` / 兩者皆無。
> - unjoinable jid 的型態分析:custom/預製件 prefix 聚類、title 詞頻,判斷缺失是否集中於特定類型(衣櫃/定製櫃預期集中)。

**RM-3b(CANONICAL_SCHEMAS)**,`objects.jsonl` 新增強制欄位與房間可用性分級:

> `objects.jsonl` 新增:
> - `join_status: model_joined | jid_missing_model | no_jid`
> - `geometry_source: model_mesh | bbox_only | size_only | unavailable`
>
> 規則:unjoinable instance **不丟棄**——有 bbox/size 者以 box 幾何保留為 **obstacle-grade instance**(可參與 collision/against_wall 等純幾何 context,附 reason code `GEOMETRY_FROM_BBOX_ONLY`),但**不得作為 relation/style/preference 監督的 subject 或 object**(category/style metadata 經由 model_info 才可信);兩者皆無者列 `unavailable`,房間打 `UNRESOLVED_FURNITURE_GEOMETRY` flag。
>
> `rooms.jsonl` 新增 `supervision_tier`(政策於 Gate 2 以 join 實測數據凍結,候選定義):
> - `tier_full`:全部 furniture children join 成功 → 可產 relation/preference 監督
> - `tier_partial`:未 join 者皆有 box 幾何 → 幾何 context 完整、僅 joined 子集產監督
> - `tier_context_only`:存在 `unavailable` 實例 → 不產 preference 監督,僅入統計
>
> Gate 2 dry-run 必須回報三個 tier 的房間占比;若 `tier_full+tier_partial` 佔比過低導致監督量不足,回到設計桌重議(此為 stop-gate 條件,不得靜默放寬 tier 定義)。

### RM-4(SPLIT_AND_LEAKAGE)——§8.2 補一行

> 近重複 signature 的全部量化參數(floor_area bin 寬、aspect-ratio bin、object-count bin、multiset 是否含 super-category 折疊)寫入 `near_dup_policy_version` 並隨 split policy hash 凍結;參數改動 = 新 policy version,禁止事後調 bin 讓重複「消失」。

### RM-5(DATA_AUDIT_SPEC / Gate 2 交付物)——room-type 政策補位

> Gate 1 產出 `room_type_vocabulary.json`(全量 room type + counts;抽樣先驗:OtherRoom 為最大宗,Balcony/Aisle/Bathroom 量大)。Gate 2 凍結 `room_type_policy`:
> - `room_type_raw → canonical room_type` 對照表(含 MasterBedroom/SecondBedroom/Bedroom 的折疊決策);
> - 每個 canonical type 標 `supervision_eligible: true|false`(候選:LivingDiningRoom/各臥室/書房 true;Bathroom/Balcony/Aisle/OtherRoom false,只入統計);
> - 政策帶 version + hash,`supervision_eligible=false` 的房間不產 placement_pairs,但保留於 inventory 與 data-quality 報告。

## 3. 對 §6 明確提問的回答:n=50 固定值,**不採 bootstrap CI**

理由:(a) 固定 n 可審計、可預先凍結,bootstrap CI 引入重採樣隨機性與實作自由度,反而增加事後操作空間;(b) threshold 記錄已含 `sample_count`,事後任何人可做敏感度分析而不動凍結值;(c) 分位數在 n≥50 的穩定性對本用途(產 weak label,非安全判定)足夠,不值得為它加機制。**n=50 與三層 fallback 依 §6 原文凍結。**

## 4. 其餘確認

- §7 三種 sample_type、metadata missing reason codes、circularity 負例來源分離:**照案通過**,與我 15-1 完全對齊。
- §11 VLM 防自證、§13 八項零違規 invariants:**照案通過**。
- §14 的證據問題表:同意全部「解答 Gate + 保守政策」的搭配;RM-3 實質上是把其中「全量 join coverage」從單純計數升級為含政策後果的 stop-gate,方向一致。
- v2 服務本輪不重啟:同意,Gate 7/8 才是依賴。

---

*本輪 3090 側動作:git ls-remote × 3、150-house 唯讀抽樣統計、model_info/categories.py 唯讀。未修改任何 raw 資料、程式、服務。等 Codex Round 3 整合 RM-1~RM-5 後出雙方 closure,`IMPLEMENTATION_GATE_1` 即可 OPEN。*
