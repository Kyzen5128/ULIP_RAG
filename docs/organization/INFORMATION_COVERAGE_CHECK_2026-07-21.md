# Information Coverage and Deletion Check — 2026-07-21

目的：證明本輪整理沒有以「檔名相似」取代人工內容判斷，也沒有因整併閱讀版而遺失原始證據。

## 1. 檢查方法

1. 人工逐份閱讀 45 份 Markdown，共 9,738 行。
2. 分別核對用途、資料 snapshot、日期、執行狀態、獨有數字、commands、hashes、review wording 與外部引用。
3. SHA-256 只用來確認 exact duplicate 與搬移前後 bytes，沒有用來做內容分類。
4. 原始 files 全部保留；reader editions 是額外導航層，不宣稱取代 normative/evidence records。
5. 既有 dirty working tree 的 11 份 tracked deletions 不 restore、不覆蓋。

## 2. Reader editions 的來源覆蓋

### PROJECT_CURRENT_STATE.md

| 整合主題 | 原始來源 | 覆蓋結果 |
|---|---|---|
| 1,546 profile 與 serving mode | HANDOFF_3090_TO_5090_ULIP_RAG2_SERVICE_READY | 已列 profile、rows、512D、hard eligible、semantic vanilla |
| Semantic train/test | CODEX closure reply、Claude ack | 已列 split、selection discipline、250 epochs、5,215 秒、一次性 test |
| RAG promotion decision | Service ready、Claude ack | 已列 Stage 1/2 與不 promotion 原因 |
| Spatial architecture | Round 2、Round 3、Round 4、final closure | 已列 learned/rules/3090/5090 邊界與 Gate 1 狀態 |
| Data roles | final closure | IKEA、3D-FRONT、3D-FUTURE、V2T、VLM、Human 均列出 |
| 尚未確認 | 現有 records 缺乏後續 completion | 明列 5090 E2E、Gate 1、front/gold 尚未確認；8321 runtime 已於本輪唯讀重驗 |

### SEMANTIC_RAG_TRAINING_AND_SERVING.md

| 獨有資訊 | 保存方式 |
|---|---|
| train 1,212／val 200／test 134 | reader edition + 原始 records |
| checkpoint selection 只能用 val、test one-shot | reader edition + closure evidence |
| checkpoint aliases SHA 相等 | reader edition + Claude ack |
| final semantic metrics | reader edition + Claude ack |
| RAG corpus SHA | reader edition + service-ready record |
| Stage 1/2 epochs/best | reader edition + Claude ack |
| RAG-vs-vanilla A/B | reader edition + service-ready record |
| 1,546 deployment five-pin tuple | reader edition + service-ready record |
| HTTP/auth/readyz contract | reader edition + service-ready record |
| 733/1,546 不可混用 | reader warning + 所有 v1/v2 records |

### SPATIAL_V2_READER_EDITION.md

| Normative section | Reader coverage |
|---|---|
| Architecture and responsibility boundary | §2–4 |
| Parser pins and differential test | §5 |
| Gate 1 source/child/join/transform/metadata/mesh/room audit | §6 |
| Canonical schemas and supervision tiers | §7 |
| Relation thresholds and negatives | §8 |
| Split and near-duplicate leakage | §9 |
| Train/serve and spot proposal parity | §10 |
| Round 4 model-blind coverage floor | §11 |
| Gate 0–9 sequence | §12 |
| V2T physical-room pins | §13 |
| All safety invariants | §14 |
| Unconfirmed implementation evidence | §15 |

Reader edition 沒有逐字重複全部 rationale、review discussion 或 schemas 的每個欄位說明，因此原始九份核准鏈必須永久保留。

## 3. 各歷史鏈為何不能刪

### 7/21 Spatial review chain

不是九份同義草稿：

- Base：完整設計 rationale。
- Claude 初審：實測 optional fields、mesh、scale/quaternion 問題。
- Round 2：主規格。
- Claude Round 2：150-house 抽樣與 RM-1～5。
- Round 3：RM-1～5 正式生效文字。
- Claude RM-6 finding：stop-gate 時序矛盾推導。
- Round 4：RM-6 正式生效文字。
- Claude approval：FULLY_APPROVED 證據。
- Final closure：hash pins、優先序、Gate 1 authorization。

刪除任一 hash-pinned file 會破壞現有核准鏈。

### 7/16 design/training chain

早期文件有過時假設，但仍保存：

- unit-sphere 與 spatial scale 問題的推導。
- Semantic、RAG 與 spatial reranker 的責任拆分。
- 已搬到 5090 的 48-capture subset／47 provisional physical-room groups／9 DA3-completed 的校正過程，以及 mapping 尚未凍結的限制。
- VLM silver 與 human gold policy。
- checkpoint/test selection discipline。
- 實際訓練、test、Stage 1/2 執行證據。

因此整組標 historical／partly superseded，而不是 hard-delete。

### Product Candidates chain

- 733 decision：v1 contract。
- 733 implementation：immutable bundle 與本機測試。
- 733 E2E：真實跨機證據。
- 1,546 service ready：新 profile 與 A/B。

後一份不能取代前三份，因商品 snapshot、hash tuple 與 E2E 狀態不同。

### IKEA legacy reports

五份分別是尺寸 parser、crawl、cleanup、TRELLIS quality 與當時 phase plan，不是同一份報告。雖然不能當 current truth，統計、failure modes 與 Mongo history 各自唯一。

## 4. 已知過時內容

| File | 過時或有風險的內容 | 現行解讀 |
|---|---|---|
| CONSENSUS_ULIP_RAG_2_0... | 7-scene split、epoch 110/115 | 依檔頭與後續 closure 作廢 |
| HANDOFF_3090_TO_CLAUDE...SPATIAL_LEARNING_DESIGN | 舊 7-scene 前提 | 只作歷史 rationale |
| ikea/loop2/README.md | 原本寫 3D-FRONT/FUTURE 不在本機、V2T 僅 7 scenes | 已於交接前修正為 raw datasets 已在 P300、48-capture subset／47 provisional groups；Gate 1 與 mapping 仍未完成 |
| IKEA_CORE_RAG_INTEGRATION.md | 後段稱正式訓練/profile 未完成 | 已由 7/16 1,546 training/service-ready 取代 |
| phase_a_dimensions_result.md | range collapse、missing handling | 現行採 fail-closed／interval semantics |
| ulip_rag_phase_plan.md | 733 不重抓、架構永遠不動 | 已被 1,546 與 Spatial V2 取代 |
| HANDOFF_5090_LIVE_E2E... | 733 E2E | 不可聲稱為 1,546 E2E |

## 5. 整理前已存在的斷鏈

以下不是本輪移動造成：

- ikea/docs/phase_a_dimensions_result.md 指向不存在的 data_architecture_and_dimension_plan.md。
- Product Candidates decision 指向 repo 外的 V2T handoff。
- scripts/quick_train_portable.sh 與 core/quick_train.sh 指向已在 dirty tree 中刪除的 docs/NEXT_TASKS_HANDOFF.md。
- core/check_environment.py 指向已在 dirty tree 中刪除的 docs/PROJECT_REPORT_3090.md。
- V3_COLLECTION_CLEANUP_REPORT.md 提及不存在的 sync_dims_v2_to_v3.py。

本輪沒有擅自重建或偽造這些缺失內容。

## 6. 移動後引用處理

三個第一輪 compatibility symlinks 已依使用者要求移除，root 不再保留任何 .md entry。canonical paths 全部列在 MOVE_MANIFEST_2026-07-21.md；repo 外 immutable historical records 若保留舊絕對路徑，需依 move manifest 解析，不修改歷史 artifact 本體。

7/21 九份與 7/16 八份 records 保持同一平面目錄，原本只寫 basename 的交叉引用仍能在同目錄解析。

PointNeXt、component README 與 ikea/docs 保持原位，因此其 mkdocs、relative image、程式註解與 component context 未被破壞。

## 7. 刪除決策

~~~text
exact duplicates:          0
safe immediate deletions:  0
files deleted this round:  0
original files preserved: 45
~~~

最低風險刪除候選是 HANDOFF_3090_TO_5090_V2T_SCENE_DATASET_AUDIT_2026-07-16.md，但它仍保存原始盤點問題與責任邊界，所以本輪保留。

未來若要 hard-delete：

1. 指名確切文件。
2. 對其每個獨有數字、hash、command、reasoning 與外部引用建立逐段 crosswalk。
3. 排除 final closure 的 pinned chain。
4. 保存 deletion manifest 與可恢復副本。
5. 再次取得明確批准。
