# ULIP_RAG 文件入口

最後整理日期：2026-07-21（Asia/Taipei）

本目錄是 ULIP_RAG 專案級 Markdown 的主要入口。根目錄原有的 22 份交接、審查與 closure 紀錄已原封不動移入 records/；套件旁的 README、IKEA 元件文件與 vendored PointNeXt 文件則保留原位，避免破壞相對連結或失去程式上下文。

## 給 Claude 的交接規則

1. 必須從本文件開始，不要直接挑 records/ 裡檔名最新的文件當現行規格。
2. 先依下方「先看哪裡」讀完三份 current reader docs，再讀 final closure 與 1,546 service handoff。
3. Spatial V2 的 normative specification 只有 Round 2＋Round 3＋Round 4；Base 是 rationale，Claude review/approval 與 final closure 是證據。
4. records/ 中的 733、早期 7-scene、舊 Mongo、舊 Phase A/B/C 內容均不可覆蓋 current 文件的修正結論。
5. SERVICE_READY 代表指定 profile 在指定時間點 ready，不等於 5090 的 1,546 real-scene E2E 已完成。
6. 文件標示「尚未確認」的內容不可自行補完；實作前必須再核對 repo、P300 artifacts、service pins 與目前 dirty working tree。

## 先看哪裡

1. [目前專案狀態](current/PROJECT_CURRENT_STATE.md)
2. [Semantic ULIP、RAG 訓練與 1,546 serving](current/SEMANTIC_RAG_TRAINING_AND_SERVING.md)
3. [Spatial V2 核准規格閱讀版](current/SPATIAL_V2_READER_EDITION.md)
4. [7/21 最終 closure](records/FINAL_CLOSURE_CLAUDE_CODEX_3DFRONT_SPATIAL_PLAN_FULLY_APPROVED_2026-07-21.md)
5. [1,546 serving handoff](records/HANDOFF_3090_TO_5090_ULIP_RAG2_SERVICE_READY_2026-07-16.md)
6. [V2T 檢索、Style／Placement RAG 實作拆分與 Claude 審查要求](records/HANDOFF_CODEX_TO_CLAUDE_V2T_RETRIEVAL_AND_PLACEMENT_RAG_REVIEW_2026-07-21.md)

第 6 份是 REVIEW REQUEST／NON-NORMATIVE EXPLANATION，用來請 Claude 核對現行設計拆分；它不取代已核准的 Round 2＋3＋4。

## 文件狀態

| 標記 | 意義 |
|---|---|
| CURRENT | 目前理解專案時應優先閱讀 |
| NORMATIVE | 已核准且對實作有約束力 |
| EVIDENCE | 訓練、測試、deployment 或 E2E 的可稽核證據 |
| HISTORICAL | 保留決策與實驗歷史，不代表現行狀態 |
| SUPERSEDED | 已被後續規格取代，不得單獨當作現行依據 |
| COMPONENT | 應留在程式或資產旁的操作文件 |
| VENDORED | 第三方上游文件，不屬於本專案交接雜訊 |

## 目前權威邊界

- IKEA 1,546 商品服務的最後成文 profile 是 semantic vanilla，不是 RAG runtime default。
- 2026-07-21 Spatial V2 設計已完成 Claude × Codex 核准；最終 closure 成文時只開放 Gate 1，Gate 1 執行狀態仍是 NOT_STARTED。
- Spatial V2 的正式規格不是單一草稿，而是 Round 2、Round 3、Round 4，優先序為 Round 4 > Round 3 > Round 2。
- current/SPATIAL_V2_READER_EDITION.md 是方便閱讀的整合說明，不取代 hash-pinned normative files。
- 舊 IKEA 733 E2E、舊 3,073/2,516 Mongo crawl 與舊 Phase A/B/C 計畫均為歷史資料，不能當作目前 1,546 pipeline 的狀態。
- 2026-07-21T12:57:29+08:00 交接前已唯讀重驗：127.0.0.1:8321、PID 2668533 與 tmux ulip-product-candidates-v2 仍存在，healthz/readyz 都是 200，1,546 profile 的七項 checks 全 true。這是時間點證據，未來使用前仍須重驗。

## 7/21 Spatial V2 核准鏈

以下九份依序構成完整設計與審查證據，全部保留原始內容：

1. [Base rationale](records/HANDOFF_CODEX_TO_CLAUDE_3DFRONT_SPATIAL_TRAINING_PLAN_REVIEW_2026-07-21.md)
2. [Claude 初審](records/CLAUDE_REVIEW_3DFRONT_SPATIAL_TRAINING_PLAN_2026-07-21.md)
3. [Round 2 normative specification](records/CODEX_REVISION_TO_CLAUDE_3DFRONT_SPATIAL_PLAN_ROUND2_2026-07-21.md)
4. [Claude Round 2 review](records/CLAUDE_ROUND2_CLOSURE_REVIEW_3DFRONT_SPATIAL_2026-07-21.md)
5. [Round 3 normative addendum](records/CODEX_ROUND3_INTEGRATED_CLOSURE_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md)
6. [Claude RM-6 finding](records/CLAUDE_FINAL_CLOSURE_CONFIRMATION_3DFRONT_SPATIAL_2026-07-21.md)
7. [Round 4 normative addendum](records/CODEX_ROUND4_RM6_FINAL_DIFF_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md)
8. [Claude full approval](records/CLAUDE_RM6_CONFIRM_FULLY_APPROVED_3DFRONT_SPATIAL_2026-07-21.md)
9. [Final closure](records/FINAL_CLOSURE_CLAUDE_CODEX_3DFRONT_SPATIAL_PLAN_FULLY_APPROVED_2026-07-21.md)

## 7/16 ULIP_RAG 2.0 設計與訓練紀錄

這一組是重要歷史與執行證據，但部分早期假設已由後續文件修正：

- [原始 spatial learning design](records/HANDOFF_3090_TO_CLAUDE_ULIP_RAG_2_0_SPATIAL_LEARNING_DESIGN_2026-07-16.md)
- [Claude spatial review](records/CLAUDE_REVIEW_ULIP_RAG_2_0_SPATIAL_LEARNING_2026-07-16.md)
- [Claude data/training review](records/CLAUDE_REVIEW_ULIP_RAG_2_0_DATA_TRAINING_V2T_2026-07-16.md)
- [Consensus，含檔頭作廢聲明](records/CONSENSUS_ULIP_RAG_2_0_CLAUDE_CODEX_2026-07-16.md)
- [Codex 最新事實校正](records/HANDOFF_CODEX_TO_CLAUDE_ULIP_RAG_2_0_LATEST_REVIEW_2026-07-16.md)
- [Claude 回覆](records/CLAUDE_REPLY_TO_CODEX_ULIP_RAG_2_0_LATEST_REVIEW_2026-07-16.md)
- [Codex closure 回覆](records/CODEX_REPLY_TO_CLAUDE_ULIP_RAG_2_0_REVIEW_CLOSURE_2026-07-16.md)
- [Claude closure ack](records/CLAUDE_ACK_CODEX_CLOSURE_ULIP_RAG_2_0_2026-07-16.md)

已確認的主要作廢內容包括：

- V2T 不是只有 7 scenes；後續盤點是已搬到 5090 的 48-capture subset、47 個 provisional physical-room groups、9 個 DA3-completed captures。physical-room mapping 尚未凍結，不能把 47 當成最終 leakage-free split 數。
- 不存在可直接使用的 epoch 110/115 checkpoint。
- checkpoint 與超參數只能使用 validation 選擇；134-product test 只允許在唯一 checkpoint 選定後執行一次。

## Product Candidates 與跨機紀錄

| 文件 | 商品庫 | 狀態 |
|---|---:|---|
| [v1 決策與執行規格](records/HANDOFF_5090_TO_3090_PRODUCT_CANDIDATES_DECISIONS_AND_EXEC_2026-07-15.md) | 733 | NORMATIVE／HISTORICAL |
| [v1 實作結果](records/HANDOFF_3090_IMPLEMENTED_PRODUCT_CANDIDATES_TO_5090.md) | 733 | EVIDENCE／HISTORICAL |
| [v1 真實跨機 E2E](records/HANDOFF_5090_LIVE_E2E_ACCEPTANCE_TO_3090_2026-07-16.md) | 733 | EVIDENCE／HISTORICAL |
| [v2 SERVICE_READY](records/HANDOFF_3090_TO_5090_ULIP_RAG2_SERVICE_READY_2026-07-16.md) | 1,546 | CURRENT／EVIDENCE |
| [V2T scene audit request](records/HANDOFF_3090_TO_5090_V2T_SCENE_DATASET_AUDIT_2026-07-16.md) | 不適用 | HISTORICAL |

733 與 1,546 是不同 snapshot/profile，不得把兩份 E2E 或 hash tuple 混用。

## 歷史 IKEA 文件

- [IKEA 733 Core RAG integration](history/ikea733_core_rag/IKEA_CORE_RAG_INTEGRATION.md)：保存 legacy1095、Stage 1/2、API、測試與舊 deployment 細節；後段的未完成狀態已被 7/16 結果取代。
- [舊 733 TRELLIS 品質盤點](../ikea/docs/3d_model_quality_audit.md)
- [舊 Mongo 尺寸解析](../ikea/docs/phase_a_dimensions_result.md)
- [舊 3,073-row crawl](../ikea/docs/phase_v3_scrape_result.md)
- [舊 3,073 → 2,516 cleanup](../ikea/docs/V3_COLLECTION_CLEANUP_REPORT.md)
- [已被 Spatial V2 取代的舊 Phase A/B/C 計畫](../ikea/docs/ulip_rag_phase_plan.md)

以上 ikea/docs 文件保留在原位，因彼此有歷史引用，且 3D model audit 也被程式註解引用。

## Component-local 文件

- [IKEA 1,546 data loop](../ikea/loop2/README.md)：COMPONENT／CURRENT；已同步修正 3D-FRONT/FUTURE 與 V2T inventory 狀態。
- [Product Candidates contracts](../ikea/product_candidates/contracts/README.md)：COMPONENT／NORMATIVE。
- [Product Candidates deployment](../ikea/product_candidates/deploy/README.md)：COMPONENT。
- [Core archive 說明](../core/_archive/README.md)：COMPONENT。

core/models/pointnext/PointNeXt 下的 Markdown 全部是 VENDORED 上游文件，由 mkdocs.yml、相對圖片和相對連結約束，維持原目錄結構。

## 整理與稽核

- [原始 Markdown inventory](organization/MARKDOWN_INVENTORY_2026-07-21.md)
- [搬移 manifest](organization/MOVE_MANIFEST_2026-07-21.md)
- [SHA-256 manifest](organization/SHA256_MANIFEST_2026-07-21.md)
- [資訊覆蓋與刪除檢查](organization/INFORMATION_COVERAGE_CHECK_2026-07-21.md)

本輪原始 Markdown source files 刪除數量為 0；root 的三個 compatibility symlinks 已依使用者要求移除。任何未來 source hard-delete 都必須先更新 coverage matrix、確認沒有外部引用、重新驗證 hash chain，並再次取得明確批准。
