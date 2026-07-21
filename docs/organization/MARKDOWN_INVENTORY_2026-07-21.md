# Markdown Inventory — 2026-07-21

盤點範圍：整理開始前 /home/kyzen/ULIP_RAG 內現存的 Markdown。  
盤點方法：逐份人工閱讀；沒有用關鍵字、相似度或自動分類腳本決定去向。  
基線：45 files、9,738 lines；root 22、non-root 23。  
本表不把本次整理新建立的索引、閱讀版與 organization manifests 算回原始基線。

## 1. Root records：22 份

以下全部已保持原檔名與原始 bytes 移入 docs/records/。

| Current file | Status | 人工判斷與唯一資訊 |
|---|---|---|
| [CLAUDE_ACK_CODEX_CLOSURE_ULIP_RAG_2_0_2026-07-16.md](../records/CLAUDE_ACK_CODEX_CLOSURE_ULIP_RAG_2_0_2026-07-16.md) | EVIDENCE／HISTORICAL | checkpoint identity、val/test 時序、一次性 test、RAG Stage 1/2 追記 |
| [CLAUDE_FINAL_CLOSURE_CONFIRMATION_3DFRONT_SPATIAL_2026-07-21.md](../records/CLAUDE_FINAL_CLOSURE_CONFIRMATION_3DFRONT_SPATIAL_2026-07-21.md) | REVIEW EVIDENCE | RM-1～5 diff confirmation 與 RM-6 跨 Gate 時序問題 |
| [CLAUDE_REPLY_TO_CODEX_ULIP_RAG_2_0_LATEST_REVIEW_2026-07-16.md](../records/CLAUDE_REPLY_TO_CODEX_ULIP_RAG_2_0_LATEST_REVIEW_2026-07-16.md) | HISTORICAL | 48-capture subset／47 provisional groups 校正、split、human gold、VLM provenance、防 leakage |
| [CLAUDE_REVIEW_3DFRONT_SPATIAL_TRAINING_PLAN_2026-07-21.md](../records/CLAUDE_REVIEW_3DFRONT_SPATIAL_TRAINING_PLAN_2026-07-21.md) | REVIEW EVIDENCE | optional fields、built-in mesh、scale/quaternion、dual parser 實測依據 |
| [CLAUDE_REVIEW_ULIP_RAG_2_0_DATA_TRAINING_V2T_2026-07-16.md](../records/CLAUDE_REVIEW_ULIP_RAG_2_0_DATA_TRAINING_V2T_2026-07-16.md) | SUPERSEDED／HISTORICAL | 訓練 log 有效；7-scene、patience 30、front prior 後來作廢 |
| [CLAUDE_REVIEW_ULIP_RAG_2_0_SPATIAL_LEARNING_2026-07-16.md](../records/CLAUDE_REVIEW_ULIP_RAG_2_0_SPATIAL_LEARNING_2026-07-16.md) | HISTORICAL | unit-sphere 無公制尺度、learn vs rule 邊界、3D-FRONT 角色 |
| [CLAUDE_RM6_CONFIRM_FULLY_APPROVED_3DFRONT_SPATIAL_2026-07-21.md](../records/CLAUDE_RM6_CONFIRM_FULLY_APPROVED_3DFRONT_SPATIAL_2026-07-21.md) | APPROVAL EVIDENCE | Claude FULLY_APPROVED；被 final closure hash-pin |
| [CLAUDE_ROUND2_CLOSURE_REVIEW_3DFRONT_SPATIAL_2026-07-21.md](../records/CLAUDE_ROUND2_CLOSURE_REVIEW_3DFRONT_SPATIAL_2026-07-21.md) | REVIEW EVIDENCE | 150-house 抽樣、join failure、mesh/room type 統計、RM-1～5 |
| [CODEX_REPLY_TO_CLAUDE_ULIP_RAG_2_0_REVIEW_CLOSURE_2026-07-16.md](../records/CODEX_REPLY_TO_CLAUDE_ULIP_RAG_2_0_REVIEW_CLOSURE_2026-07-16.md) | HISTORICAL | val-only selection、test one-shot、實存 checkpoints |
| [CODEX_REVISION_TO_CLAUDE_3DFRONT_SPATIAL_PLAN_ROUND2_2026-07-21.md](../records/CODEX_REVISION_TO_CLAUDE_3DFRONT_SPATIAL_PLAN_ROUND2_2026-07-21.md) | NORMATIVE | Gate 1～9、schemas、parity、8322 主規格 |
| [CODEX_ROUND3_INTEGRATED_CLOSURE_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md](../records/CODEX_ROUND3_INTEGRATED_CLOSURE_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md) | NORMATIVE | RM-1～5：spot parity、join tiers、room type、near duplicate |
| [CODEX_ROUND4_RM6_FINAL_DIFF_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md](../records/CODEX_ROUND4_RM6_FINAL_DIFF_CANDIDATE_3DFRONT_SPATIAL_2026-07-21.md) | NORMATIVE | RM-6：Gate-2 model-blind coverage floor |
| [CONSENSUS_ULIP_RAG_2_0_CLAUDE_CODEX_2026-07-16.md](../records/CONSENSUS_ULIP_RAG_2_0_CLAUDE_CODEX_2026-07-16.md) | PARTLY SUPERSEDED | unknown/range/front/rules 等決策；檔頭明列已作廢段落 |
| [FINAL_CLOSURE_CLAUDE_CODEX_3DFRONT_SPATIAL_PLAN_FULLY_APPROVED_2026-07-21.md](../records/FINAL_CLOSURE_CLAUDE_CODEX_3DFRONT_SPATIAL_PLAN_FULLY_APPROVED_2026-07-21.md) | NORMATIVE CLOSURE | 優先序、hashes、Gate 1 OPEN、safety invariants |
| [HANDOFF_3090_IMPLEMENTED_PRODUCT_CANDIDATES_TO_5090.md](../records/HANDOFF_3090_IMPLEMENTED_PRODUCT_CANDIDATES_TO_5090.md) | EVIDENCE／HISTORICAL | 733 v1 bundle、五 hashes、測試、dirty-state、啟動證據 |
| [HANDOFF_3090_TO_5090_ULIP_RAG2_SERVICE_READY_2026-07-16.md](../records/HANDOFF_3090_TO_5090_ULIP_RAG2_SERVICE_READY_2026-07-16.md) | CURRENT／EVIDENCE | 1,546 profile、七 checks、five-pin tuple、Semantic vs RAG A/B |
| [HANDOFF_3090_TO_5090_V2T_SCENE_DATASET_AUDIT_2026-07-16.md](../records/HANDOFF_3090_TO_5090_V2T_SCENE_DATASET_AUDIT_2026-07-16.md) | HISTORICAL | 原始 V2T inventory 問題；後續回答已吸收但仍是驗收問題來源 |
| [HANDOFF_3090_TO_CLAUDE_ULIP_RAG_2_0_SPATIAL_LEARNING_DESIGN_2026-07-16.md](../records/HANDOFF_3090_TO_CLAUDE_ULIP_RAG_2_0_SPATIAL_LEARNING_DESIGN_2026-07-16.md) | SUPERSEDED／HISTORICAL | 完整 Stage 0～8 舊設計；包含後來修正的 7-scene 前提 |
| [HANDOFF_5090_LIVE_E2E_ACCEPTANCE_TO_3090_2026-07-16.md](../records/HANDOFF_5090_LIVE_E2E_ACCEPTANCE_TO_3090_2026-07-16.md) | EVIDENCE／HISTORICAL | 733 真實跨機 request、product、pose、score 與 artifact hashes |
| [HANDOFF_5090_TO_3090_PRODUCT_CANDIDATES_DECISIONS_AND_EXEC_2026-07-15.md](../records/HANDOFF_5090_TO_3090_PRODUCT_CANDIDATES_DECISIONS_AND_EXEC_2026-07-15.md) | NORMATIVE／HISTORICAL | 733 v1 category/token/dimension/security/readyz contract |
| [HANDOFF_CODEX_TO_CLAUDE_3DFRONT_SPATIAL_TRAINING_PLAN_REVIEW_2026-07-21.md](../records/HANDOFF_CODEX_TO_CLAUDE_3DFRONT_SPATIAL_TRAINING_PLAN_REVIEW_2026-07-21.md) | PINNED RATIONALE | Spatial V2 大型設計、資料、schema、training、service、Gate rationale |
| [HANDOFF_CODEX_TO_CLAUDE_ULIP_RAG_2_0_LATEST_REVIEW_2026-07-16.md](../records/HANDOFF_CODEX_TO_CLAUDE_ULIP_RAG_2_0_LATEST_REVIEW_2026-07-16.md) | HISTORICAL | 48-capture subset／47 provisional groups／9 completed inventory、250 epochs、checkpoint facts、VLM-first |

## 2. Project historical document：1 份

| Current file | Previous file | Status | 判斷 |
|---|---|---|---|
| [IKEA_CORE_RAG_INTEGRATION.md](../history/ikea733_core_rag/IKEA_CORE_RAG_INTEGRATION.md) | docs/IKEA_CORE_RAG_INTEGRATION.md | HISTORICAL | 733/legacy1095 Core RAG、API、Stage 1/2、tests 與 hashes 的唯一完整手冊；後段未完成狀態已過時 |

## 3. IKEA historical reports：5 份，原位保留

| File | Status | 判斷 |
|---|---|---|
| [3d_model_quality_audit.md](../../ikea/docs/3d_model_quality_audit.md) | HISTORICAL | 舊 733 TRELLIS：445 clean、delisted/mislabel/duplicate/sparse 統計 |
| [V3_COLLECTION_CLEANUP_REPORT.md](../../ikea/docs/V3_COLLECTION_CLEANUP_REPORT.md) | HISTORICAL | 舊 3,073 → 2,516 cleanup 與 category correction |
| [phase_a_dimensions_result.md](../../ikea/docs/phase_a_dimensions_result.md) | HISTORICAL／PARTLY UNSAFE | 舊尺寸 parser 2,527 → 2,112；range/missing policy 已被 fail-closed 取代 |
| [phase_v3_scrape_result.md](../../ikea/docs/phase_v3_scrape_result.md) | HISTORICAL | 舊 Category API 3,073-row crawl |
| [ulip_rag_phase_plan.md](../../ikea/docs/ulip_rag_phase_plan.md) | SUPERSEDED | 舊「不重抓、不重跑、架構不動」Phase A/B/C，禁止當 2.0 現行規格 |

## 4. Component-local docs：4 份，原位保留

| File | Status | 保留理由 |
|---|---|---|
| [core/_archive/README.md](../../core/_archive/README.md) | COMPONENT | archived code manifest 與 deprecated import 警告 |
| [ikea/loop2/README.md](../../ikea/loop2/README.md) | COMPONENT／CURRENT | 1,546 data loop 指令與 gates；3D-FRONT/FUTURE 與 V2T inventory 狀態已於交接前修正 |
| [contracts/README.md](../../ikea/product_candidates/contracts/README.md) | COMPONENT／NORMATIVE | schema mirror、hash 與 runtime contract |
| [deploy/README.md](../../ikea/product_candidates/deploy/README.md) | COMPONENT | 8321 topology、security、systemd boundary |

## 5. Vendored PointNeXt/OpenPoints docs：13 份，原位保留

| File | 用途 |
|---|---|
| [PointNeXt/README.md](../../core/models/pointnext/PointNeXt/README.md) | 上游主 README |
| [docs/changes.md](../../core/models/pointnext/PointNeXt/docs/changes.md) | 上游 changelog/TODO |
| [docs/index.md](../../core/models/pointnext/PointNeXt/docs/index.md) | OpenPoints 安裝與 general usage |
| [docs/modelzoo.md](../../core/models/pointnext/PointNeXt/docs/modelzoo.md) | benchmarks、pretrained models |
| [examples/modelnet.md](../../core/models/pointnext/PointNeXt/docs/examples/modelnet.md) | ModelNet40 |
| [examples/s3dis.md](../../core/models/pointnext/PointNeXt/docs/examples/s3dis.md) | S3DIS |
| [examples/scannet.md](../../core/models/pointnext/PointNeXt/docs/examples/scannet.md) | ScanNet |
| [examples/scanobjectnn.md](../../core/models/pointnext/PointNeXt/docs/examples/scanobjectnn.md) | ScanObjectNN |
| [examples/shapenetpart.md](../../core/models/pointnext/PointNeXt/docs/examples/shapenetpart.md) | ShapeNetPart |
| [projects/pix4point.md](../../core/models/pointnext/PointNeXt/docs/projects/pix4point.md) | Pix4Point |
| [projects/pointnext.md](../../core/models/pointnext/PointNeXt/docs/projects/pointnext.md) | PointNeXt |
| [openpoints/README.md](../../core/models/pointnext/PointNeXt/openpoints/README.md) | OpenPoints package |
| [emd/README.md](../../core/models/pointnext/PointNeXt/openpoints/cpp/emd/README.md) | CUDA EMD extension |

這 13 份由 PointNeXt/mkdocs.yml、relative images 與相對 links 約束；搬動會破壞 vendored documentation。

## 6. 基線結論

| Decision | Count |
|---|---:|
| 移入中央 records/history | 23 |
| 原位保留 | 22 |
| 第一輪 hard-delete | 0 |
| Exact duplicate files | 0 |

沒有一份文件能僅因「看起來重複」安全刪除。最低風險候選是已完成的 V2T scene audit request，但仍保存原始驗收問題，因此本輪也只封存、不刪除。
