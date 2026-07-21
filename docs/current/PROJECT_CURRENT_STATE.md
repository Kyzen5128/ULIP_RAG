# ULIP_RAG 目前專案狀態

狀態基準：2026-07-21 文件紀錄  
用途：快速區分「已完成」、「已核准但未執行」與「歷史／尚未確認」。

## 結論

目前專案有兩條不能混為一談的線：

1. IKEA 1,546 Semantic ULIP 商品檢索與 Product Candidates v2：已完成訓練、向量建立、immutable serving profile、本機測試及 SERVICE_READY handoff。
2. ULIP_RAG 2.0 Spatial Reranker：架構與資料工程規格已完成 Claude × Codex 核准，但最終 closure 成文時只開放 Gate 1，Gate 1 尚未開始。

RAG Stage 1/2 已完成實驗；因 validation A/B 沒有一致勝過 vanilla semantic retrieval，所以不作 1,546 production default。

## 已完成

### IKEA 1,546 Semantic ULIP

- 商品 snapshot：1,546 rows。
- split：train 1,212、validation 200、test 134 products。
- semantic training：250 epochs，紀錄時間 5,215 秒。
- checkpoint selection：只比較 validation；唯一 checkpoint 選定後才執行一次 final test。
- final semantic test 的 point-cloud-to-text：
  - R@1：10.4%
  - R@10：59.7%
- final semantic test 的 text-to-point-cloud：
  - R@1：17.9%
  - R@10：63.4%
- image-to-point／point-to-image test R@10：
  - i2p：89.6%
  - p2i：91.0%
- 上述 final semantic test 是 134-product split gallery 內的 exact-product cross-modal retrieval，不是 V2T room-query、style compatibility 或 placement E2E 指標。
- 正式 serving checkpoint：checkpoint_best.pt。
- 正式 gallery vectors：1,546 × 512。

### RAG 實驗

- Stage 1：50 epochs。
- Stage 2：50 epochs。
- Semantic vs RAG validation A/B 已完成。
- Stage 2 best 相對 vanilla：
  - R@1：+0.005
  - R@5：-0.005
  - R@10：-0.045
  - MRR：+0.00240
- 因結果不一致，RAG 保留為實驗 artifact；正式 1,546 runtime 使用 semantic_vanilla。
- 這組 A/B 是 200-product validation、category-filtered retrieval，不是 V2T real-scene E2E。

### Product Candidates v2 profile

~~~text
deployment_profile_id:
ikea1546-product-candidates-v2-20260716-732f2a0323c1896f

catalog_rows:           1546
dimension_profile_rows: 1546
vector_shape:           [1546, 512]
hard_filter_eligible:   1122
retrieval_mode:         semantic_vanilla
~~~

已成文驗證：

- healthz、readyz 成功。
- readyz 七項 checks 全部為 true。
- Bearer authentication 正、負向測試通過。
- 真實 ULIP retrieval smoke 通過。
- exact 5090 adapter 與 regression tests 通過。

2026-07-21T12:57:29+08:00 交接前又做了一次唯讀驗證：127.0.0.1:8321、PID 2668533 與 tmux ulip-product-candidates-v2 仍存在，healthz/readyz 都是 200，profile ID 正確且七項 checks 全 true。這是時間點證據，未來使用前仍須重驗。

## 空間能力的現況

Product Candidates v2 已具備 deterministic、fail-closed 的商品資料可信度門檻，但它不接收 spot 或房間尺寸限制，也不直接判斷某商品是否放得進某個位置：

- 只有通過 production_eligible 信任閘的商品才會進 results[]。
- hard_filter_eligible 表示可交給下游 deterministic hard-fit 比較，不代表已對某個 spot 判定 pass。
- 缺尺寸、unsafe partial、range-present、unknown footprint 不會進 production results。
- Sofa 因非矩形／sectional 未逐件確認，不進 production results。
- canonical front 尚未逐件驗證。
- drawer open、chair pull-out、door/access clearance 尚未完成。
- 現行服務不宣稱 facing、front 或完整 clearance compatibility。
- 真正的 product × spot fit、verified/unknown/rejected 分流屬於後續 placement loop。

## 已核准、尚未執行完的 Spatial V2

核准 pipeline：

~~~text
V2T query + compact scene context + candidate spots
  → Semantic ULIP 對 IKEA 1,546 gallery 做 Top-M retrieval
  → 展開 product × spot × allowed footprint yaws
  → deterministic hard gate
  → Spatial Reranker 只排序 hard-valid candidates
  → 5090 full-grid/collision/path/access authoritative revalidation
  → final placement + alternatives + rejection evidence
~~~

模型只學：

- room-product style compatibility
- furniture relation compatibility
- hard-valid candidates 之間的人類偏好

規則負責：

- known／interval dimension fit
- footprint containment
- collision
- outside-floor
- observed-space validity
- 有資料時的 walkability、path、access

learned score 永遠不得復活 hard-invalid candidate。

## 資料角色

| 資料 | 正式角色 |
|---|---|
| IKEA 1,546 | production 商品、semantic retrieval/index、商品 metadata |
| 3D-FRONT | synthetic room、layout、pose、relation supervision |
| 3D-FUTURE-model | 3D-FRONT 家具 geometry、category、style/material metadata |
| V2T | real-domain E2E evaluation／calibration |
| VLM | 有完整 provenance 的 silver／weak labels |
| Human | final gold preference、front 與品質 audit |

3D-FRONT/FUTURE 不取代 IKEA production gallery。V2T 在 physical-room split 凍結前不得被視為無 leakage 的 training set。

## 目前下一個合法步驟

7/21 final closure 只開放 Gate 1：

- 對 3D-FRONT／3D-FUTURE 做全量、唯讀、CPU/IO raw audit。
- 建立 source manifest、children resolution、join、transform、mesh、metadata quality、room-type 與 storage artifacts。
- 不修改 raw datasets。
- 不產生正式 relation labels／placement pairs。
- 不開始 spatial model 或 RAG training。
- 不修改 Mongo、IKEA vectors、checkpoint 或 deployment。
- derived artifact 預估超過 30G 必須先停止回報。

Gate 1 完成後仍需書面審核，不能自動進 Gate 2。

## 已作廢或只能當歷史的內容

- 「V2T 只有 7 scenes」：作廢。
- 「epoch 110/115 checkpoint 可供選擇」：作廢，檔案不存在。
- 「test 可用來選 production checkpoint」：禁止。
- 「尺寸／碰撞／路徑由 learned heads 決定」：作廢；這些由 deterministic rules 決定。
- 「733 不必重抓、模型架構永遠不動、caption enhancement 就是 Spatial 2.0」：被新版 1,546 與 7/21 Spatial V2 規格取代。
- 舊 733 profile 的 hashes、E2E 結果不得套用到 1,546 profile。

## 尚未確認

- 5090 是否已完成 1,546 profile 的最終真實跨機 E2E；repo 內目前只有等待 5090 執行的 SERVICE_READY handoff。
- 2026-07-21 closure 之後是否已在其他位置啟動 Gate 1；本 repo 現有 Markdown 沒有 Gate 1 completion report。
- canonical front、clearance 與 human-gold 是否已有 repo 外的新標註成果。

## 權威來源

- [1,546 SERVICE_READY](../records/HANDOFF_3090_TO_5090_ULIP_RAG2_SERVICE_READY_2026-07-16.md)
- [7/16 final acknowledgement](../records/CLAUDE_ACK_CODEX_CLOSURE_ULIP_RAG_2_0_2026-07-16.md)
- [7/21 final closure](../records/FINAL_CLOSURE_CLAUDE_CODEX_3DFRONT_SPATIAL_PLAN_FULLY_APPROVED_2026-07-21.md)
- [Spatial V2 閱讀版](SPATIAL_V2_READER_EDITION.md)
