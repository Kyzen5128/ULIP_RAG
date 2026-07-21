# Semantic ULIP、RAG 訓練與 IKEA 1,546 Serving

狀態基準：2026-07-16 成文結果  
文件性質：CURRENT READER RECORD。原始執行證據與 hashes 仍以 records/ 下的 handoff 為準。

## 1. 訓練與服務是三個不同層次

~~~text
Semantic ULIP training
  → 文字／圖片／point cloud 對齊到共同 512D embedding space
  → 建立 1,546 商品 point-cloud gallery vectors
  → semantic retrieval

RAG Stage 1/2 experiment
  → 使用訓練 corpus 與 retrieved knowledge 做 embedding enhancement
  → 與 vanilla semantic retrieval 做 validation A/B
  → 未獲 production promotion

Product Candidates service
  → query embedding 與全 category pool semantic scoring
  → primary／secondary category policy
  → catalog／dimension production_eligible 信任閘
  → per-tier Top-K results[] 或 no_match
~~~

RAG 與 V2T spatial reranker不是同一件事：

- RAG 嘗試用商品知識改善語意 embedding／retrieval。
- Spatial reranker 在 semantic retrieval 之後，對 product × spot 候選學 style、relation、human preference。
- 尺寸、碰撞、路徑與 hard safety 由規則處理，不交給 RAG 或 learned reranker翻案。

## 2. Semantic ULIP 資料切分

成文 split：

| split | products | 用途 |
|---|---:|---|
| train | 1,212 | 模型訓練 |
| validation | 200 | checkpoint、模型與超參數選擇 |
| test | 134 | 唯一 checkpoint 選定後的一次性 final evaluation |

切分原則是 product-family disjoint。test 結果不能回饋 checkpoint 或超參數選擇。

實存且不重複的 checkpoint candidates：

~~~text
checkpoint_50.pt
checkpoint_100.pt
checkpoint_150.pt
checkpoint_200.pt
checkpoint_250.pt
checkpoint_best.pt
~~~

checkpoint_last.pt 與 checkpoint.pt 的 SHA-256 都等於 checkpoint_250.pt，不重複計算。

## 3. Semantic 訓練結果

- epochs：250。
- 記錄訓練時間：5,215 秒。
- validation 選出的正式版本：checkpoint_best.pt。
- best 與 epoch 250 在 validation 上統計很接近，文件沒有宣稱顯著差異。

一次性 final test：

| direction | metric | value |
|---|---|---:|
| shape-to-text | R@1 | 10.4% |
| shape-to-text | R@10 | 59.7% |
| text-to-point | R@1 | 17.9% |
| text-to-point | R@10 | 63.4% |
| image-to-point | R@10 | 89.6% |
| point-to-image | R@10 | 91.0% |

成文結論是 image 線明顯強於 text 線，且 validation → test 存在泛化落差；test 已使用，不得為了改善數字回頭重選 checkpoint。

範圍警告：這是 134-product split gallery 內的 exact-product cross-modal retrieval，不是 V2T room-query、style compatibility、product × spot fit 或 placement E2E。

## 4. RAG corpus 與訓練

RAG corpus 是商品資料的訓練 lineage，不是 V2T room corpus。正式 pin：

~~~text
product_corpus_train.jsonl
SHA-256:
1f053282712f53328a96f99fc89e3dd42a17f9347d569f3b3bcddd7a378d0467
~~~

RAG 訓練紀錄：

| stage | epochs | 成文結果 |
|---|---:|---|
| Stage 1 | 50 | best metric 0.225，epoch 45 |
| Stage 2 | 50 | best shape-to-text R@1 0.145，epoch 8；final epoch 0.12 |

Stage 2 很早出現 best，後續下降，呈現過擬合訊號。

## 5. Semantic vs RAG A/B

相同 200-product validation、category-filtered Top-10：

| checkpoint | R@1 差 | R@5 差 | R@10 差 | MRR 差 |
|---|---:|---:|---:|---:|
| Stage 2 best，RAG - vanilla | +0.005 | -0.005 | -0.045 | +0.00240 |
| Stage 2 last，RAG - vanilla | -0.015 | -0.010 | -0.015 | -0.01305 |

決策：

- RAG 沒有在主要 retrieval metrics 上一致改善。
- 本次 IKEA 1,546 正式 profile 的 retrieval_mode/default 不使用 RAG。
- RAG checkpoint、corpus 與 A/B 結果保留為實驗 artifact。
- runtime 不得在未更新 profile／policy 的情況下靜默切換成 RAG。

範圍警告：這組 A/B 是 200-product validation、category-filtered Top-10 retrieval，不是 V2T real-scene E2E，也不評估 style、relation、collision 或 placement。

A/B artifacts：

~~~text
/mnt/P300/data/ULIP/ULIP_RAG_2_0/ikea-us-20260716-v1/evaluation/rag_val_ab_best.json
SHA-256:
211de76664d8cc655ca6df7e339eb743c57c3ca719888f59461aec9e202a3523

/mnt/P300/data/ULIP/ULIP_RAG_2_0/ikea-us-20260716-v1/evaluation/rag_val_ab_last.json
SHA-256:
348eddee1bffeb9a2098f0288bb996fd34e0dd1e1d92c0d93d994728098721d4
~~~

## 6. 正式 1,546 profile

Profile：

~~~text
/mnt/P300/data/ULIP/product_candidates/deployments/
ikea1546-product-candidates-v2-20260716-732f2a0323c1896f
~~~

Identity：

~~~text
deployment_profile_id:
ikea1546-product-candidates-v2-20260716-732f2a0323c1896f

retrieval_mode:         semantic_vanilla
catalog_rows:           1546
dimension_profile_rows: 1546
vector_shape:           [1546, 512]
hard_filter_eligible:   1122
~~~

Pinned hashes：

| bundle | SHA-256 |
|---|---|
| deployment profile | d4a046263bc8f13264250a642242f691e0fbe50ef07fe8d9e040365a2fd9d323 |
| model checkpoint | 4c49ae845b972eb176decb29d398a5c5d03781270a267377b1481bbfbf2c4936 |
| vector bundle | 603814ede8d0e8ce1557731e6fc8f9a831699c5fe8c8d9449eb71feddd9a94d5 |
| catalog manifest | bfecfb675e1eba929e66ee17606be9a029eeaea306bb96a462e1620b54b4cf67 |
| dimension bundle | 732cdf381caaf11e10164c650a7fc76b49be08220b1b15b740fd260b060c56d6 |
| tokenizer bundle | 6cc2fa0afe0fa24ee7858c42635d11c43b2dcdd836afd63b07ce0e577af794e2 |

Pinned policies：

~~~text
category_policy_version:
v2t36_to_ikea24_v2

dimension_axis_policy_version:
dimension_axis_ikea_snapshot_v2
~~~

consumer 必須驗證完整 tuple，不得接受部分相符，也不得 fallback 到舊 733 profile。

## 7. HTTP contract

成文時 listener：

~~~text
127.0.0.1:8321
~~~

Endpoints：

~~~text
GET  /healthz
GET  /readyz
POST /v2/product-candidates
~~~

5090 透過 SSH tunnel 使用，不應要求 LAN bind：

~~~bash
ssh -N \
  -L 127.0.0.1:18321:127.0.0.1:8321 \
  kyzen@192.168.0.105
~~~

Bearer token 位於：

~~~text
/home/kyzen/.config/ulip-product-candidates/e2e_token
directory mode: 700
file mode:      600
~~~

成文 readiness：

- healthz = 200。
- readyz = 200。
- model、vectors、catalog、dimensions、tokenizer、row join、policy 七項 checks 全 true。
- 無 Bearer 的 POST = 401。
- 有 Bearer 的真實 retrieval = 200。

2026-07-21T12:57:29+08:00 交接前已唯讀重驗：listener 仍在 127.0.0.1:8321，PID 2668533，tmux session ulip-product-candidates-v2；healthz/readyz 都是 200，deployment profile 正確且七項 checks 全 true。未來實際使用前仍需重新驗證。

## 8. Product Candidates 與空間安全邊界

- 1,546 vector rows 都有同 row-order catalog 與 dimension row。
- API request 不包含 candidate spot、room footprint 或 placement constraints。
- runtime 對 requested category pool 全量計分，再套 production_eligible 信任閘，最後取 per-tier Top-K。
- response 是 status=ok／no_match 與 results[]；不產生 verified_pass／unknown_fallback／rejected 三條 placement lanes。
- hard_filter_eligible 表示資料可供下游 rectangular-footprint hard gate 使用，不表示商品已對特定 spot 通過 fit。
- unsafe partial、missing、range-present、unknown footprint fail closed，不進 production results。
- range 不得壓成假精確值。
- Sofa shape 尚未逐件確認，因此不進 production results。
- canonical front 與 clearance 尚未驗證。
- semantic embedding score 不得翻轉 hard rule rejection。
- 實際 product × spot 尺寸比較、collision 與 rejected reason 由 placement loop 執行。

## 9. 舊資料不得混用

以下屬於舊 IKEA 733 pipeline：

- legacy1095 Core RAG bundle。
- 733-row Product Candidates v1 profile。
- 733 E2E acceptance hashes。
- 舊 Mongo dimension snapshot。

它們可以作歷史與 regression evidence，但不得和 1,546 checkpoint、vectors、catalog、dimension 或 tokenizer tuple 混成同一 runtime。

## 10. 原始來源

- [1,546 SERVICE_READY 與 A/B](../records/HANDOFF_3090_TO_5090_ULIP_RAG2_SERVICE_READY_2026-07-16.md)
- [Checkpoint selection discipline](../records/CODEX_REPLY_TO_CLAUDE_ULIP_RAG_2_0_REVIEW_CLOSURE_2026-07-16.md)
- [Final semantic test 與 Stage 1/2 追記](../records/CLAUDE_ACK_CODEX_CLOSURE_ULIP_RAG_2_0_2026-07-16.md)
- [7/16 consensus 與 superseded notices](../records/CONSENSUS_ULIP_RAG_2_0_CLAUDE_CODEX_2026-07-16.md)
- [Legacy IKEA 733 Core RAG](../history/ikea733_core_rag/IKEA_CORE_RAG_INTEGRATION.md)
