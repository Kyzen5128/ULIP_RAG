# 3090 → 5090：ULIP_RAG 2.0 / IKEA 1,546 SERVICE_READY

日期：2026-07-16（Asia/Taipei）

## 1. 狀態

```text
SERVICE_READY
```

這是新的 IKEA 1,546-row profile，不是舊 IKEA 733 profile。3090 已完成本機 immutable bundle 驗證、模型載入、HTTP contract、Bearer auth、真實 ULIP retrieval、exact 5090 adapter 與 regression tests。尚未宣稱 5090 真實場景 placement E2E 完成；該步驟由 5090 執行。

## 2. 服務連線

3090 listener：

```text
127.0.0.1:8321
PID: 2668533
tmux session: ulip-product-candidates-v2
```

Endpoint：

```text
GET  /healthz
GET  /readyz
POST /v2/product-candidates
```

5090 請沿用 SSH tunnel，不要要求 3090 改成 LAN bind：

```bash
ssh -N \
  -L 127.0.0.1:18321:127.0.0.1:8321 \
  kyzen@192.168.0.105
```

5090 tunnel endpoint：

```text
http://127.0.0.1:18321/v2/product-candidates
```

## 3. Bearer token

3090 token 檔案：

```text
/home/kyzen/.config/ulip-product-candidates/e2e_token
directory mode: 700
file mode:      600
file bytes:     65
```

5090 可透過既有 SSH 唯讀取得。Token 內容未寫入 repo、handoff、聊天或 service log。

## 4. 正式 serving profile

```text
/mnt/P300/data/ULIP/product_candidates/deployments/ikea1546-product-candidates-v2-20260716-732f2a0323c1896f
```

```text
deployment_profile_id:
ikea1546-product-candidates-v2-20260716-732f2a0323c1896f

retrieval_mode:
semantic_vanilla

catalog_rows:            1546
dimension_profile_rows:  1546
vector_shape:            [1546, 512]
hard_filter_eligible:    1122
```

`deployment_profile.json`：

```text
SHA-256: d4a046263bc8f13264250a642242f691e0fbe50ef07fe8d9e040365a2fd9d323
```

## 5. 5090 必須加入的新 allowlist tuple

```json
{
  "deployment_profile_id": "ikea1546-product-candidates-v2-20260716-732f2a0323c1896f",
  "model_checkpoint_sha256": "4c49ae845b972eb176decb29d398a5c5d03781270a267377b1481bbfbf2c4936",
  "vector_bundle_sha256": "603814ede8d0e8ce1557731e6fc8f9a831699c5fe8c8d9449eb71feddd9a94d5",
  "catalog_manifest_sha256": "bfecfb675e1eba929e66ee17606be9a029eeaea306bb96a462e1620b54b4cf67",
  "dimension_bundle_sha256": "732cdf381caaf11e10164c650a7fc76b49be08220b1b15b740fd260b060c56d6",
  "tokenizer_bundle_sha256": "6cc2fa0afe0fa24ee7858c42635d11c43b2dcdd836afd63b07ce0e577af794e2",
  "category_policy_version": "v2t36_to_ikea24_v2",
  "dimension_axis_policy_version": "dimension_axis_ikea_snapshot_v2",
  "catalog_rows": 1546,
  "dimension_profile_rows": 1546,
  "vector_shape": [1546, 512]
}
```

請先加入 exact tuple 與一筆 hash-mismatch 負向測試，再執行真實 retrieval。不得接受部分相符或回退舊 733 tuple。

## 6. Readiness 實測

```text
GET /healthz = HTTP 200
GET /readyz  = HTTP 200
```

七項 checks：

```text
model_checkpoint = true
vector_bundle     = true
catalog_manifest  = true
dimension_bundle  = true
tokenizer_bundle  = true
row_identity_join = true
policy_versions   = true
```

Auth：

```text
POST without Bearer = HTTP 401 AUTHENTICATION_REQUIRED
POST with Bearer    = HTTP 200
```

真實 retrieval smoke：

```text
query: compact natural wood armchair + small side table
primary results:   5 Armchair
secondary results: 5 Side Table
response schema:   PASS
exact 5090 adapter: PASS
```

## 7. Semantic vs RAG A/B 決策

RAG Stage 1、Stage 2 與 A/B 都已完成，但 RAG 不設為本次 production default。理由是 validation 沒有一致改善：

```text
Stage2 best, 200 val products, category-filtered Top-10:
RAG - vanilla
R@1:  +0.005
R@5:  -0.005
R@10: -0.045
MRR:  +0.00240

Stage2 last:
R@1:  -0.015
R@5:  -0.010
R@10: -0.015
MRR:  -0.01305
```

因此正式 serving 使用 Semantic ULIP `checkpoint_best.pt` 與其 1,546 PC vectors；RAG 留作實驗 artifact，不在 runtime 靜默啟用。

A/B artifacts：

```text
/mnt/P300/data/ULIP/ULIP_RAG_2_0/ikea-us-20260716-v1/evaluation/rag_val_ab_best.json
SHA-256: 211de76664d8cc655ca6df7e339eb743c57c3ca719888f59461aec9e202a3523

/mnt/P300/data/ULIP/ULIP_RAG_2_0/ikea-us-20260716-v1/evaluation/rag_val_ab_last.json
SHA-256: 348eddee1bffeb9a2098f0288bb996fd34e0dd1e1d92c0d93d994728098721d4
```

RAG training corpus pin（實驗 lineage；production semantic serving 不讀 corpus）：

```text
product_corpus_train.jsonl SHA-256:
1f053282712f53328a96f99fc89e3dd42a17f9347d569f3b3bcddd7a378d0467
```

## 8. 新 profile 的 fail-closed 空間政策

- 1,546 個 vector rows 都有同 row-order catalog 與 dimension row。
- 完整三軸且目前 footprint 可建模者才可能 `hard_filter_eligible=true`。
- 缺尺寸、partial、range-present、unknown footprint 一律不能 hard pass。
- Sofa 因非矩形／sectional shape 尚未逐件確認，仍一律不能 hard pass。
- canonical front 與 clearance 尚未驗證；服務不宣稱已具備 facing 或 drawer/chair pull-out clearance。
- 現行 response v1 無 dimension interval 型別；range-present 商品保留 reason code 並 fail closed，不把 range 壓成單一數值。

## 9. 3090 regression 結果

```text
Product Candidates tests: 57 PASS
IKEA loop2/RAG/vector tests: 25 PASS
RAG serving/evaluation tests: 22 PASS
Placement loop tests: ALL PASS
Live exact 5090 adapter: PASS
```

## 10. 5090 下一步

依序執行：

```text
1. 建立 SSH tunnel。
2. 唯讀取得 Bearer token。
3. GET /readyz，exact 驗證本 handoff 的 1,546 tuple。
4. 將 tuple 加入 current consumer allowlist，跑 hash mismatch 負向測試。
5. 對 27 個 pre-service query bundles 執行真實 product retrieval。
6. 執行 placement loop。
7. 執行 artifact validator。
8. 回傳 per-query success/no_match/rejection、placement 統計與完整 artifact hashes。
```

`0a7cc12c0e` 仍禁止作 final test；兩個 scale-QA quarantine 場景也不可當 final result。3090 沒有擅自替 5090 凍結其餘六個場景的 calibration/final-test 分組。
