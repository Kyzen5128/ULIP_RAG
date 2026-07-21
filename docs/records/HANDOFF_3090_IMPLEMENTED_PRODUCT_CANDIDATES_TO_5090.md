# 3090 → 5090：Product Candidates v1 實作完成與跨機 E2E 交接

完成時間：2026-07-16 00:36（Asia/Taipei；工作於 2026-07-15 開始）  
3090：`edge05` / `192.168.0.105` / `/home/kyzen/ULIP_RAG`  
5090：`edge08` / `/home/kyzen/V2T_DA3`  
權威決策：`HANDOFF_5090_TO_3090_PRODUCT_CANDIDATES_DECISIONS_AND_EXEC_2026-07-15.md`

## 0. 結論與本輪邊界

Product Candidates v1 已在 3090 完成實作、正式 733-row immutable deployment profile 建置、離線完整性驗證、RTX 3090 Ti 真實 vanilla ULIP 載入、本機 HTTP 驗收，以及 5090 exact adapter 驗收。

本機 release gate 結果：

```text
Product Candidates tests                    57 / 57 PASS
既有 IKEA/Core-RAG regression tests          28 / 28 PASS
5090 handoff manifest/contract validator     PASS
5090 placement-loop reference suite          ALL PASS
真實 RTX 3090 Ti model load/inference         PASS
真實 127.0.0.1:8321 + exact 5090 adapter      PASS
獨立 acceptance audit                        NO RELEASE BLOCKER
```

本輪遵守的限制：

- MongoDB 僅執行 `ping`、`find`、`count_documents`；沒有 update、insert、delete、補值或 index 操作。
- 原 checkpoint、vectors、`meta_pc.jsonl`、schema 與 tokenizer source 未被改寫。
- 未碰 port `8000`；原未知 Docker listener 全程仍在 `0.0.0.0:8000` / `[::]:8000`。
- 未安裝或 enable systemd unit，未修改 firewall，未使用 sudo。
- 未 commit、push 或建立 PR；HEAD 保持不變。
- 未把 Bearer secret 寫進 repo、fixture、log 或本交接檔。
- 本機測試完成後已停止 `8321`；5090 真實跨機 E2E 前需由 operator 依第 10 節啟動。
- 尚未宣稱跨機 production E2E；該步驟必須由 5090 以真實 V2T scene 完成。

## 1. 唯一 serving 路徑

正式 route：

```http
POST /v2/product-candidates
```

健康與 readiness：

```http
GET /healthz
GET /readyz
```

這條 route 固定使用既有 IKEA vanilla ULIP text encoder 與 733-row point-cloud vectors：

```text
strict UTF-8 JSON + Bearer
→ Draft 2020-12 request validation
→ category trim/casefold/duplicate/tier policy
→ 75-content-token gate（76+ 直接 422，不呼叫 encoder）
→ vanilla ULIP text encoder
→ normalized 512-D query embedding
→ requested canonical category 全 pool cosine scoring
→ immutable catalog/dimension/quality join
→ hard_filter_eligible / trust fail-closed gate
→ score desc、product_id asc deterministic sort
→ primary / secondary 各自 Top-K
→ Draft 2020-12 response validation
→ JSON response
```

這個 endpoint **不會**：

- 啟用 Core RAG、FAISS corpus 或 query rewrite。
- 使用舊 `/search/text` 或早期 `/v2/retrieval`。
- 先截 raw Top-K 再讓不可信 row 佔掉 quota。
- 回 synthetic fallback。
- emit top-level `partial`；未完整執行時回 503。
- 回 `catalog_metadata`、本機 path 或 secret。

## 2. 新增實作檔案與主要物件

### 2.1 Runtime 與 HTTP service

- `ikea/app_product_candidates.py`
  - `load_runtime_from_env()`：載入正式 profile、vanilla model，並在模型載入後再做一次完整 bundle revalidation。
  - ASGI app 固定 `hard_timeout_s=25`、active inference `1`、queued `1`。
- `ikea/product_candidates/runtime.py`
  - `ULIPQueryTokenizer`：明確的 75/76 token gate 與 EOT 驗證。
  - `VanillaULIPTextEncoder`：只呼叫 base ULIP `encode_text`，輸出 L2-normalized embedding。
  - `ProductRow.production_eligible`：集中執行尺寸、quality、Sofa/special/range fail-closed。
  - `ProductCandidateRuntime.from_components(...)` / `from_profile(...)`。
  - `ProductCandidateRuntime.retrieve(...)`：category pool、全量 scoring、tier quota、deterministic ranking。
  - `ProductCandidateRuntime.revalidate_integrity(...)`：每次 ready probe 重新驗證 profile。
  - `load_immutable_tokenizer(...)` / `load_vanilla_ulip_model(...)`。
- `ikea/product_candidates/service.py`
  - `InferenceGate`：Python 3.9 event-loop-safe；一個 active、最多一個 queued。
  - `ServiceState`：startup/readiness/auth state。
  - `create_app(...)`：`/healthz`、`/readyz`、`/v2/product-candidates`。
  - strict JSON：拒絕 invalid UTF-8、duplicate object keys、NaN/Inf、unknown fields 與 oversized body。
  - timeout/client cancellation 後，CUDA worker 真正結束前不釋放 inference lease。
- `ikea/product_candidates/contracts.py`
  - 載入三份 Draft 2020-12 schema 並提供固定 validator。
- `ikea/product_candidates/__init__.py`
  - package exports。

### 2.2 Immutable bundle builder

- `ikea/product_candidates/bundles.py`
  - `build_deployment_bundle(...)`：唯讀 Mongo snapshot、canonical row join、deterministic in-memory build。
  - `write_deployment_bundle(...)`：streaming copy、copy 前後 hash 驗證、拒絕 hardlink 與既有 output overwrite。
  - `load_deployment_profile(...)`：完全離線重算 manifests/members、row order/join、vectors、policies 與 quality invariants。
  - canonical JSON：UTF-8、sorted keys、compact separators、禁止 NaN、單一 trailing newline。
  - canonical JSONL：每 row 使用同一 canonical JSON encoding。
- `ikea/product_candidates/build_deployment.py`
  - 預設 dry-run，不建立檔案。
  - 只有同時提供 `--write --output` 才 materialize。
- `ikea/product_candidates/policies/category_v2t36_to_ikea15_v1.json`
- `ikea/product_candidates/policies/dimension_axis_v1.json`
- `ikea/product_candidates/policies/quality_overrides_v1.json`

### 2.3 Contracts、部署範本與測試

- `ikea/product_candidates/contracts/`
  - request/response schema 是 5090 權威檔案的 byte-identical mirror。
  - 新增 strict `/readyz` schema。
- `ikea/product_candidates/deploy/launch_8321.sh`
  - 固定 `127.0.0.1:8321`、one worker、固定 `app_product_candidates:app`。
- `ikea/product_candidates/deploy/ulip-product-candidates.service`
  - 只是一份未安裝的 systemd template。
- `ikea/product_candidates/deploy/ulip-product-candidates.env.example`
  - 只有 placeholder，沒有 credential。
- `ikea/product_candidates/tests/`
  - 57 個 bundle、policy、runtime、service、schema 與 exact-consumer tests。

## 3. 正式 immutable deployment profile

正式路徑：

```text
/mnt/P300/data/ULIP/product_candidates/deployments/
  ikea733-product-candidates-v1-20260715-e18415000f11bce9
```

Identity：

```text
source_version:
  ikea733-product-candidates-v1-20260715

deployment_profile_id:
  ikea733-product-candidates-v1-20260715-e18415000f11bce9

deployment_identity_sha256:
  e18415000f11bce966ac2dd31b084cadeebac7d88c6f71394428996e76eac3c4

deployment_profile.json raw SHA-256:
  275d7e8c39dc46b4473dd5a8fb9f3c088b68804b275f46cebd2d2c31f66fc843
```

Canonical identity join 只有以下一種，無 fallback：

```text
vectors_pc.npy row i
→ meta_pc.jsonl line i
→ ObjectId(meta_pc[i]["id"])
→ Mongo document _id
```

禁止用 ASIN、URL、名稱、Mongo cursor order 或 fuzzy matching。

### 3.1 5090 必須 pin 的完整 tuple

下列五個 SHA 是 canonical aggregate manifest bytes 的 SHA，不是單一 raw asset hash：

| Response provenance field | Canonical manifest | SHA-256 |
|---|---|---|
| `model_checkpoint_sha256` | `manifests/model_checkpoint.json` | `6ee2b44e8647af8323dc9405604b04f37cc16d63b6e40e2034f1e9c0cb6cde0f` |
| `vector_bundle_sha256` | `manifests/vector_bundle.json` | `25472051a53b08869f3e2c458b2bc4f0ad39d0f6d12bfacbb271672ee018a0e1` |
| `catalog_manifest_sha256` | `manifests/catalog_manifest.json` | `f23fd25a6c72045ae30b5475a5d279e7ac4389b85b4cd50d3a2da3bf42624304` |
| `dimension_bundle_sha256` | `manifests/dimension_bundle.json` | `dc2b031061e77b093542e1d260572861704897e9712527f4e42a4cf41b4c3cf4` |
| `tokenizer_bundle_sha256` | `manifests/tokenizer_bundle.json` | `81a4b450a023da2330ee1763073bab05c6fd2a5c2ea80f4717678b8f469b2b4c` |

政策版本：

```text
category_policy_version       v2t36_to_ikea15_v1
dimension_axis_policy_version dimension_axis_v1
quality policy（dimension manifest 內綁定） quality_overrides_v1
```

Policy raw bytes：

```text
33671d7de3d2b6c1134336410f59f856d62f36d35b3fe1d5208e34f65771b3d4  category_v2t36_to_ikea15_v1.json
99648aaf6644944409056efa8ac7446dd6034f6a0096c3ebc67c6b4872fbcf9d  dimension_axis_v1.json
8fe07249ca41e7662ea75557f475b3f63b8c82cf078ccc205c6aab4977aff5ba  quality_overrides_v1.json
```

### 3.2 Manifest member raw hashes

```text
49ac18ab10950412f016c9e023d1c7e9b34250a257d59b70b5590d74207f3e01  model/checkpoint_last.pt
30070b4c576bf511ac498bc1f6ddb46502bd0cf4737a455447d5e85fc707d207  vectors/vectors_pc.npy
0c05805753bb085080e75fa8db204c795f46b6f2c55754e37ae3a819bd068a94  vectors/meta_pc.jsonl
8058c788f938bd5d4f74292c46962dd93cf0fc588b903fd27ff19df36e62f08f  vectors/schema.json
140c2a46989df8115f7b10206b8ef3e96f3273c1973796706f702c94deb30a40  tokenizer/tokenizer.py
924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a  tokenizer/bpe_simple_vocab_16e6.txt.gz
75eee69cfc060778f874b025e5beb307e7c773bc1c3c6eb19516465c3611a941  catalog/catalog_rows.jsonl
5968c7e08a550dafa89c49433503fd6cf0a76cd9782bcef4596c477dd32aec29  dimensions/dimension_quality_rows.jsonl
```

Copied source hashes與原始檔完全相同；builder 使用實體 copy，不是 hardlink。

## 4. 正式 inventory 與品質 gate

```text
Vector rows / shape / dtype       733 / [733, 512] / float32
Vectors finite / L2 normalized    yes / yes
Catalog canonical join            733 / 733
Dimension source join             575 / 733
Missing dimension source rows     158
Footprint rows                    520
Physical three-axis rows          401
Effective hard-filter eligible    463
```

Effective dimension status：

```text
complete            359
partial             116
missing             195
range_collapsed       8
suspect                5
special_unmodeled     50
```

`physical three-axis=401` 與 `effective complete=359` 不相等是預期結果：Sofa 與 range quality override 會把有數字的 row 降成不可安全使用的狀態。

品質政策：

- Bed：49/49 `blocked`、hard-ineligible；Bed-only 回 HTTP 200 `no_match` + `CATEGORY_DATA_QUALITY_BLOCKED`。
- Sofa：50/50 `special_unmodeled`、hard-ineligible；32 筆有已知 special evidence，18 筆 `FOOTPRINT_SHAPE_UNVERIFIED`。
- Bar Stool：50/50 category `degraded`；目前不是受支援的 V2T `stool` mapping pool。
- Range source evidence：2 hyphen、6 slash-multivalue、7 paired Min/Max、2 max-only；所有受影響 row hard-ineligible。四筆同時是 Sofa，effective status 由更強的 `special_unmodeled` 覆寫，但 range flags仍保留。
- `partial + height_policy=optional` 可進 footprint-only。
- `partial + height_policy=required|unknown` fail closed。
- missing、suspect、range、special、blocked/unknown 全部 fail closed。
- Wire dimensions固定為 mm；5090只做一次 `/1000`。
- 目前沒有可信 front direction/canonical facing；default yaws只表示矩形 footprint axis rotation，不代表商品正面。

目前可由 V2T enabled mappings 實際取回的 production-eligible row 數：

```text
Sofa               0
Bench              30
Storage Ottoman    22
Dining Table       22
Office Desk        47
Wardrobe           47
Bookshelf          35
TV Stand           40
```

## 5. Request/category/token 行為

Category term 先 trim，再 casefold。`matched_request_category` 保存最早出現 term 的大小寫與內容，但移除外圍空白；`request_id` 則 exact echo。

已支援 mappings：

```text
sofa/couch/loveseat            → Sofa（degraded；目前 0 eligible）
bench                           → Bench
ottoman                         → Storage Ottoman
dining table                    → Dining Table
desk                            → Office Desk
wardrobe/closet/armoire         → Wardrobe
bookshelf/bookcase/shelf/
shelving unit                   → Bookshelf
tv stand/media console          → TV Stand
```

`bed/bed frame → Bed` 有 mapping 但整類 blocked。其餘 decision 文件列出的 V2T controlled-but-unsupported terms 回正常 business `no_match`，不是 422；非 V2T-36 term 才回 semantic 422。

Duplicate/tier：

- 同 tier 或跨 tier 的同 raw term（trim+casefold後相同）→ 422 `CATEGORY_TERM_DUPLICATE`。
- 同 tier 不同 alias 指到同 canonical pool → 合併，保留最早 term。
- 跨 tier alias 指到同 canonical pool → primary precedence，drop secondary pool，warning `CATEGORY_CANONICAL_POOL_DEDUPED`。
- 全 response `product_id` 唯一。
- 每 tier各自遵守 `top_k_per_tier`；secondary不會被升格成 primary。

Tokenizer：

```text
token_count = SOT + content BPE + EOT
0..75 content tokens  → encode，回 truncated=false / eot_present=true
76+ content tokens    → HTTP 422，不呼叫 encoder、不 truncate
tokenizer_id          → ulip-simple-tokenizer-clip77-v1
```

## 6. `/readyz` 與 fail-closed semantics

`/readyz` 每一次 probe 都會重新執行 offline `load_deployment_profile(...)`，不是回傳 startup cache。它會重新驗證：

```text
model_checkpoint
vector_bundle
catalog_manifest
dimension_bundle
tokenizer_bundle
row_identity_join
policy_versions
```

檢查包含 canonical manifest SHA、所有 member SHA/bytes、vector shape/dtype/finite/normalization、manifest safety metadata、733-row order/ObjectId join、policy hash/version、Bed/Sofa/range invariants、tokenizer context/token/overflow policy 與 deployment identity。

任一 drift：

- `/readyz` 回 HTTP 503、`status=not_ready`、至少一個 false check、machine-readable failure。
- 後續 POST 也回 503，不會繼續 serving 或 fallback。
- `/readyz` 與 `/healthz` 都有 `Cache-Control: no-store`；不回 path/secret。

正式 profile 的 readyz完整 rehash本機實測約 `1.78s`。大量並發 readiness polling會各自進 thread executor，再由 runtime lock序列化；目前 loopback + SSH tunnel拓撲下是非阻擋 caveat，請不要高頻輪詢。

## 7. Checkpoint 與真實 GPU smoke

Checkpoint metadata：

```text
model arg       ULIP_PointBERT
state keys      462
rag_enhancer    0 keys
best_acc1       76.53478980064392
epoch           last
```

`best_acc1` 是舊訓練 artifact內的指標，不應當成 V2T placement E2E 成績。

3090 本機真實測試：

```text
GPU                         NVIDIA GeForce RTX 3090 Ti, 24564 MiB
offline profile load        1.725 s
vanilla ULIP GPU load       8.928 s
one direct text inference   0.142 s
response schema             valid
```

真實 query：

```text
query:      a compact modern desk for a home office
primary:    desk
secondary:  bookshelf
top_k:      3 per tier
result:     6 real products, 3 primary + 3 secondary
```

Returned IDs：

```text
67d161eb7270e1a09493c19d
67d1655a7270e1a09493c208
67d1640c7270e1a09493c1df
67d1546e7270e1a09493c000
67d183697270e1a09493c596
67d1543f7270e1a09493bffa
```

5090 exact `HttpProductRetriever` 已對同一 live endpoint 接受這六筆；三筆 desk 是 `partial` footprint-only，三筆 bookshelf 是 `complete`。

## 8. 測試命令與結果

### 8.1 Product Candidates 全測試

```bash
cd /home/kyzen/ULIP_RAG
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/kyzen/ULIP_RAG/core:/home/kyzen/ULIP_RAG/ikea:/home/kyzen/ULIP_RAG \
/home/kyzen/miniconda3/envs/ulip/bin/python \
  -m unittest discover \
  -s ikea/product_candidates/tests \
  -p 'test_*.py' -v
```

結果：`Ran 57 tests ... OK`。

覆蓋：

- Draft 2020-12 request/response/readyz positive/negative。
- whitespace、unknown/missing key、bool-as-number、NaN/Inf、duplicate term/ID/flag/yaw。
- 75/76 token、EOT、reject前不呼叫 encoder。
- aliases、unsupported、blocked、degraded、canonical overlap、tier quota。
- complete/partial/required/unknown/missing/suspect/range/special matrix。
- full-pool gate、deterministic ordering、product ID tiebreak。
- member/hash/order/join fault injection。
- 啟動後實際 bundle member tamper → readyz與POST都503。
- Bearer、invalid UTF-8、超過4 MiB、timeout、one-active/one-queued/excess 503。
- 同一 app 跨兩個 TestClient/event loops。
- exact 5090 strict consumer與mm只轉換一次。

### 8.2 既有 IKEA/Core-RAG regression

```text
ikea/tests/    6 / 6 PASS
tests/        22 / 22 PASS
```

Product Candidates endpoint仍固定 vanilla；這些測試只證明既有 pre-existing Core-RAG work沒有被新檔案破壞，不表示 Core RAG 被接入新 route。

### 8.3 5090 handoff與placement reference

```bash
cd /home/kyzen/V2T/HANDOFF_5090_TO_3090_PLACEMENT_LOOP_2026-07-15
./tools/verify_manifest.sh
/home/kyzen/miniconda3/envs/ulip/bin/python tools/validate_handoff.py
PYTHONDONTWRITEBYTECODE=1 \
  /home/kyzen/miniconda3/envs/ulip/bin/python \
  -m unittest discover -s reference_v2t_da3/tests -p 'test_*.py'
```

結果：

```text
package manifest                          PASS
5 Draft 2020-12 schemas                   PASS
positive/negative fixtures                PASS
5090 strict parser + mm conversion        PASS
placement loop sections [1]..[22]         ALL PASS
wall-segment sections [1]..[6]            ALL PASS
```

### 8.4 真實 local HTTP acceptance

- 啟動在 `127.0.0.1:8321`，確認沒有 bind LAN address。
- `/healthz` 200。
- `/readyz` 200、七 checks全 true、五 SHA正確。
- 5090 原始 `tools/check_live_endpoint.py`：`adapter_kind=ulip_http`，預設 unsupported nightstand fixture正常 `no_match`。
- 5090 exact adapter desk/bookshelf request：六筆真實商品，schema與tier quota通過。
- Bed-only：`no_match` + `CATEGORY_DATA_QUALITY_BLOCKED`。
- 無 Bearer POST：401。
- 測試結束已 clean shutdown，`8321` 目前沒有 listener。

### 8.5 Deployment template

```text
bash -n launch_8321.sh                    PASS
missing env拒絕                           PASS
placeholder/short token拒絕（rc=64）       PASS
systemd-analyze verify unit               PASS（host另有無關systemd警告）
fixed ASGI module，無 env override          PASS
```

## 9. Repo HEAD 與 dirty working tree

```text
branch  3090
HEAD    0eefc1f4301edccd47687e2963e3c18517b8374d
origin  git@github.com:Kyzen5128/ULIP_RAG.git
```

本輪開始前已存在、且全程保留的 dirty state：

```text
 M core/models/ULIP_models.py
 M core/utils/tokenizer.py
 D docs/CLAUDE_USAGE_GUIDE.md
 D docs/ENVIRONMENT_SETUP.md
 D docs/HANDOFF_ASK_4090.md
 D docs/HANDOFF_FOR_3090.md
 D docs/INDEX.md
 D docs/NEXT_TASKS_HANDOFF.md
 D docs/PROJECT_REPORT_3090.md
 D docs/PROJECT_REPORT_OLD_4090.md
 D docs/RESEARCH_GUIDE.md
 D docs/TRAINING_COMMANDS.txt
 D docs/USAGE_AND_FIXES_3090.md
 D docs/ikea_pipeline_code_detail.md
 M ikea/app_ikea_retrieval.py
 M ikea/build_vectors.py
 M ikea/run_app_3090.sh
?? HANDOFF_5090_TO_3090_PRODUCT_CANDIDATES_DECISIONS_AND_EXEC_2026-07-15.md
?? docs/IKEA_CORE_RAG_INTEGRATION.md
?? ikea/evaluate_rag_retrieval.py
?? ikea/main_ikea_rag.py
?? ikea/rag_serving.py
?? ikea/rag_training_utils.py
?? ikea/tests/
?? tests/
```

本輪新增、沒有冒充為 pre-existing change：

```text
?? ikea/app_product_candidates.py
?? ikea/product_candidates/
?? HANDOFF_3090_IMPLEMENTED_PRODUCT_CANDIDATES_TO_5090.md
```

另在 repo外新增正式 deployment profile（第3節路徑）。沒有修改上述既有 tracked/untracked內容，沒有 commit/push。

重要重現資訊：正式 tokenizer bundle鎖定的是本輪開始前已 dirty 的 `core/utils/tokenizer.py` bytes，raw SHA為 `140c2a...a40`；它包含長 query保留EOT的修正。Vanilla model factory相關既有程式仍由目前工作樹載入，因此五 bundle SHA之外，重現時也必須保留本節記錄的 HEAD/dirty code狀態。依使用者限制，本輪沒有把這些既有變更 commit。

## 10. 3090 啟動、停止與本機檢查

### 10.1 安全提供 Bearer

Bearer必須是至少32 random bytes的編碼值，經安全管道交付5090。不要把值貼進本檔、repo、shell history或一般log。

互動式啟動範例：

```bash
cd /home/kyzen/ULIP_RAG

export PRODUCT_CANDIDATES_PROFILE=/mnt/P300/data/ULIP/product_candidates/deployments/ikea733-product-candidates-v1-20260715-e18415000f11bce9
read -rsp 'Product Candidates bearer token: ' PRODUCT_CANDIDATES_API_TOKEN
echo
export PRODUCT_CANDIDATES_API_TOKEN

./ikea/product_candidates/deploy/launch_8321.sh
```

Token可由 operator事先用 `openssl rand -hex 32` 產生並透過安全管道共享；請勿在會被記錄的終端輸出它。

停止 foreground服務：`Ctrl-C`。若由 process manager啟動，送 `SIGINT` 並等待 CUDA work退出。

### 10.2 Health/readiness

```bash
curl -fsS http://127.0.0.1:8321/healthz
curl -fsS http://127.0.0.1:8321/readyz
```

`/readyz` 是完整 rehash，預期比一般 health probe慢。

### 10.3 3090 exact live checker

```bash
cd /home/kyzen/V2T/HANDOFF_5090_TO_3090_PLACEMENT_LOOP_2026-07-15
read -rsp 'Product Candidates bearer token: ' ULIP_API_TOKEN
echo
export ULIP_API_TOKEN

/home/kyzen/miniconda3/envs/ulip/bin/python \
  tools/check_live_endpoint.py \
  --url http://127.0.0.1:8321/v2/product-candidates \
  --timeout 30 \
  --retries 1
```

### 10.4 Systemd狀態

Repo內只有 template，沒有真正安裝。以下動作仍需使用者另行授權並由 operator執行：

- 建立 `/etc/ulip-product-candidates/environment` 且設為 `0600`。
- 複製 unit至 `/etc/systemd/system/`。
- `daemon-reload`、enable/start。
- 任何 sudo或firewall變更。

## 11. 通知 5090：真實跨機 E2E 待辦

本檔即為 3090 本機 acceptance完成通知。5090 請依序執行：

1. 在 5090 allowlist明確 pin第3.1節的五 SHA與兩個 response policy versions；禁止 TOFU/wildcard/mix-and-match。
2. 由安全管道取得與 3090 service相同的 Bearer；不要寫進 repo或交接檔。
3. 建立 SSH tunnel：

```bash
ssh -N \
  -L 127.0.0.1:18321:127.0.0.1:8321 \
  kyzen@192.168.0.105
```

4. 從5090驗證：

```bash
curl -fsS http://127.0.0.1:18321/readyz
```

5. 將5090 endpoint設為：

```text
http://127.0.0.1:18321/v2/product-candidates
```

6. 使用 `/home/kyzen/V2T_DA3` 的真實 scene occupancy grid跑 PlacementLoop，不得使用 synthetic product fixture冒充。
7. 產生並驗證真實：

```text
Outputs/<real_scene>/placement_candidates.json
Outputs/<real_scene>/placement_result.json
```

8. 執行5090 artifact validator，確認：
   - request/response/profile tuple全部匹配。
   - `adapter_kind=ulip_http`。
   - real product IDs與dimensions保留。
   - mm只轉換一次。
   - selected spot/product/center/yaw/score breakdown完整。
   - authoritative occupancy無collision。
   - primary-first/secondary fallback正確。
   - 3090 422/503絕不轉成 synthetic fallback。

## 12. 尚未確認 / TODO

以下項目仍明確標為「尚未確認」：

- **真實3090↔5090跨機 PlacementLoop E2E：尚未確認。** 本輪只完成3090本機與本機載入的5090 exact adapter。
- **5090 allowlist已寫入上述tuple：尚未確認。** 3090沒有越權修改5090 allowlist。
- **SSH tunnel、30秒client timeout與一次retry的真實跨機行為：尚未確認。**
- **真實V2T scene最終 selected pose與artifact validator結果：尚未確認。**
- **Systemd正式安裝、開機自啟與長時間監控：尚未確認，且本輪未獲sudo安裝授權。**
- **Bearer長期secret管理方式：尚未確認。** 本輪只驗證安全設定介面與ephemeral token。
- **高頻併發readyz與長時間GPU soak：尚未確認。** 目前完整rehash約1.78秒，應低頻使用。
- **商品front direction、canonical facing、非矩形footprint、drawer/chair clearance與functional relationship：尚未確認。** Product Candidates v1只提供保守矩形footprint gate；relationship仍由5090保持unverified/advisory。
- **Bed allowlist、Sofa矩形人工正向驗證與catalog-gap類別擴充：尚未確認。** 現行政策繼續fail closed。
- **Runtime source code的獨立content-addressed release artifact：尚未建立。** 目前以repo HEAD、完整dirty state、bundle tuple與本檔記錄重現；這是遵守「不commit/push」限制後的已知操作邊界。

5090完成真實跨機E2E後，請回傳：request ID、readyz tuple、真實商品IDs、selected spot/product/pose、score breakdown、所有rejection reasons、artifact validator輸出與仍未解決事項。只有該回傳通過後，才能宣稱雙機production E2E完成。
