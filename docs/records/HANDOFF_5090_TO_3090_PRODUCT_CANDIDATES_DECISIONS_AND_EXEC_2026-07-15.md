# 5090 → 3090：Product Candidates v1 正式裁定與執行授權

日期：2026-07-15（Asia/Taipei）  
5090：`edge08` / `/home/kyzen/V2T_DA3`  
3090：`edge05` / `/home/kyzen/ULIP_RAG`  
前置審核：`HANDOFF_3090_TO_5090_PRODUCT_CANDIDATES.md`  

## 0. 授權與邊界

使用者已明確授權 3090 端進入實作，範圍包含：

- 修改 `/home/kyzen/ULIP_RAG` 以實作本文件的 product-candidates v1。
- 建立新的 immutable deployment bundles 與 manifest。
- 以唯讀方式查詢 Mongo 產生 dimension/catalog profile；不得更新、補值、刪除或重建 Mongo collection。
- 實作並驗證 loopback `8321` 服務、Bearer 設定介面、readyz 與相關測試。
- 產生必要的 service/unit 範本；若真正安裝 system-wide unit、修改 firewall 或需要 sudo，先停下向使用者確認，不得猜密碼或繞過權限。

本授權不包含：

- 覆蓋、刪除或 reset 現有 dirty working tree。
- 修改 port 8000 或其未知 Docker service。
- Git commit、push、PR。
- 改寫既有 checkpoint、vectors 或原始商品資料。
- 把 secret 寫入 repo、handoff、測試 fixture 或 log。

現有使用者修改一律保留；只做與本規格直接相關的 surgical changes。

## 1. 唯一現行介面

```http
POST /v2/product-candidates
```

舊 `/search/text` 與早期 `/v2/retrieval` 不得代替。新 route 固定使用 vanilla ULIP；不得讓 client 切換 Core RAG，也不得 silent rewrite query。

權威契約：

```text
/home/kyzen/V2T/HANDOFF_5090_TO_3090_PLACEMENT_LOOP_2026-07-15/contracts/
  ulip_product_candidate_request_v1.0.schema.json
  ulip_product_candidate_response_v1.0.schema.json
```

可執行 consumer spec：

```text
reference_v2t_da3/placement_loop/retrievers.py
reference_v2t_da3/tests/test_placement_loop.py §11、§11b、§18、§19、§20
```

## 2. Category normalization 正式政策

政策版本建議名稱：`v2t36_to_ikea15_v1`。Mapping 只代表可進的 catalog retrieval pool，不證明商品 subtype；每 row 必須保留原 request term 於 `matched_request_category`。

### 2.1 Enabled

```text
sofa, couch, loveseat       → Sofa（category degraded）
bench                        → Bench
ottoman                      → Storage Ottoman
dining table                 → Dining Table
desk                         → Office Desk
wardrobe, closet, armoire   → Wardrobe
bookshelf, bookcase,
shelf, shelving unit        → Bookshelf
tv stand, media console     → TV Stand
```

`shelf → Bookshelf` 只適用目前 V2T 的具 footprint 落地單元語意，不代表 wall-mounted shelf。

### 2.2 Mapped but blocked

```text
bed, bed frame → Bed
```

Bed pool 第一版整類 suppressed。Bed-only request 回 HTTP 200：

```json
{
  "status": "no_match",
  "results": [],
  "warnings": ["CATEGORY_DATA_QUALITY_BLOCKED"]
}
```

### 2.3 Known controlled terms but unsupported

```text
armchair
accent chair
lounge chair
chair
stool
table
coffee table
side table
end table
console table
nightstand
bedside table
bedside cabinet
dresser
cabinet
lamp
floor lamp
rug
```

禁止為提高 recall 而錯映至 Recliner、Dining Chair、Bar Stool、Filing Cabinet、Sideboard、Vanity Table 等不同用途 catalog。

受控但 unsupported 的 term 是正常 business no-match，不是 422。非 V2T 36-term 才是 semantic 422。

## 3. Tier 與 duplicate 政策

輸入先 trim，再 casefold：

- 任一 tier 內同詞只差大小寫：422，error code `CATEGORY_TERM_DUPLICATE`。
- Primary/secondary 原始同詞只差大小寫：422，同 error code。
- 不同 aliases 在同 tier 映到同 canonical pool：合併 pool，`matched_request_category` 使用 request array 最早出現的詞。
- 不同 aliases 跨 tier 映到同 canonical pool：primary precedence，移除 secondary pool，回 `CATEGORY_CANONICAL_POOL_DEDUPED`。
- 全 response 的 `product_id` 必須唯一。
- Primary 沒有合法 pool但 secondary 有結果時，結果仍標 `secondary`，不得升格。

## 4. Status 與 warning

Top-level v1 server 只能 emit：

```text
ok       results 非空
no_match results 為空
```

`partial` 是保留字，v1 MUST NOT emit。若 bundle、encoder 或任一應執行 tier 未完整執行，回 HTTP 503；不可用 `partial` 掩蓋。Row-level `dimension_status=partial` 不受影響。

Producer warning allowlist：

```text
CATEGORY_UNSUPPORTED
CATEGORY_PARTIALLY_UNSUPPORTED
CATEGORY_DATA_QUALITY_BLOCKED
CATEGORY_DATA_QUALITY_DEGRADED
CATEGORY_CANONICAL_POOL_DEDUPED
```

Warnings 必須 unique 且 lexicographic sort。少於 K 本身不是錯誤。Query 超限不得回成功 response 或 `QUERY_TRUNCATED` warning，必須 4xx。

## 5. Tokenizer gate

`token_count` 定義為：

```text
len([SOT] + BPE(query_text) + [EOT])
```

政策：

```text
0..75 content BPE tokens → encode；成功 response truncated=false, eot_present=true
76+ content BPE tokens   → HTTP 422；不得呼叫 encoder、不得 silent truncate
```

必加 75 pass、76 reject、EOT retained、tokenizer ID/hash 對應 fixtures。

## 6. Dimension、height 與 eligibility

- Wire dimensions 固定 mm；5090 只做一次 `/1000`。
- `complete` 必須有 width/depth/height 三軸正數。
- `partial` 必須有 dimensions object，width/depth 正數、height=null。
- `height_policy=optional` 且 height null 可進 footprint-only。
- `height_policy=required|unknown` 且 height null 必須 fail closed；正式取代早期 Q3 unknown fallback。
- `hard_filter_eligible` 定義為「該 row 有足夠可信資料，可進目前矩形 footprint hard placement gate」。
- missing、suspect、range、special、quality blocked/unknown 一律 false。
- `catalog_metadata` v1 一律省略，避免 unrestricted object 洩漏本機 path；如未來需要，升 schema 加 safe allowlist。

Production pipeline 必須：

```text
requested canonical pool 全量 ULIP scoring
→ immutable catalog/dimension/quality join
→ catalog/dimension trust + hard-eligibility gate
→ deterministic score desc / product_id tiebreak
→ 每 tier Top-K
```

不能先截 raw Top-K 再讓 missing/special rows 耗盡 K。若要保留 ineligible rows作診斷，另設 diagnostic lane/schema，不占 production K。

## 7. Sofa、Bed、range 與 special policy

### Sofa

- Category quality=`degraded`。
- 已知 32/50 special shape 全部 `hard_filter_eligible=false`。
- 剩餘 18 個 `not_catalog_flagged` 不是已驗證 rectangle；人工正向驗證前也全部 fail closed。
- 未驗 shape 建議：`footprint_kind=special_unmodeled`、`dimension_status=special_unmodeled`、flag `FOOTPRINT_SHAPE_UNVERIFIED`。
- 只有逐 row 可重現證據確認矩形後，才可隨新版 bundle/policy/hash 解鎖。

### Range

```text
DIMENSION_RANGE_COLLAPSED_HYPHEN
DIMENSION_RANGE_COLLAPSED_SLASH
DIMENSION_RANGE_PRESENT
DIMENSION_MAX_ONLY
```

- String collapse → `dimension_status=range_collapsed`。
- Paired range／max-only → `dimension_status=suspect`。
- 全部 `hard_filter_eligible=false`。
- 每 code 至少一個 positive fixture 與 cross-field negative fixture。
- 8 筆 string rows 的 hyphen/slash 分派必須依 source profile產生，不能猜。

Special flags 只是 evidence；安全 gate 必須依 status/quality/eligibility，不能期待 consumer 認得所有 flag。

## 8. Immutable bundles 與 hash 定義

必須建立：

```text
model checkpoint identity
vector bundle
catalog manifest
733-row dimension/quality profile
tokenizer bundle
```

每個 bundle 使用 deterministic canonical manifest，明確定義 canonical bytes、members、relative paths、bytes、SHA、rows/shape/order/source version。Response bundle SHA 指向該 canonical manifest bytes；readyz 必須重算 members 與 manifest。

Raw `vectors_pc.npy` 或 tokenizer.py 單檔 hash 不得冒充 aggregate bundle SHA。

5090 之後會 pin 完整 deployment tuple：

```text
five SHA-256
+ category_policy_version
+ dimension_axis_policy_version
```

禁止 TOFU、wildcard、mix-and-match。3090 回交時必須提供完整 profile tuple，但不要修改 5090 allowlist。

## 9. `/readyz`

新增 separate strict schema。最低語意：

```text
GET /readyz
ready     → HTTP 200
not_ready → HTTP 503
Cache-Control: no-store
```

Root 至少包含：

```text
schema_version=1.0
service=ulip-product-candidates
status=ready|not_ready
deployment_profile_id
endpoint_schema_version=1.0
provenance（與 product response 同七欄；not_ready可null）
checks 固定 booleans：
  model_checkpoint
  vector_bundle
  catalog_manifest
  dimension_bundle
  tokenizer_bundle
  row_identity_join
  policy_versions
inventory：vector shape/rows、catalog rows、dimension profile rows
failures[] unique uppercase machine codes
```

Ready 必須 profile/provenance非空、全部 checks true、failures empty；not_ready 至少一 false 且 failures非空。不得回 path或secret。

`relationship_policy` 對 product-candidates v1 是 N/A/future，不是 endpoint blocker；functional relation仍由5090保持 unverified/advisory。

## 10. Transport／service target

工程驗收拓撲：

```text
3090 API bind 127.0.0.1:8321
5090 SSH tunnel 127.0.0.1:18321 → edge05 127.0.0.1:8321
```

- 不碰 port 8000。
- 不開 LAN firewall；SSH tunnel 提供傳輸加密，MVP 不另加 app TLS。
- POST 使用 Bearer token（至少 32 random bytes），只放 0600 env/secret file，不入 repo/log/handoff。
- `/healthz`、`/readyz` 可在 loopback/tunnel 無 auth，但不得洩漏敏感資料。
- Server hard timeout 25s；client 30s，僅 connect/502/503/504 retry一次。
- 單 Uvicorn worker，GPU inference concurrency=1，最多1 queued；超額回503及 `Retry-After`。
- 提供 systemd unit與安全 EnvironmentFile範本；若需要 sudo 安裝/enable，先向使用者確認。

## 11. 必做測試

至少完成：

- Request/response Draft 2020-12 positive/negative。
- Whitespace、unknown/missing key、bool-as-number、NaN/Inf、duplicate term/ID/flags/yaw。
- Token 75/76、EOT、no encoder call on reject。
- Category exact/alias/unsupported/blocked/degraded/canonical overlap/tier quota。
- Bed-only no_match；Sofa/range/special fail closed。
- Dimensions complete、partial+optional、partial+required/unknown、missing/suspect/range/special。
- Bundle member/hash/row order/join fault injection與readyz 503。
- Deterministic ordering與每 tier K。
- 422/503/timeout/>4MiB/invalid UTF-8，沒有 synthetic fallback。
- 本包 `tools/check_live_endpoint.py` 經 exact 5090 strict consumer通過。

## 12. 回交要求

完成後建立：

```text
/home/kyzen/ULIP_RAG/HANDOFF_3090_IMPLEMENTED_PRODUCT_CANDIDATES_TO_5090.md
```

至少包含：

- 修改檔案與 function/class。
- Repo HEAD與完整 dirty state；不把pre-existing changes冒充本次修改。
- Bundle paths、manifest定義、row count/shape、五SHA與兩policy versions。
- Start/stop/healthz/readyz/live checker指令。
- 所有測試命令與結果。
- Local endpoint URL及Bearer設定方式（不得貼secret本身）。
- 仍是stub/TODO/尚未確認事項。

完成3090本機驗證後先不要宣稱雙機production E2E；通知5090，由5090完成allowlist、SSH tunnel與真實場景PlacementLoop驗收。
