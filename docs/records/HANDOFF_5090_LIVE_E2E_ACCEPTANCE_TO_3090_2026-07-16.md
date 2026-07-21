# 5090 → 3090：V2T × ULIP Product Candidates 真實跨機 E2E 驗收

日期：2026-07-16（Asia/Taipei）  
5090：`edge08` / `/home/kyzen/V2T_DA3`  
3090：`edge05` / `192.168.0.105` / `/home/kyzen/ULIP_RAG`

## 0. 結論

**PASS。** 已完成下列真實 production 流程：

```text
真實 V2T/DA3 場景點雲
→ Stage 06 authoritative occupancy NPZ
→ Stage 10 SDO
→ Stage 11 Qwen3-VL query/category/candidate spots
→ SSH tunnel
→ 3090 vanilla ULIP POST /v2/product-candidates
→ 10 個真實商品與實際尺寸
→ 5090 product × spot × pose search
→ placement_result.json
→ artifact validator PASS
```

本次沒有使用 synthetic products、舊 `/search/text`、Core RAG 或 fallback。
Bearer token 只由既有 SSH 在執行程序記憶體中讀取，未寫入本檔、repo、log
或 5090 local file。

## 1. 連線與 readiness

3090 service：

```text
listener      127.0.0.1:8321
tmux session  ulip-product-candidates
healthz       HTTP 200
readyz        HTTP 200, 7/7 checks=true
no Bearer     POST HTTP 401
```

5090 tunnel：

```text
127.0.0.1:18321 → edge05 127.0.0.1:8321
```

`/readyz` 實際回傳：

```text
deployment_profile_id  ikea733-product-candidates-v1-20260715-e18415000f11bce9
vector_shape           [733, 512]
catalog_rows           733
dimension_profile_rows 733
checks                 7/7 true
failures               []
```

## 2. 5090 exact deployment pin

`placement_loop/retrievers.py` 現在只接受完整 allowlisted tuple，不能個別 hash
mix-and-match：

```text
model_checkpoint  6ee2b44e8647af8323dc9405604b04f37cc16d63b6e40e2034f1e9c0cb6cde0f
vector_bundle     25472051a53b08869f3e2c458b2bc4f0ad39d0f6d12bfacbb271672ee018a0e1
catalog_manifest  f23fd25a6c72045ae30b5475a5d279e7ac4389b85b4cd50d3a2da3bf42624304
dimension_bundle  dc2b031061e77b093542e1d260572861704897e9712527f4e42a4cf41b4c3cf4
tokenizer_bundle  81a4b450a023da2330ee1763073bab05c6fd2a5c2ea80f4717678b8f469b2b4c
category_policy   v2t36_to_ikea15_v1
dimension_policy  dimension_axis_v1
```

另依已凍結 v1 契約，consumer 拒絕 top-level `partial`。新增的 unpinned
tuple 與 reserved `partial` 負向測試均通過。

## 3. 真實 V2T scene

來源：`Outputs/0a7cc12c0e_u128`。為避免覆寫既有成果，本次建立獨立輸出：

```text
Outputs/e2e_live_0a7cc12c0e_u128_20260716
```

複製來源場景的真實 `aligned_points.ply`、layout、style、transform 與 preview
frames，再以目前程式重建 Stage 06/10/11 artifacts。

Stage 06：

```text
floor area       15.24 m²
obstacle area     8.04 m²
free area         6.67 m²
free ratio       44%
grid resolution   0.025 m
candidate rects   3
```

Candidate spots：

```text
rect:0  center=[4.4625, 1.7875]  size=3.675×0.625 m  area=2.297 m²
rect:1  center=[2.5375, 1.2250]  size=2.375×0.500 m  area=1.188 m²
rect:2  center=[5.8000, 2.4125]  size=1.000×0.625 m  area=0.625 m²
```

Stage 10：14 個場景 objects，沒有 dropped object。

Stage 11 使用原始需求 `幫我找個沙發`，由目前 Qwen3-VL 產生：

```text
primary category    ottoman
secondary category  sofa
feasibility         fits
query token_count   77（SOT + 75 content + EOT；未截斷，EOT存在）
```

Request ID：

```text
e2e_live_0a7cc12c0e_u128_20260716-20260716014513
```

## 4. 3090 真實商品

3090 回傳 10 個 production product IDs，全部為 primary `Storage Ottoman`：

```text
67d113aaa3ae78c12f05aee1  BRÄNNBOLL
67d1193fa3ae78c12f05af91  INNDYR
67d167bb7270e1a09493c251  SMÅSTAD
67d167c37270e1a09493c252  ESSEBODA
67d16afd7270e1a09493c2b7  SÖDERHAMN
67d16b257270e1a09493c2bc  GRUNDSJÖ
67d16b3d7270e1a09493c2bf  ESSEBODA
67d16b757270e1a09493c2c6  KEDJEBO
67d16f907270e1a09493c341  BESTÅ
67d17f717270e1a09493c51e  BESTÅ
```

Wire dimensions 使用 mm，5090 strict adapter 只轉換一次為 m。Retrieval status
為 `ok`；provenance 完整 tuple 與 allowlist 相同。

## 5. Selected placement

```json
{
  "spot_id": "e2e_live_0a7cc12c0e_u128_20260716:rect:0",
  "product_id": "67d16f907270e1a09493c341",
  "product_name": "BESTÅ",
  "center_xy": [5.52505, 1.88455],
  "yaw": 0,
  "score": 0.681978,
  "score_type": "heuristic_composite_v1",
  "yaw_semantics": "footprint_axis_only",
  "facing_direction": "unknown"
}
```

Selected product dimensions：

```text
width  0.6001 m
depth  0.4191 m
height 0.3810 m
status complete
```

Scoring breakdown：

```text
collision.hard_reject       false
collision.score             1.000000
retrieval.raw               0.08254727721214294
retrieval.normalized        0.541274
retrieval.weight            0.55
size_fit                    1.000000
anchor_relation             N/A（本 request 無 anchor intent）
clearance_proxy             0.275 m
clearance_score             0.458333
walkability_score           0.958830
clearance_walkability       0.683557
wall/corner                 N/A（無 wall/corner intent）
functional_fit              unverified_advisory_not_ranked
spatial_score               0.853949
final heuristic score       0.681978
```

Selected pair search stats：270 centers tested、270 collision-free poses、top 24
poses執行 walkability；`search_exhaustive=false`。

## 6. Candidates 與 rejected reasons

```text
retrieved products                 10
V2T candidate spots                3
legal placement candidates        16
rejected product×spot pairs        14
```

14 個 rejection 全部有 machine-readable `FOOTPRINT_EXCEEDED`：

- SMÅSTAD × rect:1
- GRUNDSJÖ × rect:1
- ESSEBODA `67d16b3d...` × rect:0/1/2
- BRÄNNBOLL × rect:0/1/2
- SÖDERHAMN × rect:0/1/2
- ESSEBODA `67d167c3...` × rect:2
- INNDYR × rect:1/2

每筆完整 product ID、spot ID、scope、detail 與 `failure_counts` 保存在
`placement_result.json`。

## 7. 驗證結果與 artifact hashes

```text
default Python placement suite     PASS
Open3DIS placement suite           PASS（含 production result schema）
unpinned tuple negative test       PASS
reserved partial negative test     PASS
live /healthz                      PASS
live /readyz                       PASS
Bearer authentication              PASS
live production retrieval          PASS
artifact validator                 PASS
selected occupancy collision       PASS（hard_reject=false）
```

Artifacts：

```text
340efb9b4a5c52a89f9e1f598b745ee940b1ab419c50d807c971616a064aabf1  occupancy_grid_v1.npz
4970ea09f8b42070fffc9f3b92619beb57e5bd6975c1513c7eb103ecae320a1d  scene_description.json
ed345eec793699acdf5de18ec9e81df0df1c3d583b0d386fb4e541099626874d  retrieval_request.json
e5f1475b639b1f011e3cd9b526bb31fa4e53907ffc63f20cdb3721933acb9041  placement_candidates.json
5d900efcc1d48087b51c0e167a9d34e7402fb6d9e4fe730185b72e6dc21f55c1  placement_result.json
```

修改後 source hashes：

```text
14cc2eb0cf30bbc4f06c0acfc82744f77c6602b36dcb42a89be040ea8752ed7e  placement_loop/retrievers.py
0a9f9375ffa529685976dcb5c9d769d0e2302037635a60be4adcc376c7af15af  tests/test_placement_loop.py
```

## 8. 重現命令

Bearer 不落地到 5090：

```bash
ssh -i ~/.ssh/id_ed25519_5090_to_3090 \
  -o IdentitiesOnly=yes -N \
  -L 127.0.0.1:18321:127.0.0.1:8321 \
  kyzen@192.168.0.105

cd /home/kyzen/V2T_DA3
ULIP_API_TOKEN="$(ssh -i ~/.ssh/id_ed25519_5090_to_3090 \
  -o IdentitiesOnly=yes kyzen@192.168.0.105 \
  'cat /home/kyzen/.config/ulip-product-candidates/e2e_token')" \
/home/kyzen/miniconda3/envs/open3dis/bin/python \
  tools/run_placement_loop.py \
  --scene-dir Outputs/e2e_live_0a7cc12c0e_u128_20260716 \
  --ulip-url http://127.0.0.1:18321/v2/product-candidates \
  --top-k 10 --timeout 30 --retries 1

/home/kyzen/miniconda3/envs/open3dis/bin/python \
  tools/validate_placement_artifacts.py \
  Outputs/e2e_live_0a7cc12c0e_u128_20260716
```

## 9. 完成範圍與仍有效的限制

本次已確認雙機 production E2E 可執行，但 selected 仍是「已評估離散候選中的最高
heuristic score」，不是連續空間 global optimum 或安全認證。以下限制不因本次 PASS
而消失：

- Yaw 只表示矩形 footprint axis；商品 front/facing 仍 unknown。
- Walkability 是 largest-connected-component area proxy，不是 door-aware 或人體走道認證。
- Functional relationship 仍是 unverified advisory，不參與 ranking。
- Pose center 以 0.1 m 離散搜索，walkability只算 preliminary top 24。
- 目前 Stage 11 與 Placement Loop 仍需在主 `pipeline.sh` 後分開執行。
- Systemd、自啟、長期 secret rotation、GPU soak與高頻 readiness 仍是 operation TODO。

