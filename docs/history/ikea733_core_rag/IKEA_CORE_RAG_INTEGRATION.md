# IKEA × Core RAG 串接、訓練與上線手冊

更新日期：2026-07-15（Asia/Taipei）

## 1. 結論先行

IKEA serving 現在已經能真正執行 Core RAG，不再只是「Core 有 RAG 程式、IKEA 服務沒有使用」。目前已完成的能力是：

- 原有 `/search/text` 保持相容，且可在 request 明確選擇 `vanilla` 或 `rag`。
- 新增 `/search/text/rag`，強制走 RAG query embedding。
- 新增 `/search/text/compare`，同一個 query 同時計算 vanilla 與 RAG，供 A/B 比較。
- 新增 `/healthz` 與會檢查模型、vectors、metadata、RAG adapter 的 `/readyz`。
- RAG 啟用時採 fail-closed：checkpoint、corpus、FAISS index、維度、列數或共享 embedding space 任一不相容，服務拒絕 ready，而不是假裝在跑 RAG。
- 已新增 IKEA 專用的 RAG Stage 1／Stage 2 訓練入口與 provenance 驗證。

但「可以執行」不等於「品質已通過」。100-query 暫時性 A/B 顯示舊 Core enhancer 在 IKEA 商品檢索上嚴重退化，因此 production 預設仍是：

```text
RAG_ENABLED=1
RAG_DEFAULT_MODE=vanilla
```

這代表 RAG adapter 會載入、可由明確 endpoint 或 request 使用，但既有 client 不會被靜默切換。必須完成 IKEA-specific Stage 1／Stage 2 正式訓練、重建 point-cloud vectors 並通過 promotion gate，才可把預設改成 `rag`。

## 2. 「原本 vanilla」與「現在已串接」的差異

### 2.1 原本 IKEA vanilla serving

原流程只有 IKEA checkpoint 與 IKEA 商品 PC vectors：

```text
英文 query
→ CLIP SimpleTokenizer
→ IKEA ULIP frozen text encoder
→ text_projection（512d）
→ L2 normalize
→ 與 733 × 512 vectors_pc.npy 做 dot product
→ Top-K 商品
```

這條路沒有：

- MiniLM corpus retrieval
- FAISS knowledge retrieval
- RAGEnhancer
- Core Stage 1 checkpoint
- Core Stage 2 point branch

Core RAG 當時雖存在於 `core/models/ULIP_models.py`，但它是一條獨立研究／訓練線，沒有被 `ikea/app_ikea_retrieval.py` 呼叫。

### 2.2 現在的 explicit RAG serving

現在 RAG query 路徑為：

```text
英文 query
├─→ IKEA frozen CLIP text backbone → 原始 base feature（投影前 512d）
│
└─→ all-MiniLM-L6-v2（384d）
    → legacy1095 FAISS IndexFlatIP
    → Top-K knowledge documents
    → 同一個 frozen CLIP text backbone 編碼 documents

原始 base feature + document base features
→ Core Stage 1 RAGEnhancer（Multi-Head Attention + FFN）
→ 同一個 IKEA text_projection
→ L2 normalize
→ 與 IKEA PC vectors 做 cosine/dot-product Top-K
```

關鍵點不是「兩邊都是 512 維」；系統會逐 tensor 驗證 IKEA checkpoint 與 Core Stage 1 checkpoint 的 frozen text/image base 是否完全相同。目前實際 bundle 已驗證共享 tensor `301/301` 形狀、dtype、值完全一致。

另外，長 query 的 tokenizer 已修正為一定保留 EOT token。CLIP 會在 EOT 位置取文字特徵，舊版直接切前 77 tokens 可能把 EOT 切掉，造成錯誤 pooling。

## 3. 目前正確的 legacy serving bundle

舊 Core Stage 1 enhancer 要以它原本的 corpus 執行。可重現、已驗證的 bundle 是：

```text
IKEA vanilla base checkpoint
+ Core RAG2 Stage 1 enhancer
+ archived legacy1095 corpus/index
+ IKEA vanilla PC vectors/meta
```

### 3.1 不可任意互換的規則

1. Core Stage 1 enhancer 必須配 `legacy1095`。
2. 不可因為新版 corpus 同樣是 MiniLM 384d，就把 8,256-row corpus 換進來。維度相同不代表 enhancer 訓練時的 retrieval distribution 相同。
3. 目前 serving 絕對不能拿 Core RAG2 Stage 2 的 point branch 取代 IKEA point branch。
4. IKEA PC vectors 必須與產生它們的 IKEA point encoder checkpoint 綁定；不可只看 shape 是 `(N, 512)` 就混用。

### 3.2 為何不得使用 Core Stage 2 point branch

Core RAG2 Stage 2 是用 Core／ShapeNet 資料，把 Core PointBERT point branch 對齊至 Core RAG-enhanced text space。它不是用 IKEA 的 733 件商品、IKEA PLY、IKEA caption 與 IKEA render 訓練。

如果把 Core Stage 2 point branch 直接接到目前 IKEA serving，會同時破壞兩個契約：

- 現有 `vectors_pc.npy` 是 IKEA checkpoint 產生，不是 Core Stage 2 產生。
- Core Stage 2 point branch 的資料分布與 IKEA 商品分布沒有被證明相容。

所以目前 serving 只取 Core Stage 1 的 `rag_enhancer`，不取 Core Stage 2 的 `point_encoder`／`pc_projection`。未來可以使用的 Stage 2 必須是由 `ikea/main_ikea_rag.py` 以 IKEA 資料訓練出的 IKEA Stage 2，並用該 checkpoint 重建整套 PC vectors。

## 4. Artifact 路徑、列數與 SHA-256

以下 hash 已於本機重新計算。

| Artifact | 路徑 | 內容／shape | SHA-256 |
|---|---|---:|---|
| IKEA vanilla checkpoint | `/mnt/P300/data/ULIP/checkpoint_last.pt` | 約 826 MiB | `49ac18ab10950412f016c9e023d1c7e9b34250a257d59b70b5590d74207f3e01` |
| Core RAG2 Stage 1 | `/home/kyzen/ULIP_RAG/core/outputs/RAG2_Stage1/checkpoint_best.pt` | 約 680 MiB；只取 enhancer | `c67ec093d3118f85a8498d436f50cdfa64a0cfef2e616c0786cd14595ab5f298` |
| legacy1095 corpus | `/mnt/P300/data/ULIP/4090_cheng_archive_20260715/ULIP_RAG/rag_corpus_1095/rag_corpus.jsonl` | 1,095 rows | `d9bc65b3006e7b0a085025e4a53fee775197df506ff8b0e4b7404e5fa424b537` |
| legacy1095 FAISS | `/mnt/P300/data/ULIP/4090_cheng_archive_20260715/ULIP_RAG/rag_corpus_1095/corpus_index.faiss` | 1,095 rows、384d | `2847f6638b8dda42ee52dcdfc2a5d7f2aa71b650d1d266850241f294119c3f0f` |
| IKEA PC vectors | `/mnt/P300/data/ikea_data/vectors/vectors_pc.npy` | `733 × 512`、L2-normalized | `30070b4c576bf511ac498bc1f6ddb46502bd0cf4737a455447d5e85fc707d207` |
| IKEA PC row metadata | `/mnt/P300/data/ikea_data/vectors/meta_pc.jsonl` | 733 rows | `0c05805753bb085080e75fa8db204c795f46b6f2c55754e37ae3a819bd068a94` |
| 現有 vector schema | `/mnt/P300/data/ikea_data/vectors/schema.json` | 舊 schema | `8058c788f938bd5d4f74292c46962dd93cf0fc588b903fd27ff19df36e62f08f` |

以下 artifact 存在，但不得加入上述 serving bundle：

| Artifact | 路徑 | SHA-256 | 用途限制 |
|---|---|---|---|
| Core RAG2 Stage 2 | `/home/kyzen/ULIP_RAG/core/outputs/RAG2_Stage2/checkpoint_best.pt` | `bc1b3248562550f7110fff0d324de8a42f94eab6f8da16f83c724b3aba7bbcf1` | Core／ShapeNet point branch；不可拿來生成或取代 IKEA PC vectors |

`legacy1095` profile 會 fail-closed 驗證 corpus rows、FAISS rows、FAISS dimension、corpus/index hash 及目前 legacy checkpoint bundle。任何 drift 都會使 RAG 初始化失敗。

## 5. Serving 啟動方式與環境變數

### 5.1 現行安全啟動

```bash
cd /home/kyzen/ULIP_RAG

USE_MONGO=0 \
RAG_ENABLED=1 \
RAG_DEFAULT_MODE=vanilla \
RAG_CORPUS_PROFILE=legacy1095 \
RAG_STAGE1_CKPT=/home/kyzen/ULIP_RAG/core/outputs/RAG2_Stage1/checkpoint_best.pt \
RAG_CORPUS_DIR=/mnt/P300/data/ULIP/4090_cheng_archive_20260715/ULIP_RAG/rag_corpus_1095 \
RAG_TOP_K=5 \
bash ikea/run_app_3090.sh 8321
```

`run_app_3090.sh` 另會設定：

```text
CKPT=/mnt/P300/data/ULIP/checkpoint_last.pt
VEC_DIR=/mnt/P300/data/ikea_data/vectors
ULIP_OUTPUT=/mnt/P300/data/ikea_data
CUSTOM_OUTPUT=/mnt/P300/data/custom_data
PYTHONPATH=/home/kyzen/ULIP_RAG/core:/home/kyzen/ULIP_RAG/ikea:/home/kyzen/TRELLIS
```

環境變數語意：

| 變數 | 說明 |
|---|---|
| `RAG_ENABLED=1` | 載入 RAG adapter；任一 RAG artifact 不相容就拒絕啟動 |
| `RAG_ENABLED=0` | 明確停用 RAG，只提供 vanilla；不是自動 fallback |
| `RAG_DEFAULT_MODE=vanilla` | `/search/text` 未指定 mode 時走 vanilla |
| `RAG_DEFAULT_MODE=rag` | `/search/text` 未指定 mode 時走 RAG；只有 promotion 後才可設定 |
| `RAG_STAGE1_CKPT` | query-side enhancer checkpoint |
| `RAG_CORPUS_DIR` | 同時包含 `rag_corpus.jsonl` 與 `corpus_index.faiss` |
| `RAG_CORPUS_PROFILE` | `legacy1095` 或 `provenance`；前者鎖定歷史 bundle hashes，後者驗證 IKEA Stage 1／Stage 2／vector schema provenance chain |
| `RAG_TOP_K` | 預設檢索的知識文件數 |
| `CKPT` | ULIP base checkpoint；必須與 vectors／RAG shared base 相容 |
| `VEC_DIR` | `vectors_pc.npy` 與 `meta_pc.jsonl` 所在目錄 |
| `USE_MONGO=0` | 本檢索流程不查 Mongo；本輪所有驗證均未寫 Mongo |

## 6. API 契約與 curl

### 6.1 Health 與 readiness

```bash
curl -sS http://127.0.0.1:8321/healthz | jq
curl -sS http://127.0.0.1:8321/readyz | jq
```

`/healthz` 只表示 process 有回應；production 驗收必須看 `/readyz`。`/readyz` 會回報：

- model 是否載入
- PC vector rows／metadata rows
- embedding dimension
- RAG 是否為 required
- adapter 是否 ready
- 301 個共享 base tensors 的相容結果
- checkpoint、corpus、FAISS 路徑與 hash

### 6.2 明確 vanilla

```bash
curl -sS -X POST http://127.0.0.1:8321/search/text \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "a compact modern wooden dining chair",
    "category": "Dining Chair",
    "top_k": 5,
    "embedding_mode": "vanilla"
  }' | jq
```

### 6.3 明確 RAG

```bash
curl -sS -X POST http://127.0.0.1:8321/search/text/rag \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "a compact modern wooden dining chair",
    "category": "Dining Chair",
    "top_k": 5,
    "rag_top_k": 5,
    "include_rag_documents": true
  }' | jq
```

也可以對 `/search/text` 傳入 `"embedding_mode":"rag"`。RAG response 的 `diagnostics` 會包含 tokenization、retrieved document row/score/text，以及 RAG embedding 對 vanilla embedding 的 cosine/L2 差異。

### 6.4 同 query A/B

```bash
curl -sS -X POST http://127.0.0.1:8321/search/text/compare \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "a compact modern wooden dining chair",
    "category": "Dining Chair",
    "top_k": 5,
    "rag_top_k": 5,
    "include_rag_documents": false
  }' | jq
```

回傳：

```text
vanilla: vanilla 的完整 TextSearchResp
rag: RAG 的完整 TextSearchResp
overlap_at_k: 兩邊 Top-K 商品 ID 集合的交集比例
```

輸入錯誤回 400；明確要求 RAG 但 RAG 未 ready 回 503。RAG enabled 而 artifact 不相容時，服務會在 startup 直接失敗。

## 7. 已執行的真實 serving 驗證

FastAPI 曾實際在 `127.0.0.1:8321` 啟動，並完成以下真實 HTTP 呼叫：

- `GET /readyz`：HTTP 200
- `POST /search/text` vanilla：成功
- `POST /search/text/rag`：成功
- `POST /search/text/compare`：成功

實際 query：

```text
query = "a compact modern wooden dining chair"
category = "Dining Chair"
top_k = 5
```

該 query 的診斷結果：

```text
RAG-vs-vanilla query cosine = 0.2270
RAG-vs-vanilla L2 distance = 1.243
Top-5 product overlap = 0
```

這證明 RAG path 不是把 vanilla 結果換標籤，而是真的檢索文件、經 enhancer 產生不同 query embedding 並重新排序商品。它同時也顯示舊 Core enhancer 對 IKEA query 的改動非常大，必須以品質評測決定能否 promotion。

## 8. 100-query 暫時性 A/B 結果

測試方法：從 `meta_pc.jsonl` 中有 caption 的 rows 取固定前 100 筆；每筆以該 caption 當 query、以該商品 category 做 exact category prefilter，expected target 是同一筆 product ID。vanilla 與舊 Core RAG 使用同一批 query、同一 gallery 與同一 category filter。

| 模式 | R@1 | R@5 | MRR@10 |
|---|---:|---:|---:|
| IKEA vanilla | **0.7400** | **1.0000** | **0.86417** |
| 舊 Core RAG + legacy1095 | 0.0100 | 0.1700 | 0.09849 |

其他結果：

```text
mean Top-10 vanilla/RAG overlap = 0.33
request/evaluation failures = 0
```

因此目前決策是：

```text
default = vanilla
explicit RAG = available for diagnostics/A-B
old Core RAG = not eligible for production default
```

### 8.1 這份 A/B 證據的限制

這是本輪臨時 Python／E2E 功能與 promotion-gate evidence，沒有落成 immutable query-set/result artifact。它也是 in-sample/self-retrieval 型評估：query 直接來自商品自己的 metadata caption，且 target 是同商品，不是 5090 V2T room request 的 held-out relevance test。

所以它足以否決舊 Core RAG 的預設上線，但不能用來宣稱 vanilla 的真實場景品質，也不能取代正式 held-out 評測。

後續已提供 `ikea/evaluate_rag_retrieval.py`，可把固定 seed 的 self-retrieval suite、selected IDs、caption hashes、artifact hashes、failures 與 metrics 原子化寫成 JSON：

```bash
cd /home/kyzen/ULIP_RAG
export PYTHONPATH=/home/kyzen/ULIP_RAG/core:/home/kyzen/ULIP_RAG/ikea
mkdir -p /mnt/P300/data/ULIP/evals

/home/kyzen/miniconda3/envs/ulip/bin/python \
  ikea/evaluate_rag_retrieval.py \
  --profile legacy1095 \
  --limit 100 \
  --seed 0 \
  --top-k 10 \
  --rag-top-k 5 \
  --device cuda:0 \
  --include-per-query \
  --output /mnt/P300/data/ULIP/evals/ikea_rag_legacy1095_self_retrieval.json
```

注意：這個正式 evaluator 採 SHA-256 seeded ordering；本節上方既有數字來自當時的「有 caption rows 固定前 100 筆」。兩種 selection 不同，不應假設重跑會得到完全相同的數字。promotion 比較必須固定同一份輸出中的 selected IDs。

### 8.2 可重現 evaluator 的實際 GPU 結果

已以 evaluator 預設的 SHA-256 selection、`limit=100`、`seed=0`、`device=cuda` 真實執行；本次沒有指定 `--output`，因此只驗證 JSON stdout，尚未建立不可變結果檔：

| 指標 | Vanilla | 舊 Core RAG |
|---|---:|---:|
| R@1 | **0.85** | 0.08 |
| R@5 | **1.00** | 0.22 |
| MRR@10 | **0.9217** | 0.1481 |
| Miss@10 | 0 | 62 |

```text
mean Top-10 overlap = 0.31
execution errors = 0
wall time ≈ 17.66 seconds
```

這組 selection 與第 8 節上方的固定前 100 筆不同，所以數字不可直接逐項比較；兩組獨立測試的方向一致：目前歷史 Core enhancer 不適合成為 IKEA production 預設。

## 9. IKEA RAG 訓練原理

正式入口：

```text
/home/kyzen/ULIP_RAG/ikea/main_ikea_rag.py
```

支援工具：

```text
ikea/rag_training_utils.py
ikea/tests/test_rag_training_utils.py
```

它會把 `IkeaULIP` 的 tokenized caption 與原始 caption 一起傳給 RAG retriever，不再發生 IKEA dataset 只有 tokens、RAG 卻拿不到 raw query 的斷線。

### 9.1 Dataset 與 provenance

預設以商品 ID 做 deterministic hash split，`--train-ratio 0.9` 在目前有效 729 件資料上得到：

```text
train = 663
val = 66
```

訓練啟動前會驗證並記錄：

- corpus/index 是否存在
- corpus rows 是否等於 FAISS `ntotal`
- FAISS dimension 是否為 MiniLM 的 384d
- corpus/index SHA-256
- IKEA JSON、point cloud、caption、render inventory manifest
- base checkpoint SHA-256
- ordered product IDs hash
- Git commit/dirty 狀態

每個 output directory 會寫：

```text
dataset_manifest.json
run_provenance.json
checkpoint_last.pt
checkpoint_best.pt
log.jsonl
```

Stage 2 會嚴格要求 Stage 1 checkpoint，並驗證 Stage 1 的完整 enhancer tensors、tensor shapes、corpus hash、index hash、dataset bundle hash。缺 checkpoint、錯 corpus 或 dataset drift 都會中止，不再用 random enhancer 繼續訓練。

### 9.2 Stage 1：訓練 IKEA RAG enhancer

凍結：

```text
CLIP text encoder
CLIP image encoder
text/image projections
PointBERT
pc_projection
logit_scale
```

只訓練：

```text
rag_enhancer
```

Stage 1 不需要 point cloud forward。對每筆 IKEA caption，MiniLM 先取 legacy1095 Top-K，documents 再由 frozen CLIP text encoder 編成 base features，最後由 RAGEnhancer 融合。

Loss：

```text
L_stage1
= MSE(normalize(enhanced_text), normalize(original_text).detach())
+ 0.5 × bidirectional_InfoNCE(enhanced_text, image)
```

`original_text ↔ image` contrastive loss 只作參考 metric，不加入 total loss。best checkpoint 依 validation `T2I R@1` 選擇。

### 9.3 Stage 2：訓練 IKEA point branch

載入並凍結：

```text
Stage 1 rag_enhancer
CLIP text/image encoders
text/image projections
logit_scale
```

只訓練：

```text
PointBERT point_encoder
pc_projection
```

Stage 2 會把 frozen enhancer 設為 eval，避免 dropout 讓 target space 每批漂移。

Loss：

```text
L_stage2
= bidirectional_InfoNCE(point_cloud, enhanced_text)
+ 0.5 × bidirectional_InfoNCE(point_cloud, image)
```

best checkpoint 依 validation `S2T R@1` 選擇。正式 Stage 2 完成後，舊 PC vectors 全部失效，必須用 Stage 2 checkpoint 重建。

## 10. legacy1095 完整訓練命令

以下命令全部以 3090 的 `ulip` conda environment 執行。50 epochs 是目前 trainer 預設，不代表已證明最佳；正式實驗應保留完整 manifest/log 並依 held-out 指標調整。

### 10.1 共用環境

```bash
cd /home/kyzen/ULIP_RAG
source /home/kyzen/miniconda3/etc/profile.d/conda.sh
conda activate ulip

export PYTHONPATH=/home/kyzen/ULIP_RAG/core:/home/kyzen/ULIP_RAG/ikea
export LEGACY1095=/mnt/P300/data/ULIP/4090_cheng_archive_20260715/ULIP_RAG/rag_corpus_1095
export IKEA_BASE=/mnt/P300/data/ULIP/checkpoint_last.pt
```

### 10.2 Stage 1 preflight（不建模、不訓練）

```bash
python ikea/main_ikea_rag.py \
  --stage stage1 \
  --validate-only \
  --base-ulip-checkpoint "$IKEA_BASE" \
  --rag-corpus "$LEGACY1095/rag_corpus.jsonl" \
  --rag-index "$LEGACY1095/corpus_index.faiss" \
  --rag-top-k 5 \
  --train-ratio 0.9 \
  --workers 0
```

### 10.3 正式 Stage 1

```bash
python ikea/main_ikea_rag.py \
  --stage stage1 \
  --output-dir /mnt/P300/data/ULIP/ikea_rag_legacy1095_stage1 \
  --base-ulip-checkpoint "$IKEA_BASE" \
  --rag-corpus "$LEGACY1095/rag_corpus.jsonl" \
  --rag-index "$LEGACY1095/corpus_index.faiss" \
  --rag-top-k 5 \
  --train-ratio 0.9 \
  --epochs 50 \
  --batch-size 16 \
  --workers 8 \
  --lr 3e-4 \
  --warmup-epochs 1
```

### 10.4 Stage 2 preflight

```bash
python ikea/main_ikea_rag.py \
  --stage stage2 \
  --validate-only \
  --stage1-checkpoint /mnt/P300/data/ULIP/ikea_rag_legacy1095_stage1/checkpoint_best.pt \
  --base-ulip-checkpoint "$IKEA_BASE" \
  --rag-corpus "$LEGACY1095/rag_corpus.jsonl" \
  --rag-index "$LEGACY1095/corpus_index.faiss" \
  --rag-top-k 5 \
  --train-ratio 0.9 \
  --workers 0
```

### 10.5 正式 Stage 2

```bash
python ikea/main_ikea_rag.py \
  --stage stage2 \
  --output-dir /mnt/P300/data/ULIP/ikea_rag_legacy1095_stage2 \
  --stage1-checkpoint /mnt/P300/data/ULIP/ikea_rag_legacy1095_stage1/checkpoint_best.pt \
  --base-ulip-checkpoint "$IKEA_BASE" \
  --rag-corpus "$LEGACY1095/rag_corpus.jsonl" \
  --rag-index "$LEGACY1095/corpus_index.faiss" \
  --rag-top-k 5 \
  --train-ratio 0.9 \
  --epochs 50 \
  --batch-size 16 \
  --workers 8 \
  --lr 3e-4 \
  --warmup-epochs 1
```

若中斷，只能以同 stage、同 corpus/index hash、同 dataset bundle 的 checkpoint resume：

```bash
--resume /path/to/the/same-stage/checkpoint_last.pt
```

## 11. 用 IKEA Stage 2 重建 PC vectors

不要直接覆蓋目前 production vectors。先輸出到 candidate directory：

```bash
cd /home/kyzen/ULIP_RAG
source /home/kyzen/miniconda3/etc/profile.d/conda.sh
conda activate ulip
export PYTHONPATH=/home/kyzen/ULIP_RAG/core:/home/kyzen/ULIP_RAG/ikea

python ikea/build_vectors.py \
  --json_dir /mnt/P300/data/ikea_data/json \
  --out_dir /mnt/P300/data/ikea_data/vectors_ikea_rag_candidate \
  --ckpt /mnt/P300/data/ULIP/ikea_rag_legacy1095_stage2/checkpoint_best.pt \
  --model ULIP_PointBERT \
  --modalities pc \
  --npoints 8192 \
  --pc_sampler fps \
  --batch_size_pc 32 \
  --device cuda
```

`build_vectors.py` 對 RAG checkpoint 只載入 base／point state，排除 `rag_enhancer.*`，不會為離線 PC encoding 無謂載入 MiniLM/FAISS。輸出的 `schema.json` 會包含 checkpoint path/hash、是否為 RAG checkpoint、原始 provenance、point sampler 與 embedding dimension。

重建後至少要驗證：

```bash
python - <<'PY'
import json
import numpy as np

root = "/mnt/P300/data/ikea_data/vectors_ikea_rag_candidate"
vectors = np.load(f"{root}/vectors_pc.npy")
with open(f"{root}/meta_pc.jsonl", encoding="utf-8") as handle:
    meta = [json.loads(line) for line in handle if line.strip()]
norms = np.linalg.norm(vectors, axis=1)
print("shape=", vectors.shape)
print("meta_rows=", len(meta))
print("finite=", bool(np.isfinite(vectors).all()))
print("norm_min/max=", float(norms.min()), float(norms.max()))
assert vectors.shape == (733, 512)
assert len(meta) == 733
assert np.isfinite(vectors).all()
assert np.allclose(norms, 1.0, atol=1e-4)
PY

sha256sum \
  /mnt/P300/data/ikea_data/vectors_ikea_rag_candidate/vectors_pc.npy \
  /mnt/P300/data/ikea_data/vectors_ikea_rag_candidate/meta_pc.jsonl \
  /mnt/P300/data/ikea_data/vectors_ikea_rag_candidate/schema.json
```

## 12. Promotion gate 與切換方式

### 12.1 不可省略的 gate

1. **訓練完整性**：Stage 1／Stage 2 正式 run 完成；不是 `--smoke-test` checkpoint。
2. **Provenance chain**：Stage 2 必須指向同 corpus/index/dataset hash 的 Stage 1；vectors schema 必須指向該 Stage 2。
3. **Embedding compatibility**：IKEA Stage 2 與 Stage 1 的 frozen shared base 全 tensor exact match。
4. **Immutable profile**：為 candidate 註冊 checkpoint、corpus、index、vectors、meta hashes；不得用 `custom` profile 靜默跳過已知 hash。
5. **Vector integrity**：`733 × 512`、733-row ordered meta、finite、non-zero、L2 normalized、product ID 唯一。
6. **固定 A/B suite**：把 100-query suite、expected IDs、原始 responses、metrics、執行時間與 artifact hashes落成不可變結果。
7. **非劣性門檻**：candidate 至少需在同一固定 suite 達到 vanilla 的主要指標；目前可用 guardrail 是 R@1 不低於 `0.74`、R@5 不低於 `0.99`、MRR@10 不低於 `0.85`。這些只是既有 in-sample gate，不是最終品質標準。
8. **Held-out V2T 驗收**：以 5090 room request／人工標註相關商品評估 Top-K relevance、category、尺寸 fit；不得只用商品自己的 caption。
9. **Serving 驗收**：candidate `/readyz` 200；vanilla、rag、compare 都通；無 silent fallback；記錄 latency、VRAM、併發與 timeout。
10. **Canary/rollback**：先保持 `RAG_DEFAULT_MODE=vanilla`，由 explicit RAG 小流量觀察；原 vanilla checkpoint/vector bundle 必須可立即切回。

舊 Core RAG 的 100-query R@1 只有 `0.01`，明確未通過 gate。

### 12.2 Candidate 環境切換

通過 gate 且已註冊新的 immutable profile 後，candidate 應綁定：

```bash
export CKPT=/mnt/P300/data/ULIP/ikea_rag_legacy1095_stage2/checkpoint_best.pt
export VEC_DIR=/mnt/P300/data/ikea_data/vectors_ikea_rag_candidate
export RAG_STAGE1_CKPT=/mnt/P300/data/ULIP/ikea_rag_legacy1095_stage1/checkpoint_best.pt
export RAG_CORPUS_DIR=/mnt/P300/data/ULIP/4090_cheng_archive_20260715/ULIP_RAG/rag_corpus_1095
export RAG_CORPUS_PROFILE=provenance
export RAG_TOP_K=5
export RAG_ENABLED=1
export RAG_DEFAULT_MODE=vanilla
export USE_MONGO=0
```

先以 `RAG_DEFAULT_MODE=vanilla` 啟動、跑 `/readyz` 和完整 A/B。確認 promotion 後才改：

```bash
export RAG_DEFAULT_MODE=rag
```

rollback：

```bash
export CKPT=/mnt/P300/data/ULIP/checkpoint_last.pt
export VEC_DIR=/mnt/P300/data/ikea_data/vectors
export RAG_STAGE1_CKPT=/home/kyzen/ULIP_RAG/core/outputs/RAG2_Stage1/checkpoint_best.pt
export RAG_CORPUS_PROFILE=legacy1095
export RAG_DEFAULT_MODE=vanilla
```

現行 serving loader 已支援 IKEA RAG Stage 2 checkpoint 的 base-only strict load：它能讀取 dict／Namespace 形式的 checkpoint args，建立 vanilla `ULIP_PointBERT` serving encoder、去除 `rag_enhancer.*` 後對其餘 state 做 `strict=True`，再由 RAG adapter 載入指定的 Stage 1 enhancer。`provenance` profile 會沿 Stage 2 checkpoint → Stage 1 checkpoint → corpus/index → vector schema 驗證 lineage；不得以 `strict=False` 掩蓋 checkpoint/vector 不相容。

## 13. 已完成驗證與證據等級

### 13.1 Unit tests

執行：

```bash
PYTHONPATH=/home/kyzen/ULIP_RAG/core:/home/kyzen/ULIP_RAG/ikea \
/home/kyzen/miniconda3/envs/ulip/bin/python \
  -m unittest tests.test_rag_serving ikea.tests.test_rag_training_utils -v
```

結果：`Ran 28 tests ... OK`

覆蓋：

- shared base 完全相同可通過
- 同 shape 但 tensor value 不同會拒絕
- named corpus profile hash drift 會拒絕
- legacy vector bytes／metadata hash drift 會拒絕
- provenance profile 會驗證 Stage 2 指向正確 Stage 1
- Stage 1／Stage 2 corpus 或 dataset lineage drift 會拒絕
- candidate vector schema 必須綁定產生它的 Stage 2 checkpoint SHA-256
- long query EOT preserved
- vanilla/RAG encode 與 diagnostics
- PC vector search 與 status
- raw IKEA caption 送進 retriever
- Stage 1 enhancer tensor 完整性
- corpus/index/dataset provenance drift 拒絕
- Stage 2 frozen target modules 維持 deterministic eval mode

其中 serving adapter unit tests 使用 fake 小模型／fake retriever，是邏輯與錯誤處理驗證，不是檢索品質證據。

### 13.2 真實 CPU/GPU adapter 與 FastAPI smoke

已完成真實 checkpoint/corpus/index 的 CPU/GPU load smoke：

```text
shared tensors exact: 301/301
corpus rows: 1095
FAISS rows: 1095
FAISS dimension: 384
adapter ready: true
```

並曾真實啟動 FastAPI `:8321`、取得 `/readyz` 200、成功呼叫三個 text search routes。這證明完整 serving chain 可執行；不是品質 promotion 證據。

另外已用本節下方的真實 IKEA Stage 1／Stage 2 smoke checkpoints，加上一個只供載入驗證的 1-row、512d vector fixture，實際執行 `RAG_CORPUS_PROFILE=provenance`：

```text
adapter ready: true
profile: provenance
Stage 2 lineage validated: true
base training strategy: stage_2
shared tensors exact: 301/301
retrieved documents: 5
```

這證明新 checkpoint → Stage 1 → corpus/index → vector schema 的 promotion loader 可以完整運作。1-row fixture 不是候選商品 index，也不是品質證據；正式上線仍必須以 Stage 2 重建完整 733-row vectors。

### 13.3 IKEA Stage 1／Stage 2 真實 GPU smoke

兩階段都已使用真實 IKEA checkpoint、真實 legacy1095、真實 GPU 執行：

```text
Stage 1: 1 epoch、只跑 1 training batch、8 validation items
Stage 2: 1 epoch、只跑 1 training batch、8 validation items
```

暫存輸出：

```text
/tmp/ulip_ikea_rag_stage1_smoke
/tmp/ulip_ikea_rag_stage2_smoke
```

Stage 1 與 Stage 2 均成功完成 forward、backward、optimizer step、evaluation、checkpoint/provenance 寫入與 Stage 1 → Stage 2 strict load。這只是 GPU execution/functionality proof。因為只有一個 batch 和 8 個 validation items，任何 R@1、loss 或 100% accuracy 都不可當成模型品質指標。

### 13.4 尚未完成／尚未確認

- IKEA RAG Stage 1／Stage 2 尚未完成正式長訓練。
- 尚未建立不可變的 100-query fixture/result bundle；目前結果是臨時 E2E evidence。
- 尚未完成獨立 held-out V2T room-request relevance benchmark。
- 新 IKEA RAG checkpoint 與 candidate vectors 的正式 SHA-256 尚未產生。
- 新 immutable serving profile 尚未註冊；因此不可直接把 default 切成 RAG。
- RAG top-k、學習率、epochs、loss 權重尚未做系統性 ablation。
- production latency、併發、timeout、GPU memory 與長時間穩定性尚未確認。
- 目前 text retrieval 沒有尺寸 Rule Engine；Mongo dimensions 不是這次 RAG 排序的一部分。
- 舊 Core RAG 在真實 5090 V2T query 上是否有任何局部增益尚未確認；目前 self-retrieval A/B 顯示整體大幅退化。

## 14. MongoDB 狀態

本輪串接、unit tests、CPU/GPU adapter smoke、FastAPI text search 驗證、100-query A/B 與 Stage 1／Stage 2 smoke 均沒有執行 MongoDB update、補值、index rebuild 或其他 Mongo writes。

Serving 驗證使用：

```text
USE_MONGO=0
```

RAG corpus、FAISS、checkpoint、IKEA JSON／PLY／vectors 都是檔案型 artifact；RAG query path 不依賴 MongoDB。
