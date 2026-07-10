# ULIP_RAG（3090 這台）完整分析報告

> 對象：`/home/kyzen/ULIP_RAG/`（3090, git branch = `3090`）
> 產出：kyzen + Claude，2026-07-09
> 資料來源：實際程式碼閱讀 + 實機驗證（conda/torch/faiss/checkpoint/vectors 皆實跑確認）
> 用途：3090 現況的完整交接文件，並與 4090 的舊報告（已改名為 `PROJECT_REPORT_OLD_4090.md`）逐點比對。
> 標「推測」/「尚未確認」處代表未實跑驗證。

---

## 1. 專案用途總結

這台的 ULIP_RAG **已從純研究原型演化成一個「IKEA 家具 3D 檢索產品」**，內含三條並存路線：

- **研究線（`core/`）**：Salesforce ULIP + RAG 增強的三模態（3D 點雲 / 文字 / 影像）對齊訓練。就是 4090 報告描述的那套（`main.py` + `ULIP_models.py` + RAGEnhancer + 兩階段訓練）。
- **產品線（`ikea/`）**：完整的 IKEA 產品 pipeline——爬蟲 → CLIP 篩圖 → **TRELLIS 圖轉 3D** → 用 733 件家具訓練 **純 ULIP_PointBERT（無 RAG）** → FastAPI 檢索服務（文字搜 3D + 上傳照片生 3D）。
- **語料線（`corpus/vocab_build/`）**：新一代 RAG 語料生成（WordNet 種子 → **Llama-3.2-3B** 造句 → MiniLM/FAISS 去重）。

**最重要事實**：目前實際上線用的模型是 `/mnt/P300/data/ULIP/checkpoint_last.pt`，它是 **`ULIP_PointBERT`（純 ULIP，不含 RAGEnhancer）**，在 IKEA 733 件家具上訓練，**S2T Recall@1 = 76.5%**。RAG 那套是研究分支，尚未進入 IKEA 產品。

---

## 2. 目錄結構說明

```
ULIP_RAG/                        (git branch = 3090)
├── core/                        ← 原 ULIP_RAG 研究核心（= 4090 報告整個專案）
│   ├── main.py (585)            RAG 訓練/zero-shot 入口
│   ├── main_ikea.py → ../ikea/main_ikea.py  (symlink)
│   ├── test.py (489)            MRR/NDCG 跨模態檢索評估
│   ├── test_rag_functions.py (216)
│   ├── check_environment.py     環境自檢（含過時路徑）
│   ├── quick_train.sh           (過時路徑)
│   ├── data/
│   │   ├── dataset_3d.py (987)  ShapeNet/ModelNet DataLoader
│   │   ├── dataset_3d_rag.py    舊版未用
│   │   ├── ikea_ulip.py → ../ikea/ikea_ulip.py
│   │   ├── configs/             yaml + labels/templates/catalog
│   │   ├── initialize_models → /mnt/P300/data/ULIP/ULIP-1/initialize_models
│   │   ├── modelnet40 / shapenet-55 / ULIP-1 → /mnt/P300/data/ULIP/...
│   ├── models/
│   │   ├── ULIP_models.py (877) 核心（ULIP + RAG）
│   │   ├── losses.py (65)       ULIPWithImageLoss
│   │   ├── pointbert/           PointTransformer backbone (+ dvae/misc/checkpoint/logger)
│   │   ├── pointnet2/ pointnext/ pointmlp/ customized_backbone/
│   │   └── rag_*.py / retriever.py / ULIP_models_rag.py  ← 歷史遺跡
│   ├── rag_corpus/              8256 條語料 + FAISS(384d)
│   ├── scripts/                 build_rag_corpus/index + pretrain_*.sh + download_objaverse
│   ├── Pointnet2_PyTorch/       需編譯（實測走 JIT 亦可）
│   ├── utils/                   tokenizer / utils / io / build / registry / config / logger
│   └── outputs/                 ← 空（3090 本地無 checkpoint）
├── ikea/                        ← 🆕 IKEA 產品 pipeline
│   ├── get_data.py (217)        IKEA 爬蟲(ikea_api + Playwright) → MongoDB
│   ├── build_3d.py (141)        取樣 + TRELLIS 生 3D + JSON
│   ├── ikea_ulip.py (415)       IkeaULIP Dataset（讀 ply/render/caption）
│   ├── main_ikea.py (776)       IKEA ULIP 訓練 + 六向檢索評估
│   ├── app_ikea_retrieval.py (740)  FastAPI 檢索服務
│   ├── ikea_ulip.yaml           IKEA 資料設定
│   ├── fetch_dimensions.py / phase_a_parse_dimensions.py  尺寸解析
│   ├── fetch_v3_products.py / download_v3_images.py / backfill_v3_desc_images.py  V3 重爬
│   ├── build_bad_model_manifest.py / cleanup_3d_models.py / regen_3d_models.py  品質清理
│   └── docs/                    phase plan + 品質稽核報告
├── corpus/vocab_build/          🆕 Llama 造句語料生成（4 步驟 + llm.py + build_open_knowledge_base）
├── docs/                        環境/研究/訓練/ikea-pipeline 指南（本檔在此）
├── scripts/                     data_download.py + quick_train_portable.sh
├── storage → /mnt/P300/data     實體資料
├── requirements.txt
└── PROJECT_REPORT_OLD_4090.md   ← 4090 的舊報告（已改名，避免混淆；untracked）
```

**與 4090 報告最大差異**：4090 是扁平結構、資料在 `/mnt/data1/`；3090 已把研究核心收進 `core/`、資料改到 `/mnt/P300/data`、並長出整個 `ikea/` 產品線與 `corpus/` 語料線。

---

## 3. 環境與依賴

| 項目 | 文件記載 | **實際安裝（實測）** |
|---|---|---|
| conda env | `ulip_rag` | **`ulip`** |
| Python | 3.9.25 | 3.9.23 |
| PyTorch | 2.1.0 + cu121 | **2.7.1 + cu128** |
| faiss | faiss-gpu 1.7.2 | 1.7.2（可 import） |
| pointnet2_ops | 需編譯 | 可用（走 JIT 編譯，非預編譯） |

- **依賴來源**：`requirements.txt`（釘死版本，但已落後實裝）＋ `docs/ENVIRONMENT_SETUP.md`（逐步 pip）＋ 手動編譯 `Pointnet2_PyTorch/pointnet2_ops_lib`。管理方式：**conda + pip**（無 Docker）。
- **產品線額外依賴**（`app_ikea_retrieval.py`）：`fastapi`、`rembg`（去背）、`trimesh`、`transformers` CLIP、**`trellis`（TRELLIS 圖轉 3D，另有獨立 conda env `TRELLIS`）**、`pymongo`。
- **語料線額外依賴**：`transformers` + `meta-llama/Llama-3.2-3B-Instruct`、`nltk`(WordNet)、`sentence-transformers`。
- ⚠️ **多份文件含過時資訊**：`check_environment.py`、`quick_train.sh`、`quick_train_portable.sh`、`docs/TRAINING_COMMANDS.txt` 都還寫 `conda activate ulip_rag`、`/mnt/data1/cheng/...`、`/home/kyzen/cheng/ulip_rag`——這些路徑在本機都不存在，重現時要自行改成 `ulip` env、`core/`、`/mnt/P300/data`。

---

## 4. 資料位置與資料流程

**實體資料全在 `/mnt/P300/data`（= `storage` symlink）：**

| 資料 | 位置 | 量 | 用途 |
|---|---|---|---|
| ShapeNet 點雲 | `ULIP/ULIP_Shapenet_Triplets/shapenet_pc` | 52,470 npy | 研究線 train |
| ShapeNet 渲染圖/caption | 同上 `rgb_depth_rendered_images` + `ULIP-shapenet_triplets_captions.json` | — | 研究線 |
| ShapeNet split | `train.txt`=52470 / **`test.txt`=11** | — | ⚠️ test 只有 11 筆（噪音大） |
| ModelNet40 | `ULIP/ULIP-1/modelnet40_normal_resampled` | .dat | zero-shot 評估 |
| 初始權重 | `ULIP/ULIP-1/initialize_models/` | slip 2.0GB + point_bert 527MB | backbone 初始化 |
| 官方 zero-shot ckpt | `ULIP/ULIP-1/pretrained_models/` | pointbert/pointnext… | 基準 |
| **IKEA 產品** | `ikea_data/{json,ply,glb,images,rendered_images}` | **733 件** | 產品線 train |
| **IKEA 檢索向量** | `ikea_data/vectors/{vectors_pc,vectors_img,vectors_txt}.npy + meta_{pc,img,txt}.jsonl + schema.json` | **三模態各 512 維：pc(733)、img(733)、txt(732，少 1 筆)** | 產品線 serving 索引（app 目前只載 pc） |
| custom 上傳 | `custom_data/{ply,glb,images,json,tmp,uploads}` | — | 使用者上傳生成 |
| V3 重爬 | `ikea_data_V3/images` | 進行中 | Phase A/V3 |
| RAG corpus | `core/rag_corpus/` | 8256 條 + FAISS | 研究線 RAG |
| 學長參考系統 | `CAMERA_3D/`（88k 語料 + enc1.pth） | — | 參考用 |

**資料流程**：
- **研究線點雲**：`.npy → IO.get → FPS(8192) → pc_norm → aug(dropout/scale/shift/jitter/rotate) → float tensor`；caption 優先取 `ULIP-shapenet_triplets_captions.json`（每物件多句隨機挑），找不到退回 taxonomy 類名；圖片隨機挑角度(0–348°每12°)+RGB/depth。
- **IKEA 點雲**（`ikea_ulip.py`）：讀 `.ply`（trimesh 只取 vertices）→ 同款 normalize → FPS/pad 到 8192 → train 增強；caption 用 `llava_caption_en`；render 圖隨機挑一張。JSON 內路徑是 `./ulip_output/...` 相對路徑，靠 `_PathResolver` 多策略映射（因 `ikea_data/` 內含 `ply/` 而命中）。

**路徑分派方式**：
- **CLI arg**：`--pretrain_dataset_name/--npoints/--rag_corpus_dir/--rag_top_k/--stage1_ckpt_path` 等
- **catalog + yaml**：`dataset_catalog.json → configs/*.yaml`（yaml 路徑已更新到 `/mnt/P300/data`）
- **寫死路徑**（重現時要注意）：`dataset_3d.py` 內 `ULIP-shapenet_triplets_captions.json`、`ULIP_models.py` 的 `slip_base_100ep.pt`、`point_encoder.py` 的 `point_bert_pretrained.pt`——全部硬編到 `/mnt/P300/data/ULIP/...`；`app_ikea_retrieval.py` 的 `VEC_DIR`/`CKPT` 預設仍是舊 4090 路徑（要靠環境變數覆寫）。

---

## 5. 訓練流程

有**兩個訓練入口**：

### (A) 研究線 `core/main.py`（RAG 兩階段）
```
init_distributed → wandb(project=ULIP-RAG) → seed
→ use_rag_adapter? → ULIP_PointBERT_RAG(args) + ULIP_Loss_RAG_Enhanced
→ AdamW(lr, betas=(0.9,0.98), wd=0.1；bias/ln/bn 不加 wd) + GradScaler(AMP)
→ Dataset_3D + rag_collate_fn → DataLoader
→ cosine_scheduler(warmup)
→ for epoch: train() → 每 epoch test_zeroshot_3d_core() → 存 best 或每 50 epoch
```
- **Stage 1**：凍結全部，只開 `rag_enhancer`；Loss = `MSE(enhanced, orig.detach()) + 0.5·InfoNCE(enhanced↔image)`
- **Stage 2**：只開 `point_encoder + pc_projection`，從 `--stage1_ckpt_path` 撈回 `rag_enhancer` 權重；Loss = `InfoNCE(pc↔enhanced) + 0.5·InfoNCE(pc↔image)`
- ✅ **4090 報告擔心的「main.py:322 unpacking bug」在這台已修正**：L326 正確解包 4 元素 `pc, text_data, image, labels = batch_data`。

### (B) 產品線 `ikea/main_ikea.py`（純 ULIP，IKEA）
```
wandb(project=ULIP2, entity=hj6hki123-cpu)
→ getattr(models, args.model)(args)  # 預設 ULIP_PN_SSG，實際用 ULIP_PointBERT，無 RAG
→ get_loss = ULIPWithImageLoss（標準三模態 InfoNCE，pc↔text + pc↔image）
→ IkeaULIP dataset + customized_collate_fn
→ for epoch: train() → 驗證用 eval_object_level_retrieval_6way()
             → 存 best（依 S2T Recall@1）或每 50 epoch + 最後一輪
```
- **六向檢索評估**（產品線核心指標）：S2T/T2S/S2I/I2S/T2I/I2T，各算 MRR / R@1 / R@5 / R@10 / NDCG@5，以 object_id=`category-model_id` 定義正樣本。`best_acc1 = s2t_r1×100`。
- 實際訓練成果：**best_acc1 = 76.53（S2T R@1 = 76.5%）**。

**Checkpoint/Log**：`--output-dir`（本地 `core/outputs` 目前空）；`checkpoint_best.pt` + 每 50 epoch + `log.txt`（每 epoch 一行 JSON）+ wandb。訓練好的 IKEA 模型落在 `/mnt/P300/data/ULIP/checkpoint_last.pt`。

---

## 6. 模型架構

```
ULIP_WITH_IMAGE (基類, models/ULIP_models.py:78)
 ├─ visual = timm 'vit_base_patch16_224'        (影像 encoder, 768→512)
 ├─ transformer = CLIP text (width512/layers12/heads8) + token/pos embed + ln_final
 ├─ text_projection(512→512) / image_projection(768→512)
 ├─ point_encoder = PointTransformer(768d out)  (pointbert)
 ├─ pc_projection(768→512)
 └─ logit_scale = log(1/0.07), clamp[0, 4.6052]

ULIP_with_RAG_Enhancer (繼承, :672)   ← 研究線才用
 + retriever = RAGRetriever            (MiniLM-L6-v2 384d + FAISS corpus_index.faiss)
 + rag_enhancer = RAGEnhancer          (MHA 8heads + LN + FFN 512→1024→512)
 + tokenizer(SimpleTokenizer) + rag_corpus(8256 全載入記憶體)
```

**PointBERT**（`PointTransformer_8192point.yaml`，已對 yaml 核實）：`npoints=8192` → Group(`num_group=512` centers × `group_size=32` 鄰點) → Encoder(`encoder_dims=256`) → 線性升到 `trans_dim=384` → Transformer(`depth=12`、`num_heads=6`、drop_path=0.1) + cls_token + 3D 座標 pos MLP → `concat(cls, maxpool)` = **768d**；初始化自 `point_bert_pretrained.pt`。

**工廠函數**：`ULIP_PN_SSG(256d)` / `ULIP_PN_MLP` / `ULIP_PointBERT(768d)` / `ULIP_PN_NEXT` / `ULIP2_PointBERT_Colored(bigG-14)` / `ULIP_CUSTOMIZED` / `ULIP_PointBERT_RAG(768d + RAG)`。

**RAGEnhancer.forward** 細節：`context = cat([original.unsqueeze(1), doc_features])`；`MHA(query=original, key/value=context)` → 殘差+LN → FFN+LN。⚠️ 4090 報告提的 **zero-padding 無 `key_padding_mask`** 問題仍在（檢索不足 K 時補零 row 會參與 softmax）。

---

## 7. RAG / retrieval 流程

**重點：這台有「兩種 retrieval」，用途完全不同——**

### (A) 研究線 RAG（訓練時增強文字分支，`encode_text_with_rag` :698）
```
tokens + raw_text
 ├ encode_text_base(tokens)                → base_feat[B,512]  (CLIP text 投影前)
 ├ RAGRetriever(MiniLM+FAISS).retrieve     → top-K 家具語料原文
 ├ 同一個 CLIP encoder 重編碼 docs         → doc_feat[B,K,512]
 ├ RAGEnhancer(base, docs)                 → fused[B,512]
 └ fused @ text_projection                 → enhanced_text_embed
```
- 檢索器用 MiniLM(384d)，融合用 CLIP text(512d)；CLIP encoder 每次 forward 跑兩次。
- 語料：`core/rag_corpus/`（8256 條，29 個家具類、97% 合成 + Wikipedia）。
- **FAISS 索引已實機驗證**：`faiss.read_index` 讀出 `ntotal=8256, d=384, IndexFlat` → **實際是 384 維 MiniLM**。檔名雖叫 `clip_corpus_index.faiss` 但**與 CLIP 無關**（`build_rag_corpus.py` 用 MiniLM 建的）。另一支 `scripts/build_rag_index.py` 才是用 `open_clip ViT-B-32`（512 維）建索引——但那個 512 維版本**沒接上 runtime**（RAGRetriever 用 384 維 MiniLM 查詢，維度也對不上），屬未使用的實驗腳本。

### (B) 產品線 retrieval（serving，`app_ikea_retrieval.py`）— **不用 RAG**
```
使用者文字 query
 → ULIP encode_text (純 CLIP text encoder，來自 checkpoint_last.pt)
 → F.normalize → 與預存 PC 向量 vectors_pc.npy(733,512) 做 cosine
 → argpartition Top-K → 回傳家具 id/category/caption/圖
```
本質是「這段文字最像哪個 3D 家具」，靠 ULIP 訓練把 text_embed 與 pc_embed 對齊到同一空間。**完全沒用到 FAISS/RAGEnhancer/8256 corpus**。

**上傳生成流程**（`/post/upload`）：圖 → rembg 去背 → CLIP 20 類分類 → **TRELLIS 圖轉 3D** → GLB + FPS8192 純 xyz PLY → 組 ULIP JSON（含使用者輸入 `dimensions_mm`）→ `os.replace` 原子 commit 到 `custom_data/`。

### ULIP 與 RAG 如何連接
研究線裡 RAG 只增強 text 分支，pc/image 分支不動，兩階段先訓 enhancer 再把 pc 拉到新文字空間。**但產品線目前跳過 RAG**，直接用純 ULIP 對齊 + cosine 檢索。這是本專案現階段的實際架構選擇。

---

## 8. 主要程式檔案逐一說明

**研究核心 `core/`**
- `main.py` — RAG 訓練/評估入口。`get_args_parser`(L39)、`main`(L107)、`train`(L280)、`test_zeroshot_3d_core`(L383)。呼叫 `ULIP_models` + `dataset_3d`。
- `main_ikea.py`（symlink → ikea/）— 見產品線。
- `test.py` — 跨模態檢索 MRR/NDCG 評估（研究線）；`calculate_ndcg_at_k`、`calculate_mrr`、`evaluate_cross_modal_retrieval`、`main`。
- `test_rag_functions.py` — RAG 元件單元測試（檢索/enhancer 正確性）。
- `check_environment.py` — 環境自檢（路徑過時）。
- `models/ULIP_models.py` — 核心。`ULIP_WITH_IMAGE`(78)、`ULIP2_WITH_OPENCLIP`(184)、`RAGRetriever`(473)、`RAGEnhancer`(517)、`ULIP_Loss_RAG_Enhanced`(539)、`ULIP_with_RAG_Enhancer`(672)、工廠 `ULIP_PN_SSG/…/ULIP_PointBERT_RAG`(759)。
- `models/losses.py` — `ULIPWithImageLoss`（標準三模態 InfoNCE + DDP all_gather）；`ClipLoss` 別名。
- `models/pointbert/point_encoder.py` — `PointTransformer`(113, 768d)、`PointTransformer_Colored`(239)；配套 `dvae.py`/`misc.py`/`checkpoint.py`/`logger.py`。
- `models/pointnet2|pointnext|pointmlp|customized_backbone/` — 其他 backbone。
- `data/dataset_3d.py`（987 行）— `ModelNet`(161)、`ShapeNet`(498，舊版 313–494 已註解)、`Objaverse_Lvis_Colored`(704)、`Dataset_3D`(948)、`customized_collate_fn`(800)、`rag_collate_fn`(856)。
- `utils/` — `tokenizer.py`(CLIP BPE)、`utils.py`(DDP/cosine/get_dataset)、`io.py`、`build.py`+`registry.py`(DATASETS 註冊)、`config.py`(與 utils 重複，舊遺留)、`logger.py`。
- `scripts/` — `build_rag_corpus.py`、`build_rag_index.py`、`download_objaverse.py`、`rag_corpus_expanded.py`（tricolo，無關）、`pretrain_*.sh`/`test_*.sh`。
- **歷史遺跡（主流程未 import，已逐一確認）**：
  - `rag_adapter.py`、`rag_generator.py`：**空檔 / 整支註解**（T5 生成式路線已放棄）。
  - `rag_enhancer.py`：獨立版 `RAGRetriever + RAGEnhancer`（與 `ULIP_models.py` 內同名類別重複）；**只有 `check_environment.py` 還 import 它**，實際模型用的是 `ULIP_models.py` 內定義的版本。
  - `retriever.py`：獨立版 `DenseRetriever`（faiss），未使用。
  - `ULIP_models_rag.py`：早期完整 ULIP 複本（含 Transformer/ULIP_WITH_IMAGE），已被 `ULIP_models.py` 取代。
  - `dataset_3d_rag.py`：舊版 dataset，未使用。

**IKEA 產品線 `ikea/`**
- `get_data.py` — IKEA 爬蟲（ikea_api + Playwright 抓 JS 渲染尺寸）→ MongoDB `furniture_db.ikea_product`。
- `build_3d.py` — 每類 clip_confidence Top-50 → TRELLIS 生 GLB + gaussian PLY + JSON；MongoDB `converted_3d` 狀態標記支援中斷續跑。
- `ikea_ulip.py` — `IkeaULIP` Dataset + `_PathResolver`；回傳 `(taxonomy_id, model_id, tokens, pc, img)` 5-tuple。
- `main_ikea.py` — IKEA ULIP 訓練 + `eval_object_level_retrieval_6way`(六向檢索)。
- `app_ikea_retrieval.py` — FastAPI：`/search/text`（文字→3D 檢索）、`/file`、`/get/file`、`/get/ids`、`/post/upload`（上傳生 3D）。
- **Phase A / V3（目前未提交的活躍工作）**：
  - `phase_a_parse_dimensions.py` — 解析 `size_options` → `meta.dimensions_mm`，寫入備份表 `ikea_product_v2_2026q2`（不動原表）。
  - `fetch_dimensions.py` — 舊版逐 JSON 抓尺寸。
  - `fetch_v3_products.py` / `download_v3_images.py` / `backfill_v3_desc_images.py` — V3 重爬（尺寸在 ingest 時解析，寫 `ikea_product_v3_fresh_2026q2`）。
  - `build_bad_model_manifest.py` / `cleanup_3d_models.py` / `regen_3d_models.py` — 733 個 3D 模型品質稽核 → manifest → 清理/重生。

**語料線 `corpus/vocab_build/`**（新一代 RAG 語料，推測尚未接回 runtime）
- `1_build_semantic_graph.py`(WordNet 種子) → `2_generate_sentences_llama.py`(Llama-3.2-3B 造句) → `3_build_kb_faiss.py`(MiniLM 去重+FAISS) → `4_report_coverage.py`(覆蓋率/熵)。`build_open_knowledge_base.py` 是逐句搜 Wikipedia 的舊版；`llm.py` 只是 pirate 測試 demo。

**頂層 `scripts/`**：`data_download.py`（HuggingFace 下載）、`quick_train_portable.sh`（通用版訓練啟動，路徑過時）。

---

## 9. 如何從零開始跑起來

**環境**
```bash
conda activate ulip           # 注意是 ulip 不是 ulip_rag
cd /home/kyzen/ULIP_RAG/core
```

**(A) 測試 IKEA 檢索服務（最貼近產品現況）**
```bash
# 資料與模型都已就緒，只要覆寫預設的舊路徑：
export CKPT=/mnt/P300/data/ULIP/checkpoint_last.pt        # 已訓練 ULIP_PointBERT
export VEC_DIR=/mnt/P300/data/ikea_data/vectors           # vectors_pc.npy(733,512)
export ULIP_OUTPUT=/mnt/P300/data/ikea_data
export CUSTOM_OUTPUT=/mnt/P300/data/custom_data
# app 需要 TRELLIS/rembg/fastapi 環境（純文字檢索只要 CKPT+VEC_DIR；上傳生成才需 TRELLIS）
cd /home/kyzen/ULIP_RAG/ikea && uvicorn app_ikea_retrieval:app --host 0.0.0.0 --port 8000
# 之後 POST /search/text {"query":"modern black leather sofa","top_k":10}
```

**(B) 重跑 IKEA 訓練**
```bash
cd /home/kyzen/ULIP_RAG/core
python main_ikea.py --model ULIP_PointBERT \
  --pretrain_dataset_name ikea_ulip --validate_dataset_name ikea_ulip \
  --npoints 8192 --batch-size 32 --epochs 250 --output-dir ./outputs/ikea_run --wandb
```

**(C) 研究線 RAG 兩階段**：改用 `main.py`，指令見 `docs/TRAINING_COMMANDS.txt`，但要把裡面路徑/env 改成本機。

**(D) 重建 IKEA 資料**：`get_data.py`(爬) → `build_3d.py`(TRELLIS) → 訓練 → 產生 `vectors_pc.npy`（⚠️ 見第 10 節，向量產生腳本未在 repo 找到）。

---

## 10. 還需要人工確認的地方

### ✅ 已修復（2026-07-10，均已實測）

| 原問題 | 修復內容 | 驗證 |
|---|---|---|
| 建向量腳本不在 repo | 從 4090 取回 `ikea/build_vectors.py`；**修正其 pc 分支**（原版把 FPS 改成截斷 `pts[:8192]`，會產生與部署向量幾乎正交的錯誤向量 cos≈0.03）→ 改回 FPS 並用 pointnet2_ops **GPU FPS** 加速（~2.3s/筆），加 `--pc_sampler` 參數 | 3 筆 cos(new, deployed)=0.9977/0.9938/0.9917；img/txt 全量重建 cos=1.0000 |
| `dataset_catalog.json` 缺 `ikea_ulip` → `main_ikea.py` KeyError | 補上 `ikea_ulip` 條目（config 指 `./data/configs/ikea_ulip.yaml`） | `get_dataset('ikea_ulip','train')` OK len=729；main_ikea 1-epoch 端到端實跑 |
| serving app 無法啟動（舊路徑 + trellis import） | 新增 `ikea/run_app_3090.sh`（cwd=core + PYTHONPATH=core:ikea:/home/kyzen/TRELLIS + 4 個路徑 env + USE_MONGO=0）。**實測單一 `ulip` env 即可完整跑**（先前「無單一 env」的判斷是錯的） | uvicorn 16s ready；`/search/text`/`/get/ids`/`/get/file` curl 全通 |
| `ulip` env 的 `bin/uvicorn` 是 0-byte 壞檔 | 啟動腳本改用 `python -m uvicorn` 繞過 | server 正常啟動 |
| `vectors_txt` 為何 732 筆 | 已解：1 件 caption 為空字串，`build_vectors.py` 設計上跳過（非 bug） | 4090 文件 + 3090 掃描確認 `TXT usable: 732 / no_txt: 1` |

### ✅ 端到端實測結果（2026-07-10，全綠）

| 流程 | 實測 |
|---|---|
| IKEA 訓練（`main_ikea.py`，smoke 設定 1 epoch）| 45/45 iters，loss 8.21→5.38，10GB VRAM，六向檢索評估執行，checkpoint_1/best/last.pt + log.txt 產出，exit 0 |
| IKEA serving（`ikea/run_app_3090.sh`）| uvicorn 16s ready；`/search/text`（Sofa 命中正確）、`/get/ids`（735）、`/get/file`（JPEG 200）全通 |
| RAG 研究線（ULIP_PointBERT_RAG）| forward 5 keys；Stage1 loss+backward OK；Stage2 loss OK；staged_1 凍結正確（只開 rag_enhancer）|
| 向量重建（`ikea/build_vectors.py` FPS 修正版）| pc cos vs 部署 = 0.9917–0.9977；img/txt cos = 1.0000 |

### ⚠️ 新發現：IKEA 訓練的 FPS 效能瓶頸（2026-07-10 實測）

- IKEA 的 ply 是 gaussian splat，**每顆 33 萬～67 萬點**；正式 `ikea_ulip.yaml` 設 `PC_SAMPLER: fps`，而 `ikea_ulip.py` 的 FPS 是 **numpy 純迴圈**，在 DataLoader worker（CPU）內跑 → **每筆 20-40 秒**。
- 後果：batch16 一個 batch 要 5-10 分鐘，1 epoch（729 筆）小時級——實測 4 workers 跑了 26 分鐘連第一個 batch 都沒吐出來（看起來像卡死，其實是在磨 FPS）。這也解釋當初訓練為何要 36-48 小時。
- 因此另建 `ikea_ulip_smoke`（catalog + `ikea_ulip_smoke.yaml`，`PC_SAMPLER: random`）供流程驗證；**正式重訓前建議**：離線先把 733 顆 ply 用 GPU FPS 預採樣成 8192 點存檔（`build_vectors.py` 的 `sample_points` 已有 GPU FPS 可複用，全量約 28 分鐘一次性成本），訓練時直接讀預採樣檔。

**高優先**
1. **⚠️ 向量產生腳本不在 repo（最需要補的流程）**：`ikea_data/vectors/` 內其實有**三模態**向量——`vectors_pc(733,512)`、`vectors_img(733,512)`、`vectors_txt(732,512)` + 對應 `meta_*.jsonl` + `schema.json`（建立於 2026-04-07）。但**全 repo 搜尋 + P300 依名搜尋都找不到「批次 encode pc/img/txt → 存 .npy/.jsonl」的腳本**（repo 內只有 `app_ikea_retrieval.py` 在**讀取**它）。→ 這是目前最需要補回的環節：**新增 IKEA 家具後要用什麼腳本重建/增量更新這三套向量，尚未確認**。（`txt` 少 1 筆也待查是哪個 item 失敗。）推測是用訓練好的 `checkpoint_last.pt` 對 733 件跑 `encode_pc/encode_image/encode_text` 產出，但腳本本體遺失或未入庫。
2. **產品線確定不走 RAG？** `checkpoint_last.pt` 是純 `ULIP_PointBERT`，`app` 也只用 `encode_text`。研究線的 8256 corpus/FAISS/RAGEnhancer 在產品端目前**沒被使用**。需確認是最終決策還是待整合。
3. **`app_ikea_retrieval.py` 的 `CKPT`/`VEC_DIR` 預設是舊 4090 路徑，且 repo 內查無部署設定來源**。實機搜尋確認：repo 裡**沒有** `.env`、Dockerfile、systemd `.service`、或任何設定 `VEC_DIR/CKPT` 及拉起 `uvicorn app_ikea_retrieval:app` 的 shell 腳本。→ 目前只能靠手動 `export` 環境變數啟動（見第 9 節 A），或有 repo 外的部署設定（需向持有者確認）。正式部署前建議補一份 `.env` 或啟動腳本入庫。

**中優先**
4. RAGEnhancer 的 zero-padding 仍無 `key_padding_mask`（可能影響融合品質）。
5. ShapeNet `test.txt` 只有 11 筆 → 研究線用 shapenet 當驗證時指標噪音大。
6. `ikea_data` 的 PLY 是 Gaussian Splat、`custom_data` 的 PLY 是純 xyz FPS8192——兩者格式不一致（dataset 只取 vertices xyz 所以訓練不受影響，但值得留意）。
7. `main_ikea.py` L409/L415 讀 `./data/templates.json`、`./data/labels.json`（少了 `configs/`），只有非 IKEA zero-shot 路徑才會踩到——IKEA 走六向檢索沒觸發。**推測**是遺留不一致。
8. `corpus/vocab_build`（Llama 造句）產出的語料**是否已接回 `core/rag_corpus`**？看起來是獨立實驗，未確認。

**低優先**
9. `check_environment.py`、`quick_train*.sh`、`TRAINING_COMMANDS.txt`、`ENVIRONMENT_SETUP.md` 全含過時路徑/env 名，建議統一更新。
10. `utils/config.py` 與 `utils/utils.py` 重複的 `cfg_from_yaml_file`（舊遺留）。

---

## 11. 建議接下來優先讀的檔案

若重心是**產品線（最可能）**：
1. `ikea/app_ikea_retrieval.py` — 服務入口與 serving 檢索邏輯。
2. `docs/ikea_pipeline_code_detail.md` — 5 步驟講最清楚，搭配看。
3. `ikea/main_ikea.py` 的 `eval_object_level_retrieval_6way`（L600+）— 評估指標定義。
4. `ikea/ikea_ulip.py` — 要改 caption / 加尺寸語意（Phase C）時的入口。
5. `ikea/docs/ulip_rag_phase_plan.md` + `phase_a_parse_dimensions.py` — 目前未提交的 Phase A（尺寸 metadata），接著才是 Phase B Rule Engine（在 `app_ikea_retrieval.py` 加 `filter_by_dimensions`）。

若要**接續研究線 RAG**：
6. `core/models/ULIP_models.py` L473–865（RAG 全段）+ `core/main.py` `train`/loss。

---

## 附：4090 報告 → 3090 現況 對照速查

| 項目 | 4090 報告 | 3090 現況 |
|---|---|---|
| 專案性質 | 純 RAG 研究 | 研究線 + **IKEA 產品線 + 語料線** |
| 目錄 | 扁平 | `core/` + `ikea/` + `corpus/` |
| conda env | ulip_rag | **ulip** |
| torch/cuda | 2.0.1/cu118 | **2.7.1/cu128（實裝）** |
| 資料根 | /mnt/data1 | **/mnt/P300/data** |
| RAG corpus 條數 | 疑慮只 1095 | **8256（已確認，非 1095 也非 52K）** |
| main.py:322 unpack bug | 疑似有 | **已修正** |
| 上線模型 | 無 | **checkpoint_last.pt = 純 ULIP_PointBERT，S2T R@1 76.5%** |
| 產品是否用 RAG | — | **否（serving 用純 ULIP + cosine）** |

---

---

## 附錄：檔案涵蓋度盤點（全檔清點結果）

全 repo `.py` 檔案總數（排除 `.git`/`__pycache__`，已實機清點）：**258（含 symlink）/ 256（不含 symlink）**。另有 `.sh` 21、`.md` 24（含本檔與舊報告）。其中：

| 類別 | 數量 | 說明 |
|---|---|---|
| **第一方 .py（專案自身）** | **57**（含 2 個 symlink：`core/main_ikea.py`、`core/data/ikea_ulip.py`） | 全部逐一檢查過（見下） |
| vendored `core/models/pointnext/PointNeXt/**` | 180 | 第三方 PointNeXt 完整庫，非本專案邏輯；只有包裝 `pointnext.py` 相關 |
| vendored `core/Pointnet2_PyTorch/**` | 21 | 第三方 CUDA ops 庫（需編譯），非本專案邏輯 |
| （小計 vendored） | 201 | 57 + 201 = 258（含 symlink） |

**57 個第一方 .py 涵蓋狀態：**
- **完整精讀**：`core/main.py`、`ikea/main_ikea.py`、`core/data/dataset_3d.py`、`ikea/ikea_ulip.py`、`ikea/app_ikea_retrieval.py`、`core/models/ULIP_models.py`（RAG 全段 + 基類 + 工廠）、`core/models/losses.py`、`core/check_environment.py`。
- **重點驗證/略讀**：`core/test.py`、`core/models/pointbert/point_encoder.py`、`core/scripts/build_rag_corpus.py`、`core/scripts/build_rag_index.py`、`corpus/vocab_build/*`（4 步驟 + build_open_knowledge_base + llm.py）、`ikea/` 其餘腳本（get_data / build_3d / fetch_* / phase_a / cleanup / regen / backfill / download_v3 / build_bad_model_manifest）、`scripts/data_download.py`。
- **簽名核對（確認角色，未逐行）**：backbone 包裝 `pointnet2.py`/`pointnet2_utils.py`/`pointMLP.py`/`pointnext.py`/`customized_backbone.py`；pointbert 輔助 `dvae.py`(Group/Encoder)/`misc.py`/`checkpoint.py`/`logger.py`；`utils/*`(build/config/io/registry/tokenizer/utils/logger/__init__)；`core/scripts/download_objaverse.py`/`rag_corpus_expanded.py`/`test_rag_functions.py`；歷史遺跡 6 檔（見第 8 節，已確認未被主流程使用）。

**尚未開啟閱讀的檔案（不影響程式理解，屬結果/說明文件）：**
- `docs/CLAUDE_USAGE_GUIDE.md`（164 行）
- `ikea/docs/3d_model_quality_audit.md`（259 行）
- `ikea/docs/phase_a_dimensions_result.md`（227 行）
- `ikea/docs/phase_v3_scrape_result.md`（234 行）
- `ikea/docs/V3_COLLECTION_CLEANUP_REPORT.md`（155 行）
- `ikea/docs/bad_3d_models_manifest.json`（品質稽核 manifest 資料）
- 各 `.sh` 腳本（`core/scripts/pretrain_*.sh`/`test_*.sh`、`core/quick_train.sh`、`scripts/quick_train_portable.sh`）— 已知為訓練/測試啟動腳本，內含過時路徑。

**結論：程式邏輯相關的第一方檔案 100% 都已檢查過，沒有遺漏。** 未開啟的僅為結果報告類 .md 與 vendored 第三方庫，不影響對專案流程的理解。

---

**報告結束。** 本報告基於實際讀碼與實機驗證（含 FAISS 維度、checkpoint metadata、conda/torch 版本、歷史檔 import 關係逐一確認），未修改任何原始程式檔案。
