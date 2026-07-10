# ULIP_RAG 專案完整分析報告

> 對象：`/home/klooom/cheng/3d_retrival/ULIP_RAG/`
> 產出：kyzen，2026-07-09
> 資料來源：實際程式碼閱讀 + HANDOFF_3090.md
> 用途：搬到 3090 前的完整交接文件（HANDOFF_3090.md 的補充，兩份互相參照）

---

## 目錄

1. [專案用途總結](#1-專案用途總結)
2. [目錄結構](#2-目錄結構)
3. [環境與依賴](#3-環境與依賴)
4. [資料位置與資料流程](#4-資料位置與資料流程)
5. [訓練流程](#5-訓練流程)
6. [模型架構](#6-模型架構)
7. [RAG / retrieval 流程](#7-rag--retrieval-流程)
8. [主要程式檔案逐一說明](#8-主要程式檔案逐一說明)
9. [如何從零開始跑起來](#9-如何從零開始跑起來)
10. [還需要人工確認的地方](#10-還需要人工確認的地方)
11. [建議接下來優先讀的檔案](#11-建議接下來優先讀的檔案)
12. [附錄：核心資料流全景圖](#12-附錄核心資料流全景圖)

---

## 1. 專案用途總結

**ULIP_RAG** 是 Salesforce ULIP (CVPR'23, arXiv:2212.05171) 的 cheng 學長魔改版，把 **RAG (Retrieval-Augmented Generation)** 引入 3D↔Text↔Image 的三模態對齊訓練。

- **原始 ULIP**：SLIP (CLIP 變體) 的文字/影像編碼器 ↔ PointBERT 的 3D 點雲編碼器，用 InfoNCE 對齊到 512 維 embedding
- **RAG 增強**：在 CLIP 文字分支後插入 `RAGEnhancer`
  1. 用 MiniLM-L6-v2 將輸入 caption 編碼查 FAISS 索引（家具描述 corpus）
  2. Top-K 檢索文件用**同一個 CLIP text encoder 重新編碼**為 512 維
  3. 用 cross-attention + FFN 融合到原始文字 EOT 特徵
  4. 再過 CLIP 的 `text_projection` 得到 `enhanced_text_embed`
- **兩階段訓練**：
  - **Stage 1**：只訓 `rag_enhancer`；Loss = MSE(enhanced, orig) + 0.5·InfoNCE(enhanced↔image)
  - **Stage 2**：只訓 `point_encoder + pc_projection`；Loss = InfoNCE(pc↔enhanced) + 0.5·InfoNCE(pc↔image)

預期在 3D zero-shot 分類與 cross-modal 檢索上超越原版 ULIP（HANDOFF 標稱 +2% Acc@1）。

---

## 2. 目錄結構

```
ULIP_RAG/
├── main.py                        訓練 + zero-shot 評估入口 (579 行)
├── main_old.py                    舊版備份 (716 行，未使用)
├── test.py                        Cross-modal 檢索 MRR/NDCG 評估 (489 行)
├── mini_test.py                   驗證 MRR/NDCG 實作的 toy case (31 行)
├── check_env.py                   環境檢查 (121 行)
├── download_data.py               HuggingFace snapshot 下載 (18 行)
├── install.sh                     一鍵安裝
├── quick_test.sh                  建 symlink + mini_test + zero-shot
├── minimal_test.sh
├── requirements.txt
├── HANDOFF_3090.md
├── PROJECT_REPORT.md              (本檔)
├── QUICKSTART_GUIDE.md
├── RUN_DEMO.md
├── data -> /mnt/data1/cheng/ULIP/data   symlink，內有雙層 data/data/
│   └── data/                            (雙層是 HuggingFace 解壓造成)
│       ├── dataset_3d.py                DataLoader 本體 (983 行)
│       ├── dataset_3d_rag.py            舊版 (622 行，未使用)
│       ├── dataset_catalog.json
│       ├── ShapeNet-55.yaml
│       ├── ModelNet40.yaml
│       ├── Objaverse_Lvis_Colored.yaml  路徑指 Salesforce 內部，本機無法用
│       ├── labels.json / templates.json
│       ├── initialize_models/           SLIP + PointBERT 預訓練權重
│       ├── pretrained_models/           官方 zero-shot ckpt
│       ├── rag_corpus/                  1095 條家具 corpus + FAISS index
│       ├── modelnet40_normal_resampled/ ~3 GB
│       └── shapenet-55/                 ~795 GB
│           ├── shapenet_pc/             52470 個 .npy
│           ├── rendered_images/         52535 資料夾，每個含 30x2 張渲染圖
│           ├── taxonomy.json
│           ├── train.txt / test.txt     52468 / 11
│           └── ULIP-shapenet_triplets_captions.json  每物件 10 條 caption
├── models/
│   ├── ULIP_models.py             核心模型 (877 行)
│   ├── ULIP_models_rag.py         早期版 (665 行，未使用)
│   ├── rag_enhancer.py            獨立版 RAGEnhancer (114 行，未使用)
│   ├── rag_adapter.py             整支註解 (歷史遺跡)
│   ├── rag_generator.py           整支註解 (T5 生成式，已放棄)
│   ├── retriever.py               獨立版 DenseRetriever (85 行，未使用)
│   ├── losses.py                  原版 ULIPWithImageLoss (62 行)
│   ├── pointbert/                 PointTransformer + config
│   ├── pointnet2/ pointnext/ pointmlp/ customized_backbone/
├── utils/
│   ├── utils.py                   分散式 / cosine / get_dataset
│   ├── tokenizer.py               CLIP BPE
│   ├── config.py io.py build.py registry.py logger.py
│   └── bpe_simple_vocab_16e6.txt.gz
├── scripts/
│   ├── build_rag_corpus.py        Wikipedia+spaCy 建 corpus
│   ├── build_rag_index.py         CLIP 版索引 (未接上)
│   ├── rag_corpus_expanded.py     tricolo 專案的，跟本專案無關
│   └── pretrain_*.sh / test_*.sh
├── Pointnet2_PyTorch/             需編譯 pointnet2_ops (CUDA)
└── outputs/                       學長已跑完的 checkpoints
    ├── RAG_Stage1/
    ├── RAG_Stage2/ RAG2_Stage2/ RAG3_Stage2/
    ├── generative_rag_stage1/
    └── reproduce_pointbert*/      對照組
```

---

## 3. 環境與依賴

### 版本組合（HANDOFF 官方驗證）

- **Python 3.9**（腳本容許 3.8–3.11）
- **PyTorch 2.0.1 + CUDA 11.8**（3090 用 cu118；4090 也 OK）
- `timm==0.4.12`（不能升，會爆 ViT 介面）
- `easydict==1.9`、`open-clip-torch==2.24.0`、`wandb==0.13.3`、`open3d==0.16.0`

### 三份互補依賴清單

1. **requirements.txt**：11 個釘死版本的套件
2. **install.sh 額外裝**：PyTorch、faiss-gpu/cpu、sentence-transformers、huggingface-hub
3. **手動編譯**：`Pointnet2_PyTorch/pointnet2_ops_lib`（CUDA 擴展）
4. **選配（僅重建 corpus 時）**：`spacy` + `en_core_web_sm`、`scikit-learn`、`wikipedia`

### GPU 記憶體實測

| 情境 | 記憶體 |
|---|---|
| Zero-shot 測試 | ~6 GB |
| Stage 1 訓練 | ~10 GB |
| Stage 2 訓練 (batch 32 / npoints 8192) | ~22 GB |

3090 (24 GB) 邊緣可跑；OOM 時降 `--batch-size 16 --npoints 4096`。

### 已知陷阱

- `./data/...` 硬編路徑 → **一定要在 ULIP_RAG/ 根目錄執行**
- `data/data/` 雙層 → 用 symlink 解
- `faiss-gpu` 常裝不起 → 改 `faiss-cpu` 完全夠
- pointnet2_ops 編譯失敗 → 設 `TORCH_CUDA_ARCH_LIST="8.6"`（3090 是 SM 8.6）
- Stage 2 忘記 `--stage1_ckpt_path` 只 warning 不會停 → `rag_enhancer` 會用隨機初始化，一定要檢查 log 有 `Stage 1 rag_enhancer weights loaded successfully`

---

## 4. 資料位置與資料流程

### 位置

- 主資料：`/mnt/data1/cheng/ULIP/data/data/`（透過 `ULIP_RAG/data` symlink）
- 註冊表：`data/dataset_catalog.json`（shapenet / modelnet40 / objaverse_lvis_colored）
- Config：三份 yaml + labels.json + templates.json

### 資料量

| 資料集 | 量 | 用途 |
|---|---|---|
| ShapeNet-55 | 52,470 npy 點雲 + 52,535 渲染圖資料夾 + 10 caption/物件 | Train (pretrain) |
| ModelNet40 | 預處理過的 .dat (1024 / 8192 pts) | Zero-shot 分類評估 |
| Objaverse Lvis Colored | 路徑指 Salesforce 內部 | 本機不可用 |
| RAG corpus | **1095 條**（94% 合成、6% Wikipedia）→ 17 種家具類別 | RAG 檢索 |

**警告**：HANDOFF 標稱 52K corpus 但實際只有 1095 條，見第 10 節。

### 資料流程

```
[點雲 .npy]  -> IO.get() -> FPS/random sample (npoints=8192) -> pc_norm
   ↓
[augmentation]  dropout -> scale -> shift -> jitter -> rotate
   ↓
[torch.float32 tensor]

[渲染圖]  隨機挑角度 (0-348°, 每 12°) + 隨機選 RGB or depth
   -> PIL -> RandomResizedCrop(224) + Normalize(ImageNet mean/std)

[Caption]  優先從 ULIP-shapenet_triplets_captions.json 隨機挑 1 句
                 找不到 -> taxonomy.json 類別名 (可套 template)
   -> SimpleTokenizer (CLIP BPE) -> (tokens_77, [raw_str])
```

### 路徑分派方式

- **CLI arg**：`--pretrain_dataset_name / --validate_dataset_name / --npoints / --rag_corpus_dir / --rag_top_k`
- **catalog + yaml**：`dataset_catalog.json -> *.yaml (DATA_PATH/PC_PATH/IMAGE_PATH)`
- **完全寫死**：`./data/labels.json`、`./data/templates.json`、`./data/dataset_catalog.json`、`./data/initialize_models/slip_base_100ep.pt`、`./data/shapenet-55/ULIP-shapenet_triplets_captions.json`

---

## 5. 訓練流程

### 入口：main.py:main()

```
init_distributed_mode -> wandb.init -> set seed
   ↓
if evaluate_3d: test_zeroshot_3d() -> return
   ↓
建模型: ULIP_PointBERT_RAG(args) 或 ULIP_PointBERT(args)
建 loss: ULIP_Loss_RAG_Enhanced 或 losses.ULIPWithImageLoss
建 optimizer: AdamW(lr=3e-3, betas=(0.9,0.98), wd=0.1)   [wd 分兩組]
scaler = amp.GradScaler
resume from checkpoint if --resume
   ↓
Dataset_3D -> DataLoader (collate: rag_collate_fn 或 customized_collate_fn)
lr_schedule = cosine_scheduler (warmup 1 epoch)
   ↓
for epoch in range(start, epochs):
    train(train_loader, model, criterion, ...)   # AMP + grad accumulation
    if epoch % eval_freq == 0:
        val_stats = test_zeroshot_3d_core(...)
        save checkpoint (best 或每 50 epoch) -> save_on_master
    寫 log.txt / wandb.log
```

### 兩階段訓練指令（HANDOFF 提供）

**Stage 1**（~2-3h）：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model ULIP_PointBERT --use_rag_adapter --training_strategy staged_1 \
    --rag_corpus_dir data/rag_corpus --rag_top_k 5 \
    --pretrain_dataset_name shapenet --pretrain_dataset_prompt shapenet_64 \
    --validate_dataset_name shapenet  --validate_dataset_prompt shapenet_64 \
    --npoints 8192 --epochs 50 --batch-size 32 --lr 1e-4 \
    --output-dir ./outputs/rag_stage1 --wandb
```

**Stage 2**（~10-12h）：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model ULIP_PointBERT --use_rag_adapter --training_strategy staged_2 \
    --stage1_ckpt_path ./outputs/rag_stage1/checkpoint_best.pt \
    --rag_corpus_dir data/rag_corpus --rag_top_k 5 \
    --pretrain_dataset_name shapenet --npoints 8192 \
    --epochs 250 --batch-size 32 --lr 3e-3 \
    --output-dir ./outputs/rag_stage2 --wandb
```

### 凍結策略（寫死在 ULIP_PointBERT_RAG(args) L789-843）

- **staged_1**：全部凍結 → 只開 `rag_enhancer.parameters()`
- **staged_2**：只開 `point_encoder.*` + `pc_projection` → 從 `--stage1_ckpt_path` 撈 `rag_enhancer.*` 權重載回去

### Loss（ULIP_Loss_RAG_Enhanced.forward）

- **Stage 1**：`consistency_loss (MSE) + 0.5 · enhanced_contrast_loss (InfoNCE)`
- **Stage 2**：`pc_text_loss (InfoNCE) + 0.5 · pc_image_loss (InfoNCE)`
- 兩階段的 `0.5×` 常數寫死在 L623 / L659

### Checkpoint / Log 輸出

- 目錄：`--output-dir`（default `./outputs`）
- 檔案：`checkpoint_{epoch}.pt`（best 或每 50 epoch）+ `checkpoint_best.pt` + `log.txt`（每 epoch 一行 JSON）
- Wandb：project=`ULIP-RAG`，id=output_dir 最後一段

### 已跑 outputs

- `RAG_Stage1`、`RAG_Stage2`、`RAG2_Stage2`、`RAG3_Stage2`
- `generative_rag_stage1`（早期實驗）
- `reproduce_pointbert*`（對照組：原版 ULIP）

---

## 6. 模型架構

### 類別繼承層次

```
ULIP_WITH_IMAGE (基類)
   ├── visual = timm 'vit_base_patch16_224' (image encoder)
   ├── transformer = CLIP text Transformer (width=512, layers=12, heads=8)
   ├── token_embedding (49408 -> 512) + positional_embedding (77) + ln_final
   ├── image_projection (768 -> 512)
   ├── text_projection (512 -> 512)
   ├── point_encoder = PointTransformer (768d output)
   ├── pc_projection (768 -> 512)
   └── logit_scale = log(1/0.07), clamp [0, log(100)]

ULIP_with_RAG_Enhancer (繼承並擴充)
   + retriever = RAGRetriever   (FAISS IndexFlatIP + MiniLM-L6-v2)
   + rag_enhancer = RAGEnhancer (MHA head=8, dropout=0.1 + FFN)
   + tokenizer = SimpleTokenizer
   + rag_corpus (全部載入記憶體)
```

### PointBERT 內部（PointTransformer_8192point.yaml）

- **輸入**：8192 個 3D 點
- **Group**：Farthest Point Sampling → 512 個 group centers，每個含 32 個鄰近點
- **Encoder** (PointNet-like) → 每 group 256d → linear → 384d
- **Transformer** 12 層 depth，6 heads，384d，含 cls_token + pos_embed（3D 座標 MLP）
- **輸出**：`concat([cls_token, max_pool(tokens)])` → **768d**
- 初始化：`data/initialize_models/point_bert_pretrained.pt`

### Backbone 工廠

| 工廠 | Backbone | pc_dims | 基類 |
|---|---|---|---|
| ULIP_PN_SSG | Pointnet2_Ssg | 256 | ULIP_WITH_IMAGE |
| ULIP_PN_MLP | pointMLP | 256 | ULIP_WITH_IMAGE |
| ULIP_PointBERT | PointTransformer (8192pt) | 768 | ULIP_WITH_IMAGE |
| ULIP_PN_NEXT | PointNEXT | 256 | ULIP_WITH_IMAGE |
| ULIP_CUSTOMIZED | custom | 512 | ULIP_WITH_IMAGE |
| ULIP2_PointBERT_Colored | PointTransformer_Colored (10k pts) | 768 | ULIP2_WITH_OPENCLIP (bigG-14) |
| **ULIP_PointBERT_RAG** | PointTransformer + RAGEnhancer | 768 | **ULIP_with_RAG_Enhancer** |

---

## 7. RAG / retrieval 流程

### 三個 RAG 組件

**RAGRetriever**（ULIP_models.py L473）

- `query_encoder = SentenceTransformer('all-MiniLM-L6-v2')` (384d)
- `index = faiss.read_index('corpus_index.faiss')` (IndexFlatIP, cosine)
- 查詢：MiniLM encode → L2 norm → search top-K → 從 `corpus[idx].text` 取原文

**RAGEnhancer**（ULIP_models.py L517）

```python
# 512d 特徵空間，8 heads MHA + LayerNorm + FFN (512 -> 1024 -> 512)
context = concat([original.unsqueeze(1), doc_features], dim=1)     # [B, 1+K, 512]
attn = MHA(query=original, key=context, value=context)
fused = LN(original + attn)
fused = LN(fused + FFN(fused))
return fused                                                        # [B, 512]
```

**encode_text_with_rag**（ULIP_models.py L698）— 完整前向

```
tokenized_text + raw_text
   ├─ encode_text_base(tokens) [CLIP text, 未投影]     -> base_feat [B, 512]
   ├─ retrieve_docs(raw_text) -> retrieved [B, K str]
   ├─ SimpleTokenizer(docs) -> encode_text_base(docs)   -> doc_feat [B, K, 512]
   ├─ rag_enhancer(base_feat, doc_feat)                -> fused [B, 512]
   └─ fused @ text_projection                          -> enhanced_text_embed [B, 512]
```

**關鍵設計**：檢索用 MiniLM (384d)，融合用 CLIP text (512d)。檢索到的原文被 **CLIP text encoder 重新編碼**過才進 cross-attn，所以 CLIP encoder 每次 forward 被跑 2 次。

### RAG Pipeline 圖（訓練時）

```
[dataset_3d.py:ShapeNet.__getitem__]
   raw_caption <- ULIP-shapenet_triplets_captions.json (隨機 1/10 句)
   tokens = SimpleTokenizer(raw_caption)
   return (taxonomy_id, model_id, (tokens, [raw]), pc, image)
                    ↓ rag_collate_fn
   (pc[B], (tokens[B,77], raw_strings[B]), image[B,3,224,224], labels)

[Model.forward]
   ├─ RAG side: enhanced_text_embed [B, 512]
   ├─ Original text (training only): encode_text(tokens) -> original_text_embed
   ├─ Point cloud: PointBERT(pc) -> pc_projection -> pc_embed [B, 512]
   └─ Image: SLIP visual + image_projection -> image_embed [B, 512]

[Loss] 依 training_strategy 選 stage1_loss 或 stage2_loss
```

### ULIP 與 RAG 如何連接

1. ULIP 是三模態 (text, pc, image) 對齊框架，在 512d embed 空間用 InfoNCE
2. RAG 只增強 text 分支：`text -> enhanced_text = RAGEnhancer(text, docs)`
3. pc 分支經 pc_projection、image 分支經 image_projection 都不動
4. Stage 1 先訓好 enhancer，Stage 2 再把 pc 拉到新的 text 空間對齊

---

## 8. 主要程式檔案逐一說明

### 頂層入口檔

- **main.py** (579 行) — 訓練 + zero-shot 評估入口
  - `get_args_parser` L38、`main` L106、`train` L279、`test_zeroshot_3d_core` L377、`test_zeroshot_3d` L464
- **main_old.py** (716 行) — 舊版備份，未使用
- **test.py** (489 行) — Cross-modal 檢索評估
  - `calculate_ndcg_at_k` L40、`calculate_mrr` L67、`calculate_retrieval_metrics_streaming` L117
  - `evaluate_rag_quality` L149、`evaluate_cross_modal_retrieval` L228、`main` L316
  - 結果存 `evaluation_results_{dataset}_{model}[_RAG].json`
- **mini_test.py** (31 行) — 驗證 MRR/NDCG 實作的單元測試（不需 GPU）
- **check_env.py** (121 行) — 環境自檢
- **download_data.py** (18 行) — HuggingFace `SFXX/ulip` 下載

### models/

- **ULIP_models.py** (877 行) — **核心**
  - L1-460 是原版 ULIP、L460-877 是學長 RAG 擴充
  - class：`ULIP_WITH_IMAGE` L78、`ULIP2_WITH_OPENCLIP` L184、`RAGRetriever` L473、`RAGEnhancer` L517、`ULIP_Loss_RAG_Enhanced` L539、`ULIP_with_RAG_Enhancer` L672
  - 工廠：`ULIP_PN_SSG` / `ULIP_PN_MLP` / `ULIP_PointBERT` / `ULIP_PN_NEXT` / `ULIP2_PointBERT_Colored` / `ULIP_CUSTOMIZED` / `ULIP_PointBERT_RAG` L759
- **ULIP_models_rag.py** (665 行) — 早期版，已被合併，未使用
- **rag_enhancer.py / rag_adapter.py / rag_generator.py / retriever.py** — 全部歷史遺跡，未被 main import
- **losses.py** (62 行) — 原版 `ULIPWithImageLoss`（用 all_gather_batch 支援 DDP）
- **pointbert/point_encoder.py** — `PointTransformer` L113、`PointTransformer_Colored` L239
- **pointbert/PointTransformer_8192point.yaml** — PointBERT config
- **pointnet2/pointnet2.py** — Pointnet2_Ssg（需要編譯過的 pointnet2_ops）
- **pointnext/ pointmlp/ customized_backbone/** — 其他 backbone

### utils/

- **utils.py** (241 行) — 核心工具
  - `cfg_from_yaml_file` L32、`get_model` L42（DDP unwrap）、`save_on_master` L89
  - `init_distributed_mode` L98、`scaled_all_reduce` L123、`all_gather_batch` L148
  - `cosine_scheduler` L215、`get_dataset` L240
- **tokenizer.py** (150 行) — CLIP BPE tokenizer，讀 `bpe_simple_vocab_16e6.txt.gz`
- **build.py** (17 行) — `DATASETS = Registry('dataset')` + `build_dataset_from_cfg`
- **registry.py** (287 行) — 通用註冊模式
- **io.py** (41 行) — 統一讀 `.npy / .pcd / .h5 / .txt`
- **config.py** (62 行) — yaml 讀取（與 utils.py 重複，推測舊版遺留）
- **logger.py** (126 行) — `print_log` log 工具

### scripts/

- **build_rag_corpus.py** (462 行) — 從 Wikipedia + spaCy + sklearn KMeans 建 `rag_corpus.jsonl`
- **build_rag_index.py** (53 行) — 用 OpenCLIP ViT-B-32 建 `clip_corpus_index.faiss`（未接上 runtime）
- **rag_corpus_expanded.py** (47 行) — tricolo 專案用，跟本專案無關
- **pretrain_\*.sh / test_\*.sh** — 各 backbone shell 腳本

### data/data/（symlink 進去的資料目錄）

- **dataset_3d.py** (983 行) — DataLoader 本體
  - class：`ModelNet` L161、`ShapeNet` L498、`Objaverse_Lvis_Colored` L700、`Dataset_3D` L944
  - function：`customized_collate_fn` L796、`rag_collate_fn` L852、`farthest_point_sample`、各種點雲 augmentation
- **dataset_3d_rag.py** (622 行) — 舊版，未使用

---

## 9. 如何從零開始跑起來

### Step 1：環境

```bash
conda create -n ulip_rag python=3.9 -y
conda activate ulip_rag

cd /path/to/ULIP_RAG
bash install.sh                            # 會互動問 PyTorch CUDA 版本
python check_env.py                        # 全綠代表 OK
```

### Step 2：資料 symlink

```bash
# data/ 是 symlink 指向資料實體目錄（HuggingFace 解壓後是雙層 data/data/）
ln -sfn /path/on/3090/ULIP_data data/data

# 程式碼路徑是 ./data/xxx（單層），需再建第二層 symlink
bash quick_test.sh                         # 自動建 initialize_models, rag_corpus,
                                            # shapenet-55, modelnet40_normal_resampled,
                                            # labels.json, templates.json, dataset_catalog.json
```

### Step 3：驗證（最快 5 分鐘）

```bash
python mini_test.py                        # MRR/NDCG 實作正確性
bash scripts/test_pointbert.sh \
   data/data/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointbert.pt
# 預期 Acc@1 ≈ 87%, Acc@5 ≈ 96%
```

### Step 4：測試學長的 RAG checkpoint

```bash
python test.py --model ULIP_PointBERT_RAG --use_rag_adapter \
    --training_strategy staged_2 \
    --test_ckpt_addr outputs/RAG2_Stage2/checkpoint_best.pt \
    --validate_dataset_name shapenet --validate_dataset_prompt shapenet_64 \
    --rag_corpus_dir data/rag_corpus --batch-size 16 --eval_zero_shot
```

### Step 5：兩階段從頭訓（可選，見第 5 節指令）

### Step 6：重建 corpus（可選）

```bash
pip install scikit-learn wikipedia spacy
python -m spacy download en_core_web_sm
python scripts/build_rag_corpus.py         # 產 rag_corpus.jsonl + corpus_index.faiss
```

### 搬到 3090 需要同步的檔案

| 相對路徑 | 大小 | 必要 | 用途 |
|---|---|---|---|
| `data/data/initialize_models/slip_base_100ep.pt` | 2.0 GB | **Yes** | CLIP backbone 初始化 |
| `data/data/initialize_models/point_bert_pretrained.pt` | 527 MB | Yes (訓練) | PointBERT 初始化 |
| `data/data/pretrained_models/ckpt_zero-sho_classification/*.pt` | ~200MB | 測試 yes | 官方 zero-shot ckpt |
| `data/data/rag_corpus/`（4 個檔） | 2.4 MB | **Yes** | FAISS + JSONL 語料 |
| `data/data/labels.json`, `templates.json`, `*.yaml`, `dataset_catalog.json`, `dataset_3d.py` | 小 | **Yes** | 資料管線配置 + DataLoader |
| `data/data/modelnet40_normal_resampled/` | 3.1 GB | 測試 yes | ModelNet40 zero-shot |
| `data/data/shapenet-55/shapenet_pc/` (52470 npy) | ~200 GB | Stage2 yes | 點雲 |
| `data/data/shapenet-55/rendered_images/` (52535 資料夾) | 大部分 | Stage1/2 yes | 多視角渲染圖 |
| `data/data/shapenet-55/taxonomy.json`, `train.txt`, `test.txt`, `ULIP-shapenet_triplets_captions.json` | 小 | **Yes** | 類別與 caption |
| 既有 checkpoints（`outputs/RAG_Stage1/checkpoint_best.pt` 680 MB, `outputs/RAG2_Stage2/checkpoint_best.pt` 834 MB） | 1.5 GB | 可選 | 直接做測試不必重訓 |

**最小搬遷組合**（只做 ModelNet40 zero-shot + RAG 小試）：
`initialize_models/` + `rag_corpus/` + `modelnet40_normal_resampled/` + `pretrained_models/` + `labels.json / templates.json / dataset_catalog.json / ShapeNet-55.yaml / ModelNet40.yaml / dataset_3d.py`（總計約 8 GB）。

---

## 10. 還需要人工確認的地方

### 高優先（會影響實驗結論）

1. **RAG corpus 只有 1095 條，不是 HANDOFF 說的 52K**
   - 全部是家具（17 類：cabinet, table, wardrobe, shelf, chair, bed, lamp, stool, dresser, bookshelf, sofa, desk, bench, coffee table, dining table, nightstand, armchair）
   - 對 ShapeNet 55 類的其他類別（airplane, animal, car...）**檢索沒意義**
   - 需要問 kyzen：HANDOFF 寫錯？還是有另一個大的 index 沒同步過來？

2. **main.py:322 疑似 unpacking bug**
   - `if args.use_rag_adapter: pc, text_data, image = batch_data` 只解 3 元素
   - 但 `rag_collate_fn` 明明 `return pc_tensor, text_batch, img_tensor, label_list`（4 元素）
   - `outputs/RAG*_Stage2/` 有大量成功 checkpoint，說明能跑 — 需確認為何不 crash

3. **RAGEnhancer 的 zero-padding 沒 mask**
   - `doc_features_padded = torch.zeros(...)` 少於 K 的位置填 0
   - `nn.MultiheadAttention` 沒傳 `key_padding_mask`，全 0 的 key 會參與 softmax → 可能 bias 融合結果

### 中優先

4. `build_rag_index.py` 產的 CLIP 版索引沒接上 runtime — 是設計還是遺留？
5. `objaverse_lvis_colored` 路徑寫死到 Salesforce 內部 → 本機無法用，是否有補丁？
6. `train_cleaned_preview.txt`（2.4MB）是什麼？—— 推測是 caption 預覽，未確認
7. Stage 1 用 `validate_dataset_name=shapenet` + `validate_dataset_prompt=shapenet_64` 時，`test.txt` 只有 11 行 → 每次 val 只有 11 個樣本，指標容易噪 — 未確認實際跑起來如何
8. `~/miniconda3/envs/ulip` 是否已完整可用（faiss、pointnet2_ops）— 未實測

### 低優先

9. `utils/config.py` 與 `utils/utils.py` 有重複的 `cfg_from_yaml_file` — 推測是舊版遺留
10. `main_old.py` 與 `main.py` 具體差多少 — 未逐行對照

---

## 11. 建議接下來優先讀的檔案

按閱讀順序（每個大約 30 分鐘 - 2 小時）：

### 第一輪：搞懂訓練 & 推理骨架

1. **main.py**（579 行）— 必讀
   - 重點：`get_args_parser` L38、`main` L106、`train` L279、`test_zeroshot_3d_core` L377
   - 順便確認第 10 節「main.py:322 bug」問題

2. **models/ULIP_models.py L460 之後**（RAG 相關 400 行）— 必讀
   - `RAGRetriever` L473 / `RAGEnhancer` L517 / `ULIP_Loss_RAG_Enhanced` L539
   - `ULIP_with_RAG_Enhancer` L672（尤其 `encode_text_with_rag` L698、`forward` L731）
   - `ULIP_PointBERT_RAG` 工廠 L759（凍結策略）

### 第二輪：搞懂資料

3. **data/data/dataset_3d.py L498-698 (ShapeNet class)**
   - 尤其 `__getitem__` L618（RAG caption 產生邏輯）
   - `rag_collate_fn` L852

4. **HANDOFF_3090.md 全文**
   - 學長寫的最完整說明，除了 corpus 條數之外都很準

### 第三輪：搞懂 backbone & RAG corpus

5. **models/pointbert/point_encoder.py L113-237 (PointTransformer)**
   - 只需要知道 forward 產 768d 特徵就好

6. **scripts/build_rag_corpus.py**（462 行）
   - 想改 corpus 或搞懂為何只有 1095 條就必讀

### 第四輪：評估流程

7. **test.py L228 evaluate_cross_modal_retrieval + L117 calculate_retrieval_metrics_streaming**
   - 想改指標時看

### 可跳過（除非要移植/擴充）

- `main_old.py`、`ULIP_models_rag.py`、`rag_adapter.py`、`rag_generator.py`、`rag_enhancer.py`、`retriever.py`、`dataset_3d_rag.py` — 全部是歷史遺跡

---

## 12. 附錄：核心資料流全景圖

```
[Shell 指令]
  python main.py --use_rag_adapter --training_strategy staged_1 --stage1_ckpt_path ...
      │
      ▼
[main.py:main()]
      │
      ├── models.ULIP_PointBERT_RAG(args)
      │       ├── PointTransformer (載入 point_bert_pretrained.pt)
      │       ├── ULIP_with_RAG_Enhancer
      │       │    ├── RAGRetriever ─── SentenceTransformer('MiniLM')
      │       │    │                └── FAISS IndexFlatIP (corpus_index.faiss)
      │       │    ├── RAGEnhancer  ─── MHA + FFN (512d)
      │       │    └── SimpleTokenizer
      │       ├── 載入 slip_base_100ep.pt
      │       └── 凍結策略 (staged_1: 只開 rag_enhancer)
      │           或 (staged_2: 只開 point_encoder+pc_projection + 從 stage1_ckpt 撈 rag_enhancer)
      │
      ├── models.ULIP_Loss_RAG_Enhanced(args)
      │       ├── stage1_loss: MSE + 0.5·InfoNCE(enhanced↔img)
      │       └── stage2_loss: InfoNCE(pc↔enhanced) + 0.5·InfoNCE(pc↔img)
      │
      ├── Dataset_3D -> ShapeNet
      │       ├── 讀 shapenet_pc/*.npy -> FPS -> aug
      │       ├── 隨機挑 rendered_images/*/…png
      │       └── 從 ULIP-shapenet_triplets_captions.json 挑 caption
      │
      ├── AdamW + cosine scheduler + AMP GradScaler
      │
      └── for epoch:
              train() -> forward -> loss -> backward -> step
              test_zeroshot_3d_core() -> save_on_master()
              寫 log.txt / wandb
```

---

**報告結束。**

本報告完全根據對程式碼的實際閱讀，並在有疑問處標註「推測」/「尚未確認」。最重要的兩件事是：

1. 確認 RAG corpus 是否只有 1095 條（HANDOFF 標稱 52K）
2. 確認 main.py:322 的 unpacking 是否真的有 bug

這兩點會直接影響後續要不要延伸這個專案。搬到 3090 後可以在第 9 節的 Step 4 用學長留的 `outputs/RAG2_Stage2/checkpoint_best.pt` 直接跑 test.py 做基準測試，比較 MRR/NDCG 是否跟學長的評估結果對得上。
