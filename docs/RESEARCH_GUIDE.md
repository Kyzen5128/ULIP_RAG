> ⚠️ **過時警告(2026-07-10)**:本檔為 4090 時代文件。專案位置/env 名/路徑均已失效(現為 /home/kyzen/ULIP_RAG + conda ulip + /mnt/P300/data)。現況請看 docs/INDEX.md。

# CAMERA Core 研究接手指南

> 撰寫人：Claude（依 Kyzen 需求整理）
> 日期：2026-03-30
> 目標：從學長留下的程式碼出發，加入空間屬性（AABB/dimensions），實現空間感知的家具 RAG 檢索。

---

## 1. 你繼承了什麼

你手上有**兩個不同的系統**，功能互補但目前尚未整合：

### 系統 A：ULIP_RAG（你的主戰場）
位置：`~/ULIP_RAG/ulip_rag/`

```
輸入：3D 點雲 (.npy) + 文字描述
核心：PointBERT（3D encoder）+ RAGEnhancer（cross-attention 融合）
訓練：對比學習，讓 3D 向量和文字向量對齊
RAG：SentenceTransformer (MiniLM 384D) + FAISS 檢索語義 corpus
Corpus：8,256 條家具文字描述（無空間屬性）
```

**目前狀態**：Pipeline 已修通，Stage 1 可訓練，1 epoch 結果：
- loss: 1.15, enhanced_image_acc: 30%, ModelNet40 zero-shot Acc@1: 1.7%

---

### 系統 B：CAMERA_3D（學長的研究，參考用）
位置：`~/ULIP_RAG/data/CAMERA_3D/`

```
輸入：多視角渲染圖（12 views）+ 文字描述
核心：GFMVEncoder（VGG16 + Gate Fusion）+ RAGTextEncoder（BERT + CrossFusion）
訓練：兩階段（對比預訓練 → Reranker 微調）
RAG：DenseRetriever（BERT 768D）+ 88,422 條 semantic corpus
Corpus：來自 Objaverse 篩選的家具，有 LLaVA 描述，仍無空間屬性
```

**目前狀態**：學長已訓練完成（ckpts_110/enc1.pth 1.7GB），但路徑寫死在 4090 上。

---

## 2. 關鍵差異對照表

| 項目 | ULIP_RAG（你的） | CAMERA_3D（學長的）|
|------|------------------|--------------------|
| 3D 輸入 | 點雲 .npy（8192 pts）| 多視角 PNG（12 views）|
| 3D Encoder | PointBERT（768D）| GFMVEncoder/VGG16（512D）|
| 文字 Encoder | CLIP text（512D）| BERT（768D）|
| RAG 檢索器 | MiniLM 384D + FAISS | BERT 768D + FAISS |
| Corpus 大小 | 8,256 條 | 88,422 條 |
| 訓練資料 | ShapeNet-55 52k | Objaverse 30k 家具 |
| 空間屬性 | **沒有** | **沒有** |
| Checkpoint | 無（要自己訓）| enc1.pth（已訓完）|

**結論：兩個系統都沒有空間屬性，這就是你的研究貢獻點。**

---

## 3. 資料資產地圖

```
/mnt/P300/data/  (= ~/ULIP_RAG/data/)
├── ULIP/
│   ├── ULIP_Shapenet_Triplets/
│   │   ├── shapenet_pc/          ← 52,470 個 .npy 點雲（你的訓練資料）
│   │   └── train.txt / test.txt  ← 檔案清單與類別標籤
│   ├── ULIP-1/
│   │   ├── initialize_models/    ← point_bert_pretrained.pt, slip_base_100ep.pt
│   │   └── modelnet40_normal_resampled/  ← 評估資料集
│   └── dataset_catalog.json
│
├── CAMERA_3D/
│   ├── datasets/
│   │   ├── semantic_corpus.jsonl  ← 88,422 條語料（學長的，可擴充用）
│   │   ├── selected_data.jsonl    ← ~30k RAG 訓練資料
│   │   └── unified_data.jsonl    ← 完整資料（含多視角）
│   ├── models/
│   │   ├── rag_text_encoder.py   ← RAG 文字編碼器（CrossFusion）
│   │   ├── dense_retriever.py    ← DPR 風格可微檢索器
│   │   ├── gf_mv_encoder.py      ← GateFusion 多視角視覺編碼器
│   │   └── cross_modal_reranker.py
│   └── ckpts_110/
│       └── enc1.pth              ← 學長訓練好的 checkpoint（1.7GB）
│
└── semantic_corpus.jsonl         ← 同上，備份在根目錄

~/ULIP_RAG/ulip_rag/
├── rag_corpus/
│   ├── rag_corpus.jsonl          ← 現有 8,256 條語料
│   └── corpus_index.faiss        ← MiniLM 384D FAISS index
├── models/
│   ├── ULIP_models.py            ← RAGRetriever + RAGEnhancer + ULIP_PointBERT_RAG
│   ├── retriever.py              ← DenseRetriever（SentenceTransformer）
│   └── losses.py                 ← InfoNCE 對比損失
├── data/dataset_3d.py            ← ShapeNet/ModelNet 資料載入
└── main.py                       ← 訓練入口
```

---

## 4. 你的研究目標（空間感知 RAG）

### 現在的問題
SpatialLM 輸出一個家具位置的 AABB：
```
{ "category": "sofa", "bbox": [x_min, y_min, z_min, x_max, y_max, z_max] }
```
但 RAG corpus 完全沒有尺寸資訊，沒辦法根據「空間能放得下」來篩選家具。

### 目標架構
```
SpatialLM 輸出的空位 AABB
        ↓
空間查詢 = 文字描述 + 空間約束 (W×H×D)
        ↓
雙通道 RAG 檢索
  ├── 語義通道：MiniLM 384D → FAISS → 相似文字描述
  └── 空間通道：比較 footprint_area / 尺寸是否符合空位
        ↓
Cross-attention 融合 → 推薦家具
```

---

## 5. 執行計畫（Phase 0～5）

### Phase 0：建立 Baseline（1～2 天）
跑完整訓練，記錄基準數字。

```bash
cd ~/ULIP_RAG/ulip_rag

# Stage 1：只訓 RAGEnhancer（100 epochs）
conda run -n ulip_rag python main.py \
  --model ULIP_PointBERT \
  --use_rag_adapter \
  --training_strategy staged_1 \
  --npoints 8192 --batch-size 64 --epochs 100 \
  --rag_corpus_dir ./rag_corpus \
  --output-dir ./outputs/baseline_stage1

# Stage 2：解凍 PointBERT（150 epochs）
conda run -n ulip_rag python main.py \
  --model ULIP_PointBERT \
  --use_rag_adapter \
  --training_strategy staged_2 \
  --npoints 8192 --batch-size 32 --epochs 150 \
  --rag_corpus_dir ./rag_corpus \
  --resume ./outputs/baseline_stage1/checkpoint_best.pt \
  --output-dir ./outputs/baseline_stage2
```

目標指標：ModelNet40 zero-shot Acc@1 > 50%（原始 ULIP 約 60%）

---

### Phase 1：建立帶空間屬性的 Corpus（3～5 天）

**方法：從 ShapeNet-55 點雲計算 AABB**

每個 .npy 點雲都可以直接用 numpy 算出 bounding box：
```python
pts = np.load('xxx.npy')  # (8192, 3)
aabb_min = pts.min(axis=0)  # [x_min, y_min, z_min]
aabb_max = pts.max(axis=0)  # [x_max, y_max, z_max]
W, H, D = aabb_max - aabb_min
```

**注意**：ShapeNet 點雲是正規化的（大約在 [-1, 1]），不是真實公尺。需要後處理轉換。

**新 corpus 格式：**
```json
{
  "text": "A wooden dining chair with ...",
  "obj_id": "xxx",
  "category": "chair",
  "attributes": { "colors": [...], "materials": [...] },
  "dimensions": { "W": 0.6, "H": 0.9, "D": 0.6 },
  "aabb": { "min": [x, y, z], "max": [x, y, z] },
  "footprint_area": 0.36,
  "spatial_constraints": { "fits_small_room": true, "fits_doorway": true }
}
```

腳本位置（待寫）：`ulip_rag/scripts/build_spatial_corpus.py`

---

### Phase 2：擴充 Corpus 並重建 FAISS Index（1～2 天）

- 合併現有 8,256 條 + 學長的 88,422 條（過濾非家具）
- 加入 Phase 1 計算的空間屬性
- 重建 FAISS index（仍用 MiniLM 384D，文字部分）
- 空間屬性另存 numpy array，用於空間通道檢索

---

### Phase 3：雙通道 Retriever（3～5 天）

修改 `ulip_rag/models/retriever.py`：

```
現在：  query_text → MiniLM → FAISS → top-k 文字
目標：  query_text → MiniLM → FAISS → 語義候選
        空間約束 (W,H,D) → 過濾 → 空間候選
        兩者取交集或加權合併 → 最終 top-k
```

---

### Phase 4：修改 CrossFusion 融合空間特徵（3～5 天）

修改 `ulip_rag/models/ULIP_models.py` 的 `RAGEnhancer`：

```
現在：[query_vec, doc1_vec, doc2_vec, doc3_vec] → CrossAttention → 融合向量
目標：[query_vec, doc1_vec+spatial1, doc2_vec+spatial2, ...] → CrossAttention → 融合向量
```

---

### Phase 5：接上 SpatialLM（2～3 天）

SpatialLM 的輸出 → 轉換成 RAG 查詢格式 → 走完整 CAMERA pipeline

---

## 6. 立刻可以做的事

1. **讀學長的 `dense_retriever.py`**：理解 DPR 風格可微分檢索器，這個比你現在的 MiniLM 更先進，可以考慮借鑑。
2. **看 `semantic_corpus.jsonl` 的格式**：`head -n 3 ~/ULIP_RAG/data/semantic_corpus.jsonl | python3 -m json.tool`
3. **跑 Phase 0 baseline**：先有基準數字，才知道你的修改有沒有效。

---

## 7. 關鍵檔案快速導航

| 想做什麼 | 去哪裡改 |
|----------|----------|
| 修改 RAG 檢索邏輯 | `ulip_rag/models/retriever.py` |
| 修改 CrossFusion 融合 | `ulip_rag/models/ULIP_models.py` L517+ |
| 修改訓練流程/Loss | `ulip_rag/main.py` |
| 新增 corpus 欄位 | `ulip_rag/scripts/build_rag_corpus.py` |
| 重建 FAISS index | `ulip_rag/scripts/build_rag_index.py` |
| 資料載入 | `ulip_rag/data/dataset_3d.py` |
| 參考學長的 RAG encoder | `data/CAMERA_3D/models/rag_text_encoder.py` |
| 參考學長的 retriever | `data/CAMERA_3D/models/dense_retriever.py` |

---

## 8. 已知問題與注意事項

- `UnifiedDataset` 的 `view_dir` 路徑寫死在 `/home/klooom/cheng/...`，要用 CAMERA_3D 的訓練需要先改這個
- ShapeNet 點雲是正規化尺寸，用來做空間約束前需要乘上真實 scale factor
- `dataset_3d.py` 已加入缺失檔案 fallback（2 個 .npy 不存在）
- corpus_index.faiss 是 MiniLM 384D，不能跟 CLIP 512D 混用
- 學長的 `enc1.pth` 對應的 objaverse 渲染圖在你電腦上不存在，無法直接跑評估
