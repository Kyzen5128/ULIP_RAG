# Codex → Claude：重新審閱 ULIP_RAG 1.0 論文並解決 2.0 訓練品質問題

> 日期：2026-07-21（Asia/Taipei）  
> 任務性質：先做唯讀 engineering / experiment audit；未經使用者另外授權，不重訓、不修改 dataset、checkpoint、vectors、MongoDB 或 serving。  
> 主要論文：`/home/kyzen/跨模態增強模型嵌入與檢索架構_完稿.docx`

## 1. 使用者目前真正要解決的問題

目前新版 IKEA 1,546 件 Semantic ULIP 已完成技術訓練，但 held-out 結果不足以視為可用：

| Direction | R@1 | R@5 | R@10 | MRR@10 |
|---|---:|---:|---:|---:|
| PC → Text | 10.4478% | 44.0299% | 59.7015% | 23.5608% |
| Text → PC | 17.9104% | 44.7761% | 63.4328% | 29.5422% |
| PC → Image | 31.3433% | 76.8657% | 91.0448% | 49.5143% |
| Image → PC | 29.8507% | 74.6269% | 89.5522% | 47.6342% |

此外，Semantic vs RAG validation A/B 沒有穩定提升：

| Mode | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| Semantic vanilla | 2.0% | 12.0% | 24.0% | 6.3921% |
| RAG | 2.5% | 11.5% | 19.5% | 6.6323% |

RAG 的 R@5、R@10 反而退步，因此目前不能 promotion，也不能宣稱 ULIP_RAG 2.0 已達可用等級。

## 2. 必須先更正的比較錯誤

先前曾把舊 IKEA serving snapshot 的 733 件資料直接當成「ULIP_RAG 1.0 完整訓練資料」，並與新 IKEA 1,546 件比較。這是不正確的。

目前能確認的是：

- `/mnt/P300/data/ikea_data/json`：733 files。
- `/mnt/P300/data/ikea_data/glb`：733 files。
- `/mnt/P300/data/ikea_data/ply`：734 files。
- 733 只代表舊 IKEA serving / vector snapshot。
- 它不能自動代表論文 ULIP_RAG 1.0 的完整訓練全集。
- 舊 733 指標使用非 object-disjoint 評估，新 1,546 使用 family-disjoint held-out test，兩者不能直接計算提升或退步。

請 Claude 不得再用 `76.53% vs 17.91%` 作為 1.0/2.0 效能比較。

## 3. 請完整重新審閱 ULIP_RAG 1.0 論文

請完整閱讀，不只依現有摘要：

```text
/home/kyzen/跨模態增強模型嵌入與檢索架構_完稿.docx
```

需要逐項抽取並核對：

1. 論文聲稱的完整 architecture。
2. 各模型、encoder、fusion、retriever 與 FAISS index 設定。
3. 所有 training stages、loss、optimizer、learning rate、batch、epoch、freeze/unfreeze 設定。
4. 所有 dataset 名稱、來源、數量、類別、split、caption、corpus 與資料生成方式。
5. Furn3D-Ecom 的定義、實際物件數、query 數、corpus 數及是否有 materialized artifact。
6. Table 4 至最後一個表格的所有 metrics、單位與實驗條件。
7. RR 是否實際表示 Recall；NDCG、MRR、classification accuracy 是否有混用。
8. 論文內 Python / CUDA / PyTorch / GPU 環境是否自相矛盾。
9. Ablation 的 K、fusion layers、corpus source 與 index scaling 是否有原始實驗支持。
10. 論文描述與 repo 實際 code、checkpoint args、wandb、logs、evaluation JSON 是否一致。

## 4. 目前發現的 1.0 高風險疑點，請 Claude 獨立驗證

### 4.1 論文宣稱

- Furn3D-Ecom 約 30,000 個家具 3D models。
- 約 30,000 個 LLaVA query descriptions。
- 約 160,000 個 LLaMA-3.2-3B-Instruct corpus rows。
- 16 categories。
- WordNet 3.0 + ConceptNet。
- Dual BERT-base retriever、768D、兩塔 fine-tune。
- 2-layer Cross Fusion、8 heads、hidden 768、FFN 3072。
- FAISS IVFFlat。

### 4.2 現有 4090 唯讀盤點則顯示三條不同 lineage

參考：

```text
/home/kyzen/HANDOFF_4090_TO_3090_ULIP_RAG_1_0_TRAINING_DATASETS.md
```

目前已知：

1. 舊 IKEA serving：733 商品、純 ULIP InfoNCE、無 RAG corpus。
2. ShapeNet55 / ModelNet40 research RAG：ShapeNet train 約 52,468；ModelNet40 test 2,468；RAG corpus 實際只有 1,095 rows。
3. CAMERA_3D / Objaverse prototype：8,632 furniture UIDs、88,434 unified rows，但 `train_two_stage.py` 硬編只取前 1,000 samples。

目前沒有找到一份完整 materialized `Furn3D-Ecom 30K + 160K corpus` manifest。

請 Claude 獨立確認這三條是否被論文混合描述成單一實驗，或是否存在尚未被找到的第四條正式 lineage。

### 4.3 Table 4 數值可能與 ModelNet40 classification logs 混用

目前部分論文數值可在舊 log 中對應到 `test_acc1` / `test_acc5` / training accuracy，而不是 retrieval R@K / NDCG。例如：

- 12.7228525 = 314 / 2468 × 100。
- 13.6547812 = 337 / 2468 × 100。
- 32.9821718 = 814 / 2468 × 100。
- 17.5040519 = 432 / 2468 × 100。
- 42.9902755 = 1061 / 2468 × 100。

這只能暫時標為「高度疑似 metric/source mixing」，不得直接指控資料造假。請 Claude 尋找：

- 原始 table generation script。
- 原始 CSV / JSON。
- query/gallery 定義。
- metric implementation。
- checkpoint lineage。
- exact dataset split。

若仍找不到，請明確標成「尚未確認／不可重現」。

## 5. P300 現有 3D-FUTURE 實體盤點

實際路徑：

```text
/mnt/P300/data/ULIP/datasets/3D-FUTURE
```

本機 README：

```text
/mnt/P300/data/ULIP/datasets/3D-FUTURE/docs/3D-FUTURE-readme.md
```

唯讀實查：

| Item | Count |
|---|---:|
| model directories | 16,563 |
| raw_model.obj | 16,563 |
| normalized_model.obj | 16,560 |
| model.mtl | 16,563 |
| texture.png | 16,562 |
| image.jpg | 16,563 |
| scene RGB images | 20,240 |
| id maps | 20,240 |
| train annotations | 73,513 |
| test annotations | 28,484 |
| total annotations | 101,997 |
| fine categories | 49 |
| styles | 19 |
| materials | 16 |

Super-category distribution：

| Super-category | Count |
|---|---:|
| Cabinet/Shelf/Desk | 5,725 |
| Sofa | 2,701 |
| Lighting | 1,921 |
| Chair | 1,775 |
| Others | 1,740 |
| Bed | 1,124 |
| Table | 1,090 |
| Pier/Stool | 487 |

排除 Lighting 與 Others 後，家具領域候選為 12,902 件。

Metadata completeness：

| Field | Completeness |
|---|---:|
| super-category | 16,563 / 16,563 |
| style | 16,563 / 16,563 |
| fine category | 14,812 / 16,563 |
| material | 12,841 / 16,563 |
| theme | 12,841 / 16,563 |
| explicit dimensions | absent |
| canonical front | absent |
| clearance | absent |
| natural-language captions | absent |

Scene GT：

- Train images 14,761；test images 5,479。
- Train unique referenced models 6,701；test 8,043。
- Train/test model ID overlap 4,750，因此官方 image split 不可直接當 object-disjoint ULIP retrieval split。
- Scene annotations有 bbox、segmentation、model ID、texture ID、style、material、6DoF pose、FoV。
- 3D-FUTURE scene pose 主要是 rendering / camera annotation；不得未驗證就當成完整 room-placement world coordinates。

## 6. P300 現有 3D-FRONT 實體盤點

路徑：

```text
/mnt/P300/data/ULIP/datasets/3D-FRONT
```

目前實查：

| Item | Count / Size |
|---|---:|
| scene JSON files | 6,813 |
| scene JSON data | ~17 GB |
| texture directories | 1,425 |
| total | ~19 GB |

抽樣 JSON 確認包含：

- room type / instance ID / area。
- furniture inventory，`uid` → `jid` model reference。
- room child instances。
- furniture position、quaternion rotation、scale。
- mesh、material、lights、scene bounding box。

完整 room count、有效 furniture instance count、3D-FUTURE `jid` join coverage、有效 relation pair 數目前尚未完成 Gate 1 canonical audit，不得自行假設。

## 7. 請 Claude 解決與回答的核心問題

### P0-A：1.0 真實實驗是什麼

1. ULIP_RAG 1.0 的真正主模型 lineage 是哪一條？
2. 論文 30K / 160K 是否有實體資料？若無，論文結果究竟使用哪一份資料？
3. Table 4 是否可從 retained artifacts 重現？
4. 哪些指標是 retrieval，哪些是 ModelNet classification？
5. 舊 IKEA 733 在 1.0 中到底只是 serving snapshot，還是正式訓練／評估資料？
6. 論文中的 1.0 方法哪些能保留，哪些需要作廢或重寫？

### P0-B：如何把目前 2.0 訓練補到可用

請審查並提出可執行方案，不要只寫概念：

1. 目前 Text→PC R@1=17.91%、R@5=44.78% 的主要原因排序。
2. 是否同意先修 Semantic embedding，再做 RAG；RAG 不可掩蓋基礎 embedding 不良。
3. 每商品要多少 query / caption，如何生成及 QA。
4. Multi-positive、hard-negative、category-balanced sampling 應如何實作。
5. 是否應加入 Image↔Text、category auxiliary loss。
6. 官方 ULIP checkpoint vs 舊 IKEA checkpoint 如何做公平 initialization A/B。
7. Effective batch、LR groups、freeze/unfreeze、early stopping 的實驗矩陣。
8. 3D-FUTURE 12,902 家具是否足夠作 domain pretraining；若不足，缺什麼，不可只說「再找更多資料」。
9. 如何從 3D-FUTURE 產 point clouds、multi-view renders、captions 與 object-disjoint split。
10. 如何把 3D-FRONT 轉成 relation / placement preference supervision。
11. 哪些能力必須 learned，哪些必須 deterministic hard rule。
12. 真實 V2T benchmark 的 query、gallery、positive set、metrics 與 promotion thresholds。

### P1：資料與安全 contract

延續 Claude 上一輪 finding，Gate 7 v3 contract 應加入：

```json
{
  "authoritative_revalidation_required": true,
  "verified_candidates": [
    {
      "safety_status": "provisional_local_only"
    }
  ]
}
```

3090 local rule pass 不得被視為最終安全；5090 full-grid authoritative revalidation 必須保留。

## 8. 請 Claude 交付的文件

請產出：

```text
/home/kyzen/ULIP_RAG/CLAUDE_REPLY_ULIP_RAG_1_0_AUDIT_AND_2_0_TRAINING_RECOVERY_2026-07-21.md
```

文件必須包含：

1. 1.0 論文逐項 audit 結論。
2. 論文 claim ↔ actual artifact 對照表。
3. Table 4 數值可重現性判定。
4. 1.0 真實 dataset / count / split / metrics 的最終判定。
5. 目前 2.0 成績不佳的 root-cause ranking。
6. 3D-FUTURE / 3D-FRONT 是否足夠的明確回答。
7. 可直接執行的 data build、training、evaluation stages。
8. 每個 stage 的 input、output、loss、checkpoint、promotion gate。
9. 尚未確認項目與取得證據的方法。
10. Claude 對 Codex 設計的反對意見、修正或問題；若無問題，明確寫 `NO_BLOCKING_OBJECTION`。

## 9. 安全與範圍限制

- 本輪先唯讀審查。
- 不修改原始 DOCX。
- 不修改 MongoDB。
- 不重建或覆寫 vectors。
- 不啟動 serving。
- 不使用舊 733 結果冒充完整 1.0。
- 不把目前 1,546 baseline 宣稱為 production-ready。
- 不把 weak VLM labels 宣稱為人工 ground truth。
- 所有無 artifact 支持的敘述標註「尚未確認」。

