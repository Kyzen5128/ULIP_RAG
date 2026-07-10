> ⚠️ **危險警告(2026-07-10)**:本檔「一鍵 rsync」段落**勿執行**——會用 4090 扁平結構覆蓋 3090 的巢狀 configs(已按最小手術處理完,見 PROJECT_REPORT_3090)。本檔僅留作交接歷史。

# 3090 補齊清單（4090 實查結果）

> 產出：2026-07-10，於 4090 實機 (`/home/klooom/cheng/3d_retrival/`) 逐項確認。
> 對照文件：`HANDOFF_ASK_4090.md`（3090 提問清單）。
> 用途：把 4090 上找到的檔案 / 設定值 / 說明帶回 3090，補齊 IKEA 訓練 + serving 缺口。

---

## 一句話總結

**P0-1 / P0-2 / P1 / P2-1 全部找到答案**，把下方「一鍵複製指令」跑完，3090 的 IKEA 訓練入口與 serving app 應該就能通。**P2-2（Objaverse）建議放棄**。

---

## P0-1：建向量腳本 — 找到

### 檔案位置

`/home/klooom/cheng/3d_retrival/ULIP/build_vectors.py`（693 行）

### 已驗證符合 3090 需求

- 用 `models.ULIP_PointBERT`，跑 `encode_pc / encode_image / encode_text`（L302, L333, L374, L549）
- 輸出檔名正是 `vectors_{pc,img,txt}.npy` + `meta_{pc,img,txt}.jsonl` + `schema.json`（L612, L686）
- meta 欄位 = `id / json_path / category / caption / preview_image`（L341-347）
- schema = `{"modalities":[...], "dims":{"pc":512,"img":512,"txt":512}}`（L678-687）

### 4090 上已跑出的產物（可直接搬）

`/mnt/data1/ikea_data/vectors/`：

```
vectors_pc.npy   (733, 512)   1.5 MB
vectors_img.npy  (733, 512)   1.5 MB
vectors_txt.npy  (732, 512)   1.5 MB
meta_pc.jsonl    733 行
meta_img.jsonl   733 行
meta_txt.jsonl   732 行
schema.json
```

### 732 vs 733 差 1 筆的原因

`build_vectors.py:337-339` 對空 caption 直接 `continue` 跳過：

```python
cap = (it["caption"] or "").strip()
if not cap:
    continue
```

733 件 IKEA 家具裡有 1 件 caption 是空字串 → txt 少 1 筆。**這是設計行為，不是 bug**。

### 執行範例（3090 上重跑）

```bash
conda activate ulip
cd /path/to/ULIP
python build_vectors.py \
    --json_dir  <path>/ulip_output/json \
    --out_dir   <path>/vectors \
    --ckpt      <path>/outputs/ULIP2_ikea_b/checkpoint_last.pt \
    --model     ULIP_PointBERT \
    --npoints   8192
```

---

## P0-2：dataset_catalog.json 的 ikea_ulip 條目 — 找到

### 檔案位置

`/home/klooom/cheng/3d_retrival/ULIP/data/dataset_catalog.json`

### 4090 上實際內容

```json
{
    "shapenet": {
        "config": "./data/ShapeNet-55.yaml",
        "train": "train", "test": "test", "usage": "train"
    },
    "modelnet40": {
        "config": "./data/ModelNet40.yaml",
        "train": "train", "test": "test", "usage": "test"
    },
    "objaverse_lvis_colored": {
        "config": "./data/Objaverse_Lvis_Colored.yaml",
        "train": "train", "test": "test", "usage": "train"
    },
    "ikea_ulip": {
        "usage": "train",
        "train": "train",
        "val":   "val",
        "config": "./data/ikea_ulip.yaml"
    }
}
```

### 對應 yaml 也一起搬

`data/ikea_ulip.yaml`：

```yaml
NAME: IkeaULIP
JSON_DIR: ./data/ulip_output/json
SPLIT_METHOD: hash      # hash / filelist / fixed
TRAIN_RATIO: 1.0
REQUIRE_IMAGE: false
RENDER_PICK: random     # random / fixed_000 / fixed_012 ...
MIN_POINTS: 8192

PC_SAMPLER: fps         # 或 random
AUGMENT: true           # 只在 train 時生效
PAD_IF_SHORT: true
```

### Dataset 實作也在同目錄

`data/ikea_ulip.py`（3090 可能也缺，一起搬）

### 3090 放到哪

- catalog → `data/dataset_catalog.json`（覆蓋現有的，或手動 merge 這條 key）
- yaml → `data/ikea_ulip.yaml`
- py → `data/ikea_ulip.py`

---

## P1：Serving 部署形態 — 釐清

### 關鍵發現：TRELLIS 是專案內 submodule，不是外部 repo

4090 app 用的 `from TRELLIS.trellis.pipelines import ...` 是相對 import 到 `/home/klooom/cheng/3d_retrival/ULIP/TRELLIS/`（ULIP 專案內的 vendor 目錄）。

跟 `/home/kyzen/TRELLIS`（3090 那個）或 conda envs 的 `trellis` / `TRELLIS` **完全無關**。

### 結論：3090 用單一 `ulip` conda env 就搞定

4090 上實測 `ulip` env（Python 3.9）：

```
import timm, fastapi, uvicorn, trimesh, rembg  →  OK
from transformers import CLIPModel             →  OK
```

TRELLIS 是 vendor submodule → `from TRELLIS.trellis import ...` 自動找到。**不需另裝**。

### Trellis 是 lazy load

`app_ikea_retrieval.py:108 _lazy_trellis()` 第一次呼叫才 `from_pretrained("microsoft/TRELLIS-image-large")`（從 HuggingFace 抓 pipeline weights）。App 開起來時**不會 crash**，只有 `/post/upload` 端點觸發時才真正載入。

### 4 個環境變數（app_ikea_retrieval.py:49-71）

全部有 default，可直接跑；要覆蓋就 export 出來：

| 變數 | 4090 default | 3090 建議設 |
|---|---|---|
| `VEC_DIR` | `/mnt/data1/ikea_data/vectors` | 3090 上對應路徑 |
| `CKPT` | `/home/klooom/cheng/3d_retrival/ULIP/outputs/ULIP2_ikea_b/checkpoint_last.pt` | 3090 的 `/mnt/P300/data/ULIP/checkpoint_last.pt` |
| `ULIP_OUTPUT` | `/mnt/data1/ikea_data` | 3090 對應 |
| `CUSTOM_OUTPUT` | `/mnt/data1/custom_data` | 3090 對應 |
| `DEVICE` | `cuda` | `cuda` |
| `TMP_DIR` | `${CUSTOM_OUTPUT}/tmp` | 自動 |
| `UPLOAD_DIR` | `${CUSTOM_OUTPUT}/uploads` | 自動 |
| `MONGO_URL` | `mongodb://localhost:27017/` | 3090 若無 mongo，設 `USE_MONGO=0` |
| `MONGO_DB` | `furniture_db` | 同上 |
| `MONGO_COL` | `ikea_product` | 同上 |
| `USE_MONGO` | `1` | 沒 mongo 就設 `0` |

### 4090 沒有正式啟動腳本

搜尋 `.env` / `.service` / `run*.sh` / `start*.sh` / systemd 全部找不到。**推測學長就是手動 uvicorn 或 shell alias 跑**。

### 3090 建議啟動指令

```bash
conda activate ulip
cd /path/to/ULIP

# 若無 mongo 先關掉
export USE_MONGO=0

# 指到 3090 實際路徑
export VEC_DIR=/mnt/P300/data/ikea_data/vectors
export CKPT=/mnt/P300/data/ULIP/checkpoint_last.pt
export ULIP_OUTPUT=/mnt/P300/data/ikea_data
export CUSTOM_OUTPUT=/mnt/P300/data/custom_data

uvicorn app_ikea_retrieval:app --host 0.0.0.0 --port 8000
```

---

## P2-1：RAG stage1 / stage2 checkpoints — 全部在 4090 上

### 4090 上實際列表

`/home/klooom/cheng/3d_retrival/ULIP_RAG/outputs/`：

| 路徑 | 用途 | 大小 |
|---|---|---|
| `RAG_Stage1/checkpoint_best.pt` | 第一輪 Stage 1 訓好的 | ~680 MB |
| `RAG_Stage2/checkpoint_best.pt` | 第一輪 Stage 2 | ~830 MB |
| `RAG2_Stage1/checkpoint_best.pt` | 第二輪 Stage 1 | |
| **`RAG2_Stage2/checkpoint_best.pt`** | 第二輪 Stage 2（**HANDOFF_3090 主推**） | ~834 MB |
| `RAG3_Stage2/checkpoint_best.pt` | 第三輪 Stage 2 | |
| `generative_rag_stage1/checkpoint_best.pt` | 早期 T5 生成式（已放棄） | |
| `reproduce_pointbert/checkpoint_best.pt` | 對照組：原版 ULIP | |
| `reproduce_pointbert_wandb/checkpoint_best.pt` | 對照組（wandb 版） | |
| `reproduce_pointbert_8kpts/checkpoint_best.pt` | 對照組（8192pt 版） | |

### 建議帶回

- **必帶**：`RAG2_Stage2/checkpoint_best.pt`（一個就能跑 test.py）
- **想從 Stage 2 續訓**：加帶 `RAG_Stage1/checkpoint_best.pt` 或 `RAG2_Stage1/checkpoint_best.pt`
- **要對照原版效果**：加帶 `reproduce_pointbert/checkpoint_best.pt`

---

## P2-2：Objaverse 資料 — 部分有，建議放棄

### 現況

- **沒有** ULIP-2 用的 colored 點雲（yaml 指 Salesforce 內部 `/export/einstein-vision-hs/...`，本機拿不到）
- **只有** `/home/klooom/cheng/3d_retrival/objaverse-rendering/views/` — 8633 個資料夾，每個含 000.png ~ N.png（是 objaverse-rendering 這個工具的多視角 PNG，**不是 ULIP-2 需要的 colored 點雲格式**）

### 建議

放棄 Objaverse LVIS 評估，不影響主線（IKEA 產品線 + ShapeNet zero-shot 都不需要）。要用就得自己重新做 colored 點雲。

---

## 一鍵複製指令（rsync 拉檔）

在 **3090 上執行**（假設目標 repo 根目錄為 `/mnt/P300/repos/ULIP/`）：

```bash
# ===== P0-1 + P0-2：訓練骨架 =====
rsync -avz klooom@4090:/home/klooom/cheng/3d_retrival/ULIP/build_vectors.py \
          klooom@4090:/home/klooom/cheng/3d_retrival/ULIP/app_ikea_retrieval.py \
          /mnt/P300/repos/ULIP/

rsync -avz klooom@4090:/home/klooom/cheng/3d_retrival/ULIP/data/dataset_catalog.json \
          klooom@4090:/home/klooom/cheng/3d_retrival/ULIP/data/ikea_ulip.yaml \
          klooom@4090:/home/klooom/cheng/3d_retrival/ULIP/data/ikea_ulip.py \
          /mnt/P300/repos/ULIP/data/

# ===== P1：TRELLIS submodule（若 3090 沒有） =====
rsync -avz klooom@4090:/home/klooom/cheng/3d_retrival/ULIP/TRELLIS/ \
          /mnt/P300/repos/ULIP/TRELLIS/

# ===== P0-1：已跑出的向量（可省重跑步驟） =====
mkdir -p /mnt/P300/data/ikea_data/vectors
rsync -avz klooom@4090:/mnt/data1/ikea_data/vectors/ \
          /mnt/P300/data/ikea_data/vectors/

# ===== P2-1：RAG checkpoint =====
mkdir -p /mnt/P300/repos/ULIP_RAG/outputs/RAG2_Stage2
rsync -avz klooom@4090:/home/klooom/cheng/3d_retrival/ULIP_RAG/outputs/RAG2_Stage2/checkpoint_best.pt \
          /mnt/P300/repos/ULIP_RAG/outputs/RAG2_Stage2/
```

---

## 3090 驗證 checklist（照順序做）

```bash
# 1. dataset_catalog.json 有 ikea_ulip 條目
python -c "import json; c=json.load(open('data/dataset_catalog.json')); assert 'ikea_ulip' in c; print('OK')"

# 2. main_ikea.py 建 dataset 不再 KeyError
python -c "
from data.dataset_3d import Dataset_3D
import argparse
args = argparse.Namespace(pretrain_dataset_name='ikea_ulip', validate_dataset_name='ikea_ulip',
                          pretrain_dataset_prompt='shapenet_64', validate_dataset_prompt='shapenet_64',
                          npoints=8192, use_height=False, model='ULIP_PointBERT')
ds = Dataset_3D(args, None, 'train', None)
print('dataset len =', len(ds.dataset))
"

# 3. build_vectors.py 能建向量（可先 dry-run 5 筆）
python build_vectors.py --json_dir data/ulip_output/json --out_dir /tmp/vectors_test \
    --ckpt outputs/ULIP2_ikea_b/checkpoint_last.pt --debug_paths --debug_limit 5

# 4. app import 無誤
python -c "import app_ikea_retrieval; print('OK')"

# 5. app 啟動（用 4090 帶來的 vectors）
export USE_MONGO=0
export VEC_DIR=/mnt/P300/data/ikea_data/vectors
export CKPT=/mnt/P300/data/ULIP/checkpoint_last.pt
uvicorn app_ikea_retrieval:app --host 0.0.0.0 --port 8000
```

---

## 附：4090 上尚未確認 / 有疑慮的地方

1. **4090 沒有 `.env` / systemd unit / launch script** — 學長應該是手動 uvicorn 跑，沒正式部署設定
2. **`microsoft/TRELLIS-image-large` HuggingFace 權重** 會在 `/post/upload` 第一次觸發時下載（幾 GB），3090 若無法對外請先 pre-download 或設 `HF_HOME`
3. **MongoDB 是選配** — 4090 default `USE_MONGO=1` 指 `localhost:27017`，3090 沒 mongo 就 `USE_MONGO=0`（僅影響用戶端商品 metadata 查詢，檢索本身不受影響）
4. **RAG corpus 家具限定的問題**（僅 1095 條）— 見 `ULIP_RAG/PROJECT_REPORT.md §10`，若要延伸到非家具類別要重建 corpus

---

**文件結束**。如果 3090 上跑起來還有卡住的細節，回頭在這份 md 附加問題，我可以再回 4090 查。
