# 去 4090 詢問 / 取檔清單（3090 可跑性缺口）

> 產出：2026-07-10，依 3090 實機 smoke test 結果整理。
> 背景線索：從 3090 上 `ikea/app_ikea_retrieval.py` 的舊預設路徑推斷，**4090 專案根大概在 `/home/klooom/cheng/3d_retrival/`**（底下有 `ULIP` / `ULIP_RAG`），資料在 `/mnt/data1/ikea_data/`。下列搜尋指令以此為起點，找不到就換成 4090 實際專案根。
> 使用方式：按優先序（P0 最急）逐項在 4090 上找、確認、搬回 3090。

---

## 3090 現況一句話

模型層、資料、RAG 檢索、IKEA 訓練 compute、serving 檢索邏輯**都已實測可跑**；卡住的是**幾個「缺設定 / 缺腳本 / 缺 env 整合」**的洞——這份清單就是去 4090 補齊它們。

---

## 🔴 P0-1：建向量的腳本（最重要）

**要什麼**：用訓練好的 ULIP 模型把 733 件 IKEA 家具各自 encode 成 pc / img / txt 三組 512 維向量，並輸出 `.npy` + `.jsonl` + `schema.json` 的腳本。

**為什麼**：3090 的 `app_ikea_retrieval.py` 只會**讀** `vectors_pc.npy` 做檢索，repo 裡完全沒有**產生**它的程式。新增或重訓家具後無法更新檢索庫。

**在 4090 上這樣找**：
```bash
cd /home/klooom/cheng/3d_retrival        # 換成實際專案根
grep -rln "vectors_pc"  --include=*.py .
grep -rln "schema.json" --include=*.py .
grep -rln "encode_pc"   --include=*.py . | xargs grep -l "np.save"
find . -iname '*.py'    | xargs grep -l "meta_pc.jsonl" 2>/dev/null
find . -iname '*.ipynb' | xargs grep -l "vectors_pc\|encode_pc" 2>/dev/null
```
檔名可能是 `build_vectors.py` / `make_index.py` / `encode_ikea.py` / `export_vectors.py` 之類。

**怎麼確認是對的**（打開後應看到）：
- 載入 checkpoint（`ULIP_PointBERT`），跑 `encode_pc` / `encode_image` / `encode_text`；
- 輸出檔名正好是 `vectors_{pc,img,txt}.npy`、`meta_{pc,img,txt}.jsonl`、`schema.json`；
- `meta_*.jsonl` 每行欄位 = `id / json_path / category / caption / preview_image`（3090 實測格式）；
- `schema.json` = `{"modalities":["pc","img","txt"],"dims":{"pc":512,"img":512,"txt":512}}`。

**搬回 3090**：放 `ikea/`（例如 `ikea/build_vectors.py`）。
**順便問**：為什麼 `vectors_txt` 是 732、比 pc/img 的 733 少 1 筆（推測某件 caption 空/編碼失敗被跳過）——看腳本跳過邏輯即知。

---

## 🔴 P0-2：`dataset_catalog.json` 的 `ikea_ulip` 條目

**要什麼**：`dataset_catalog.json` 裡對應 `ikea_ulip` 的那段。

**為什麼**：3090 的 `main_ikea.py` 走 `get_dataset('ikea_ulip')` → `Dataset_3D` 會查 catalog，但 3090 的 catalog 只有 `shapenet/modelnet40/objaverse_lvis_colored`，**缺 `ikea_ulip`** → `KeyError`（已實測）。訓練 compute 本身（forward+loss+backward）3090 已驗證可跑，只差這條 key。

**在 4090 上這樣找**：
```bash
grep -rn "ikea_ulip" --include=*.json .
cat <專案>/data/configs/dataset_catalog.json   # 或 data/dataset_catalog.json
```
**怎麼確認**：看 `ikea_ulip` 段的 `config` 指向哪個 yaml、`usage` 值。

**搬回 3090**：補進 `core/data/configs/dataset_catalog.json`。若 4090 也沒有，直接用這段（config 指到 3090 已存在的 symlink yaml）：
```json
"ikea_ulip": { "config": "./data/configs/ikea_ulip.yaml", "train": "train", "test": "test", "usage": "train" }
```

---

## 🟡 P1：serving 部署形態 + 啟動設定

**要什麼**：(a) 4090 上 `app_ikea_retrieval.py` 用哪個 conda env、怎麼啟動；(b) 有沒有 `.env` / 啟動腳本 / systemd。

**為什麼**：3090 的 app 在 top-level `import trellis`，但 `trellis` 在 `/home/kyzen/TRELLIS`（Python 3.11 repo），而 ULIP 需要的 `timm/faiss/sentence_transformers` 在 `ulip`（Python 3.9）env——**3090 現況沒有單一 env 同時滿足**。需知道 4090 怎麼解：單一大 env？還是「文字檢索」與「上傳生成3D」拆兩個服務？

**在 4090 上這樣找**：
```bash
grep -rn "app_ikea_retrieval:app\|uvicorn\|VEC_DIR\|CKPT=" --include=*.sh --include=*.service --include=*.env --include=*.md . 2>/dev/null
find . -name '*.env' -o -name '*.service' -o -name 'run*.sh' -o -name 'start*.sh' 2>/dev/null
systemctl list-units 2>/dev/null | grep -i ikea
conda env list
conda list -n <跑app的env> | grep -iE "trellis|timm|faiss|sentence|fastapi|uvicorn|open_clip"
```
**我要的答案**：
- 那個 env 是否 **trellis 和 timm/faiss 都有**？（是 → 3090 照它建合併 env）
- 或**兩個服務分開跑**？（是 → 3090 把 app 的 trellis import 改 lazy，`/search/text` 在 ulip env 跑；`/post/upload` 另開）
- `CKPT / VEC_DIR / ULIP_OUTPUT / CUSTOM_OUTPUT` 四個環境變數在 4090 設成什麼值。

**搬回 3090**：把 `.env`/啟動腳本搬來，路徑改成 `/mnt/P300/data/...`。

---

## 🟢 P2-1：研究線 RAG 的 stage1 / stage2 checkpoint（選配）

**要什麼**：訓練好的 RAG checkpoint。
**為什麼**：3090 `core/outputs/` 是空的；跑 `test.py`（RAG 評估）或 Stage2（`--stage1_ckpt_path`）才需要。**IKEA 產品線不需要**（它用已在的 `/mnt/P300/data/ULIP/checkpoint_last.pt`）。

**在 4090 上這樣找**：
```bash
find /home/klooom/cheng/3d_retrival -path '*outputs*' -name 'checkpoint_best.pt' 2>/dev/null
ls -la <專案>/outputs/RAG*_Stage*/     # 4090 筆記提過 RAG_Stage1 / RAG2_Stage2
```
**怎麼確認**：`torch.load(...)['args']` 的 `model==ULIP_PointBERT`、`use_rag_adapter==True`、`training_strategy`（staged_1/2）。
**搬回 3090**：放 `core/outputs/<name>/checkpoint_best.pt`。

---

## 🟢 P2-2：Objaverse 資料（可略）

`Objaverse_Lvis_Colored.yaml` 指 `/export/einstein-vision-hs/...`（Salesforce 內部），4090 八成也拿不到。**建議放棄這個評估集**，不影響主線。只需問一句「4090 上 objaverse 能不能跑」。

---

## 帶去 4090 的三句話重點

1. **最關鍵**：找**建向量腳本**（`grep -rln "vectors_pc\|schema.json" --include=*.py`）與 **catalog 的 `ikea_ulip` 條目**。
2. **其次**：確認 4090 的 app 是**單一 env 還是拆兩個服務**、四個環境變數設什麼。
3. 這三樣拿到，3090 的 IKEA 訓練＋檢索就能完整重現；研究線 RAG ckpt 為選配。

---

## 附：3090 smoke test 結果對照（哪些已通、哪些卡）

| 流程 | 3090 狀態 | 卡在哪 |
|---|---|---|
| 環境 import / GPU | 🟢 通 | — |
| 研究線 RAG 檢索（faiss+MiniLM）| 🟢 通 | — |
| ULIP_PointBERT build+forward | 🟢 通（timm 1.0.15 無礙）| — |
| ShapeNet / ModelNet / IkeaULIP dataset | 🟢 通 | — |
| IKEA 訓練 compute（forward+loss+backward）| 🟢 通 | — |
| IKEA serving 檢索邏輯 | 🟢 通（查詢語意正確）| — |
| `main_ikea.py` 訓練入口 | 🔴 斷 | **P0-2** catalog 缺 ikea_ulip |
| 完整 serving app import | 🔴 斷 | **P1** trellis/ULIP 跨 env |
| `/post/upload` 生成 3D | 🔴 斷 | **P1** 同上 |
| 重建 `vectors_*.npy` | 🔴 無法 | **P0-1** 腳本缺 |
| Objaverse 評估 | 🔴 無法 | **P2-2** 資料缺（可略）|
