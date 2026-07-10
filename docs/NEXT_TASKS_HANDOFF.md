# ULIP_RAG 3090 — 後續任務交接（給下一個接手的 AI/工程師）

> 產出：2026-07-10。此時點系統已全線可跑（見下方「已驗證狀態」）。
> 必讀順序：本檔 → `docs/PROJECT_REPORT_3090.md`（完整專案理解＋修復記錄）→ `ikea/docs/ulip_rag_phase_plan.md`（產品 roadmap）。
> ⚠️ 舊文件陷阱：`docs/PROJECT_REPORT_OLD_4090.md`、`TRAINING_COMMANDS.txt`、`check_environment.py`、`quick_train*.sh` 內的路徑/env 名全是 4090 舊資訊，**不要照抄**。

---

## 0. 鐵則（每次動手前）

- conda env 一律 **`ulip`**（不是 ulip_rag）；跑任何 core 程式 **cwd 必須是 `/home/kyzen/ULIP_RAG/core`**（相對路徑 `./models/...`、`./data/configs/...` 綁 cwd）。
- 資料實體都在 `/mnt/P300/data`（repo 內 `storage`、`core/data/*` 是 symlink）。
- 上線模型：`/mnt/P300/data/ULIP/checkpoint_last.pt`（純 ULIP_PointBERT，S2T R@1 76.5%）。
- 檢索向量：`/mnt/P300/data/ikea_data/vectors/`（pc/img 733、txt 732——1 件空 caption 是設計跳過）。
- 4090 機器：`ssh cheng4090`（192.168.0.106, klooom，**要密碼**，AI 無法自己連，需請使用者操作）。

## 1. 已驗證狀態（2026-07-10 全部實測綠燈）

| 流程 | 怎麼跑 | 實測結果 |
|---|---|---|
| **Serving API** | `bash ikea/run_app_3090.sh [port]` | 16s ready；/search/text、/get/ids(735)、/get/file 全通 |
| **IKEA 訓練** | `cd core && python main_ikea.py --model ULIP_PointBERT --pretrain_dataset_name ikea_ulip --validate_dataset_name ikea_ulip ...` | 1-epoch smoke（用 `ikea_ulip_smoke`）全通，ckpt/log 正常產出 |
| **向量重建** | `cd core && PYTHONPATH=core python ../ikea/build_vectors.py --json_dir /mnt/P300/data/ikea_data/json --out_dir <新目錄> --ckpt /mnt/P300/data/ULIP/checkpoint_last.pt --pc_sampler fps` | FPS 修正版與部署向量 cos 0.99+；全量約 28 分鐘 |
| **RAG 研究線** | `core/main.py --use_rag_adapter --training_strategy staged_1/2 ...` | 模型/loss/凍結/檢索 smoke 全通（無 stage ckpt，要長訓才有產物）|

今天修過的東西（細節見 PROJECT_REPORT_3090 §10 修復表）：
`ikea/build_vectors.py`（pc 截斷→GPU FPS）、`core/data/configs/dataset_catalog.json`（補 ikea_ulip、ikea_ulip_smoke）、`core/data/configs/ikea_ulip_smoke.yaml`（新）、`ikea/run_app_3090.sh`（新；內含 `python -m uvicorn` 繞過 ulip env 0-byte uvicorn 壞檔）。

## 2. 待辦任務（按優先序）

### T1 ✅ 已完成（2026-07-10）：點雲預採樣（解 FPS 瓶頸）
- 已寫 `ikea/presample_ply.py`（GPU FPS，可續跑）→ 產出 `/mnt/P300/data/ikea_data/ply_8192/`（原 ply 未動）。
- `ikea/ikea_ulip.py:_build_samples` 已接自動 override：偵測到 `ply_8192/<id>.ply` 就用它（log 有 `presampled_used` 計數）；無該目錄則行為完全不變。
- 正式 `ikea_ulip.yaml`（fps）設定下訓練已可用（N==npoints 直接跳過 FPS）。

### T2 🟡 產品 roadmap Phase A/B（見 `ikea/docs/ulip_rag_phase_plan.md`）
- **MongoDB 現況（2026-07-10 已解決）**：docker 容器 `mongodb`（mongo 4.4，綁 127.0.0.1:27017，--auth）。帳密＝`admin` / authSource=admin，**憑證已存 `~/.ulip_mongo.env`（chmod 600，勿進 git）**。pymongo 直連 `mongodb://admin:<pwd>@127.0.0.1:27017/?authSource=admin` 即可（不需 sudo）。
- **furniture_db 實查**：`ikea_product` 2527 筆（完整爬蟲，有 size_options）、`ikea_product_v2_2026q2` 2112 筆、`ikea_product_v3_fresh_2026q2` 2516 筆。
- **✅ Phase A 其實已完成**：`ikea_product_v2_2026q2` 全部 **2112/2112 筆都已有 `meta.dimensions_mm`**。→ 不用再跑 `phase_a_parse_dimensions.py`（除非要重解或補 v3）。
- **→ 直接做 Phase B**：在 `ikea/app_ikea_retrieval.py` 加 `filter_by_dimensions()`（ULIP Top-N → 讀 mongo 的 dimensions_mm → 尺寸過濾 → Top-K）。原則：尺寸住 metadata，不進模型。注意 app 需要能連 mongo（用 ~/.ulip_mongo.env 的 URI；serving 目前 USE_MONGO=0，Phase B 要開）。
- Phase B：在 `ikea/app_ikea_retrieval.py` 加 `filter_by_dimensions()`（ULIP Top-N → 尺寸過濾 → Top-K）。原則：**尺寸住 metadata，不進模型**。

### T3 🟡 git 整理（需使用者同意才 commit）
- 現況：branch `3090`；staged＝ikea V3/PhaseA 批次；modified＝`app_ikea_retrieval.py`(+一行註解)、`dataset_catalog.json`；untracked＝`build_vectors.py`、`run_app_3090.sh`、`ikea_ulip_smoke.yaml`、docs 四份（PROJECT_REPORT_3090 / OLD_4090 / HANDOFF_ASK_4090 / HANDOFF_FOR_3090 / 本檔）。
- 建議一次 commit 今天的修復＋文件。**先問使用者**。

### T4 🟢 選配
- 研究線 RAG 評估：需從 4090 抓 `ULIP_RAG/outputs/RAG2_Stage2/checkpoint_best.pt` → 放 `core/outputs/RAG2_Stage2/` → 跑 `core/test.py`（指令見 PROJECT_REPORT_3090 §9）。
- ~~uvicorn 0-byte~~ ✅ 已根治（force-reinstall，entrypoint 241 bytes、0.39.0）。
- ~~RAGEnhancer 無 key_padding_mask~~ ✅ 已修（`forward` 接受 `doc_padding_mask`、`encode_text_with_rag` 產 mask；無新參數、舊 ckpt 相容；單元+回歸測試通過）。
- ~~main_ikea.py `./data/*.json` 路徑 bug~~ ✅ 已修（改 `./data/configs/`）。
- `corpus/vocab_build/`（Llama 語料）尚未接回 runtime；產品線是否要接 RAG 屬設計決策，問使用者。
- Objaverse 評估：資料拿不到（Salesforce 內部），放棄。

## 3. 常見坑（前人踩過的）

1. **看起來卡死其實是 numpy FPS 在磨**（GPU 0%、worker CPU 100%）→ 見 T1。
2. `main_ikea.py` 的 val split：catalog `usage=train` → val dataset 實際上= train 全集（TRAIN_RATIO 1.0 時 val 為空，程式會自動退回）。沿用 4090 行為，別驚訝。
3. smoke/實驗用 `--pretrain_dataset_name ikea_ulip_smoke`（random sampler，秒級）；正式訓練才用 `ikea_ulip`（fps）。
4. `IkeaULIP` dataset init 會逐顆 trimesh 掃 733 ply（~24s），train+val 各掃一次，屬正常。
5. 4090 交接文件 `HANDOFF_FOR_3090.md` 的「一鍵 rsync」**不要照跑**——會用 4090 扁平結構覆蓋 3090 的巢狀 config（詳見對話記錄；已按最小手術方式處理完）。
6. timm 1.0.15 / torch 2.7.1 與 ULIP 相容（已實測），別被 requirements.txt 的舊版本嚇到回退。

## 4. 快速自檢（接手後先跑這三行確認環境沒被動過）

```bash
cd /home/kyzen/ULIP_RAG/core && conda activate ulip
python -c "import json;c=json.load(open('data/configs/dataset_catalog.json'));assert 'ikea_ulip' in c;print('catalog OK')"
bash ../ikea/run_app_3090.sh 8321 &  sleep 25 && curl -s -X POST localhost:8321/search/text -H 'Content-Type: application/json' -d '{"query":"sofa","top_k":1}' | head -c 200; kill %1
```
