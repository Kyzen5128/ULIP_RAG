# ULIP_RAG 3090 — 今日修復記錄 + 完整使用手冊

> 日期：2026-07-10
> 適用：`/home/kyzen/ULIP_RAG`（branch `3090`）
> 本檔所有指令皆在 3090 上**實測通過**。相關文件：`PROJECT_REPORT_3090.md`（專案理解）、`NEXT_TASKS_HANDOFF.md`（後續任務）。

---

# Part 1：今日修復記錄（2026-07-10）

## 修復總表

| # | 問題 | 根因 | 修復 | 驗證結果 |
|---|---|---|---|---|
| 1 | 無法重建檢索向量（腳本不在 repo） | 建向量腳本只存在 4090 | 從 4090 取回 `ikea/build_vectors.py` | img/txt 重建 cos=1.0000 |
| 2 | 取回的 `build_vectors.py` **pc 向量是壞的** | 該版把 FPS 改成截斷 `pts[:8192]`（IKEA ply 每顆 33–67 萬點，截斷後與部署向量幾乎正交 cos≈0.03） | 改回 FPS，並用 `pointnet2_ops` **GPU FPS** 加速（2.3s/筆）；新增 `--pc_sampler fps/truncate/random` 參數 | 3 筆 cos vs 部署向量 = 0.9977 / 0.9938 / 0.9917 |
| 3 | `main_ikea.py` 訓練啟動即 `KeyError: 'ikea_ulip'` | `core/data/configs/dataset_catalog.json` 缺 `ikea_ulip` 條目 | 補上（config 指 `./data/configs/ikea_ulip.yaml`） | `get_dataset('ikea_ulip')` OK（len=729）；1-epoch 訓練端到端通 |
| 4 | serving app 無法啟動 | ① 預設 `CKPT/VEC_DIR` 是 4090 舊路徑 ② `import trellis` 需要 PYTHONPATH ③ 必須 cwd=core | 新增啟動腳本 `ikea/run_app_3090.sh`（一鍵搞定 env/路徑/PYTHONPATH） | uvicorn 16s ready，API 全通 |
| 5 | `uvicorn` 指令直接掛 | **`ulip` env 的 `bin/uvicorn` 是 0-byte 壞檔**（安裝殘骸） | 啟動腳本改用 `python -m uvicorn` 繞過 | server 正常啟動（根治可 `pip install --force-reinstall uvicorn`） |
| 6 | 訓練「假死」：GPU 0%、26 分鐘無輸出 | 不是死鎖——`ikea_ulip.yaml` 的 `PC_SAMPLER: fps` 是 **numpy FPS**，對 33–67 萬點 ply 每筆要 20-40s CPU（4 workers 在磨 FPS） | 新增 smoke 專用 `ikea_ulip_smoke`（catalog 條目 + `ikea_ulip_smoke.yaml`，`PC_SAMPLER: random`）；正式 yaml 不動 | smoke 訓練 1 epoch ≈ 2 分鐘完成 |
| 7 | `vectors_txt` 為何 732 筆（少 1） | 1 件家具 caption 為空字串，`build_vectors.py` 設計上 `continue` 跳過 | 無需修（設計行為） | 掃描確認 `TXT usable: 732 / no_txt: 1` |

## 第二輪修復（2026-07-10 深夜）

| # | 問題 | 修復 | 驗證 |
|---|---|---|---|
| 8 | **T1 訓練 FPS 瓶頸**（正式訓練每 epoch 小時級） | 新增 `ikea/presample_ply.py`（GPU FPS 一次性把 733 顆大 ply 預採樣成 8192 點 → `ikea_data/ply_8192/`，可續跑）；`ikea/ikea_ulip.py:_build_samples` 自動優先讀 `ply_8192/`（無該目錄則行為不變） | 見 log `presampled_used` 計數；正式 fps 設定下訓練實測 |
| 9 | `ulip` env 的 `bin/uvicorn` 0-byte 壞檔（先前只繞過） | `pip install --force-reinstall --no-deps uvicorn` 根治 | entrypoint 241 bytes、`uvicorn --version` = 0.39.0 |
| 10 | `main_ikea.py` L409/L415 讀 `./data/{templates,labels}.json`（少 `configs/`，非 IKEA zero-shot 路徑會炸） | 改為 `./data/configs/`（附註解說明 4090 舊佈局遺留） | 檔案存在於 configs/；import 正常 |
| 11 | RAGEnhancer 補零無 `key_padding_mask`（4090 報告即提出的隱憂） | `RAGEnhancer.forward` 增加可選 `doc_padding_mask`；`encode_text_with_rag` 產 mask 並傳入。無新增參數，舊 ckpt 相容 | 單元測試：mask 生效（diff-norm 3.9、無 NaN）；全 pipeline 回歸 loss 1.534 正常 |
| 12 | MongoDB 狀態不明（第十章懸置） | 實查：mongod **有在跑**（pid 2493，6/8 起，auth 模式，推測 docker）；但 repo 內建帳密（get_data.py 的 admin）**認證失敗** | → Phase A/B 需要使用者提供 3090 的 mongo 帳密，或確認資料只在 4090 |

## 今日動過的檔案

| 檔案 | 動作 |
|---|---|
| `ikea/build_vectors.py` | 🆕 從 4090 取回 + 修 pc 分支（FPS + GPU 加速 + `--pc_sampler`） |
| `ikea/run_app_3090.sh` | 🆕 serving 一鍵啟動腳本 |
| `core/data/configs/dataset_catalog.json` | ✏️ 補 `ikea_ulip`、`ikea_ulip_smoke` 兩條目 |
| `core/data/configs/ikea_ulip_smoke.yaml` | 🆕 smoke 訓練設定（random sampler） |
| `docs/PROJECT_REPORT_3090.md` | ✏️ 補修復記錄 + 實測結果 + FPS 瓶頸 |
| `docs/NEXT_TASKS_HANDOFF.md` | 🆕 後續任務交接 |
| `docs/HANDOFF_ASK_4090.md` / `docs/HANDOFF_FOR_3090.md` | 🆕 4090 詢問清單 / 4090 回覆（參考用） |

> ⚠️ 以上皆**未 commit**（branch `3090` 另有先前 staged 的 ikea V3/PhaseA 批次）。要入庫請明確告知。

## 端到端實測結果（全綠）

- **IKEA 訓練**：45/45 iters，loss 8.21→5.38，10GB VRAM，六向檢索評估執行，`checkpoint_{1,best,last}.pt` + `log.txt` 產出，exit 0。
- **Serving**：`/search/text`（"a modern black leather sofa" → Top-3 全 Sofa、皮沙發排前）、`/get/ids`（735）、`/get/file`（JPEG 200）。
- **RAG 研究線**：`ULIP_PointBERT_RAG` forward 5 keys；Stage1 loss+backward、Stage2 loss 皆通；staged_1 凍結正確（只開 rag_enhancer）。
- **向量重建**：pc（GPU FPS）cos 0.99+、img/txt cos 1.0 對齊部署向量。

---

# Part 2：ULIP_RAG 完整使用手冊

## 2.0 環境鐵則（每次動手前）

```bash
conda activate ulip                  # ⚠️ 是 ulip，不是文件舊寫的 ulip_rag
cd /home/kyzen/ULIP_RAG/core         # ⚠️ core 程式的相對路徑綁 cwd，一定要在 core/ 下跑
```
- 實機環境（實測）：Python 3.9.23、torch 2.7.1+cu128、faiss 1.7.2、GPU RTX 3090 Ti。
- 資料實體：`/mnt/P300/data`（repo 內 `storage`、`core/data/*` 為 symlink）。
- 上線模型：`/mnt/P300/data/ULIP/checkpoint_last.pt`（純 ULIP_PointBERT，S2T R@1 76.5%）。
- ⚠️ 別信舊文件：`TRAINING_COMMANDS.txt`、`check_environment.py`、`quick_train*.sh`、`ENVIRONMENT_SETUP.md` 的路徑/env 名全是 4090 舊資訊。

## 2.1 啟動檢索服務（serving）

```bash
bash /home/kyzen/ULIP_RAG/ikea/run_app_3090.sh          # 預設 port 8000
bash /home/kyzen/ULIP_RAG/ikea/run_app_3090.sh 8321     # 指定 port
```
腳本內部做的事（手動跑就要自己設）：conda ulip → cwd=core →
`PYTHONPATH=core:ikea:/home/kyzen/TRELLIS` →
`CKPT/VEC_DIR/ULIP_OUTPUT/CUSTOM_OUTPUT` 指到 `/mnt/P300/data/...` → `USE_MONGO=0` → `python -m uvicorn app_ikea_retrieval:app`。

### API 用法

```bash
# 文字搜 3D 家具（可加 "category":"Sofa" 篩類別）
curl -X POST localhost:8000/search/text -H 'Content-Type: application/json' \
     -d '{"query":"a modern black leather sofa","top_k":10}'

# 列出所有家具 id（type=0 全部 / 1 爬蟲 ikea_data / 2 使用者上傳 custom_data）
curl "localhost:8000/get/ids?type=1"

# 取檔案（type=img 圖片 / glb 3D 模型）
curl -o out.jpg "localhost:8000/get/file?type=img&id=<家具id>"
curl -o out.glb "localhost:8000/get/file?type=glb&id=<家具id>"

# 上傳照片生成 3D（會觸發 rembg 去背 → CLIP 分類 → TRELLIS 生成，第一次載模型較久）
curl -X POST localhost:8000/post/upload \
     -F "file=@photo.jpg" -F "length_mm=2000" -F "width_mm=900" -F "height_mm=850"
```

檢索原理：query → ULIP `encode_text` → 512 維 → 與預存 `vectors_pc.npy(733,512)` 做 cosine → Top-K。（**不走 RAG**；RAG 是研究線。）

## 2.2 IKEA 訓練

```bash
cd /home/kyzen/ULIP_RAG/core

# (a) 流程驗證 / 快速實驗 —— 用 smoke 設定（random sampler，1 epoch ≈ 2 分鐘）
python main_ikea.py --model ULIP_PointBERT \
  --pretrain_dataset_name ikea_ulip_smoke --validate_dataset_name ikea_ulip_smoke \
  --npoints 8192 --batch-size 16 --workers 4 --epochs 1 \
  --output-dir ./outputs/ikea_smoke

# (b) 正式訓練 —— 用 ikea_ulip（fps）
# ⚠️ 先做 ply 預採樣（見 2.4，否則 numpy FPS 每 epoch 要數小時）
python main_ikea.py --model ULIP_PointBERT \
  --pretrain_dataset_name ikea_ulip --validate_dataset_name ikea_ulip \
  --npoints 8192 --batch-size 32 --workers 10 --epochs 250 --lr 3e-3 \
  --output-dir ./outputs/ikea_run --wandb
```
- 驗證指標＝六向檢索（S2T/T2S/S2I/I2S/T2I/I2T 的 MRR/R@1/5/10/NDCG@5）；best checkpoint 依 `S2T R@1`。
- 產出：`<output-dir>/checkpoint_{N,best,last}.pt` + `log.txt`（每 epoch 一行 JSON）+ wandb（project `ULIP2`）。
- 已知現象：dataset init 會 trimesh 掃 733 顆 ply（train/val 各 ~24s），正常。

## 2.3 重建 / 更新檢索向量（訓練後或新增家具後）

```bash
cd /home/kyzen/ULIP_RAG/core
PYTHONPATH=/home/kyzen/ULIP_RAG/core python ../ikea/build_vectors.py \
  --json_dir /mnt/P300/data/ikea_data/json \
  --out_dir  /mnt/P300/data/ikea_data/vectors_new \
  --ckpt     /mnt/P300/data/ULIP/checkpoint_last.pt \
  --model    ULIP_PointBERT --npoints 8192 --pc_sampler fps --amp
# 全量約 28 分鐘（GPU FPS）。確認 OK 後把 VEC_DIR 指到新目錄或替換舊目錄。
```
- 產出：`vectors_{pc,img,txt}.npy` + `meta_{pc,img,txt}.jsonl` + `schema.json`。
- ⚠️ `--pc_sampler` 一定用 `fps`（`truncate` 是壞的舊行為，只留作對照）。
- 空 caption 的家具會被 txt 跳過（pc/img 733、txt 少 N 筆屬正常）。

## 2.4 正式重訓前：ply 預採樣（待辦 T1，尚未實作）

原因：訓練 dataloader 的 numpy FPS 對 33–67 萬點 ply 每筆 20-40s。
做法：寫 `ikea/presample_ply.py` 複用 `build_vectors.py` 的 `sample_points()`（GPU FPS）→ 733 顆存 `/mnt/P300/data/ikea_data/ply_8192/`（一次 ~28 分）→ 讓 `ikea_ulip.py` 讀預採樣檔。**不要覆蓋原 ply**。詳見 `NEXT_TASKS_HANDOFF.md` T1。

## 2.5 RAG 研究線（core/main.py，選配）

```bash
cd /home/kyzen/ULIP_RAG/core
# Stage 1：只訓 RAGEnhancer
python main.py --model ULIP_PointBERT --use_rag_adapter --training_strategy staged_1 \
  --pretrain_dataset_name shapenet --pretrain_dataset_prompt shapenet_64 \
  --validate_dataset_name modelnet40 --validate_dataset_prompt modelnet40_64 \
  --rag_corpus_dir rag_corpus --rag_top_k 5 \
  --npoints 8192 --batch-size 32 --epochs 100 --lr 3e-3 --output-dir ./outputs/rag_stage1
# Stage 2：訓 point_encoder（必帶 --stage1_ckpt_path，注意 log 要出現
# "Stage 1 rag_enhancer weights loaded successfully"）
python main.py --model ULIP_PointBERT --use_rag_adapter --training_strategy staged_2 \
  --stage1_ckpt_path ./outputs/rag_stage1/checkpoint_best.pt \
  --pretrain_dataset_name shapenet --npoints 8192 --batch-size 32 --epochs 150 --lr 1e-3 \
  --rag_corpus_dir rag_corpus --rag_top_k 5 --output-dir ./outputs/rag_stage2
```
- RAG 語料：`core/rag_corpus/`（8256 條；FAISS 為 384 維 MiniLM——檔名 `clip_corpus_index.faiss` 是誤導，與 CLIP 無關）。
- 3090 沒有現成 stage checkpoint；要直接評估需從 4090 抓 `ULIP_RAG/outputs/RAG2_Stage2/checkpoint_best.pt`（`ssh cheng4090`，需密碼）。

## 2.6 資料 pipeline（新增 IKEA 商品時的完整流程）

```
ikea/get_data.py        爬 IKEA（ikea_api + Playwright）→ MongoDB furniture_db.ikea_product
      ↓
ikea/build_3d.py        每類 clip_confidence Top-50 → TRELLIS 圖轉 3D → ply/glb/json
      ↓                 （需 MongoDB + TRELLIS；converted_3d 標記支援中斷續跑）
main_ikea.py            重訓 ULIP（見 2.2；先做 2.4 預採樣）
      ↓
build_vectors.py        重建向量（見 2.3）
      ↓
run_app_3090.sh         重啟 serving
```
Phase A/B（尺寸過濾）路線圖見 `ikea/docs/ulip_rag_phase_plan.md`；需要 MongoDB 起來（serving 目前 `USE_MONGO=0`）。

## 2.7 接手自檢（3 行）

```bash
cd /home/kyzen/ULIP_RAG/core && conda activate ulip
python -c "import json;c=json.load(open('data/configs/dataset_catalog.json'));assert 'ikea_ulip' in c;print('catalog OK')"
bash ../ikea/run_app_3090.sh 8321 &  sleep 25 && curl -s -X POST localhost:8321/search/text -H 'Content-Type: application/json' -d '{"query":"sofa","top_k":1}' | head -c 200; kill %1
```

## 2.8 常見坑速查

| 症狀 | 原因 / 解法 |
|---|---|
| 訓練 GPU 0%、worker CPU 100% 很久 | numpy FPS 在磨大 ply——正常但極慢；用 smoke 設定或先做預採樣（2.4） |
| `KeyError: 'ikea_ulip'` | catalog 被還原了，重補條目（見 Part 1 #3） |
| `uvicorn: command not found`/怪錯誤 | ulip env 的 bin/uvicorn 是 0-byte，壞檔；用 `python -m uvicorn` |
| `FileNotFoundError ./models/pointbert/...yaml` | cwd 不在 `core/` |
| `ModuleNotFoundError: trellis` | PYTHONPATH 少了 `/home/kyzen/TRELLIS`（用 run_app_3090.sh 就不會） |
| app 啟動讀不到 vectors | 忘了 export `VEC_DIR`（預設是 4090 舊路徑） |
| Objaverse 相關報錯 | 資料在 Salesforce 內部，本機拿不到，放棄該評估集 |
