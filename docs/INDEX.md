# ULIP_RAG 文件與入口總索引

> 建立:2026-07-10(第一階段整理)。任何人(含 AI)接手前先讀本檔。
> 完整架構審查:`/home/kyzen/daily_notion/ULIP_RAG_architecture_audit.md`

---

## 一、三條產品/研究線

| 線 | 目的 | 目錄 | 狀態 |
|---|---|---|---|
| **IKEA 產品線** | 爬蟲 → MongoDB → TRELLIS 生 3D → ULIP 訓練 → 向量 → FastAPI 檢索 | `ikea/` | ✅ 全線可跑(2026-07-10 實測) |
| **core RAG 研究線** | ULIP + RAGRetriever/RAGEnhancer 兩階段訓練 | `core/` | ✅ 可訓練;stage ckpt 尚在 4090 |
| **corpus 語料線** | WordNet/ConceptNet + Llama 生成語料(實驗) | `corpus/vocab_build/` | ⏸ 未接回 runtime,步驟 3/4 產物缺 |

## 二、主流程入口(鐵則:conda env = `ulip`,core 程式 cwd 必須 = `core/`)

| 動作 | 指令 |
|---|---|
| Serving API | `bash ikea/run_app_3090.sh [port]` |
| IKEA 訓練(正式 fps) | `cd core && python main_ikea.py --model ULIP_PointBERT --pretrain_dataset_name ikea_ulip --validate_dataset_name ikea_ulip ...` |
| IKEA 訓練(smoke 秒級) | 同上,dataset 改 `ikea_ulip_smoke` |
| 向量重建 | `cd core && PYTHONPATH=$PWD python ../ikea/build_vectors.py --json_dir /mnt/P300/data/ikea_data/json --out_dir <新目錄> --ckpt /mnt/P300/data/ULIP/checkpoint_last.pt --pc_sampler fps` |
| RAG 研究線訓練 | `cd core && python main.py --use_rag_adapter --training_strategy staged_1(→staged_2) ...` |
| RAG 研究線評估 | `cd core && python test.py --test_ckpt_addr <ckpt> --eval_zero_shot ...` |
| ply 預採樣(已完成) | `python ikea/presample_ply.py`(產出 ply_8192/,可續跑) |
| 爬蟲 v3 | `python ikea/fetch_v3_products.py`(需先 export MONGO_URL,見下) |

**MongoDB**:已啟用認證(127.0.0.1:27017)。憑證在 `~/.ulip_mongo.env`(chmod 600,勿進 git)。
腳本用法:`export MONGO_URL="$(grep '^MONGO_URI=' ~/.ulip_mongo.env | cut -d= -f2-)"`。

## 三、文件現況表

### ✅ 現行有效(3090,2026-07-10 更新)

| 文件 | 內容 |
|---|---|
| `docs/PROJECT_REPORT_3090.md` | 專案完整理解 + 修復記錄(主文件) |
| `docs/NEXT_TASKS_HANDOFF.md` | 交接:已驗證狀態、待辦 T1-T4、常見坑 |
| `docs/USAGE_AND_FIXES_3090.md` | 使用流程與修復細節 |
| `ikea/docs/ulip_rag_phase_plan.md` | 產品 roadmap(Phase A ✅ / Phase B 待做) |
| `docs/HANDOFF_ASK_4090.md` | 要向 4090 索取的清單 |

### ⚠️ 歷史文件(路徑/env 名已過時,勿照抄)

| 文件 | 過時原因 |
|---|---|
| `docs/PROJECT_REPORT_OLD_4090.md` | 4090 舊報告(原 PROJECT_REPORT.md) |
| `docs/TRAINING_COMMANDS.txt` | 4090 指令(env 名 ulip_rag、/mnt/data1) |
| `docs/ENVIRONMENT_SETUP.md`、`docs/CLAUDE_USAGE_GUIDE.md`、`docs/RESEARCH_GUIDE.md` | 4090 時代 |
| `docs/HANDOFF_FOR_3090.md` | ⚠️ 其中「一鍵 rsync」會覆蓋 3090 configs,**勿執行** |
| `docs/ikea_pipeline_code_detail.md` | 4090 時代 pipeline 細節 |

## 四、DEPRECATED 檔案清單(已加檔頭註記,勿引用勿照跑)

| 檔案 | 原因 |
|---|---|
| `core/models/ULIP_models_rag.py` | 舊版 RAG 模型,無人 import;現役在 ULIP_models.py:473-883 |
| `core/models/rag_adapter.py` / `rag_generator.py` / `rag_enhancer.py` / `retriever.py` | 獨立版 RAG 元件,無人 import |
| `core/data/dataset_3d_rag.py` | 舊版 RAG dataset,無人 import |
| `core/scripts/build_rag_index.py` | CLIP 512d,與現役 384d faiss 不相容,**照跑會蓋掉現役索引** |
| `core/quick_train.sh`、`core/check_environment.py`、`core/test_rag_functions.py` | 4090 env/路徑 |
| `scripts/quick_train_portable.sh` | 同上(env 名 ulip_rag) |

## 五、關鍵資產位置

| 資產 | 位置 |
|---|---|
| 上線模型 | `/mnt/P300/data/ULIP/checkpoint_last.pt`(S2T R@1 76.5%) |
| 檢索向量 | `/mnt/P300/data/ikea_data/vectors/`(pc/img 733、txt 732) |
| IKEA 3D 資料 | `/mnt/P300/data/ikea_data/{ply(26G),ply_8192,json,glb,images}` |
| RAG 語料+索引 | `core/rag_corpus/`(jsonl 8256 條 + faiss 384d;**faiss 生成腳本已佚失,此檔不可再生,已納入 git**) |
| SLIP/PointBERT 初始權重 | `/mnt/P300/data/ULIP/ULIP-1/initialize_models/` |
| MongoDB | furniture_db:ikea_product(2527)/v2_2026q2(2112,含 dimensions_mm)/v3_fresh(2516) |
