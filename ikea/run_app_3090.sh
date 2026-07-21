#!/bin/bash
# =============================================================================
# IKEA ULIP + Core RAG 檢索服務啟動腳本（3090 專用）
# 用法：bash ikea/run_app_3090.sh [port]        # 預設 port 8000
#
# 為什麼需要這些設定（缺一不可）：
#   - cwd 必須是 core/：模型工廠讀 ./models/pointbert/*.yaml、資料設定讀 ./data/configs/*
#   - PYTHONPATH 含 core/（ULIP 的 models/utils/data）、ikea/（app 模組本身）、
#     /home/kyzen/TRELLIS（app 的 `from trellis...` 由此解析；已實測 ulip env 可載入）
#   - 4 個路徑環境變數：app 內建 default 是 4090 舊路徑，必須覆寫成 3090 的
#   - USE_MONGO=0：3090 未跑 MongoDB；只影響商品 metadata 查詢，檢索不受影響
# =============================================================================
set -e

PORT="${1:-8000}"
REPO=/home/kyzen/ULIP_RAG

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ulip

export PYTHONPATH="$REPO/core:$REPO/ikea:/home/kyzen/TRELLIS${PYTHONPATH:+:$PYTHONPATH}"
export CKPT=/mnt/P300/data/ULIP/checkpoint_last.pt
export VEC_DIR=/mnt/P300/data/ikea_data/vectors
export ULIP_OUTPUT=/mnt/P300/data/ikea_data
export CUSTOM_OUTPUT=/mnt/P300/data/custom_data
export USE_MONGO="${USE_MONGO:-0}"

# Core RAG serving bundle.  The RAG2 enhancer was trained with the archived
# 1,095-row corpus; the newer 8,256-row corpus is not a drop-in replacement.
export RAG_ENABLED="${RAG_ENABLED:-1}"
# RAG is loaded and available at /search/text/rag and /search/text/compare.
# Keep the backward-compatible route on vanilla until an IKEA-specific RAG
# Stage 2 checkpoint passes the retrieval promotion gate.
export RAG_DEFAULT_MODE="${RAG_DEFAULT_MODE:-vanilla}"
export RAG_CORPUS_PROFILE="${RAG_CORPUS_PROFILE:-legacy1095}"
export RAG_STAGE1_CKPT="${RAG_STAGE1_CKPT:-$REPO/core/outputs/RAG2_Stage1/checkpoint_best.pt}"
export RAG_CORPUS_DIR="${RAG_CORPUS_DIR:-/mnt/P300/data/ULIP/4090_cheng_archive_20260715/ULIP_RAG/rag_corpus_1095}"
export RAG_TOP_K="${RAG_TOP_K:-5}"

cd "$REPO/core"
echo "[run_app_3090] port=$PORT  CKPT=$CKPT  RAG_ENABLED=$RAG_ENABLED  RAG_PROFILE=$RAG_CORPUS_PROFILE"
# 注意：ulip env 的 bin/uvicorn entrypoint 是 0-byte 壞檔，須用 python -m uvicorn
exec python -m uvicorn app_ikea_retrieval:app --host 0.0.0.0 --port "$PORT"
