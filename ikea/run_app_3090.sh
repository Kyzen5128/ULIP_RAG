#!/bin/bash
# =============================================================================
# IKEA ULIP 檢索服務啟動腳本（3090 專用）
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

cd "$REPO/core"
echo "[run_app_3090] port=$PORT  CKPT=$CKPT"
# 注意：ulip env 的 bin/uvicorn entrypoint 是 0-byte 壞檔，須用 python -m uvicorn
exec python -m uvicorn app_ikea_retrieval:app --host 0.0.0.0 --port "$PORT"
