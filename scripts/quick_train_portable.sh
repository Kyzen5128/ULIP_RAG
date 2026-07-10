#!/bin/bash
# =====================================================================
# ⚠️ DEPRECATED(2026-07-10 標記):conda env 名 ulip_rag 已不存在(現為 ulip),
# 「通用版」假設已失效。3090 正確指令見 docs/NEXT_TASKS_HANDOFF.md §1。勿照跑。
# =====================================================================
# =============================================================================
# ULIP RAG 訓練快速啟動腳本（通用版本）
# 適用於任何新環境，自動檢測項目路徑
# =============================================================================

set -e  # 遇到錯誤立即停止

echo "🚀 ULIP RAG 訓練快速啟動"
echo "======================================"

# 獲取腳本所在目錄（項目根目錄）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

echo "📂 項目路徑: $PROJECT_ROOT"

# 檢查環境
echo ""
echo "📋 Step 1: 檢查 Conda 環境..."
if ! command -v conda &> /dev/null; then
    echo "❌ Conda 未安裝！請先安裝 Miniconda。"
    exit 1
fi

# 激活環境
eval "$(conda shell.bash hook)"
conda activate ulip_rag || { echo "❌ 環境 ulip_rag 不存在！請先創建環境。"; exit 1; }

cd "$PROJECT_ROOT"

# 環境驗證（如果存在檢查腳本）
if [ -f "check_environment.py" ]; then
    python check_environment.py || { echo "⚠️  環境檢查有警告，繼續執行..."; }
fi

# 自動檢測數據目錄
echo ""
echo "📊 Step 2: 檢查數據目錄..."

# 檢查是否已有符號鏈接或實際目錄
if [ -L "data" ]; then
    DATA_DIR=$(readlink -f data)
    echo "✓ 數據目錄（符號鏈接）: $DATA_DIR"
elif [ -d "data" ]; then
    DATA_DIR="$PROJECT_ROOT/data"
    echo "✓ 數據目錄: $DATA_DIR"
else
    echo "⚠️  數據目錄未找到！"
    echo "請手動創建符號鏈接，例如："
    echo "  ln -s /mnt/data1/your_path/data ./data"
    exit 1
fi

# 檢查 RAG 語料庫
RAG_CORPUS_DIR="$DATA_DIR/rag_corpus"
if [ ! -f "$RAG_CORPUS_DIR/rag_corpus.jsonl" ]; then
    echo "❌ RAG 語料庫未找到！"
    echo "期望位置: $RAG_CORPUS_DIR/rag_corpus.jsonl"
    echo ""
    echo "請執行以下步驟："
    echo "  1. cd scripts"
    echo "  2. python build_rag_corpus.py"
    echo "  3. python build_rag_index.py"
    exit 1
fi

echo "✅ RAG 語料庫就緒: $RAG_CORPUS_DIR"

# 檢查預訓練模型
echo ""
echo "🎯 Step 3: 檢查預訓練模型..."
PRETRAIN_DIR="$DATA_DIR/initialize_models"

if [ ! -f "$PRETRAIN_DIR/slip_base_100ep.pt" ]; then
    echo "⚠️  SLIP 預訓練模型未找到！"
    echo "期望位置: $PRETRAIN_DIR/slip_base_100ep.pt"
    echo ""
    read -p "是否下載？(約 2GB) [y/N]: " download_choice
    if [[ $download_choice =~ ^[Yy]$ ]]; then
        mkdir -p "$PRETRAIN_DIR"
        cd "$PRETRAIN_DIR"
        wget https://dl.fbaipublicfiles.com/slip/slip_base_100ep.pt
        cd "$PROJECT_ROOT"
    else
        echo "⚠️  跳過下載，訓練可能失敗"
    fi
else
    echo "✅ 預訓練模型就緒"
fi

# 創建輸出目錄
echo ""
echo "📁 Step 4: 創建輸出目錄..."
mkdir -p outputs/stage1_rag
mkdir -p outputs/stage2_rag

# 顯示訓練選項
echo ""
echo "======================================"
echo "📝 訓練選項："
echo "  1) 快速測試（1 epoch，~10分鐘）"
echo "  2) Stage 1 完整訓練（100 epochs，~24-36小時）"
echo "  3) Stage 2 完整訓練（150 epochs，需先完成 Stage 1）"
echo "  4) 僅評估（使用已有 checkpoint）"
echo ""
read -p "請選擇 [1-4]: " choice

# 檢測可用 GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "✓ 檢測到 $GPU_COUNT 個 GPU"
    read -p "使用哪個 GPU？(0-$((GPU_COUNT-1))) [0]: " GPU_ID
    GPU_ID=${GPU_ID:-0}
    DEVICE="CUDA_VISIBLE_DEVICES=$GPU_ID"
else
    echo "⚠️  未檢測到 GPU，將使用 CPU（非常慢）"
    DEVICE=""
fi

case $choice in
    1)
        echo ""
        echo "🧪 開始快速測試..."
        $DEVICE python main.py \
          --model ULIP_PointBERT \
          --use_rag_adapter \
          --training_strategy staged_1 \
          --pretrain_dataset_name shapenet \
          --pretrain_dataset_prompt shapenet_64 \
          --validate_dataset_name modelnet40 \
          --validate_dataset_prompt modelnet40_64 \
          --rag_corpus_dir "$RAG_CORPUS_DIR" \
          --rag_top_k 5 \
          --npoints 8192 \
          --batch-size 8 \
          --lr 3e-3 \
          --epochs 1 \
          --workers 4 \
          --output-dir ./outputs/test_stage1 \
          --print-freq 1
        ;;
    
    2)
        echo ""
        echo "🏋️ 開始 Stage 1 訓練..."
        echo "預計時間: 24-36 小時 (單GPU)"
        echo ""
        read -p "是否啟用 WandB 監控？[y/N]: " wandb_choice
        
        WANDB_FLAG=""
        if [[ $wandb_choice =~ ^[Yy]$ ]]; then
            WANDB_FLAG="--wandb"
        fi
        
        $DEVICE python main.py \
          --model ULIP_PointBERT \
          --use_rag_adapter \
          --training_strategy staged_1 \
          --pretrain_dataset_name shapenet \
          --pretrain_dataset_prompt shapenet_64 \
          --validate_dataset_name modelnet40 \
          --validate_dataset_prompt modelnet40_64 \
          --rag_corpus_dir "$RAG_CORPUS_DIR" \
          --rag_top_k 5 \
          --npoints 8192 \
          --batch-size 32 \
          --lr 3e-3 \
          --epochs 100 \
          --warmup-epochs 1 \
          --workers 10 \
          --output-dir ./outputs/stage1_rag \
          --print-freq 50 \
          $WANDB_FLAG
        ;;
    
    3)
        # 檢查 Stage 1 checkpoint
        if [ ! -f "outputs/stage1_rag/checkpoint_best.pt" ]; then
            echo "❌ 未找到 Stage 1 checkpoint！"
            echo "請先完成 Stage 1 訓練。"
            exit 1
        fi
        
        echo ""
        echo "🏋️ 開始 Stage 2 訓練..."
        echo "預計時間: 36-48 小時 (單GPU)"
        echo ""
        read -p "是否啟用 WandB 監控？[y/N]: " wandb_choice
        
        WANDB_FLAG=""
        if [[ $wandb_choice =~ ^[Yy]$ ]]; then
            WANDB_FLAG="--wandb"
        fi
        
        $DEVICE python main.py \
          --model ULIP_PointBERT \
          --use_rag_adapter \
          --training_strategy staged_2 \
          --stage1_ckpt_path ./outputs/stage1_rag/checkpoint_best.pt \
          --pretrain_dataset_name shapenet \
          --pretrain_dataset_prompt shapenet_64 \
          --validate_dataset_name modelnet40 \
          --validate_dataset_prompt modelnet40_64 \
          --rag_corpus_dir "$RAG_CORPUS_DIR" \
          --rag_top_k 5 \
          --npoints 8192 \
          --batch-size 32 \
          --lr 1e-3 \
          --epochs 150 \
          --warmup-epochs 1 \
          --workers 10 \
          --output-dir ./outputs/stage2_rag \
          --print-freq 50 \
          $WANDB_FLAG
        ;;
    
    4)
        # 檢查 Stage 2 checkpoint
        if [ ! -f "outputs/stage2_rag/checkpoint_best.pt" ]; then
            echo "❌ 未找到 Stage 2 checkpoint！"
            echo "請先完成訓練。"
            exit 1
        fi
        
        echo ""
        echo "📊 開始評估..."
        $DEVICE python test.py \
          --model ULIP_PointBERT_RAG \
          --use_rag_adapter \
          --test_ckpt_addr ./outputs/stage2_rag/checkpoint_best.pt \
          --validate_dataset_name modelnet40 \
          --validate_dataset_prompt modelnet40_64 \
          --npoints 8192 \
          --batch-size 64 \
          --rag_corpus_dir "$RAG_CORPUS_DIR" \
          --rag_top_k 5 \
          --eval_zero_shot \
          --gpu ${GPU_ID:-0}
        ;;
    
    *)
        echo "❌ 無效選擇！"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "✅ 完成！"
echo ""
echo "📊 查看結果："
echo "  - 訓練日誌: outputs/*/log.txt"
echo "  - Checkpoint: outputs/*/checkpoint_*.pt"
echo ""
