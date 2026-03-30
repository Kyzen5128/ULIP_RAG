#!/bin/bash
# =============================================================================
# ULIP RAG 訓練快速啟動腳本
# =============================================================================

set -e  # 遇到錯誤立即停止

echo "🚀 ULIP RAG 訓練快速啟動"
echo "======================================"

# 檢查環境
echo "📋 Step 1: 檢查環境..."
conda activate ulip_rag
cd /home/kyzen/ULIP_RAG/ulip_rag

# 環境驗證
python check_environment.py || { echo "❌ 環境檢查失敗！請先修復環境問題。"; exit 1; }

# 檢查數據
echo ""
echo "📊 Step 2: 檢查數據..."
if [ ! -d "data" ]; then
    echo "創建數據符號鏈接..."
    ln -s /mnt/data1/cheng/ULIP/data/data ./data
fi

if [ ! -f "data/rag_corpus/rag_corpus.jsonl" ]; then
    echo "❌ RAG 語料庫未找到！"
    echo "請確認: /mnt/data1/cheng/ULIP/data/data/rag_corpus/rag_corpus.jsonl"
    exit 1
fi

echo "✅ 數據檢查通過"

# 檢查預訓練模型
echo ""
echo "🎯 Step 3: 檢查預訓練模型..."
if [ ! -f "data/initialize_models/slip_base_100ep.pt" ]; then
    echo "⚠️  SLIP 預訓練模型未找到！"
    echo "正在下載... (約 400MB)"
    mkdir -p data/initialize_models
    cd data/initialize_models
    wget https://dl.fbaipublicfiles.com/slip/slip_base_100ep.pt
    cd ../..
fi

echo "✅ 預訓練模型就緒"

# 創建輸出目錄
echo ""
echo "📁 Step 4: 創建輸出目錄..."
mkdir -p outputs/stage1_rag
mkdir -p outputs/stage2_rag

# 詢問用戶
echo ""
echo "======================================"
echo "📝 訓練選項："
echo "  1) 快速測試（1 epoch，~10分鐘）"
echo "  2) Stage 1 完整訓練（100 epochs，~24-36小時）"
echo "  3) Stage 2 完整訓練（150 epochs，需先完成 Stage 1）"
echo "  4) 僅評估（使用已有 checkpoint）"
echo ""
read -p "請選擇 [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "🧪 開始快速測試..."
        CUDA_VISIBLE_DEVICES=0 python main.py \
          --model ULIP_PointBERT \
          --use_rag_adapter \
          --training_strategy staged_1 \
          --pretrain_dataset_name shapenet \
          --pretrain_dataset_prompt shapenet_64 \
          --validate_dataset_name modelnet40 \
          --validate_dataset_prompt modelnet40_64 \
          --rag_corpus_dir /mnt/data1/cheng/ULIP/data/data/rag_corpus \
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
        CUDA_VISIBLE_DEVICES=0 python main.py \
          --model ULIP_PointBERT \
          --use_rag_adapter \
          --training_strategy staged_1 \
          --pretrain_dataset_name shapenet \
          --pretrain_dataset_prompt shapenet_64 \
          --validate_dataset_name modelnet40 \
          --validate_dataset_prompt modelnet40_64 \
          --rag_corpus_dir /mnt/data1/cheng/ULIP/data/data/rag_corpus \
          --rag_top_k 5 \
          --npoints 8192 \
          --batch-size 32 \
          --lr 3e-3 \
          --epochs 100 \
          --warmup-epochs 1 \
          --workers 10 \
          --output-dir ./outputs/stage1_rag \
          --print-freq 50 \
          --wandb
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
        CUDA_VISIBLE_DEVICES=0 python main.py \
          --model ULIP_PointBERT \
          --use_rag_adapter \
          --training_strategy staged_2 \
          --stage1_ckpt_path ./outputs/stage1_rag/checkpoint_best.pt \
          --pretrain_dataset_name shapenet \
          --pretrain_dataset_prompt shapenet_64 \
          --validate_dataset_name modelnet40 \
          --validate_dataset_prompt modelnet40_64 \
          --rag_corpus_dir /mnt/data1/cheng/ULIP/data/data/rag_corpus \
          --rag_top_k 5 \
          --npoints 8192 \
          --batch-size 32 \
          --lr 1e-3 \
          --epochs 150 \
          --warmup-epochs 1 \
          --workers 10 \
          --output-dir ./outputs/stage2_rag \
          --print-freq 50 \
          --wandb
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
        python test.py \
          --model ULIP_PointBERT_RAG \
          --use_rag_adapter \
          --test_ckpt_addr ./outputs/stage2_rag/checkpoint_best.pt \
          --validate_dataset_name modelnet40 \
          --validate_dataset_prompt modelnet40_64 \
          --npoints 8192 \
          --batch-size 64 \
          --rag_corpus_dir /mnt/data1/cheng/ULIP/data/data/rag_corpus \
          --rag_top_k 5 \
          --eval_zero_shot \
          --gpu 0
        ;;
    
    *)
        echo "❌ 無效選擇！"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "✅ 完成！"
