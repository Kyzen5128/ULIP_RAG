#!/usr/bin/env bash
set -euo pipefail

REPO=/home/kyzen/ULIP_RAG
SNAPSHOT=/mnt/P300/data/ULIP/ULIP_RAG_2_0/ikea-us-20260716-v1
PYTHON=/home/kyzen/miniconda3/envs/ulip/bin/python
CONFIG="$REPO/ikea/loop2/ikea_ulip_v2.yaml"
TRAIN_CORPUS="$SNAPSHOT/corpus/product_corpus_train.jsonl"
TRAIN_INDEX="$SNAPSHOT/corpus/corpus_index_train.faiss"
OUTPUT_ROOT="$SNAPSHOT/checkpoints/rag"

: "${BASE_ULIP_CHECKPOINT:?Set BASE_ULIP_CHECKPOINT to the accepted ULIP_RAG 2.0 semantic checkpoint}"
EXPECTED_CORPUS_SHA="$(sha256sum "$TRAIN_CORPUS" | awk '{print $1}')"
EXPECTED_INDEX_SHA="$(sha256sum "$TRAIN_INDEX" | awk '{print $1}')"
printf 'train corpus sha256=%s\ntrain index sha256=%s\n' "$EXPECTED_CORPUS_SHA" "$EXPECTED_INDEX_SHA"

cd "$REPO"
export PYTHONPATH="$REPO:$REPO/core:$REPO/ikea${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" ikea/main_ikea_rag.py \
  --stage stage1 \
  --dataset-config "$CONFIG" \
  --rag-corpus "$TRAIN_CORPUS" \
  --rag-index "$TRAIN_INDEX" \
  --require-rag-split train \
  --base-ulip-checkpoint "$BASE_ULIP_CHECKPOINT" \
  --output-dir "$OUTPUT_ROOT/stage1" \
  --epochs 50 --batch-size 16 --workers 8 --npoints 8192 --lr 3e-4

STAGE1_CHECKPOINT="$OUTPUT_ROOT/stage1/checkpoint_best.pt"
test -s "$STAGE1_CHECKPOINT"
"$PYTHON" ikea/main_ikea_rag.py \
  --stage stage2 \
  --dataset-config "$CONFIG" \
  --rag-corpus "$TRAIN_CORPUS" \
  --rag-index "$TRAIN_INDEX" \
  --require-rag-split train \
  --base-ulip-checkpoint "$BASE_ULIP_CHECKPOINT" \
  --stage1-checkpoint "$STAGE1_CHECKPOINT" \
  --output-dir "$OUTPUT_ROOT/stage2" \
  --epochs 50 --batch-size 16 --workers 8 --npoints 8192 --lr 3e-4
