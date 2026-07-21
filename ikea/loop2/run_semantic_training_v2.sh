#!/usr/bin/env bash
set -euo pipefail

REPO=/home/kyzen/ULIP_RAG
VERSION=${ULIP2_IKEA_VERSION:-ikea-us-20260716-v1}
ROOT=${ULIP2_IKEA_ROOT:-/mnt/P300/data/ULIP/ULIP_RAG_2_0}
SNAPSHOT="$ROOT/$VERSION"
DATASET_MANIFEST="$SNAPSHOT/manifests/training_dataset.json"
INIT_CHECKPOINT=${ULIP2_INIT_CHECKPOINT:-/mnt/P300/data/ULIP/checkpoint_last.pt}
OUTPUT_DIR=${ULIP2_OUTPUT_DIR:-$SNAPSHOT/training/semantic_ulip_v2}

test -f "$DATASET_MANIFEST"
test -f "$INIT_CHECKPOINT"

EXPECTED_INIT_SHA=49ac18ab10950412f016c9e023d1c7e9b34250a257d59b70b5590d74207f3e01
ACTUAL_INIT_SHA=$(sha256sum "$INIT_CHECKPOINT" | awk '{print $1}')
if [[ "$ACTUAL_INIT_SHA" != "$EXPECTED_INIT_SHA" ]]; then
  echo "Refusing unrecognized IKEA 1.0 init checkpoint: $ACTUAL_INIT_SHA" >&2
  exit 2
fi

TRAINING_ROWS=$(/home/kyzen/miniconda3/envs/ulip/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["training_rows"])' "$DATASET_MANIFEST")
if (( TRAINING_ROWS < 100 )); then
  echo "Refusing undersized dataset: training_rows=$TRAINING_ROWS" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/core:$REPO/ikea${PYTHONPATH:+:$PYTHONPATH}"
exec /home/kyzen/miniconda3/envs/ulip/bin/python ikea/main_ikea.py \
  --model ULIP_PointBERT \
  --npoints 8192 \
  --lr 3e-3 \
  --batch-size 32 \
  --epochs 250 \
  --save-freq 5 \
  --archive-freq 50 \
  --workers 8 \
  --output-dir "$OUTPUT_DIR" \
  --pretrain_dataset_name ikea_ulip_v2 \
  --validate_dataset_name ikea_ulip_v2 \
  --init-checkpoint "$INIT_CHECKPOINT"
