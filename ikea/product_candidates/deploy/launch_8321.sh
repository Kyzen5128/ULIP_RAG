#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/home/kyzen/ULIP_RAG"
readonly SERVICE_DIR="${REPO_ROOT}/ikea"
readonly BIND_HOST="127.0.0.1"
readonly BIND_PORT="8321"
readonly WORKERS="1"

PYTHON_BIN="${PRODUCT_CANDIDATES_PYTHON:-/home/kyzen/miniconda3/envs/ulip/bin/python}"
readonly APP_MODULE="app_product_candidates:app"

: "${PRODUCT_CANDIDATES_PROFILE:?PRODUCT_CANDIDATES_PROFILE must name an immutable deployment profile}"
: "${PRODUCT_CANDIDATES_API_TOKEN:?PRODUCT_CANDIDATES_API_TOKEN must come from the external 0600 EnvironmentFile}"

if [[ "${PRODUCT_CANDIDATES_API_TOKEN}" == *REPLACE* ]] ||
   [[ "${PRODUCT_CANDIDATES_API_TOKEN}" == *CHANGE_ME* ]] ||
   (( ${#PRODUCT_CANDIDATES_API_TOKEN} < 32 )); then
    echo "PRODUCT_CANDIDATES_API_TOKEN is a placeholder or shorter than 32 characters" >&2
    exit 64
fi

if [[ "${PRODUCT_CANDIDATES_PROFILE}" != /* ]]; then
    echo "PRODUCT_CANDIDATES_PROFILE must be an absolute path" >&2
    exit 64
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Configured Python interpreter is not executable: ${PYTHON_BIN}" >&2
    exit 69
fi

export PYTHONPATH="${REPO_ROOT}/core:${REPO_ROOT}/ikea${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE="1"
export PYTHONUNBUFFERED="1"

# These are fixed by the 2026-07-15 transport decision.  The application owns
# the 25-second request deadline and the inference semaphore/one-item queue.
export PRODUCT_CANDIDATES_BIND_HOST="${BIND_HOST}"
export PRODUCT_CANDIDATES_BIND_PORT="${BIND_PORT}"
export PRODUCT_CANDIDATES_REQUEST_TIMEOUT_SECONDS="25"
export PRODUCT_CANDIDATES_INFERENCE_CONCURRENCY="1"
export PRODUCT_CANDIDATES_MAX_QUEUE="1"

cd "${SERVICE_DIR}"
exec "${PYTHON_BIN}" -m uvicorn "${APP_MODULE}" \
    --host "${BIND_HOST}" \
    --port "${BIND_PORT}" \
    --workers "${WORKERS}" \
    --no-access-log \
    --no-server-header
