#!/usr/bin/env bash
# COCO val2017 localnet E2E: 200 images (20 golden / 180 pool), 3 quality-tier miners, validator fusion.
# Isolated under artifacts/localnet/ — does not modify production data paths.
#
# Usage (from repo root):
#   export R2_* or use MinIO defaults below
#   ./scripts/localnet/run_coco_localnet_e2e.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/staging.env" ]]; then
  set -a && source "$ROOT_DIR/staging.env" && set +a
fi

NEURON_PYTHON="${NEURON_PYTHON:-$ROOT_DIR/.venv-neurons/bin/python}"
BTCLI_BIN="${BTCLI_BIN:-$ROOT_DIR/.venv-btcli/bin/btcli}"
PYTHON_BIN="${PYTHON_BIN:-$NEURON_PYTHON}"

CHAIN_ENDPOINT="${CHAIN_ENDPOINT:-ws://127.0.0.1:9944}"
NETUID="${NETUID:-2}"
WALLET_PATH="${WALLET_PATH:-$HOME/.bittensor/wallets}"

COCO_OUT="${COCO_OUT:-$ROOT_DIR/artifacts/localnet/coco200}"
MANIFEST="${COCO_MANIFEST:-$COCO_OUT/manifest.json}"
COCO_SIZE="${COCO_DATASET_SIZE:-200}"
GOLDEN_RATIO="${COCO_GOLDEN_RATIO:-0.1}"

# MinIO defaults for local S3 (override with real Cloudflare R2_* in staging.env)
export R2_ACCOUNT_ID="${R2_ACCOUNT_ID:-local}"
export R2_BUCKET_NAME="${R2_BUCKET_NAME:-localnet-hazard}"
export R2_S3_ENDPOINT="${R2_S3_ENDPOINT:-http://127.0.0.1:9000}"
export R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID:-minioadmin}"
export R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY:-minioadmin123}"

RUN_SECONDS="${RUN_SECONDS:-2400}"
FORWARD_STEP_SLEEP_SECONDS="${FORWARD_STEP_SLEEP_SECONDS:-300}"
MINER_COUNT="${MINER_COUNT:-3}"
NEURON_SAMPLE_SIZE="${NEURON_SAMPLE_SIZE:-3}"

# Backends: all localnet miners use the real YOLO detector path.
MINER_BACKENDS="${MINER_BACKENDS:-yolo,yolo,yolo}"
R2_PREFIXES="${R2_PREFIXES:-localnet/coco_m1_yolo,localnet/coco_m2_yolo,localnet/coco_m3_yolo}"

COMMERCIAL_PREFIX="${COMMERCIAL_PREFIX:-file://$ROOT_DIR/artifacts/localnet/coco_commercial}"
export FORCE_LOCAL_SET_WEIGHTS="${FORCE_LOCAL_SET_WEIGHTS:-1}"

log() { printf '[coco-e2e] %s\n' "$*"; }

log "=== 0) Subtensor RPC ==="
if ! curl -sS -m 3 -H 'Content-Type: application/json' \
  --data '{"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}' \
  "${CHAIN_ENDPOINT/ws:\/\//http://}" >/dev/null; then
  log "ERROR: start subtensor first (e.g. docker restart subtensor-devnet-stable)"
  exit 1
fi

log "=== 1) Prepare COCO subset ==="
"$PYTHON_BIN" "$ROOT_DIR/scripts/localnet/prepare_coco_val2017_subset.py" \
  --out-dir "$COCO_OUT" \
  --dataset-size "$COCO_SIZE" \
  --golden-ratio "$GOLDEN_RATIO"

log "=== 2) Offline acceptance sim (fusion + gates) ==="
"$PYTHON_BIN" "$ROOT_DIR/scripts/run_annotation_only_coco_sim.py" \
  --images-dir "$COCO_OUT/images" \
  --annotations-file "${COCO_UPSTREAM_ANNOTATIONS:-$ROOT_DIR/artifacts/localnet/coco_upstream/annotations/instances_val2017.json}" \
  --dataset-size "$COCO_SIZE" \
  --golden-ratio "$GOLDEN_RATIO" \
  --rounds 3 \
  --seed 7 \
  | tee "$ROOT_DIR/artifacts/localnet/coco_offline_sim.json"

log "=== 3) Chain registration (spaced) + operator matrix ==="
export CHAIN_ENDPOINT NETUID WALLET_PATH SKIP_FUND="${SKIP_FUND:-1}" REGISTER_AUTO_FUND=0
export MINER_COUNT NEURON_SAMPLE_SIZE RUN_SECONDS FORWARD_STEP_SLEEP_SECONDS
export R2_PREFIXES MINER_BACKENDS

# Validator uses COCO manifest; isolated commercial export dir
export FLYWHEEL_COCO_MANIFEST="$MANIFEST"
export FLYWHEEL_GOLDEN_MISSING_PENALTY=0.5
export FLYWHEEL_COMMERCIAL_EXPORT_EVERY=1
export FLYWHEEL_COMMERCIAL_DATASET_PREFIX="$COMMERCIAL_PREFIX"
export FLYWHEEL_IMAGE_CACHE_ROOT="$ROOT_DIR/artifacts/localnet/coco_image_cache"
export FLYWHEEL_ANNOTATION_REQUEST_SIZE="${FLYWHEEL_ANNOTATION_REQUEST_SIZE:-15}"
export FLYWHEEL_GOLDEN_INJECTION_PER_REQUEST="${FLYWHEEL_GOLDEN_INJECTION_PER_REQUEST:-5}"
export NEURON_ANNOTATION_TIMEOUT=3600
mkdir -p "$ROOT_DIR/artifacts/localnet/coco_commercial" "$ROOT_DIR/artifacts/localnet/coco_image_cache"

bash "$ROOT_DIR/scripts/run_operator_localnet_matrix.sh"

log "=== 4) Post-run export evaluation (180 pool holdout) ==="
EXPORT_JSONL="$ROOT_DIR/artifacts/commercial_dataset/commercial-dataset.jsonl"
if [[ ! -f "$EXPORT_JSONL" ]]; then
  EXPORT_JSONL="$(ls -1t "$ROOT_DIR/artifacts/commercial_dataset/"*.jsonl 2>/dev/null | head -1 || true)"
fi
if [[ -z "$EXPORT_JSONL" || ! -f "$EXPORT_JSONL" ]]; then
  log "WARN: no commercial export jsonl found for holdout eval"
else
  "$PYTHON_BIN" "$ROOT_DIR/scripts/localnet/evaluate_coco_commercial_export.py" \
    --manifest "$MANIFEST" \
    --export "$EXPORT_JSONL" \
    | tee "$ROOT_DIR/artifacts/localnet/coco_holdout_eval.json"
fi

log "=== done ==="
