#!/usr/bin/env bash
# Full subnet staging simulation (localnet): chain check, shared R2, 3 miners + validator.
#
# Prerequisites:
#   - Subtensor RPC at CHAIN_ENDPOINT (default ws://127.0.0.1:9944) with blocks producing
#   - Wallets: owner, miner+minerhk, miner2+minerhk2, miner3+minerhk3, validator+valhk
#   - Subnet created and all above registered on NETUID (see bootstrap_localnet_and_register.sh)
#   - staging.env OR environment: R2_ACCOUNT_ID, R2_BUCKET_NAME, R2_S3_ENDPOINT,
#     R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
#
# This script does NOT start Docker subtensor by default (subtensor/ is often external).
# Set START_SUBTENSOR_COMPOSE=1 if you have $ROOT_DIR/subtensor with docker-compose.localnet.yml.
# Set START_LOCALNET_DOCKER=1 (default) to run scripts/start_localnet_subtensor_docker.sh when RPC is down.
#
# Tuning:
#   FORWARD_STEP_SLEEP_SECONDS=300  # ~5 min between validator steps (default in matrix script)
#   RUN_SECONDS=3600               # wall-clock before teardown + gate (raise for more rounds)
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/staging.env" ]]; then
  set -a && source "$ROOT_DIR/staging.env" && set +a
fi

CHAIN_ENDPOINT="${CHAIN_ENDPOINT:-ws://127.0.0.1:9944}"
http="${CHAIN_ENDPOINT/ws:\/\//http://}"

log() { printf '[staging-e2e] %s\n' "$*"; }

if ! curl -sS -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}' "$http" | grep -q result; then
  log "ERROR: no JSON-RPC result at $http"
  if [[ "${START_SUBTENSOR_COMPOSE:-0}" == "1" && -d "$ROOT_DIR/subtensor" ]]; then
    log "Starting subtensor compose..."
    (cd "$ROOT_DIR/subtensor" && docker compose -f docker-compose.localnet.yml up -d)
    sleep 5
  elif [[ "${START_LOCALNET_DOCKER:-1}" == "1" && -x "$ROOT_DIR/scripts/start_localnet_subtensor_docker.sh" ]]; then
    log "Trying scripts/start_localnet_subtensor_docker.sh ..."
    bash "$ROOT_DIR/scripts/start_localnet_subtensor_docker.sh" || true
    if ! curl -sS -H "Content-Type: application/json" \
      --data '{"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}' "$http" | grep -q result; then
      log "ERROR: RPC still down after docker start."
      exit 1
    fi
  else
    log "Start your local node, set START_LOCALNET_DOCKER=1, or clone subtensor and set START_SUBTENSOR_COMPOSE=1"
    exit 1
  fi
fi

NEURON_PYTHON="${NEURON_PYTHON:-$ROOT_DIR/.venv-neurons/bin/python}"
if [[ ! -x "$NEURON_PYTHON" ]]; then
  log "ERROR: missing $NEURON_PYTHON (create .venv-neurons and pip install -r requirements.txt)"
  exit 1
fi

# Confirm block height advances (weak liveness check)
read_block() {
  curl -sS -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}' "$http" \
    | "$NEURON_PYTHON" -c 'import json,sys; print(int(json.load(sys.stdin)["result"]["number"],16))'
}
b1="$(read_block)"
sleep 4
b2="$(read_block)"
if [[ "$b2" -le "$b1" ]]; then
  log "WARN: chain header did not advance ($b1 -> $b2); registrations may hang."
fi

export FORWARD_STEP_SLEEP_SECONDS="${FORWARD_STEP_SLEEP_SECONDS:-300}"
export RUN_SECONDS="${RUN_SECONDS:-3600}"
export MAX_TRAINING_SECONDS="${MAX_TRAINING_SECONDS:-300}"
export TRAINING_TIMEOUT="${TRAINING_TIMEOUT:-900}"
export NEURON_SAMPLE_SIZE="${NEURON_SAMPLE_SIZE:-3}"
export MINER_MAX_TRAIN_SAMPLES="${MINER_MAX_TRAIN_SAMPLES:-24}"
export MINER_MAX_VAL_SAMPLES="${MINER_MAX_VAL_SAMPLES:-8}"
export MINER_MAX_EPOCHS="${MINER_MAX_EPOCHS:-1}"

log "CHAIN_ENDPOINT=$CHAIN_ENDPOINT NETUID=${NETUID:-2} FORWARD_STEP_SLEEP_SECONDS=$FORWARD_STEP_SLEEP_SECONDS RUN_SECONDS=$RUN_SECONDS"
log "Invoking run_operator_localnet_matrix.sh"

exec bash "$ROOT_DIR/scripts/run_operator_localnet_matrix.sh"
