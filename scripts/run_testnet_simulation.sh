#!/usr/bin/env bash
set -euo pipefail

# End-to-end testnet simulation launcher for this subnet.
# It intentionally does not auto-register wallets or mutate chain state beyond
# launching neurons; registration must be done explicitly by the operator.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-neurons/bin/python}"
CHAIN_ENDPOINT="${CHAIN_ENDPOINT:-wss://test.finney.opentensor.ai:443}"
NETUID="${NETUID:-1}"

MINER_WALLET_NAME="${MINER_WALLET_NAME:-}"
MINER_WALLET_HOTKEY="${MINER_WALLET_HOTKEY:-}"
VALIDATOR_WALLET_NAME="${VALIDATOR_WALLET_NAME:-}"
VALIDATOR_WALLET_HOTKEY="${VALIDATOR_WALLET_HOTKEY:-}"

MINER_PORT="${MINER_PORT:-8091}"
VALIDATOR_PORT="${VALIDATOR_PORT:-8092}"
VALIDATOR_AXON_OFF="${VALIDATOR_AXON_OFF:-1}"

DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/hazard}"
MAX_TRAINING_SECONDS="${MAX_TRAINING_SECONDS:-120}"
ENABLE_AUTORESEARCH="${ENABLE_AUTORESEARCH:-0}"
MINER_RESPONSE_MODE="${MINER_RESPONSE_MODE:-standard}"

INCENTIVE_TEMPERATURE="${INCENTIVE_TEMPERATURE:-0.25}"
INCENTIVE_FLOOR="${INCENTIVE_FLOOR:-0.002}"
INCENTIVE_MIN_SCORE="${INCENTIVE_MIN_SCORE:-0.05}"

RUN_MINER="${RUN_MINER:-1}"
RUN_VALIDATOR="${RUN_VALIDATOR:-1}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  Configure env vars, then run:
    bash scripts/run_testnet_simulation.sh

Required env vars:
  MINER_WALLET_NAME
  MINER_WALLET_HOTKEY
  VALIDATOR_WALLET_NAME
  VALIDATOR_WALLET_HOTKEY

Common optional env vars:
  CHAIN_ENDPOINT (default: wss://test.finney.opentensor.ai:443)
  NETUID (default: 1)
  DATASET_ROOT (default: ./data/hazard)
  MAX_TRAINING_SECONDS (default: 120)
  DRY_RUN=1 (print commands only)
EOF
}

require_env() {
  local name="$1"
  if [[ -z "${!name}" ]]; then
    echo "Missing required env var: $name"
    usage
    exit 1
  fi
}

require_env MINER_WALLET_NAME
require_env MINER_WALLET_HOTKEY
require_env VALIDATOR_WALLET_NAME
require_env VALIDATOR_WALLET_HOTKEY

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Dataset root does not exist: $DATASET_ROOT"
  exit 1
fi

export PYTHONPATH="$ROOT_DIR"

MINER_CMD=(
  "$PYTHON_BIN" neurons/miner.py
  --netuid "$NETUID"
  --subtensor.chain_endpoint "$CHAIN_ENDPOINT"
  --wallet.name "$MINER_WALLET_NAME"
  --wallet.hotkey "$MINER_WALLET_HOTKEY"
  --axon.port "$MINER_PORT"
  --blacklist.force_validator_permit
  --miner.annotation_workspace "$ROOT_DIR/artifacts/miner_annotation/$MINER_WALLET_HOTKEY"
)

VALIDATOR_CMD=(
  "$PYTHON_BIN" neurons/validator.py
  --netuid "$NETUID"
  --subtensor.chain_endpoint "$CHAIN_ENDPOINT"
  --wallet.name "$VALIDATOR_WALLET_NAME"
  --wallet.hotkey "$VALIDATOR_WALLET_HOTKEY"
  --axon.port "$VALIDATOR_PORT"
  --neuron.incentive_temperature "$INCENTIVE_TEMPERATURE"
  --neuron.incentive_floor "$INCENTIVE_FLOOR"
  --neuron.incentive_min_score "$INCENTIVE_MIN_SCORE"
)

if [[ "$VALIDATOR_AXON_OFF" == "1" ]]; then
  VALIDATOR_CMD+=(--neuron.axon_off)
fi

echo "=== Testnet Simulation Configuration ==="
echo "CHAIN_ENDPOINT=$CHAIN_ENDPOINT"
echo "NETUID=$NETUID"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "BASELINE_URI=$BASELINE_URI"
echo "RUN_MINER=$RUN_MINER RUN_VALIDATOR=$RUN_VALIDATOR DRY_RUN=$DRY_RUN"
echo "ENABLE_AUTORESEARCH=$ENABLE_AUTORESEARCH"
echo
echo "Miner command:"
printf '  %q ' "${MINER_CMD[@]}"
echo
echo "Validator command:"
printf '  %q ' "${VALIDATOR_CMD[@]}"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete."
  exit 0
fi

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT INT TERM

if [[ "$RUN_MINER" == "1" ]]; then
  "${MINER_CMD[@]}" &
  PIDS+=("$!")
fi

if [[ "$RUN_VALIDATOR" == "1" ]]; then
  "${VALIDATOR_CMD[@]}" &
  PIDS+=("$!")
fi

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "Nothing to run. Set RUN_MINER=1 and/or RUN_VALIDATOR=1."
  exit 1
fi

wait "${PIDS[@]}"
