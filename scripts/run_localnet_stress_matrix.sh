#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-neurons/bin/python}"

CHAIN_ENDPOINT="${CHAIN_ENDPOINT:-ws://127.0.0.1:9944}"
NETUID="${NETUID:-2}"
WALLET_PATH="${WALLET_PATH:-$HOME/.bittensor/wallets}"
VALIDATOR_WALLET_NAME="${VALIDATOR_WALLET_NAME:-validator}"
VALIDATOR_WALLET_HOTKEY="${VALIDATOR_WALLET_HOTKEY:-valhk}"
VALIDATOR_PORT="${VALIDATOR_PORT:-8092}"
RUN_SECONDS="${RUN_SECONDS:-1800}"
MAX_TRAINING_SECONDS="${MAX_TRAINING_SECONDS:-300}"
TRAINING_TIMEOUT="${TRAINING_TIMEOUT:-900}"
MINER_WALLET_NAMES="${MINER_WALLET_NAMES:-miner,miner2,miner3}"
MINER_WALLET_HOTKEYS="${MINER_WALLET_HOTKEYS:-minerhk,minerhk2,minerhk3}"
MINER_PORTS="${MINER_PORTS:-8091,8093,8094}"
ADVERSARIAL_MINER_INDEX="${ADVERSARIAL_MINER_INDEX:--1}"
ADVERSARIAL_MODE="${ADVERSARIAL_MODE:-malformed_manifest}"
# Annotation-only miner matrix.
AUTORESEARCH_MINER_INDEX="${AUTORESEARCH_MINER_INDEX:--1}"
AUTORESEARCH_MAX_ITERS="${AUTORESEARCH_MAX_ITERS:-2}"
AUTORESEARCH_EXPERIMENT_MINUTES="${AUTORESEARCH_EXPERIMENT_MINUTES:-1}"

export PYTHONPATH="$ROOT_DIR"
export MINER_MAX_TRAIN_SAMPLES="${MINER_MAX_TRAIN_SAMPLES:-16}"
export MINER_MAX_VAL_SAMPLES="${MINER_MAX_VAL_SAMPLES:-8}"
export MINER_MAX_EPOCHS="${MINER_MAX_EPOCHS:-1}"
export GOLDEN_MAX_SAMPLES="${GOLDEN_MAX_SAMPLES:-8}"

LOG_DIR="$ROOT_DIR/artifacts/stress"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

IFS=',' read -r -a NAMES <<< "$MINER_WALLET_NAMES"
IFS=',' read -r -a HOTKEYS <<< "$MINER_WALLET_HOTKEYS"
IFS=',' read -r -a PORTS <<< "$MINER_PORTS"

if [[ "${#NAMES[@]}" -ne "${#HOTKEYS[@]}" || "${#NAMES[@]}" -ne "${#PORTS[@]}" ]]; then
  echo "MINER_WALLET_NAMES, MINER_WALLET_HOTKEYS, MINER_PORTS lengths must match."
  exit 1
fi

# Align dendrite targets with real axon ports (metagraph often stale on single-host localnet).
SS58_MAP=""
for i in "${!NAMES[@]}"; do
  ss58="$("$PYTHON_BIN" - "$WALLET_PATH" "${NAMES[$i]}" "${HOTKEYS[$i]}" <<'PY'
import bittensor as bt, sys
path, name, hk = sys.argv[1:4]
w = bt.wallet(name=name, hotkey=hk, path=path)
print(w.hotkey.ss58_address)
PY
)"
  [[ -n "$SS58_MAP" ]] && SS58_MAP+=","
  SS58_MAP+="${ss58}=${PORTS[$i]}"
done
export LOCALNET_MINER_PORT_BY_SS58="$SS58_MAP"
echo "[stress] LOCALNET_MINER_PORT_BY_SS58=$LOCALNET_MINER_PORT_BY_SS58"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT INT TERM

for i in "${!NAMES[@]}"; do
  mode="standard"
  if [[ "$i" -eq "$ADVERSARIAL_MINER_INDEX" ]]; then
    mode="$ADVERSARIAL_MODE"
  fi
  log="$LOG_DIR/miner_${i}_${STAMP}.log"
  echo "[stress] starting miner[$i] mode=$mode log=$log"
  miner_cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/neurons/miner.py"
    --wallet.name "${NAMES[$i]}"
    --wallet.hotkey "${HOTKEYS[$i]}"
    --wallet.path "$WALLET_PATH"
    --subtensor.network local
    --subtensor.chain_endpoint "$CHAIN_ENDPOINT"
    --netuid "$NETUID"
    --axon.port "${PORTS[$i]}"
    --miner.annotation_workspace "$ROOT_DIR/artifacts/miner_annotation/${HOTKEYS[$i]}"
    --logging.debug
  )
  echo "[stress] miner[$i] annotation-only mode=$mode"
  "${miner_cmd[@]}" >"$log" 2>&1 &
  PIDS+=("$!")
done

MINER_WARMUP_SECONDS="${MINER_WARMUP_SECONDS:-20}"
echo "[stress] waiting ${MINER_WARMUP_SECONDS}s for miner axons (MINER_WARMUP_SECONDS)"
sleep "$MINER_WARMUP_SECONDS"

VLOG="$LOG_DIR/validator_${STAMP}.log"
echo "[stress] starting validator log=$VLOG"
"$PYTHON_BIN" "$ROOT_DIR/neurons/validator.py" \
  --wallet.name "$VALIDATOR_WALLET_NAME" \
  --wallet.hotkey "$VALIDATOR_WALLET_HOTKEY" \
  --wallet.path "$WALLET_PATH" \
  --subtensor.network local \
  --subtensor.chain_endpoint "$CHAIN_ENDPOINT" \
  --netuid "$NETUID" \
  --axon.port "$VALIDATOR_PORT" \
  --logging.debug >"$VLOG" 2>&1 &
PIDS+=("$!")

echo "[stress] running for ${RUN_SECONDS}s"
sleep "$RUN_SECONDS"
cleanup

echo "[stress] summary:"
echo "  validator_log=$VLOG"
python3 - <<PY
from pathlib import Path
import re
log = Path("$VLOG").read_text(encoding="utf-8", errors="ignore")
annotation_rounds = len(re.findall(r"event=annotation_flywheel_round_done", log))
golden_payloads = len(re.findall(r"event=annotation_flywheel_round_start", log))
validation_errors = len(re.findall(r"Error during validation step|Challenge nonce mismatch|Mismatched task_id|Miner response missing annotations_uri", log))
print(f"  annotation_rounds={annotation_rounds}")
print(f"  golden_payloads={golden_payloads}")
print(f"  validation_errors={validation_errors}")
PY
