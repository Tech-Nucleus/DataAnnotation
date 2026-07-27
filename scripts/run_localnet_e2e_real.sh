#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
NEURON_PYTHON="${NEURON_PYTHON:-$ROOT_DIR/.venv-neurons/bin/python}"

CHAIN_ENDPOINT="${CHAIN_ENDPOINT:-ws://127.0.0.1:9944}"
NETUID="${NETUID:-2}"
WALLET_PATH="${WALLET_PATH:-/home/komail/.bittensor/wallets}"
MINER_WALLET_NAME="${MINER_WALLET_NAME:-miner}"
MINER_WALLET_HOTKEY="${MINER_WALLET_HOTKEY:-minerhk}"
VALIDATOR_WALLET_NAME="${VALIDATOR_WALLET_NAME:-validator}"
VALIDATOR_WALLET_HOTKEY="${VALIDATOR_WALLET_HOTKEY:-valhk}"
MINER_PORT="${MINER_PORT:-8091}"
VALIDATOR_PORT="${VALIDATOR_PORT:-8092}"
MAX_TRAINING_SECONDS="${MAX_TRAINING_SECONDS:-600}"
RUN_SECONDS="${RUN_SECONDS:-180}"
ENABLE_AUTORESEARCH="${ENABLE_AUTORESEARCH:-0}"
AUTORESEARCH_MAX_ITERS="${AUTORESEARCH_MAX_ITERS:-1}"
AUTORESEARCH_EXPERIMENT_MINUTES="${AUTORESEARCH_EXPERIMENT_MINUTES:-1}"
# Optional explicit dendrite timeout for annotation tasks (seconds). 0 = use validator default logic.
TRAINING_TIMEOUT="${TRAINING_TIMEOUT:-0}"
MINER_RESPONSE_MODE="${MINER_RESPONSE_MODE:-standard}"
# Single-miner E2E: only sample one UID and pin queries to this hotkey (avoids stale on-chain "serving" peers).
NEURON_SAMPLE_SIZE="${NEURON_SAMPLE_SIZE:-1}"

export PYTHONPATH="$ROOT_DIR"
export CHAIN_ENDPOINT NETUID WALLET_PATH \
  MINER_WALLET_NAME MINER_WALLET_HOTKEY \
  VALIDATOR_WALLET_NAME VALIDATOR_WALLET_HOTKEY

log() {
  printf '[localnet-e2e] %s\n' "$*"
}

http_endpoint_from_ws() {
  local ws="$1"
  echo "${ws/ws:\/\//http://}"
}

block_number() {
  local http="$1"
  local out
  out="$(curl -sS -H "Content-Type: application/json" --data '{"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}' "$http")"
  "$NEURON_PYTHON" -c 'import json,sys; r=json.loads(sys.argv[1]); print(int(r["result"]["number"],16))' "$out"
}

assert_chain_progressing() {
  local http
  local baseline
  local current
  local i
  http="$(http_endpoint_from_ws "$CHAIN_ENDPOINT")"
  baseline="$(block_number "$http")"
  for ((i = 1; i <= 12; i++)); do
    sleep 5
    current="$(block_number "$http")"
    if [[ "$current" -gt "$baseline" ]]; then
      log "Chain progressing: $baseline -> $current"
      return 0
    fi
  done
  log "ERROR: chain did not advance within 60s ($baseline -> ${current:-$baseline})"
  exit 1
}

assert_registrations() {
  "$NEURON_PYTHON" - <<'PY'
import os
import bittensor as bt

chain = os.environ["CHAIN_ENDPOINT"]
netuid = int(os.environ["NETUID"])
wallet_path = os.environ["WALLET_PATH"]
miner_name = os.environ["MINER_WALLET_NAME"]
miner_hk = os.environ["MINER_WALLET_HOTKEY"]
validator_name = os.environ["VALIDATOR_WALLET_NAME"]
validator_hk = os.environ["VALIDATOR_WALLET_HOTKEY"]

st = bt.subtensor(network=chain)
mg = st.metagraph(netuid)
miner_addr = bt.wallet(name=miner_name, hotkey=miner_hk, path=wallet_path).hotkey.ss58_address
validator_addr = bt.wallet(name=validator_name, hotkey=validator_hk, path=wallet_path).hotkey.ss58_address
if miner_addr not in mg.hotkeys:
    raise SystemExit(f"Miner hotkey not registered on netuid {netuid}: {miner_addr}")
if validator_addr not in mg.hotkeys:
    raise SystemExit(f"Validator hotkey not registered on netuid {netuid}: {validator_addr}")
print(f"registrations_ok miner_uid={mg.hotkeys.index(miner_addr)} validator_uid={mg.hotkeys.index(validator_addr)}")
PY
}

cleanup() {
  if [[ -n "${MINER_PID:-}" ]]; then kill "${MINER_PID}" >/dev/null 2>&1 || true; fi
  if [[ -n "${VALIDATOR_PID:-}" ]]; then kill "${VALIDATOR_PID}" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT INT TERM

log "Verifying chain status..."
assert_chain_progressing

log "Verifying miner/validator registrations..."
assert_registrations

export LOCALNET_TARGET_MINER_SS58="$(
  "$NEURON_PYTHON" - <<'PY'
import os
import bittensor as bt
path = os.environ["WALLET_PATH"]
name = os.environ["MINER_WALLET_NAME"]
hk = os.environ["MINER_WALLET_HOTKEY"]
print(bt.wallet(name=name, hotkey=hk, path=path).hotkey.ss58_address)
PY
)"
log "Pinned localnet miner hotkey for sampling: $LOCALNET_TARGET_MINER_SS58"

MINER_LOG="$ROOT_DIR/artifacts/miner_e2e.log"
VALIDATOR_LOG="$ROOT_DIR/artifacts/validator_e2e.log"
mkdir -p "$ROOT_DIR/artifacts"

MINER_CMD=(
  "$NEURON_PYTHON" "$ROOT_DIR/neurons/miner.py"
  --wallet.name "$MINER_WALLET_NAME"
  --wallet.hotkey "$MINER_WALLET_HOTKEY"
  --wallet.path "$WALLET_PATH"
  --subtensor.network local
  --subtensor.chain_endpoint "$CHAIN_ENDPOINT"
  --netuid "$NETUID"
  --axon.port "$MINER_PORT"
  --miner.annotation_workspace "$ROOT_DIR/artifacts/miner_annotation"
  --logging.debug
)

VALIDATOR_CMD=(
  "$NEURON_PYTHON" "$ROOT_DIR/neurons/validator.py"
  --wallet.name "$VALIDATOR_WALLET_NAME"
  --wallet.hotkey "$VALIDATOR_WALLET_HOTKEY"
  --wallet.path "$WALLET_PATH"
  --subtensor.network local
  --subtensor.chain_endpoint "$CHAIN_ENDPOINT"
  --netuid "$NETUID"
  --axon.port "$VALIDATOR_PORT"
  --neuron.sample_size "$NEURON_SAMPLE_SIZE"
  --logging.debug
)

log "Starting miner..."
"${MINER_CMD[@]}" >"$MINER_LOG" 2>&1 &
MINER_PID="$!"
log "Starting validator..."
"${VALIDATOR_CMD[@]}" >"$VALIDATOR_LOG" 2>&1 &
VALIDATOR_PID="$!"

log "Running for ${RUN_SECONDS}s..."
sleep "$RUN_SECONDS"

log "Checking metagraph serving ports and incentive distribution..."
"$NEURON_PYTHON" - <<'PY'
import os
from pathlib import Path
import numpy as np
import bittensor as bt
from template.hazard.incentives import broad_softmax_scores

chain = os.environ["CHAIN_ENDPOINT"]
netuid = int(os.environ["NETUID"])
wallet_path = os.environ["WALLET_PATH"]
validator_name = os.environ["VALIDATOR_WALLET_NAME"]
validator_hk = os.environ["VALIDATOR_WALLET_HOTKEY"]

st = bt.subtensor(network=chain)
mg = st.metagraph(netuid)
vaddr = bt.wallet(name=validator_name, hotkey=validator_hk, path=wallet_path).hotkey.ss58_address
vuid = mg.hotkeys.index(vaddr)
if int(mg.axons[vuid].port) == 0:
    raise SystemExit("Validator axon is not served on-chain.")

state = Path.home() / ".bittensor" / "miners" / validator_name / validator_hk / f"netuid{netuid}" / "validator" / "state.npz"
if not state.exists():
    raise SystemExit(f"Validator state file missing: {state}")
arr = np.load(state, allow_pickle=False)
scores = arr["scores"].astype(np.float32)
dist = broad_softmax_scores(scores, temperature=0.25, floor=0.002, min_score=0.05)
nonzero = int((dist > 0).sum())
print(f"distribution_nonzero={nonzero} total_miners={len(dist)}")
print("distribution_values=", dist.tolist())
PY

log "E2E test complete. Logs:"
log "  miner: $MINER_LOG"
log "  validator: $VALIDATOR_LOG"
