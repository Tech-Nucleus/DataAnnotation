#!/usr/bin/env bash
# After host or subtensor restart: wait for RPC, optionally create subnet netuid 2,
# fund coldkeys from alice, register validator + miners with spacing to avoid
# Subtensor RateLimitExceeded (custom error 6, see learnbittensor errors doc).
#
# Many local devnet images cap each subnet at 4 UIDs with a genesis neuron on UID 0,
# leaving room for validator + 2 miners only. In that case set MINER_COUNT=2.
# For three distinct miners you need a chain/subnet with max UIDs >= 5 or no
# genesis neuron consuming a slot.
#
# Usage (from repo root, MinIO or real R2_* already exported or in staging.env):
#   CHAIN_ENDPOINT=ws://127.0.0.1:9944 NETUID=2 ./scripts/localnet_post_restart_operator.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/staging.env" ]]; then
  set -a && source "$ROOT_DIR/staging.env" && set +a
fi

NEURON_PYTHON="${NEURON_PYTHON:-$ROOT_DIR/.venv-neurons/bin/python}"
BTCLI_BIN="${BTCLI_BIN:-$ROOT_DIR/.venv-btcli/bin/btcli}"

CHAIN_ENDPOINT="${CHAIN_ENDPOINT:-ws://127.0.0.1:9944}"
WALLET_PATH="${WALLET_PATH:-$HOME/.bittensor/wallets}"
NETUID="${NETUID:-2}"

OWNER_WALLET="${OWNER_WALLET_NAME:-owner}"
OWNER_HOTKEY="${OWNER_HOTKEY:-ownerhk}"
VALIDATOR_WALLET="${VALIDATOR_WALLET_NAME:-validator}"
VALIDATOR_HOTKEY="${VALIDATOR_WALLET_HOTKEY:-valhk}"
MINER_WALLET="${MINER_WALLET_NAME:-miner}"
MINER_HOTKEY="${MINER_HOTKEY:-minerhk}"
MINER2_WALLET="${MINER2_WALLET_NAME:-miner2}"
MINER2_HOTKEY="${MINER2_HOTKEY:-minerhk2}"
MINER3_WALLET="${MINER3_WALLET_NAME:-miner3}"
MINER3_HOTKEY="${MINER3_HOTKEY:-minerhk3}"

FUND_AMOUNT="${FUND_AMOUNT:-500}"
REGISTER_GAP_SECONDS="${REGISTER_GAP_SECONDS:-50}"
CREATE_SUBNET_IF_MISSING="${CREATE_SUBNET_IF_MISSING:-1}"
MINER_COUNT="${MINER_COUNT:-3}"

log() { printf '[localnet-post-restart] %s\n' "$*"; }

http_from_ws() { echo "${1/ws:\/\//http://}"; }

wait_rpc_and_blocks() {
  local http out i
  http="$(http_from_ws "$CHAIN_ENDPOINT")"
  for ((i = 1; i <= 90; i++)); do
    out="$(curl -sS -m 2 -H "Content-Type: application/json" \
      --data '{"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}' "$http" 2>/dev/null || true)"
    if [[ -n "$out" && "$out" == *"result"* ]]; then
      log "RPC up at $CHAIN_ENDPOINT"
      break
    fi
    sleep 2
  done
  local b0 b1 j
  for ((j = 1; j <= 40; j++)); do
    b0="$("$NEURON_PYTHON" - <<PY
import json,urllib.request
u="${http}"
req=urllib.request.Request(u,data=json.dumps({"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}).encode(),headers={"Content-Type":"application/json"})
print(int(json.loads(urllib.request.urlopen(req,timeout=5).read())["result"]["number"],16))
PY
)"
    sleep 4
    b1="$("$NEURON_PYTHON" - <<PY
import json,urllib.request
u="${http}"
req=urllib.request.Request(u,data=json.dumps({"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}).encode(),headers={"Content-Type":"application/json"})
print(int(json.loads(urllib.request.urlopen(req,timeout=5).read())["result"]["number"],16))
PY
)"
    if [[ "${b1:-0}" -gt "${b0:-0}" ]]; then
      log "block production ok ($b0 -> $b1)"
      return 0
    fi
    log "waiting for blocks... ($b0)"
  done
  log "WARN: chain may not be producing blocks; continuing anyway"
}

coldkey_dest() {
  "$NEURON_PYTHON" - "$WALLET_PATH" "$1" "$2" <<'PY'
import bittensor as bt, sys
path, cold, hot = sys.argv[1:4]
print(bt.wallet(name=cold, hotkey=hot, path=path).coldkey.ss58_address)
PY
}

fund_from_alice() {
  local name hk dest
  if [[ "${SKIP_FUND:-0}" == "1" ]]; then
    log "SKIP_FUND=1 — not transferring from alice"
    return 0
  fi
  for pair in \
    "$OWNER_WALLET:$OWNER_HOTKEY" \
    "$VALIDATOR_WALLET:$VALIDATOR_HOTKEY" \
    "$MINER_WALLET:$MINER_HOTKEY" \
    "$MINER2_WALLET:$MINER2_HOTKEY" \
    "$MINER3_WALLET:$MINER3_HOTKEY"; do
    name="${pair%%:*}"
    hk="${pair##*:}"
    dest="$(coldkey_dest "$name" "$hk")"
    log "alice -> $name coldkey ($dest) amount=$FUND_AMOUNT"
    printf '\n\n\n\n\n' | "$BTCLI_BIN" wallet transfer \
      --wallet-path "$WALLET_PATH" --wallet-name alice \
      --network "$CHAIN_ENDPOINT" --destination "$dest" \
      --amount "$FUND_AMOUNT" --no-prompt || log "WARN: transfer failed for ${name%%:*}"
  done
}

subnet_exists() {
  "$NEURON_PYTHON" - "$CHAIN_ENDPOINT" "$1" <<'PY'
import bittensor as bt, sys
st = bt.subtensor(network=sys.argv[1])
try:
    st.metagraph(int(sys.argv[2]))
    print("yes")
except Exception:
    print("no")
PY
}

create_subnet_if_needed() {
  if [[ "$CREATE_SUBNET_IF_MISSING" != "1" ]]; then
    return 0
  fi
  if [[ "$(subnet_exists "$NETUID")" == "yes" ]]; then
    log "subnet netuid=$NETUID already exists"
    return 0
  fi
  log "creating subnet on netuid $NETUID (owner=$OWNER_WALLET) ..."
  printf '\n\n\n\n\n' | "$BTCLI_BIN" subnets create \
    --wallet-name "$OWNER_WALLET" --wallet-path "$WALLET_PATH" --hotkey "$OWNER_HOTKEY" \
    --network "$CHAIN_ENDPOINT" \
    --subnet-name "${SUBNET_NAME:-hazard-localnet}" \
    --github-repo "${SUBNET_GITHUB_REPO:-https://github.com/opentensor/subtensor}" \
    --subnet-contact "${SUBNET_CONTACT:-eng@subnet.local}" \
    --subnet-url "${SUBNET_URL:-https://subnet.local}" \
    --description "${SUBNET_DESCRIPTION:-Localnet dual-flywheel}" \
    --json-output --no-prompt --no-mev-protection
}

register_one() {
  local w="$1" h="$2"
  log "subnets register netuid=$NETUID $w/$h"
  printf '\n\n\n\n\n' | "$BTCLI_BIN" subnets register \
    --wallet-name "$w" --wallet-path "$WALLET_PATH" --hotkey "$h" \
    --network "$CHAIN_ENDPOINT" --netuid "$NETUID" \
    --no-prompt --unsafe-register
}

register_sequence() {
  register_one "$VALIDATOR_WALLET" "$VALIDATOR_HOTKEY" || true
  sleep "$REGISTER_GAP_SECONDS"
  register_one "$MINER_WALLET" "$MINER_HOTKEY" || true
  sleep "$REGISTER_GAP_SECONDS"
  register_one "$MINER2_WALLET" "$MINER2_HOTKEY" || true
  if [[ "$MINER_COUNT" -ge 3 ]]; then
    sleep "$REGISTER_GAP_SECONDS"
    register_one "$MINER3_WALLET" "$MINER3_HOTKEY" || true
  fi
}

wait_rpc_and_blocks
fund_from_alice
create_subnet_if_needed
register_sequence

log "Metagraph snapshot:"
"$NEURON_PYTHON" - "$CHAIN_ENDPOINT" "$NETUID" <<'PY'
import bittensor as bt, sys
st = bt.subtensor(network=sys.argv[1])
mg = st.metagraph(int(sys.argv[2]))
print("n", int(mg.n))
for i in range(int(mg.n)):
    print(i, mg.hotkeys[i])
PY

log "Starting operator matrix (set RUN_SECONDS / FORWARD_STEP_SLEEP_SECONDS / MINER_COUNT as needed)."
exec bash "$ROOT_DIR/scripts/run_operator_localnet_matrix.sh"
