#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBTENSOR_DIR="$ROOT_DIR/subtensor"
LOCALNET_MODE="${LOCALNET_MODE:-single-node}"
NEURON_PYTHON="${NEURON_PYTHON:-$ROOT_DIR/.venv-neurons/bin/python}"
BTCLI_BIN="${BTCLI_BIN:-$ROOT_DIR/.venv-btcli/bin/btcli}"
WALLET_PATH="${WALLET_PATH:-/home/komail/.bittensor/wallets}"
CHAIN_ENDPOINT="${CHAIN_ENDPOINT:-ws://127.0.0.1:9944}"

OWNER_WALLET_NAME="${OWNER_WALLET_NAME:-owner}"
OWNER_HOTKEY="${OWNER_HOTKEY:-ownerhk}"
MINER_WALLET_NAME="${MINER_WALLET_NAME:-miner}"
MINER_HOTKEY="${MINER_HOTKEY:-minerhk}"
VALIDATOR_WALLET_NAME="${VALIDATOR_WALLET_NAME:-validator}"
VALIDATOR_HOTKEY="${VALIDATOR_HOTKEY:-valhk}"

SUBNET_NAME="${SUBNET_NAME:-hazard-localnet}"
SUBNET_GITHUB_REPO="${SUBNET_GITHUB_REPO:-https://github.com/opentensor/subtensor}"
SUBNET_CONTACT="${SUBNET_CONTACT:-local@subnet.test}"
SUBNET_DESCRIPTION="${SUBNET_DESCRIPTION:-Localnet hazard subnet smoke deployment}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_NEURON_START="${SKIP_NEURON_START:-0}"
PYTHONPATH_VALUE="$ROOT_DIR"

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    eval "$*"
  fi
}

wait_for_chain() {
  local retries=60
  local i
  for ((i = 1; i <= retries; i++)); do
    if "$BTCLI_BIN" wallet balance --wallet-name alice --wallet-path "$WALLET_PATH" --network "$CHAIN_ENDPOINT" >/dev/null 2>&1; then
      echo "Localnet RPC is available at $CHAIN_ENDPOINT"
      return 0
    fi
    sleep 2
  done
  echo "Localnet RPC did not come up at $CHAIN_ENDPOINT"
  return 1
}

http_endpoint_from_chain() {
  local endpoint="$1"
  echo "${endpoint/ws:\/\//http://}"
}

chain_block_number() {
  local http_endpoint="$1"
  local response
  response="$(curl -sS -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}' \
    "$http_endpoint")"
  "$NEURON_PYTHON" -c 'import json,sys; r=json.loads(sys.argv[1]); print(int(r["result"]["number"],16))' "$response"
}

wait_for_block_production() {
  local http_endpoint
  local first_block
  local second_block
  local retries=30
  local i

  http_endpoint="$(http_endpoint_from_chain "$CHAIN_ENDPOINT")"
  for ((i = 1; i <= retries; i++)); do
    first_block="$(chain_block_number "$http_endpoint" 2>/dev/null || echo -1)"
    sleep 3
    second_block="$(chain_block_number "$http_endpoint" 2>/dev/null || echo -1)"
    if [[ "$first_block" -ge 0 && "$second_block" -gt "$first_block" ]]; then
      echo "Block production confirmed: #$first_block -> #$second_block"
      return 0
    fi
  done

  echo "ERROR: RPC is up but chain is not producing blocks."
  echo "This localnet cannot finalize transactions, so subnet create/register will fail."
  echo "Check subtensor logs and fix consensus before continuing."
  return 1
}

echo "=== Step 1: Start localnet node ==="
if [[ "$LOCALNET_MODE" == "compose" ]]; then
  if [[ ! -d "$SUBTENSOR_DIR" ]]; then
    echo "Missing subtensor repo at $SUBTENSOR_DIR"
    exit 1
  fi
  run_cmd "cd \"$SUBTENSOR_DIR\" && docker compose -f docker-compose.localnet.yml up -d"
else
  echo "Using externally managed single-node chain at $CHAIN_ENDPOINT"
fi

echo "=== Step 2: Wait for RPC ==="
if [[ "$DRY_RUN" != "1" ]]; then
  wait_for_chain
  wait_for_block_production
else
  echo "Dry run: skipping RPC wait"
fi

echo "=== Step 3: Create wallets ==="
run_cmd "\"$BTCLI_BIN\" wallet create --wallet-name \"$OWNER_WALLET_NAME\" --wallet-path \"$WALLET_PATH\" --hotkey \"$OWNER_HOTKEY\" --n-words 12 --no-use-password --overwrite"
run_cmd "\"$BTCLI_BIN\" wallet create --wallet-name \"$MINER_WALLET_NAME\" --wallet-path \"$WALLET_PATH\" --hotkey \"$MINER_HOTKEY\" --n-words 12 --no-use-password --overwrite"
run_cmd "\"$BTCLI_BIN\" wallet create --wallet-name \"$VALIDATOR_WALLET_NAME\" --wallet-path \"$WALLET_PATH\" --hotkey \"$VALIDATOR_HOTKEY\" --n-words 12 --no-use-password --overwrite"

echo "=== Step 4: Create subnet ==="
CREATE_JSON="$(mktemp)"
run_cmd "printf \"\\n\\n\\n\\n\\n\" | \"$BTCLI_BIN\" subnets create --wallet-name \"$OWNER_WALLET_NAME\" --wallet-path \"$WALLET_PATH\" --hotkey \"$OWNER_HOTKEY\" --network \"$CHAIN_ENDPOINT\" --subnet-name \"$SUBNET_NAME\" --github-repo \"$SUBNET_GITHUB_REPO\" --subnet-contact \"$SUBNET_CONTACT\" --description \"$SUBNET_DESCRIPTION\" --json-output --no-prompt > \"$CREATE_JSON\""

if [[ "$DRY_RUN" != "1" ]]; then
  NETUID="$("$NEURON_PYTHON" - "$CREATE_JSON" <<'PY'
import json,sys
p=sys.argv[1]
with open(p,'r',encoding='utf-8') as f:
    data=json.load(f)
for key in ("netuid","subnet_netuid","created_netuid"):
    if key in data:
        print(data[key]); break
else:
    raise SystemExit("Could not determine netuid from btcli JSON output")
PY
)"
  echo "Created subnet netuid=$NETUID"
else
  NETUID="${NETUID:-1}"
  echo "Dry run: assuming netuid=$NETUID"
fi

echo "=== Step 5: Register miner and validator ==="
run_cmd "printf \"\\n\\n\\n\\n\\n\" | \"$BTCLI_BIN\" subnets register --wallet-name \"$MINER_WALLET_NAME\" --wallet-path \"$WALLET_PATH\" --hotkey \"$MINER_HOTKEY\" --network \"$CHAIN_ENDPOINT\" --netuid \"$NETUID\" --no-prompt --unsafe-register"
run_cmd "printf \"\\n\\n\\n\\n\\n\" | \"$BTCLI_BIN\" subnets register --wallet-name \"$VALIDATOR_WALLET_NAME\" --wallet-path \"$WALLET_PATH\" --hotkey \"$VALIDATOR_HOTKEY\" --network \"$CHAIN_ENDPOINT\" --netuid \"$NETUID\" --no-prompt --unsafe-register"

echo "=== Step 6: Run subnet neurons against localnet ==="
if [[ "$SKIP_NEURON_START" == "1" ]]; then
  echo "Skipping neuron process startup (SKIP_NEURON_START=1)."
else
  run_cmd "cd \"$ROOT_DIR\" && env PYTHONPATH=\"$PYTHONPATH_VALUE\" \"$NEURON_PYTHON\" neurons/miner.py --wallet.name \"$MINER_WALLET_NAME\" --wallet.hotkey \"$MINER_HOTKEY\" --subtensor.chain_endpoint \"$CHAIN_ENDPOINT\" --netuid \"$NETUID\" --blacklist.force_validator_permit"
  run_cmd "cd \"$ROOT_DIR\" && env PYTHONPATH=\"$PYTHONPATH_VALUE\" \"$NEURON_PYTHON\" neurons/validator.py --wallet.name \"$VALIDATOR_WALLET_NAME\" --wallet.hotkey \"$VALIDATOR_HOTKEY\" --subtensor.chain_endpoint \"$CHAIN_ENDPOINT\" --netuid \"$NETUID\""
fi

echo "Bootstrap sequence complete."
