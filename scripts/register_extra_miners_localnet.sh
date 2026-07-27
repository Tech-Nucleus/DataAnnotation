#!/usr/bin/env bash
# Register additional miner wallets on an existing localnet subnet (same pattern as bootstrap).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BTCLI_BIN="${BTCLI_BIN:-$ROOT_DIR/.venv-btcli/bin/btcli}"
WALLET_PATH="${WALLET_PATH:-$HOME/.bittensor/wallets}"
CHAIN_ENDPOINT="${CHAIN_ENDPOINT:-ws://127.0.0.1:9944}"
NETUID="${NETUID:-2}"

# Comma-separated: wallet_name:hotkey_name
EXTRA_MINERS="${EXTRA_MINERS:-miner2:minerhk2,miner3:minerhk3}"
# Pause between register extrinsics to avoid Subtensor RateLimitExceeded (custom error 6).
REGISTER_SLEEP_SECONDS="${REGISTER_SLEEP_SECONDS:-45}"
# Set AUTO_FUND=1 to send TAO from a funded local wallet before register (avoids recycle balance errors).
AUTO_FUND="${AUTO_FUND:-0}"
FUNDER_WALLET="${FUNDER_WALLET:-owner}"
FUND_AMOUNT="${FUND_AMOUNT:-1.0}"
NEURON_PYTHON="${NEURON_PYTHON:-$ROOT_DIR/.venv-neurons/bin/python}"

IFS=',' read -r -a PAIRS <<< "$EXTRA_MINERS"
nonempty=0
for pair in "${PAIRS[@]}"; do
  [[ -z "$pair" ]] && continue
  nonempty=$((nonempty + 1))
done
seen=0
for pair in "${PAIRS[@]}"; do
  [[ -z "$pair" ]] && continue
  name="${pair%%:*}"
  hk="${pair##*:}"
  echo "[register-extra] ensuring wallet $name hotkey $hk"
  if [[ ! -d "$WALLET_PATH/$name" ]]; then
    printf '\n\n\n\n\n' | "$BTCLI_BIN" wallet create \
      --wallet-name "$name" \
      --wallet-path "$WALLET_PATH" \
      --hotkey "$hk" \
      --n-words 12 \
      --no-use-password \
      --overwrite
  elif [[ ! -f "$WALLET_PATH/$name/hotkeys/$hk" ]]; then
    printf '\n\n\n\n\n' | "$BTCLI_BIN" wallet create \
      --wallet-name "$name" \
      --wallet-path "$WALLET_PATH" \
      --hotkey "$hk" \
      --n-words 12 \
      --no-use-password \
      --overwrite
  fi
  if [[ "$AUTO_FUND" == "1" ]]; then
    dest="$("$NEURON_PYTHON" - "$WALLET_PATH" "$name" "$hk" <<'PY'
import bittensor as bt, sys
path, cold, hot = sys.argv[1:4]
w = bt.wallet(name=cold, hotkey=hot, path=path)
print(w.coldkey.ss58_address)
PY
)"
    echo "[register-extra] auto_fund $FUND_AMOUNT τ from $FUNDER_WALLET -> $dest"
    "$BTCLI_BIN" wallet transfer \
      --wallet-path "$WALLET_PATH" \
      --wallet-name "$FUNDER_WALLET" \
      --network "$CHAIN_ENDPOINT" \
      --destination "$dest" \
      --amount "$FUND_AMOUNT" \
      --no-prompt
  fi
  echo "[register-extra] subnets register netuid=$NETUID $name/$hk"
  printf '\n\n\n\n\n' | "$BTCLI_BIN" subnets register \
    --wallet-name "$name" \
    --wallet-path "$WALLET_PATH" \
    --hotkey "$hk" \
    --network "$CHAIN_ENDPOINT" \
    --netuid "$NETUID" \
    --no-prompt \
    --unsafe-register
  seen=$((seen + 1))
  if [[ "$seen" -lt "$nonempty" ]]; then
    echo "[register-extra] sleeping ${REGISTER_SLEEP_SECONDS}s before next register (rate-limit spacing)"
    sleep "${REGISTER_SLEEP_SECONDS}"
  fi
done

echo "[register-extra] done"
