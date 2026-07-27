#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-neurons/bin/python}"
CHAIN_ENDPOINT="${CHAIN_ENDPOINT:-wss://test.finney.opentensor.ai:443}"
NETUID="${NETUID:-1}"
WALLET_PATH="${WALLET_PATH:-$HOME/.bittensor/wallets}"
VALIDATOR_WALLET_NAME="${VALIDATOR_WALLET_NAME:-}"
VALIDATOR_WALLET_HOTKEY="${VALIDATOR_WALLET_HOTKEY:-}"
VALIDATOR_PORT="${VALIDATOR_PORT:-8092}"
RUN_SECONDS="${RUN_SECONDS:-1800}"
MIN_SET_WEIGHTS_SUCCESSES="${MIN_SET_WEIGHTS_SUCCESSES:-2}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/hazard}"
BASELINE_URI="${BASELINE_URI:-yolov8s.pt}"
MAX_TRAINING_SECONDS="${MAX_TRAINING_SECONDS:-300}"
TRAINING_TIMEOUT="${TRAINING_TIMEOUT:-900}"

if [[ -z "$VALIDATOR_WALLET_NAME" || -z "$VALIDATOR_WALLET_HOTKEY" ]]; then
  echo "Missing VALIDATOR_WALLET_NAME / VALIDATOR_WALLET_HOTKEY"
  exit 1
fi

export NETUID CHAIN_ENDPOINT WALLET_PATH VALIDATOR_WALLET_NAME VALIDATOR_WALLET_HOTKEY

SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  if ! "$PYTHON_BIN" - <<PY
import os
import sys
import bittensor as bt
netuid = int(os.environ.get("NETUID", "1"))
endpoint = os.environ.get("CHAIN_ENDPOINT", "")
wallet_path = os.environ.get("WALLET_PATH", "")
name = os.environ.get("VALIDATOR_WALLET_NAME", "")
hk = os.environ.get("VALIDATOR_WALLET_HOTKEY", "")
w = bt.wallet(name=name, hotkey=hk, path=wallet_path)
st = bt.subtensor(network=endpoint)
if not st.is_hotkey_registered(netuid=netuid, hotkey_ss58=w.hotkey.ss58_address):
    print(f"[set-weights-check] PREFLIGHT FAIL: hotkey not registered on netuid {netuid}. Run: btcli subnets register ...", file=sys.stderr)
    sys.exit(1)
print(f"[set-weights-check] preflight_ok netuid={netuid} hotkey={w.hotkey.ss58_address[:16]}...")
PY
  then
    exit 4
  fi
fi

LOG_DIR="$ROOT_DIR/artifacts/stress"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/testnet_set_weights_${STAMP}.log"

export PYTHONPATH="$ROOT_DIR"
CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/neurons/validator.py"
  --wallet.name "$VALIDATOR_WALLET_NAME"
  --wallet.hotkey "$VALIDATOR_WALLET_HOTKEY"
  --wallet.path "$WALLET_PATH"
  --subtensor.chain_endpoint "$CHAIN_ENDPOINT"
  --subtensor.network finney
  --netuid "$NETUID"
  --axon.port "$VALIDATOR_PORT"
  --logging.debug
)

echo "[set-weights-check] logging to: $LOG_FILE"
"${CMD[@]}" >"$LOG_FILE" 2>&1 &
PID="$!"
cleanup() { kill "$PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
sleep "$RUN_SECONDS"
kill "$PID" >/dev/null 2>&1 || true
wait "$PID" 2>/dev/null || true

read -r successes failures < <("$PYTHON_BIN" - "$LOG_FILE" <<'PY'
import re
import sys
from pathlib import Path
p = Path(sys.argv[1])
text = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
ok = len(re.findall(r"set_weights on chain successfully!", text))
bad = len(re.findall(r"set_weights failed|Error during validation step", text))
print(ok, bad)
PY
)
echo "[set-weights-check] successes=$successes failures=$failures log=$LOG_FILE"

if (( successes < MIN_SET_WEIGHTS_SUCCESSES )); then
  echo "[set-weights-check] FAIL: insufficient successful set_weights events."
  exit 2
fi
if (( failures > 0 )); then
  echo "[set-weights-check] FAIL: detected set_weights/validation errors."
  exit 3
fi
echo "[set-weights-check] PASS"
