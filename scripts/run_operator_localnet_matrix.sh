#!/usr/bin/env bash
# Operator matrix: 3 annotation-only miners + validator with paced steps.
# First-time chain: scripts/bootstrap_localnet_and_register.sh (or your own subnet + register flow).
# Prerequisites: local subtensor RPC (default ws://127.0.0.1:9944), miner+validator registered on NETUID;
# Cloudflare R2 env vars set for real uploads.
# Quick dev pacing: FORWARD_STEP_SLEEP_SECONDS=30 RUN_SECONDS=300 ./scripts/run_operator_localnet_matrix.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Optional: shared staging credentials (same bucket, per-miner prefixes below).
if [[ -f "$ROOT_DIR/staging.env" ]]; then
  # shellcheck source=/dev/null
  set -a && source "$ROOT_DIR/staging.env" && set +a
fi

NEURON_PYTHON="${NEURON_PYTHON:-$ROOT_DIR/.venv-neurons/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-$NEURON_PYTHON}"
BTCLI_BIN="${BTCLI_BIN:-$ROOT_DIR/.venv-btcli/bin/btcli}"

CHAIN_ENDPOINT="${CHAIN_ENDPOINT:-ws://127.0.0.1:9944}"
NETUID="${NETUID:-2}"
WALLET_PATH="${WALLET_PATH:-$HOME/.bittensor/wallets}"

VALIDATOR_WALLET_NAME="${VALIDATOR_WALLET_NAME:-validator}"
VALIDATOR_WALLET_HOTKEY="${VALIDATOR_WALLET_HOTKEY:-valhk}"
VALIDATOR_PORT="${VALIDATOR_PORT:-8092}"

# All miners follow the annotation-only path and may use different external VLM configurations.
# MINER_COUNT=2 when local subnet is full at 4 neurons (owner + validator + 2 miners).
MINER_COUNT="${MINER_COUNT:-3}"
if [[ "$MINER_COUNT" -eq 2 ]]; then
  MINER_WALLET_NAMES="${MINER_WALLET_NAMES:-miner,miner2}"
  MINER_WALLET_HOTKEYS="${MINER_WALLET_HOTKEYS:-minerhk,minerhk2}"
  MINER_PORTS="${MINER_PORTS:-8091,8093}"
  R2_PREFIXES="${R2_PREFIXES:-miners/m1_karpathy,miners/m2_rand_hpo}"
  NEURON_SAMPLE_SIZE="${NEURON_SAMPLE_SIZE:-2}"
else
  MINER_WALLET_NAMES="${MINER_WALLET_NAMES:-miner,miner2,miner3}"
  MINER_WALLET_HOTKEYS="${MINER_WALLET_HOTKEYS:-minerhk,minerhk2,minerhk3}"
  MINER_PORTS="${MINER_PORTS:-8091,8093,8094}"
  R2_PREFIXES="${R2_PREFIXES:-miners/m1_karpathy,miners/m2_rand_hpo,miners/m3_rand_hpo}"
fi

RUN_SECONDS="${RUN_SECONDS:-600}"
MAX_TRAINING_SECONDS="${MAX_TRAINING_SECONDS:-300}"
TRAINING_TIMEOUT="${TRAINING_TIMEOUT:-900}"
NEURON_SAMPLE_SIZE="${NEURON_SAMPLE_SIZE:-3}"
# Pacing between successful validator steps (not a passive R2 poller).
# Default 300s ~= "5-minute watch" in operator docs — forward_step_sleep_seconds only.
FORWARD_STEP_SLEEP_SECONDS="${FORWARD_STEP_SLEEP_SECONDS:-300}"

AUTORESEARCH_MAX_ITERS="${AUTORESEARCH_MAX_ITERS:-2}"
AUTORESEARCH_EXPERIMENT_MINUTES="${AUTORESEARCH_EXPERIMENT_MINUTES:-1}"

log() { printf '[operator-matrix] %s\n' "$*"; }

require_r2() {
  local miss=0
  for v in R2_ACCOUNT_ID R2_BUCKET_NAME R2_S3_ENDPOINT R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY; do
    if [[ -z "${!v:-}" ]]; then
      log "ERROR: export $v for dual-flywheel R2 uploads."
      miss=1
    fi
  done
  if [[ "$miss" -ne 0 ]]; then
    exit 2
  fi
}

http_from_ws() { echo "${1/ws:\/\//http://}"; }

chain_ok() {
  local http out
  http="$(http_from_ws "$CHAIN_ENDPOINT")"
  out="$(curl -sS -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}' "$http" 2>/dev/null || true)"
  [[ -n "$out" ]] && [[ "$out" == *"result"* ]]
}

fund_from_alice() {
  local amount="${FUND_AMOUNT:-500}"
  local dest name hk
  log "Funding coldkeys from wallet 'alice' ($amount τ each) — set SKIP_FUND=1 to skip."
  if [[ "${SKIP_FUND:-0}" == "1" ]]; then
    return 0
  fi
  IFS=',' read -r -a NAMES <<< "$MINER_WALLET_NAMES"
  IFS=',' read -r -a HOTKEYS <<< "$MINER_WALLET_HOTKEYS"
  local targets=(owner "$VALIDATOR_WALLET_NAME")
  for n in "${NAMES[@]}"; do targets+=("$n"); done
  for name in "${targets[@]}"; do
    hk="${VALIDATOR_WALLET_HOTKEY}"
    case "$name" in
      owner) hk="${OWNER_HOTKEY:-ownerhk}" ;;
      validator) hk="$VALIDATOR_WALLET_HOTKEY" ;;
      *)
        idx=-1
        for i in "${!NAMES[@]}"; do
          if [[ "${NAMES[$i]}" == "$name" ]]; then idx=$i; break; fi
        done
        if [[ "$idx" -lt 0 ]]; then continue; fi
        hk="${HOTKEYS[$idx]}"
        ;;
    esac
    dest="$("$NEURON_PYTHON" - "$WALLET_PATH" "$name" "$hk" <<'PY'
import bittensor as bt, sys
path, cold, hot = sys.argv[1:4]
w = bt.wallet(name=cold, hotkey=hot, path=path)
print(w.coldkey.ss58_address)
PY
)"
    log "transfer $amount τ alice -> $name coldkey $dest"
    printf '\n\n\n\n\n' | "$BTCLI_BIN" wallet transfer \
      --wallet-path "$WALLET_PATH" \
      --wallet-name alice \
      --network "$CHAIN_ENDPOINT" \
      --destination "$dest" \
      --amount "$amount" \
      --no-prompt || log "WARN: transfer failed (already funded or no alice wallet?)"
  done
}

register_extra_if_needed() {
  log "Registering extra miners (AUTO_FUND from owner if set)..."
  local extra_default="miner2:minerhk2,miner3:minerhk3"
  if [[ "$MINER_COUNT" -eq 2 ]]; then
    extra_default="miner2:minerhk2"
  fi
  AUTO_FUND="${REGISTER_AUTO_FUND:-1}" FUNDER_WALLET="${FUNDER_WALLET:-owner}" \
    FUND_AMOUNT="${FUND_AMOUNT_REGISTER:-2.0}" \
    NETUID="$NETUID" CHAIN_ENDPOINT="$CHAIN_ENDPOINT" WALLET_PATH="$WALLET_PATH" \
    EXTRA_MINERS="${EXTRA_MINERS:-$extra_default}" \
    bash "$ROOT_DIR/scripts/register_extra_miners_localnet.sh" || true
}

require_r2
if ! chain_ok; then
  log "ERROR: no JSON-RPC at $(http_from_ws "$CHAIN_ENDPOINT"). Start subtensor / local node first."
  exit 1
fi

fund_from_alice
register_extra_if_needed

unset LOCALNET_TARGET_MINER_SS58 || true
export PYTHONPATH="$ROOT_DIR"
# On local ws RPC, validator skips set_weights unless forced (neurons/validator.py).
case "${CHAIN_ENDPOINT:-}" in
  ws://127.0.0.1:*|ws://localhost:*)
    export FORCE_LOCAL_SET_WEIGHTS="${FORCE_LOCAL_SET_WEIGHTS:-1}"
    ;;
esac
export MINER_MAX_TRAIN_SAMPLES="${MINER_MAX_TRAIN_SAMPLES:-24}"
export MINER_MAX_VAL_SAMPLES="${MINER_MAX_VAL_SAMPLES:-8}"
export MINER_MAX_EPOCHS="${MINER_MAX_EPOCHS:-1}"

LOG_DIR="$ROOT_DIR/artifacts/operator_matrix"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

IFS=',' read -r -a NAMES <<< "$MINER_WALLET_NAMES"
IFS=',' read -r -a HOTKEYS <<< "$MINER_WALLET_HOTKEYS"
IFS=',' read -r -a PORTS <<< "$MINER_PORTS"
IFS=',' read -r -a PREFIXES <<< "$R2_PREFIXES"
IFS=',' read -r -a BACKENDS <<< "${MINER_BACKENDS:-yolo,yolo,yolo}"

if [[ "${#NAMES[@]}" -ne "$MINER_COUNT" || "${#HOTKEYS[@]}" -ne "$MINER_COUNT" || "${#PORTS[@]}" -ne "$MINER_COUNT" || "${#PREFIXES[@]}" -ne "$MINER_COUNT" ]]; then
  log "ERROR: need MINER_COUNT=$MINER_COUNT entries each: MINER_WALLET_NAMES, MINER_WALLET_HOTKEYS, MINER_PORTS, R2_PREFIXES."
  exit 1
fi

# Route dendrite to real listen ports (chain axon ports are often stale on single-host localnet).
SS58_MAP=""
for ((i = 0; i < MINER_COUNT; i++)); do
  ss58="$("$NEURON_PYTHON" - "$WALLET_PATH" "${NAMES[$i]}" "${HOTKEYS[$i]}" <<'PY'
import bittensor as bt, sys
path, name, hk = sys.argv[1:4]
w = bt.wallet(name=name, hotkey=hk, path=path)
print(w.hotkey.ss58_address)
PY
)"
  if [[ -n "$SS58_MAP" ]]; then
    SS58_MAP+=","
  fi
  SS58_MAP+="${ss58}=${PORTS[$i]}"
done
export LOCALNET_MINER_PORT_BY_SS58="$SS58_MAP"
log "LOCALNET_MINER_PORT_BY_SS58=$LOCALNET_MINER_PORT_BY_SS58"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do kill "$pid" >/dev/null 2>&1 || true; done
}
trap cleanup EXIT INT TERM

for ((i = 0; i < MINER_COUNT; i++)); do
  mlog="$LOG_DIR/miner_${i}_${STAMP}.log"
  cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/neurons/miner.py"
    --wallet.name "${NAMES[$i]}"
    --wallet.hotkey "${HOTKEYS[$i]}"
    --wallet.path "$WALLET_PATH"
    --subtensor.network local
    --subtensor.chain_endpoint "$CHAIN_ENDPOINT"
    --netuid "$NETUID"
    --axon.port "${PORTS[$i]}"
    --miner.annotation_workspace "$ROOT_DIR/artifacts/miner_annotation/${HOTKEYS[$i]}"
    --miner.dual_flywheel_r2_prefix "${PREFIXES[$i]}"
    --miner.annotation_backend "${BACKENDS[$i]:-${MINER_ANNOTATION_BACKEND:-yolo}}"
    --logging.debug
  )
  log "miner[$i] backend=${BACKENDS[$i]:-yolo} R2=${PREFIXES[$i]} log=$mlog"
  "${cmd[@]}" >"$mlog" 2>&1 &
  PIDS+=("$!")
done

MINER_WARMUP_SECONDS="${MINER_WARMUP_SECONDS:-20}"
log "waiting ${MINER_WARMUP_SECONDS}s for miner axons before validator (MINER_WARMUP_SECONDS)"
sleep "$MINER_WARMUP_SECONDS"

vlog="$LOG_DIR/validator_${STAMP}.log"
log "validator forward_step_sleep=${FORWARD_STEP_SLEEP_SECONDS}s log=$vlog"

VALIDATOR_CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/neurons/validator.py"
  --wallet.name "$VALIDATOR_WALLET_NAME"
  --wallet.hotkey "$VALIDATOR_WALLET_HOTKEY"
  --wallet.path "$WALLET_PATH"
  --subtensor.network local
  --subtensor.chain_endpoint "$CHAIN_ENDPOINT"
  --netuid "$NETUID"
  --axon.port "$VALIDATOR_PORT"
  --neuron.sample_size "$NEURON_SAMPLE_SIZE"
  --neuron.forward_step_sleep_seconds "$FORWARD_STEP_SLEEP_SECONDS"
  --logging.debug
)
[[ -n "${FLYWHEEL_COCO_MANIFEST:-}" ]] && VALIDATOR_CMD+=(--neuron.flywheel_coco_manifest "$FLYWHEEL_COCO_MANIFEST")
[[ -n "${FLYWHEEL_COMMERCIAL_DATASET_PREFIX:-}" ]] && VALIDATOR_CMD+=(--neuron.flywheel_commercial_dataset_prefix "$FLYWHEEL_COMMERCIAL_DATASET_PREFIX")
[[ -n "${FLYWHEEL_IMAGE_CACHE_ROOT:-}" ]] && VALIDATOR_CMD+=(--neuron.flywheel_image_cache_root "$FLYWHEEL_IMAGE_CACHE_ROOT")
[[ -n "${FLYWHEEL_GOLDEN_MISSING_PENALTY:-}" ]] && VALIDATOR_CMD+=(--neuron.flywheel_golden_missing_penalty "$FLYWHEEL_GOLDEN_MISSING_PENALTY")
[[ -n "${FLYWHEEL_COMMERCIAL_EXPORT_EVERY:-}" ]] && VALIDATOR_CMD+=(--neuron.flywheel_commercial_export_every "$FLYWHEEL_COMMERCIAL_EXPORT_EVERY")
[[ -n "${FLYWHEEL_ANNOTATION_REQUEST_SIZE:-}" ]] && VALIDATOR_CMD+=(--neuron.flywheel_annotation_request_size "$FLYWHEEL_ANNOTATION_REQUEST_SIZE")
[[ -n "${FLYWHEEL_GOLDEN_INJECTION_PER_REQUEST:-}" ]] && VALIDATOR_CMD+=(--neuron.flywheel_golden_injection_per_request "$FLYWHEEL_GOLDEN_INJECTION_PER_REQUEST")
[[ -n "${NEURON_ANNOTATION_TIMEOUT:-}" ]] && VALIDATOR_CMD+=(--neuron.annotation_timeout "$NEURON_ANNOTATION_TIMEOUT")
"${VALIDATOR_CMD[@]}" >"$vlog" 2>&1 &
PIDS+=("$!")

log "running ${RUN_SECONDS}s (set RUN_SECONDS to change)"
sleep "$RUN_SECONDS"
cleanup
log "done. Logs under $LOG_DIR"

if [[ "${SKIP_VALIDATOR_GATE:-0}" != "1" ]]; then
  export GATE_JSON_OUT="${GATE_JSON_OUT:-$LOG_DIR/gate_${STAMP}.json}"
  log "validator release gate -> $GATE_JSON_OUT"
  if ! "$PYTHON_BIN" "$ROOT_DIR/scripts/validator_gate_report.py" "$vlog"; then
    log "VALIDATOR GATE: NO-GO (set SKIP_VALIDATOR_GATE=1 to bypass)"
    exit 1
  fi
  log "VALIDATOR GATE: GO"
fi
