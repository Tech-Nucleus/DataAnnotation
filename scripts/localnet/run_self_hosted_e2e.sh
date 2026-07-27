#!/usr/bin/env bash
# Self-Hosted Backend E2E (REAL chain path): reference YOLO server + N self_hosted miners
# + validator, exercising the full annotation flywheel with REAL dendrite/axon networking.
#
# IMPORTANT: This does NOT use --test-mode. In test mode the validator uses MockDendrite, which
# fabricates the miner's annotations_uri locally and never calls the miner process or the
# self-hosted server — so the real train->infer->upload->score round cannot happen.
#
# Two miners are launched by default because the validator's Bayesian fusion needs >=2
# voters per image to accept an annotation into the commercial dataset (single-miner rounds
# are escalated, never exported).
#
# Prerequisites:
#   - Local subtensor RPC at $CHAIN_ENDPOINT (default ws://127.0.0.1:9944)
#   - Wallets registered on $NETUID: validator (owner/ownerhk = uid0) + miners (miner*, ...)
#   - R2 credentials in .env (real Cloudflare R2, or MinIO)
#
# Usage:
#   ./scripts/localnet/run_self_hosted_e2e.sh
#   MINER_COUNT=2 RUN_SECONDS=1500 ./scripts/localnet/run_self_hosted_e2e.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then set -a && source "$ROOT_DIR/.env" && set +a; fi
if [[ -f "$ROOT_DIR/staging.env" ]]; then set -a && source "$ROOT_DIR/staging.env" && set +a; fi

NEURON_PYTHON="${NEURON_PYTHON:-$ROOT_DIR/.venv-neurons/bin/python}"
CHAIN_ENDPOINT="${CHAIN_ENDPOINT:-ws://127.0.0.1:9944}"
NETUID="${NETUID:-2}"
WALLET_PATH="${WALLET_PATH:-$HOME/.bittensor/wallets}"

# --- Validator wallet (must be registered on NETUID; uid0 = owner/ownerhk on this localnet) ---
VALIDATOR_WALLET_NAME="${VALIDATOR_WALLET_NAME:-owner}"
VALIDATOR_WALLET_HOTKEY="${VALIDATOR_WALLET_HOTKEY:-ownerhk}"
VALIDATOR_PORT="${VALIDATOR_PORT:-8092}"

# --- Miners (comma-separated, must be registered) ---
MINER_COUNT="${MINER_COUNT:-2}"
if [[ "$MINER_COUNT" -eq 1 ]]; then
  MINER_WALLET_NAMES="${MINER_WALLET_NAMES:-miner}"
  MINER_WALLET_HOTKEYS="${MINER_WALLET_HOTKEYS:-minerhk}"
  MINER_PORTS="${MINER_PORTS:-8091}"
  MINER_R2_PREFIXES="${MINER_R2_PREFIXES:-localnet/self_hosted_m1}"
else
  MINER_WALLET_NAMES="${MINER_WALLET_NAMES:-miner,miner2}"
  MINER_WALLET_HOTKEYS="${MINER_WALLET_HOTKEYS:-minerhk,minerhk2}"
  MINER_PORTS="${MINER_PORTS:-8091,8093}"
  MINER_R2_PREFIXES="${MINER_R2_PREFIXES:-localnet/self_hosted_m1,localnet/self_hosted_m2}"
fi

# --- Miner backend: self_hosted (reference YOLO server) | yolo_local (in-miner YOLO) ---
MINER_BACKEND="${MINER_BACKEND:-self_hosted}"

# --- Self-hosted server (used when MINER_BACKEND=self_hosted) ---
SELF_HOSTED_PORT="${SELF_HOSTED_PORT:-8081}"
SELF_HOSTED_API_KEY="${SELF_HOSTED_API_KEY:-dummy}"
SELF_HOSTED_ADVERSARIAL_RANDOM_BOXES="${SELF_HOSTED_ADVERSARIAL_RANDOM_BOXES:-0}"
SELF_HOSTED_TRAIN_EPOCHS="${SELF_HOSTED_TRAIN_EPOCHS:-5}"
SELF_HOSTED_TRAIN_IMGSZ="${SELF_HOSTED_TRAIN_IMGSZ:-640}"
# Adversarial good-vs-bad test: start a 2nd server returning random boxes and point
# the LAST miner at it; that miner should earn near-zero reward.
ADVERSARIAL_SECOND_SERVER="${ADVERSARIAL_SECOND_SERVER:-0}"
SELF_HOSTED_PORT2="${SELF_HOSTED_PORT2:-8082}"

# --- yolo_local backend params (used when MINER_BACKEND=yolo_local) ---
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolov8n.pt}"
YOLO_EPOCHS="${YOLO_EPOCHS:-3}"
YOLO_IMGSZ="${YOLO_IMGSZ:-416}"

# --- Dataset / flywheel ---
COCO_OUT="${COCO_OUT:-$ROOT_DIR/artifacts/localnet/coco200}"
MANIFEST="${COCO_MANIFEST:-$COCO_OUT/manifest.json}"
COCO_SIZE="${COCO_DATASET_SIZE:-200}"
GOLDEN_RATIO="${COCO_GOLDEN_RATIO:-0.1}"
ANNOTATION_REQUEST_SIZE="${ANNOTATION_REQUEST_SIZE:-15}"
GOLDEN_INJECTION_PER_REQUEST="${GOLDEN_INJECTION_PER_REQUEST:-5}"
TRAIN_SPLIT_PCT="${TRAIN_SPLIT_PCT:-70}"
SELF_HOSTED_POLL_INTERVAL_SECONDS="${SELF_HOSTED_POLL_INTERVAL_SECONDS:-5}"

# --- Run pacing ---
RUN_SECONDS="${RUN_SECONDS:-1500}"
FORWARD_STEP_SLEEP_SECONDS="${FORWARD_STEP_SLEEP_SECONDS:-45}"
STOP_AFTER_FIRST_ROUND="${STOP_AFTER_FIRST_ROUND:-1}"
export NEURON_ANNOTATION_TIMEOUT="${NEURON_ANNOTATION_TIMEOUT:-1800}"
export FORCE_LOCAL_SET_WEIGHTS="${FORCE_LOCAL_SET_WEIGHTS:-1}"
export PYTHONPATH="$ROOT_DIR"

COMMERCIAL_PREFIX="${COMMERCIAL_PREFIX:-file://$ROOT_DIR/artifacts/localnet/self_hosted_commercial}"
IMAGE_CACHE_ROOT="${IMAGE_CACHE_ROOT:-$ROOT_DIR/artifacts/localnet/self_hosted_image_cache}"
LOGDIR="$ROOT_DIR/artifacts/localnet/logs"
mkdir -p "$LOGDIR" "$ROOT_DIR/artifacts/localnet/self_hosted_commercial" "$IMAGE_CACHE_ROOT"

IFS=',' read -r -a NAMES <<< "$MINER_WALLET_NAMES"
IFS=',' read -r -a HOTKEYS <<< "$MINER_WALLET_HOTKEYS"
IFS=',' read -r -a PORTS <<< "$MINER_PORTS"
IFS=',' read -r -a PREFIXES <<< "$MINER_R2_PREFIXES"
NEURON_SAMPLE_SIZE="${NEURON_SAMPLE_SIZE:-$MINER_COUNT}"

SERVER_PID=""; SERVER2_PID=""; VALIDATOR_PID=""; MINER_PIDS=()
cleanup() {
  echo "[e2e] Cleaning up..."
  [[ -n "$VALIDATOR_PID" ]] && kill "$VALIDATOR_PID" 2>/dev/null || true
  for p in "${MINER_PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true
  [[ -n "$SERVER2_PID" ]] && kill "$SERVER2_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM
log() { printf '[e2e] %s\n' "$*"; }

# ---------------------------------------------------------------------------
log "=== 0) Subtensor health check ==="
HTTP_ENDPOINT="${CHAIN_ENDPOINT/ws:\/\//http://}"; HTTP_ENDPOINT="${HTTP_ENDPOINT/wss:\/\//https://}"
if curl -sS -m 5 -H 'Content-Type: application/json' \
  --data '{"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}' "$HTTP_ENDPOINT" >/dev/null 2>&1; then
  log "Subtensor reachable at $CHAIN_ENDPOINT"
else
  log "ERROR: subtensor not reachable at $CHAIN_ENDPOINT. Start the local node first."; exit 1
fi

# ---------------------------------------------------------------------------
log "=== 1) COCO subset (reuse if present) ==="
if [[ -f "$MANIFEST" ]]; then
  log "Manifest present, reusing: $MANIFEST"
else
  "$NEURON_PYTHON" "$ROOT_DIR/scripts/localnet/prepare_coco_val2017_subset.py" \
    --out-dir "$COCO_OUT" --dataset-size "$COCO_SIZE" --golden-ratio "$GOLDEN_RATIO"
  [[ -f "$MANIFEST" ]] || { log "ERROR: manifest not found after prep"; exit 1; }
fi

# ---------------------------------------------------------------------------
log "=== 2) Resolve miner hotkeys + dendrite port routing ==="
SS58_MAP=""
for ((i = 0; i < MINER_COUNT; i++)); do
  ss58="$("$NEURON_PYTHON" - "$WALLET_PATH" "${NAMES[$i]}" "${HOTKEYS[$i]}" <<'PY'
import bittensor as bt, sys
path, name, hk = sys.argv[1:4]
print(bt.wallet(name=name, hotkey=hk, path=path).hotkey.ss58_address)
PY
)"
  [[ -n "$SS58_MAP" ]] && SS58_MAP+=","
  SS58_MAP+="${ss58}=${PORTS[$i]}"
  log "miner[$i] ${NAMES[$i]}/${HOTKEYS[$i]} ss58=$ss58 -> 127.0.0.1:${PORTS[$i]}"
done
export LOCALNET_MINER_PORT_BY_SS58="$SS58_MAP"
log "LOCALNET_MINER_PORT_BY_SS58=$LOCALNET_MINER_PORT_BY_SS58"

SERVER_URL="http://localhost:$SELF_HOSTED_PORT"
SERVER2_URL="http://localhost:$SELF_HOSTED_PORT2"
SERVER2_PID=""
wait_server() { # url logfile
  for i in $(seq 1 30); do
    curl -sS -m 2 "$1/health" >/dev/null 2>&1 && { log "  server ready: $1 (attempt $i)"; return 0; }
    [[ $i -eq 30 ]] && { log "ERROR: server not ready: $1"; tail -20 "$2"; exit 1; }
    sleep 1
  done
}

# ---------------------------------------------------------------------------
if [[ "$MINER_BACKEND" == "self_hosted" ]]; then
  log "=== 3) Start reference self-hosted server(s) ==="
  SERVER_CMD=(
    "$NEURON_PYTHON" "$ROOT_DIR/scripts/reference_self_hosted_server.py"
    --host 0.0.0.0 --port "$SELF_HOSTED_PORT"
    --train-epochs "$SELF_HOSTED_TRAIN_EPOCHS" --train-imgsz "$SELF_HOSTED_TRAIN_IMGSZ"
  )
  [[ "$SELF_HOSTED_ADVERSARIAL_RANDOM_BOXES" == "1" ]] && SERVER_CMD+=(--test-mode --adversarial-random-boxes)
  "${SERVER_CMD[@]}" > "$LOGDIR/self_hosted_server.log" 2>&1 &
  SERVER_PID=$!
  log "Primary server PID=$SERVER_PID (:$SELF_HOSTED_PORT)"
  wait_server "$SERVER_URL" "$LOGDIR/self_hosted_server.log"

  if [[ "$ADVERSARIAL_SECOND_SERVER" == "1" ]]; then
    "$NEURON_PYTHON" "$ROOT_DIR/scripts/reference_self_hosted_server.py" \
      --host 0.0.0.0 --port "$SELF_HOSTED_PORT2" \
      --train-epochs "$SELF_HOSTED_TRAIN_EPOCHS" --train-imgsz "$SELF_HOSTED_TRAIN_IMGSZ" \
      --test-mode --adversarial-random-boxes > "$LOGDIR/self_hosted_server_adversarial.log" 2>&1 &
    SERVER2_PID=$!
    log "Adversarial server PID=$SERVER2_PID (:$SELF_HOSTED_PORT2) — last miner points here"
    wait_server "$SERVER2_URL" "$LOGDIR/self_hosted_server_adversarial.log"
  fi
else
  log "=== 3) MINER_BACKEND=$MINER_BACKEND — no external server needed ==="
fi

# ---------------------------------------------------------------------------
log "=== 4) Start $MINER_COUNT $MINER_BACKEND miner(s) ==="
for ((i = 0; i < MINER_COUNT; i++)); do
  mlog="$LOGDIR/miner_${i}_self_hosted.log"
  MINER_CMD=(
    "$NEURON_PYTHON" "$ROOT_DIR/neurons/miner.py"
    --wallet.name "${NAMES[$i]}" --wallet.hotkey "${HOTKEYS[$i]}" --wallet.path "$WALLET_PATH"
    --subtensor.network local --subtensor.chain_endpoint "$CHAIN_ENDPOINT"
    --netuid "$NETUID" --axon.port "${PORTS[$i]}"
    --miner.model_backend "$MINER_BACKEND"
    --miner.train_split_pct "$TRAIN_SPLIT_PCT"
    --miner.force_retrain
    --miner.annotation_workspace "$ROOT_DIR/artifacts/miner_annotation/${HOTKEYS[$i]}"
    --miner.dual_flywheel_r2_prefix "${PREFIXES[$i]}"
    --logging.debug
  )
  if [[ "$MINER_BACKEND" == "self_hosted" ]]; then
    murl="$SERVER_URL"; tag="good"
    if [[ "$ADVERSARIAL_SECOND_SERVER" == "1" && "$i" -eq $((MINER_COUNT - 1)) ]]; then
      murl="$SERVER2_URL"; tag="ADVERSARIAL"
    fi
    MINER_CMD+=(
      --miner.self_hosted_train_url "$murl/train"
      --miner.self_hosted_infer_url "$murl/infer"
      --miner.self_hosted_api_key "$SELF_HOSTED_API_KEY"
      --miner.self_hosted_poll_interval_seconds "$SELF_HOSTED_POLL_INTERVAL_SECONDS"
    )
    log "miner[$i] ${NAMES[$i]} backend=self_hosted server=$murl ($tag)"
  else
    MINER_CMD+=(
      --miner.yolo_pretrained_weights "$YOLO_WEIGHTS"
      --miner.yolo_epochs "$YOLO_EPOCHS"
      --miner.yolo_imgsz "$YOLO_IMGSZ"
    )
    log "miner[$i] ${NAMES[$i]} backend=yolo_local weights=$YOLO_WEIGHTS epochs=$YOLO_EPOCHS"
  fi
  "${MINER_CMD[@]}" > "$mlog" 2>&1 &
  MINER_PIDS+=("$!")
  log "miner[$i] ${NAMES[$i]} PID=${MINER_PIDS[$i]} axon=:${PORTS[$i]} R2=${PREFIXES[$i]} log=$mlog"
done

# Wait for all miner axons
for ((i = 0; i < MINER_COUNT; i++)); do
  for t in $(seq 1 30); do
    (exec 3<>"/dev/tcp/127.0.0.1/${PORTS[$i]}") 2>/dev/null && { exec 3>&- 3<&-; log "miner[$i] axon listening (attempt $t)"; break; }
    if ! kill -0 "${MINER_PIDS[$i]}" 2>/dev/null; then log "ERROR: miner[$i] died"; tail -40 "$LOGDIR/miner_${i}_self_hosted.log"; exit 1; fi
    [[ $t -eq 30 ]] && { log "ERROR: miner[$i] axon not listening"; tail -40 "$LOGDIR/miner_${i}_self_hosted.log"; exit 1; }
    sleep 2
  done
done

# ---------------------------------------------------------------------------
log "=== 5) Start validator ($VALIDATOR_WALLET_NAME/$VALIDATOR_WALLET_HOTKEY) sample_size=$NEURON_SAMPLE_SIZE ==="
"$NEURON_PYTHON" "$ROOT_DIR/neurons/validator.py" \
  --wallet.name "$VALIDATOR_WALLET_NAME" --wallet.hotkey "$VALIDATOR_WALLET_HOTKEY" --wallet.path "$WALLET_PATH" \
  --subtensor.network local --subtensor.chain_endpoint "$CHAIN_ENDPOINT" \
  --netuid "$NETUID" --axon.port "$VALIDATOR_PORT" \
  --neuron.sample_size "$NEURON_SAMPLE_SIZE" \
  --neuron.forward_step_sleep_seconds "$FORWARD_STEP_SLEEP_SECONDS" \
  --neuron.annotation_timeout "$NEURON_ANNOTATION_TIMEOUT" \
  --neuron.flywheel_coco_manifest "$MANIFEST" \
  --neuron.flywheel_annotation_request_size "$ANNOTATION_REQUEST_SIZE" \
  --neuron.flywheel_golden_injection_per_request "$GOLDEN_INJECTION_PER_REQUEST" \
  --neuron.flywheel_commercial_export_every 1 \
  --neuron.flywheel_commercial_dataset_prefix "$COMMERCIAL_PREFIX" \
  --neuron.flywheel_image_cache_root "$IMAGE_CACHE_ROOT" \
  --logging.debug \
  > "$LOGDIR/validator_self_hosted.log" 2>&1 &
VALIDATOR_PID=$!
log "Validator PID=$VALIDATOR_PID"

# ---------------------------------------------------------------------------
log "=== 6) Running ${RUN_SECONDS}s (annotation_timeout=${NEURON_ANNOTATION_TIMEOUT}s) ==="
ELAPSED=0; POLL=30
while [[ $ELAPSED -lt $RUN_SECONDS ]]; do
  sleep "$POLL"; ELAPSED=$((ELAPSED + POLL))
  if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "ERROR: server died at ${ELAPSED}s"; tail -30 "$LOGDIR/self_hosted_server.log"; exit 1
  fi
  kill -0 "$VALIDATOR_PID" 2>/dev/null || { log "WARNING: validator exited at ${ELAPSED}s"; tail -40 "$LOGDIR/validator_self_hosted.log"; }
  rounds=$(grep -c "event=annotation_flywheel_round_done" "$LOGDIR/validator_self_hosted.log" 2>/dev/null) || rounds=0
  log "Heartbeat ${ELAPSED}/${RUN_SECONDS}s — rounds_done=$rounds"
  [[ "$STOP_AFTER_FIRST_ROUND" == "1" && "$rounds" -ge 1 ]] && { log "First round complete — stopping early"; break; }
done

# ---------------------------------------------------------------------------
log "=== 7) Verification ==="
PASS=true
SL="$LOGDIR/self_hosted_server.log"; VL="$LOGDIR/validator_self_hosted.log"; M0="$LOGDIR/miner_0_self_hosted.log"
check() { if grep -qiE "$2" "$1" 2>/dev/null; then log "  ✓ $3"; else log "  ✗ $3"; PASS=false; fi; }

if [[ "$MINER_BACKEND" == "self_hosted" ]]; then
  log "-- server --"
  check "$SL" "\[train\]"  "server handled /train"
  check "$SL" "\[infer\]"  "server handled /infer"
  check "$SL" "Training completed — version=" "server captured fine-tuned checkpoint"
  log "-- miner[0] --"
  check "$M0" "SelfHostedBackend: POST .*/train" "miner sent /train"
  check "$M0" "SelfHostedBackend: POST .*/infer" "miner sent /infer"
else
  log "-- miner[0] (yolo_local) --"
  check "$M0" "YoloLocalBackend: starting training" "miner trained YOLO locally"
  check "$M0" "YoloLocalBackend: running inference" "miner ran local inference"
fi
check "$M0" "task .* complete .* uri=" "miner uploaded annotations + returned uri"
log "-- validator --"
check "$VL" "event=training_pool_built" "validator built training pool"
check "$VL" "event=evaluator_golden_score_payload" "validator scored golden set"
check "$VL" "event=annotation_flywheel_round_done" "validator completed reward round"
# A commercial export needs >=2 honest miners forming consensus; not expected for a
# single-miner run or an adversarial good-vs-bad run (random boxes don't cluster).
if [[ "$MINER_COUNT" -ge 2 && "$ADVERSARIAL_SECOND_SERVER" != "1" ]]; then
  if grep -qE "event=annotation_flywheel_commercial_export uri=.+" "$VL" 2>/dev/null; then
    log "  ✓ validator exported commercial dataset (non-empty uri)"
  else
    log "  ✗ validator commercial export empty (need >=2 voters per image)"; PASS=false
  fi
else
  log "  • commercial export not required for this config (single honest miner / adversarial)"
fi

log "==================================="
$PASS && log "✓ REAL E2E PASSED (backend=$MINER_BACKEND)" || log "✗ REAL E2E FAILED (backend=$MINER_BACKEND) — see $LOGDIR/"
log "==================================="
$PASS || exit 1
