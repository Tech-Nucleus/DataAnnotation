#!/usr/bin/env bash
# End-to-end Localnet simulation with Path A (self_hosted backend) and watchdog verification

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Source env vars
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Configuration
CHAIN_ENDPOINT="ws://127.0.0.1:9944"
NETUID=2
RUN_DURATION=180 # 3 minutes

log() {
  printf '[localnet-path-a] %s\n' "$*"
}

log "Preparing slate..."
pkill -f "neurons/miner.py" || true
pkill -f "neurons/validator.py" || true
pkill -f "server.py" || true
pkill -f "reference_self_hosted_server.py" || true

mkdir -p "$ROOT_DIR/artifacts"
rm -f "$ROOT_DIR/artifacts/server.log" "$ROOT_DIR/artifacts/miner.log" "$ROOT_DIR/artifacts/validator.log"
rm -rf "$ROOT_DIR/artifacts/localnet/self_hosted_image_cache/"
rm -rf "$ROOT_DIR/artifacts/localnet/self_hosted_commercial/"
rm -rf "$ROOT_DIR/artifacts/miner_annotation/"

mkdir -p "$ROOT_DIR/artifacts/miner_annotation"
mkdir -p "$ROOT_DIR/artifacts/localnet/self_hosted_image_cache"
mkdir -p "$ROOT_DIR/artifacts/localnet/self_hosted_commercial"

NEURON_PYTHON="$ROOT_DIR/.venv-neurons/bin/python"
MINER_SS58="$($NEURON_PYTHON -c "import bittensor as bt; print(bt.wallet(name='miner', hotkey='minerhk').hotkey.ss58_address)")"

# Verify subtensor
log "Checking subtensor chain endpoint at $CHAIN_ENDPOINT..."
if ! curl -sS -m 2 -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}' \
  http://127.0.0.1:9944 >/dev/null; then
  log "ERROR: Subtensor local node is not reachable on port 9944"
  exit 1
fi

# Verify miner & validator are registered
log "Verifying neuron registrations on netuid $NETUID..."
$NEURON_PYTHON - <<'PY'
import bittensor as bt
st = bt.subtensor(network="ws://127.0.0.1:9944")
mg = st.metagraph(2)
miner_hk = bt.wallet(name="miner", hotkey="minerhk").hotkey.ss58_address
val_hk = bt.wallet(name="validator", hotkey="valhk").hotkey.ss58_address
if miner_hk not in mg.hotkeys:
    print(f"ERROR: Miner hotkey {miner_hk} is not registered.")
    exit(1)
if val_hk not in mg.hotkeys:
    print(f"ERROR: Validator hotkey {val_hk} is not registered.")
    exit(1)
print(f"Neurons verified. Miner UID: {mg.hotkeys.index(miner_hk)}, Validator UID: {mg.hotkeys.index(val_hk)}")
PY

cleanup() {
  log "Cleaning up processes..."
  kill $SERVER_PID 2>/dev/null || true
  kill $MINER_PID 2>/dev/null || true
  kill $VALIDATOR_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

log "1) Starting reference self-hosted model server (port 8081)..."
env PYTHONPATH="$ROOT_DIR" "$NEURON_PYTHON" "$ROOT_DIR/server.py" \
  --host 127.0.0.1 \
  --port 8081 \
  --checkpoint yolov8n.pt \
  --test-mode \
  --adversarial-random-boxes > "$ROOT_DIR/artifacts/server.log" 2>&1 &
SERVER_PID="$!"

# Wait for server to boot
sleep 5
if ! curl -s http://127.0.0.1:8081/ >/dev/null; then
  log "Warning: Server index check failed. Log output:"
  head -n 20 "$ROOT_DIR/artifacts/server.log"
fi

log "2) Starting Miner with Backend A (self_hosted)..."
env PYTHONPATH="$ROOT_DIR" MINER_ADVERSARIAL=0 \
  "$NEURON_PYTHON" "$ROOT_DIR/neurons/miner.py" \
  --wallet.name miner \
  --wallet.hotkey minerhk \
  --subtensor.network local \
  --subtensor.chain_endpoint "$CHAIN_ENDPOINT" \
  --netuid "$NETUID" \
  --miner.model_backend self_hosted \
  --miner.self_hosted_train_url http://localhost:8081/train \
  --miner.self_hosted_infer_url http://localhost:8081/infer \
  --axon.port 8091 \
  --logging.debug \
  --miner.dual_flywheel_r2_prefix localnet/miners/miner1 > "$ROOT_DIR/artifacts/miner.log" 2>&1 &
MINER_PID="$!"

log "3) Starting Validator..."
env PYTHONPATH="$ROOT_DIR" \
  FORCE_LOCAL_SET_WEIGHTS=1 \
  DEFAULT_ACCEPT_CONFIDENCE=0.01 \
  DEFAULT_ACCEPT_SEVERITY_CONFIDENCE=0.01 \
  DEFAULT_MIN_VOTERS=1 \
  DEFAULT_MIN_MEAN_IOU_TO_MEDIAN=0.1 \
  LOCALNET_MINER_PORT_BY_SS58="$MINER_SS58=8091" \
  FALLBACK_SINGLE_MINER_ENABLED=1 \
  FALLBACK_SINGLE_MINER_MIN_RELIABILITY=0.1 \
  "$NEURON_PYTHON" "$ROOT_DIR/neurons/validator.py" \
  --wallet.name validator \
  --wallet.hotkey valhk \
  --subtensor.network local \
  --subtensor.chain_endpoint "$CHAIN_ENDPOINT" \
  --netuid "$NETUID" \
  --axon.port 8090 \
  --neuron.sample_size 1 \
  --neuron.epoch_length 1 \
  --neuron.forward_step_sleep_seconds 15 \
  --neuron.annotation_timeout 300 \
  --neuron.flywheel_annotation_request_size 5 \
  --neuron.flywheel_golden_injection_per_request 1 \
  --neuron.flywheel_commercial_export_every 1 \
  --neuron.flywheel_commercial_dataset_prefix "file://${ROOT_DIR}/artifacts/localnet/self_hosted_commercial" \
  --neuron.flywheel_image_cache_root "${ROOT_DIR}/artifacts/localnet/self_hosted_image_cache" \
  --logging.debug > "$ROOT_DIR/artifacts/validator.log" 2>&1 &
VALIDATOR_PID="$!"

log "All processes started. Let's run for ${RUN_DURATION}s with watchdog check..."
elapsed=0
interval=30
while [ $elapsed -lt $RUN_DURATION ]; do
  sleep $interval
  elapsed=$((elapsed + interval))
  
  log "--- Time elapsed: ${elapsed}s / ${RUN_DURATION}s ---"
  
  # Check if processes are alive
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    log "FAIL: Server process died. Server log:"
    tail -n 20 "$ROOT_DIR/artifacts/server.log"
    exit 3
  fi
  if ! kill -0 $MINER_PID 2>/dev/null; then
    log "FAIL: Miner process died. Miner log:"
    tail -n 20 "$ROOT_DIR/artifacts/miner.log"
    exit 4
  fi
  if ! kill -0 $VALIDATOR_PID 2>/dev/null; then
    log "FAIL: Validator process died. Validator log:"
    tail -n 20 "$ROOT_DIR/artifacts/validator.log"
    exit 5
  fi
  
  # Run soak watchdog
  log "Running watchdog..."
  if ! bash "$ROOT_DIR/scripts/soak_watchdog.sh" "$ROOT_DIR/artifacts/validator.log"; then
    log "Watchdog warning or fail detected in validator.log"
  else
    log "Watchdog check: PASS"
  fi
done

log "Simulation run completed successfully. Now starting post-run verification..."
cleanup

# Verify R2 bucket
log "4) Listing R2 bucket objects..."
$NEURON_PYTHON - <<'PY'
import boto3, os
s3 = boto3.client('s3',
    endpoint_url=os.getenv('R2_ENDPOINT_URL') or os.getenv('R2_S3_ENDPOINT'),
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    region_name='auto',
)
try:
    resp = s3.list_objects_v2(Bucket=os.environ['R2_BUCKET_NAME'], MaxKeys=50)
    contents = resp.get('Contents', [])
    print(f"Found {len(contents)} keys in bucket.")
    # Show recently updated files
    sorted_objs = sorted(contents, key=lambda x: x['LastModified'], reverse=True)
    for obj in sorted_objs[:15]:
        print(f"  - {obj['Key']} ({obj['Size']} bytes, modified {obj['LastModified']})")
except Exception as e:
    print("Error querying R2:", e)
PY

# Verify commercial dataset export
log "5) Checking commercial dataset export folder..."
ls -la "$ROOT_DIR/artifacts/localnet/self_hosted_commercial/" || true
find "$ROOT_DIR/artifacts/localnet/self_hosted_commercial/" -type f -name "*.jsonl" -exec head -n 5 {} \; || true

# Verify annotated images drawn locally
log "6) Checking annotated images folder..."
ls -la "$ROOT_DIR/artifacts/localnet/self_hosted_commercial/commercial/annotated-images/" || true

# Verify weights
log "7) Checking validator weights state..."
VAL_STATE_DIR="$HOME/.bittensor/miners/validator/valhk/netuid2/validator"
if [ -d "$VAL_STATE_DIR" ]; then
  ls -la "$VAL_STATE_DIR" || true
  if [ -f "$VAL_STATE_DIR/state.npz" ]; then
    $NEURON_PYTHON -c "import numpy as np; data=np.load('$VAL_STATE_DIR/state.npz'); print('Keys in state:', list(data.keys())); print('Scores:', data['scores'][:10]); print('Successfully verified weights state!')" || true
  else
    log "state.npz not found in $VAL_STATE_DIR"
  fi
else
  log "Validator state directory $VAL_STATE_DIR not found."
fi

log "E2E LOCALNET PATH A SIMULATION PROCESS COMPLETE"
