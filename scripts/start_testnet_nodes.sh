#!/usr/bin/env bash
set -e

REPO_DIR="/home/komail/DataAnnotation"
cd "$REPO_DIR"

mkdir -p logs

echo "=================================================================="
echo " Starting Subnet 498 Testnet Validator & 3 SegFormer Miners"
echo " Time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "=================================================================="

# Check if any nodes are already running
EXISTING=$(pgrep -f "neurons/(miner|validator).py" || true)
if [ -n "$EXISTING" ]; then
    echo "Warning: Detected running nodes (PIDs: $EXISTING)."
    echo "Please stop existing nodes before restarting or run ./scripts/stop_testnet_nodes.sh"
    exit 1
fi

echo "[1/5] Launching Miner 2 (SegFormer, port 8092, hotkey miner2hk)..."
setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv-neurons/bin/python neurons/miner.py \
  --wallet.name miner2 \
  --wallet.hotkey miner2hk \
  --subtensor.network test \
  --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 \
  --netuid 498 \
  --miner.model_backend segformer \
  --axon.port 8092 \
  --logging.debug </dev/null >> logs/miner2.log 2>&1 &
MINER2_PID=$!
echo "   -> Miner 2 started with PID: $MINER2_PID"

echo "[2/5] Launching Miner 3 (SegFormer, port 8093, hotkey miner3hk)..."
setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv-neurons/bin/python neurons/miner.py \
  --wallet.name miner3 \
  --wallet.hotkey miner3hk \
  --subtensor.network test \
  --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 \
  --netuid 498 \
  --miner.model_backend segformer \
  --axon.port 8093 \
  --logging.debug </dev/null >> logs/miner3.log 2>&1 &
MINER3_PID=$!
echo "   -> Miner 3 started with PID: $MINER3_PID"

echo "[3/5] Launching Miner 5 (SegFormer, port 8095, hotkey miner5hk)..."
setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv-neurons/bin/python neurons/miner.py \
  --wallet.name miner5 \
  --wallet.hotkey miner5hk \
  --subtensor.network test \
  --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 \
  --netuid 498 \
  --miner.model_backend segformer \
  --axon.port 8095 \
  --logging.debug </dev/null >> logs/miner5.log 2>&1 &
MINER5_PID=$!
echo "   -> Miner 5 started with PID: $MINER5_PID"

echo "Waiting 6 seconds for miner axons to initialize..."
sleep 6

echo "[4/5] Launching Validator (CPU-only, port 8090, hotkey valhk)..."
setsid env CUDA_VISIBLE_DEVICES="" .venv-neurons/bin/python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey valhk \
  --subtensor.network test \
  --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 \
  --netuid 498 \
  --axon.port 8090 \
  --neuron.sample_size 12 \
  --neuron.flywheel_golden_dataset_id climate_mrv \
  --neuron.flywheel_golden_ratio 0.10 \
  --neuron.flywheel_annotation_request_size 20 \
  --neuron.flywheel_golden_injection_per_request 2 \
  --neuron.flywheel_image_cache_root ./data/flywheel/image_cache \
  --neuron.flywheel_commercial_dataset_prefix file:///home/komail/DataAnnotation/artifacts/commercial_dataset \
  --neuron.flywheel_commercial_export_every 1 \
  --neuron.annotation_timeout 180 \
  --neuron.vpermit_tao_limit 50000 \
  --neuron.incentive_min_score 0.0 \
  --neuron.incentive_floor 0.08 \
  --neuron.incentive_temperature 1.0 \
  --neuron.epoch_length 5 \
  --logging.debug </dev/null >> logs/validator.log 2>&1 &
VAL_PID=$!
echo "   -> Validator started with PID: $VAL_PID"

echo "[5/5] Launching 15s Metagraph & R2 Sync Daemon..."
pkill -f "sync_daemon_15s.sh" || true
setsid bash scripts/sync_daemon_15s.sh </dev/null >> /tmp/canopy_sync_daemon.log 2>&1 &
SYNC_PID=$!
echo "   -> Sync Daemon started with PID: $SYNC_PID"

disown -a 2>/dev/null || true

echo "=================================================================="
echo " All Subnet 498 Testnet nodes successfully launched!"
echo "=================================================================="
