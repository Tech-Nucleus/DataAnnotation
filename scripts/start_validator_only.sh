#!/usr/bin/env bash
set -e

REPO_DIR="/home/komail/DataAnnotation"
cd "$REPO_DIR"

mkdir -p logs

EXISTING=$(pgrep -f "neurons/validator.py" || true)
if [ -n "$EXISTING" ]; then
    echo "Warning: Validator is already running (PID: $EXISTING)."
    exit 0
fi

echo "Launching Validator (CPU-only, port 8090, hotkey valhk)..."
nohup env CUDA_VISIBLE_DEVICES="" .venv-neurons/bin/python neurons/validator.py \
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
echo "Validator started successfully with PID: $VAL_PID"
disown $VAL_PID 2>/dev/null || true
