#!/usr/bin/env bash
set -euo pipefail

# This script queries the live metagraph on Bittensor testnet (NetUID 498)
# and synchronizes the clean cache to the production VPS (/var/www/canopymrv/artifacts/metagraph_cache.json).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
WEB_DIR="${WEB_DIR:-$HOME/data-annotation-web}"
VPS_HOST="root@209.74.66.135"
VPS_DEST="/var/www/canopymrv/artifacts/metagraph_cache.json"
SSH_SOCK="/tmp/ssh-canopy-root@209.74.66.135:22"

TMP_FILE="/tmp/canopy_metagraph_cache_tmp.json"
FINAL_FILE="/tmp/canopy_metagraph_cache.json"

"$REPO_DIR/.venv-neurons/bin/python3" "$WEB_DIR/scripts/get_subnet_state.py" > "$TMP_FILE" 2>/dev/null || true

if [ -s "$TMP_FILE" ] && grep -q '"success": true' "$TMP_FILE"; then
    mv "$TMP_FILE" "$FINAL_FILE"
    mkdir -p "$WEB_DIR/artifacts"
    cp "$FINAL_FILE" "$WEB_DIR/artifacts/metagraph_cache.json"

    # Fast SSH sync using multiplexed socket
    scp -o StrictHostKeyChecking=no \
        -o ControlMaster=auto \
        -o ControlPath="$SSH_SOCK" \
        -o ControlPersist=10m \
        -o ConnectTimeout=5 \
        "$FINAL_FILE" "$VPS_HOST:$VPS_DEST" >/dev/null 2>&1 || true

    # Fast sync current_round.json both to root and climate_mrv
    if [ -f "$REPO_DIR/artifacts/commercial_dataset/current_round.json" ]; then
        scp -o StrictHostKeyChecking=no \
            -o ControlMaster=auto \
            -o ControlPath="$SSH_SOCK" \
            -o ControlPersist=10m \
            -o ConnectTimeout=5 \
            "$REPO_DIR/artifacts/commercial_dataset/current_round.json" \
            "$VPS_HOST:/var/www/canopymrv/artifacts/commercial_dataset/current_round.json" >/dev/null 2>&1 || true
        scp -o StrictHostKeyChecking=no \
            -o ControlMaster=auto \
            -o ControlPath="$SSH_SOCK" \
            -o ControlPersist=10m \
            -o ConnectTimeout=5 \
            "$REPO_DIR/artifacts/commercial_dataset/current_round.json" \
            "$VPS_HOST:/var/www/canopymrv/artifacts/commercial_dataset/climate_mrv/current_round.json" >/dev/null 2>&1 || true
    fi

    # Fast sync batch_summary.json if present
    if [ -f "$REPO_DIR/artifacts/commercial_dataset/climate_mrv/batch_summary.json" ]; then
        scp -o StrictHostKeyChecking=no \
            -o ControlMaster=auto \
            -o ControlPath="$SSH_SOCK" \
            -o ControlPersist=10m \
            -o ConnectTimeout=5 \
            "$REPO_DIR/artifacts/commercial_dataset/climate_mrv/batch_summary.json" \
            "$VPS_HOST:/var/www/canopymrv/artifacts/commercial_dataset/climate_mrv/batch_summary.json" >/dev/null 2>&1 || true
    fi

    # Fast sync get_commercial_dataset.py script to VPS
    if [ -f "$WEB_DIR/scripts/get_commercial_dataset.py" ]; then
        scp -o StrictHostKeyChecking=no \
            -o ControlMaster=auto \
            -o ControlPath="$SSH_SOCK" \
            -o ControlPersist=10m \
            -o ConnectTimeout=5 \
            "$WEB_DIR/scripts/get_commercial_dataset.py" \
            "$VPS_HOST:/var/www/canopymrv/scripts/get_commercial_dataset.py" >/dev/null 2>&1 || true
    fi

    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Metagraph sync complete: $(grep -o '"active_miners": [0-9]*' "$FINAL_FILE") | $(grep -o '"validator_online": [a-z]*' "$FINAL_FILE" || echo "")"
else
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Metagraph sync skipped: output incomplete or invalid"
    rm -f "$TMP_FILE"
fi
