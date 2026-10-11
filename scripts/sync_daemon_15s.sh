#!/usr/bin/env bash
set -euo pipefail

# Continuous 15-second metagraph synchronization daemon.
# Mirrors live Bittensor Testnet Subnet 498 state to local and production VPS every 15 seconds.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_SCRIPT="$SCRIPT_DIR/sync_metagraph_to_vps.sh"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting 15s Subnet Metagraph Sync Daemon..."

counter=0
while true; do
    bash "$SYNC_SCRIPT" || true
    counter=$((counter + 1))
    if [ "$counter" -ge 240 ]; then
        counter=0
        "$SCRIPT_DIR/../.venv-neurons/bin/python" "$SCRIPT_DIR/prune_r2_storage.py" --execute >> /tmp/r2_prune_daemon.log 2>&1 &
    fi
    sleep 15
done
