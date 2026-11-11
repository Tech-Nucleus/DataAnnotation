#!/usr/bin/env bash
# Dedicated supervisor for 15s Metagraph & R2 Sync Daemon
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/../logs/sync_daemon.log"
mkdir -p "$(dirname "$LOG_FILE")"

while true; do
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting sync_daemon_15s.sh..." >> "$LOG_FILE"
    bash "$SCRIPT_DIR/sync_daemon_15s.sh" >> "$LOG_FILE" 2>&1 || true
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] sync_daemon exited. Restarting in 5s..." >> "$LOG_FILE"
    sleep 5
done
