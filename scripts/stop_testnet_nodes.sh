#!/usr/bin/env bash
# Stop script for Subnet 498 Testnet Validator, Miners, and Sync Daemon

echo "Stopping Subnet 498 Testnet nodes..."
pkill -f "neurons/validator.py" || echo "Validator was not running."
pkill -f "neurons/miner.py" || echo "Miners were not running."
pkill -f "sync_daemon_15s.sh" || echo "Sync daemon was not running."

echo "Waiting for processes to terminate..."
sleep 2

REMAINING=$(pgrep -f "neurons/(miner|validator).py|sync_daemon_15s.sh" || true)
if [ -n "$REMAINING" ]; then
    echo "Force killing remaining PIDs: $REMAINING"
    kill -9 $REMAINING 2>/dev/null || true
fi

echo "All Subnet 498 Testnet nodes stopped cleanly."
