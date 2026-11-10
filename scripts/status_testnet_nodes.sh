#!/usr/bin/env bash
# Status checker for Subnet 498 Testnet Validator & 3 SegFormer Miners

REPO_DIR="/home/komail/DataAnnotation"
cd "$REPO_DIR"

echo "=================================================================="
echo " Subnet 498 (Bittensor Testnet) Node Status"
echo " Time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "=================================================================="

echo ""
echo "--- RUNNING SUPERVISORS & PROCESSES ---"
ps aux | grep -E "canopy_supervisor|neurons/miner.py|neurons/validator.py|sync_daemon_15s.sh" | grep -v grep || echo "No nodes currently running."

echo ""
echo "--- LISTENING PORTS ---"
ss -tulpn | grep -E "8090|8092|8093|8095" || echo "No ports active."

echo ""
echo "--- GPU ALLOCATION ---"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || echo "No GPU compute processes."

echo ""
echo "=================================================================="
