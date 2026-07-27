#!/usr/bin/env bash
# Start (or restart) the local subtensor dev node used for this repo's E2E scripts.
# Expects an existing container named SUBTENSOR_CONTAINER (default: subtensor-devnet-stable).
set -euo pipefail
CONTAINER="${SUBTENSOR_CONTAINER:-subtensor-devnet-stable}"
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: docker not running."
  exit 1
fi
if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "ERROR: no container named $CONTAINER."
  echo "Create one once, e.g.:"
  echo "  docker run -d --name $CONTAINER -p 9944:9944 ghcr.io/opentensor/subtensor:latest \\"
  echo "    --dev --rpc-external --rpc-methods=unsafe --rpc-cors=all --alice --force-authoring"
  exit 1
fi
docker start "$CONTAINER"
echo "Waiting for JSON-RPC on http://127.0.0.1:9944 ..."
for i in $(seq 1 40); do
  if curl -sS -m 2 -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}' \
    http://127.0.0.1:9944 | grep -q '"result"'; then
    echo "RPC up (try $i)"
    exit 0
  fi
  sleep 1
done
echo "ERROR: RPC did not become ready."
exit 1
