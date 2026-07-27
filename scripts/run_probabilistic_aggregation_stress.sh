#!/usr/bin/env bash
# Run Phase-2 probabilistic aggregation acceptance tests (synthetic, fast).
# For full-scale spec runs (1000 golden holdout, 50 Sybil miners on localnet),
# see scripts/run_probabilistic_aggregation_stress_localnet_matrix.sh and
# docs/PROBABILISTIC_AGGREGATION_REPRODUCIBILITY.md

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

pytest tests/stress/test_probabilistic_aggregation_acceptance.py -v --tb=short "$@"
