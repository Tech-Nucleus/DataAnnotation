#!/usr/bin/env bash
# Scale stress matrix (operator): requires localnet with >=10 miners per subnet spec.
# This script does not start subtensor; it documents env vars and delegates to your E2E driver.
#
# Acceptance criteria (from technical spec) — map to operations:
#
# 1) Golden-holdout accuracy
#    - Hold out 1000 golden images never used for reliability updates.
#    - Measure per-class F1 >= 0.85 and exact class match >= 90% on ACCEPTED rows only.
#    - Set: GOLDEN_HOLDOUT_SEED, GOLDEN_HOLDOUT_COUNT, run validator with export + audit JSONL.
#
# 2) Sybil attack
#    - Register N low-quality miners (e.g. 50) with random boxes.
#    - Compare accepted-set accuracy vs baseline; delta <= 2%.
#    - Use: ADVERSARIAL_MINER_INDEX / extra miner registration scripts (see scripts/register_extra_miners_localnet.sh).
#
# 3) Minority hazard preservation
#    - Inject rare-class golden and expert-weighted miners; verify rare class appears in commercial JSONL when gates pass.
#
# 4) Collusion resistance
#    - Coordinate K miners on wrong label; ensure escalation or wrong class not accepted when reliable miners disagree.
#
# 5) Low miner count
#    - With 2 miners and strong disagreement → escalation.
#    - Covered in synthetic harness: tests/stress/test_probabilistic_aggregation_acceptance.py
#
# 6) Uncertainty calibration
#    - Over >=100 samples, mean(accepted confidence) vs empirical accuracy within ±5%.
#    - Requires labeled evaluation harness (extend validator eval to compare accepted posteriors to golden).
#
# 7) Metadata completeness
#    - JSONL lines match schema in docs/PROBABILISTIC_AGGREGATION_REPRODUCIBILITY.md
#
# Implemented harness:
#   - docs/PRODUCTION_READINESS_LOCALNET.md — full mapping + env vars
#   - ./scripts/run_production_readiness_localnet.sh — ≥10 miners + evaluators
#   - ./scripts/production_readiness_eval.py — schema | golden-holdout | calibration | simulate
#
# Example:
#   export RUN_SECONDS=3600 MIN_MINERS=10
#   ./scripts/run_production_readiness_localnet.sh

set -euo pipefail
echo "Use ./scripts/run_production_readiness_localnet.sh (see docs/PRODUCTION_READINESS_LOCALNET.md)."
echo "Fast synthetic gate: ./scripts/run_probabilistic_aggregation_stress.sh"
exit 0
