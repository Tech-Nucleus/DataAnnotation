#!/usr/bin/env bash
# Production-readiness gate for localnet (≥10 miners) + offline evaluators.
#
# Prerequisites:
#   - Local subtensor + subnet registered (same as other localnet scripts).
#   - Wallets for validator + MIN_MINERS miners (default 10) registered on netuid.
#   - PYTHONPATH / venv per repo conventions.
#
# What this script does:
#   1) Verifies miner count ≥ MIN_MINERS (default 10).
#   2) Runs the stress matrix (validator + miners) for RUN_SECONDS.
#   3) Runs offline synthetic scenarios (Sybil / collusion / minority / low-miner / calibration toy).
#   4) Validates schema on commercial JSONL under artifacts/commercial_dataset (if present).
#   5) If GOLDEN_MANIFEST and COMMERCIAL_JSONL are set, runs golden-holdout + calibration evaluators.
#
# Full-spec Sybil A/B (baseline vs +50 random miners) requires two orchestrated runs and an
# external accuracy oracle; see docs/PRODUCTION_READINESS_LOCALNET.md.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MIN_MINERS="${MIN_MINERS:-10}"
RUN_SECONDS="${RUN_SECONDS:-600}"
COMM_DIR="${COMM_DIR:-$ROOT_DIR/artifacts/commercial_dataset}"
GOLDEN_MANIFEST="${GOLDEN_MANIFEST:-}"
EXTRA_LABELS="${EXTRA_LABELS:-}"
COMMERCIAL_JSONL="${COMMERCIAL_JSONL:-}"

fail() { echo "ERROR: $*" >&2; exit 1; }

# --- Build default 10-miner env if caller did not export lists ---
if [[ -z "${MINER_WALLET_NAMES:-}" ]]; then
  names=()
  hks=()
  ports=()
  for i in $(seq 1 "$MIN_MINERS"); do
    if [[ "$i" -eq 1 ]]; then
      names+=("miner")
      hks+=("minerhk")
      ports+=("8091")
    else
      names+=("miner${i}")
      hks+=("minerhk${i}")
      ports+=("$((8090 + i))")
    fi
  done
  IFS=','; export MINER_WALLET_NAMES="${names[*]}"; export MINER_WALLET_HOTKEYS="${hks[*]}"; export MINER_PORTS="${ports[*]}"; unset IFS
  echo "[prod-ready] Using default ${MIN_MINERS} miners: MINER_WALLET_NAMES=$MINER_WALLET_NAMES"
fi

IFS=',' read -r -a _NAMES <<< "$MINER_WALLET_NAMES"
if [[ "${#_NAMES[@]}" -lt "$MIN_MINERS" ]]; then
  fail "Need at least ${MIN_MINERS} miners (got ${#_NAMES[@]}). Extend MINER_WALLET_NAMES or lower MIN_MINERS."
fi

echo "[prod-ready] step 1: localnet stress matrix (${RUN_SECONDS}s)…"
export RUN_SECONDS
"$ROOT_DIR/scripts/run_localnet_stress_matrix.sh"

echo "[prod-ready] step 2: offline scenario suite (pytest stress)…"
"$PYTHON_BIN" "$ROOT_DIR/scripts/production_readiness_eval.py" simulate

echo "[prod-ready] step 3: schema validation on commercial JSONL (if any)…"
SCHEMA_GLOB="${SCHEMA_GLOB:-$COMM_DIR/commercial-dataset-step-*.jsonl}"
if compgen -G "$SCHEMA_GLOB" > /dev/null; then
  if ! "$PYTHON_BIN" "$ROOT_DIR/scripts/production_readiness_eval.py" schema --glob "$SCHEMA_GLOB"; then
    if [[ "${ALLOW_LEGACY_COMMERCIAL_JSONL:-0}" == "1" ]]; then
      echo "[prod-ready] WARN: schema validation failed (ALLOW_LEGACY_COMMERCIAL_JSONL=1); fix exports before mainnet."
    else
      fail "Commercial JSONL schema validation failed (set ALLOW_LEGACY_COMMERCIAL_JSONL=1 to warn-only during migration)."
    fi
  fi
else
  echo "[prod-ready] WARN: no files matched SCHEMA_GLOB=$SCHEMA_GLOB (export may be gated or no pool winners)."
fi

if [[ -n "$GOLDEN_MANIFEST" && -n "$COMMERCIAL_JSONL" ]]; then
  echo "[prod-ready] step 4: golden holdout + calibration…"
  GH=(golden-holdout --golden "$GOLDEN_MANIFEST" --commercial "$COMMERCIAL_JSONL")
  CAL=(calibration --golden "$GOLDEN_MANIFEST" --commercial "$COMMERCIAL_JSONL")
  if [[ -n "$EXTRA_LABELS" ]]; then
    GH+=(--extra-labels "$EXTRA_LABELS")
    CAL+=(--extra-labels "$EXTRA_LABELS")
  fi
  "$PYTHON_BIN" "$ROOT_DIR/scripts/production_readiness_eval.py" "${GH[@]}"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/production_readiness_eval.py" "${CAL[@]}"
else
  echo "[prod-ready] skip golden-holdout/calibration (set GOLDEN_MANIFEST + COMMERCIAL_JSONL, optional EXTRA_LABELS)."
fi

echo "[prod-ready] done. See docs/PRODUCTION_READINESS_LOCALNET.md for Sybil A/B and 1000-image holdout."
