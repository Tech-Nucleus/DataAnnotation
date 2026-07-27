#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${1:-}"
MAX_STEP_STALL_SECONDS="${MAX_STEP_STALL_SECONDS:-180}"
MAX_ERRORS="${MAX_ERRORS:-0}"

if [[ -z "$LOG_FILE" || ! -f "$LOG_FILE" ]]; then
  echo "Usage: bash scripts/soak_watchdog.sh <validator_log_path>"
  exit 1
fi

if command -v rg >/dev/null 2>&1; then
  last_step_line="$( (rg -n "step\\(" "$LOG_FILE" || true) | tail -n 1)"
else
  last_step_line="$( (grep -n "step(" "$LOG_FILE" || true) | tail -n 1)"
fi

if [[ -z "$last_step_line" ]]; then
  echo "[watchdog] FAIL: no step() lines found."
  exit 2
fi

last_epoch="$(date +%s)"
line_epoch="$(stat -c %Y "$LOG_FILE")"
stall="$(( last_epoch - line_epoch ))"

if command -v rg >/dev/null 2>&1; then
  errors="$( (rg -n "Error during validation step|set_weights failed|Traceback" "$LOG_FILE" || true) | wc -l | tr -d ' ')"
else
  errors="$( (grep -E -n "Error during validation step|set_weights failed|Traceback" "$LOG_FILE" || true) | wc -l | tr -d ' ')"
fi

echo "[watchdog] last_step='$last_step_line'"
echo "[watchdog] file_stall_seconds=$stall errors=$errors"

if (( stall > MAX_STEP_STALL_SECONDS )); then
  echo "[watchdog] FAIL: validator appears stalled."
  exit 3
fi
if (( errors > MAX_ERRORS )); then
  echo "[watchdog] FAIL: error budget exceeded."
  exit 4
fi
echo "[watchdog] PASS"
