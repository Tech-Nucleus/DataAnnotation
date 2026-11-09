#!/usr/bin/env python3
"""Interactive inspection and self-healing CLI for the autonomous miner agent.

Usage:
    python scripts/miner_agent_cli.py status
    python scripts/miner_agent_cli.py step
    python scripts/miner_agent_cli.py history
    python scripts/miner_agent_cli.py rollback [tag]
    python scripts/miner_agent_cli.py set-sleep <seconds>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from template.miner.agentic_controller import (
    MinerAgenticController,
    load_miner_params,
)
from template.utils.agentic_watchdog import AgenticWatchdog


def cmd_status(args):
    watchdog = AgenticWatchdog(role="miner")
    params = load_miner_params()
    ledger = watchdog.get_ledger_summary()
    sleep_sec = watchdog.get_sleep_seconds()

    history_path = Path(".agentic_state/intervention_history.json")
    last_record = None
    if history_path.is_file():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if history:
                last_record = history[-1]
        except Exception:
            pass

    print("\n" + "=" * 65)
    print("        CANOPYMRV (SUBNET 498) MINER AGENT STATUS")
    print("=" * 65)
    print(f"Watchdog State:          {watchdog.current_state}")
    print(f"Watchdog Sleep Interval: {sleep_sec:.1f}s")
    print("\n[ACTIVE PARAMETERS]")
    for k, v in params.items():
        print(f"  - {k:<20}: {v}")

    print("\n[DAILY API BUDGET CONSUMPTION]")
    role_stats = ledger.get("calls_by_role", {}).get("miner", {})
    calls = role_stats.get("calls", 0)
    p_tokens = role_stats.get("prompt_tokens", 0)
    c_tokens = role_stats.get("completion_tokens", 0)
    tripped = ledger.get("circuit_breaker_tripped", False)
    print(f"  - Daily API Calls:     {calls} / {watchdog.max_daily_calls}")
    print(f"  - Prompt Tokens:       {p_tokens}")
    print(f"  - Completion Tokens:   {c_tokens}")
    print(f"  - Circuit Breaker:     {'TRIPPED (Fallback Mode)' if tripped else 'HEALTHY (OK)'}")

    if last_record:
        receipt = last_record.get("receipt", {})
        lb = receipt.get("leaderboard", {})
        print("\n[LATEST ROUND FEEDBACK RECEIPT]")
        print(f"  - Round:               {receipt.get('round_num', 'N/A')}")
        print(f"  - Rank:                {lb.get('self_rank', 'N/A')} of {lb.get('total_active_miners', 'N/A')}")
        print(f"  - Self Fidelity Score: {lb.get('self_fidelity_score', 'N/A')}")
        print(f"  - Score Delta:         {lb.get('score_delta', 'N/A')}")
        print(f"  - Leader Gap:          {lb.get('leader_score_gap', 'N/A')}")
        print(f"  - Validator 1-Liner:   \"{receipt.get('validator_feedback_oneliner', '')}\"")
        print("\n[LAST APPLIED INTERVENTION]")
        inter = last_record.get("intervention", {})
        print(f"  - Tier:                {inter.get('tier', 'N/A')}")
        print(f"  - Rationale:           {inter.get('rationale', 'N/A')}")
        print(f"  - Params:              {inter.get('params', {})}")
    else:
        print("\n[ROUND FEEDBACK] No intervention records found yet.")
    print("=" * 65 + "\n")


def cmd_step(args):
    print("[AGENTIC CLI] Triggering immediate miner autopilot step...")
    controller = MinerAgenticController(enabled=True)
    res = controller.step()
    if res:
        print(f"[AGENTIC CLI] Step successfully executed: Round {res.get('round_num')}")
        print(f"  Intervention: {res.get('intervention')}")
    else:
        print("[AGENTIC CLI] Step completed with no active intervention needed or standby triggered.")


def cmd_history(args):
    history_path = Path(".agentic_state/intervention_history.json")
    if not history_path.is_file():
        print("No history found.")
        return
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        print("History is empty.")
        return

    print("\n" + "=" * 80)
    print(f"{'Round':<8}{'Fidelity':<12}{'Rank':<8}{'Tier':<6}{'Intervention / Delta':<46}")
    print("=" * 80)
    for rec in history:
        r_num = rec.get("round_num", 0)
        receipt = rec.get("receipt", {})
        lb = receipt.get("leaderboard", {})
        fid = lb.get("self_fidelity_score", 0.0)
        rank = f"{lb.get('self_rank', '-')}/{lb.get('total_active_miners', '-')}"
        inter = rec.get("intervention", {})
        tier = inter.get("tier", 1)
        params_str = str(inter.get("params", {}))[:44]
        print(f"{r_num:<8}{fid:<12.4f}{rank:<8}{tier:<6}{params_str:<46}")
    print("=" * 80 + "\n")


def cmd_rollback(args):
    tag = args.tag
    controller = MinerAgenticController()
    if not tag:
        # Pick latest snapshot
        snap_dir = Path(".agentic_snapshots")
        if snap_dir.is_dir():
            dirs = sorted([d.name for d in snap_dir.iterdir() if d.is_dir()])
            if dirs:
                tag = dirs[-1]
    if not tag:
        print("[AGENTIC CLI] No snapshots available for rollback.")
        return
    print(f"[AGENTIC CLI] Rolling back to snapshot: {tag}...")
    success = controller.rollback(tag)
    if success:
        print(f"[AGENTIC CLI] Rollback to {tag} complete.")
    else:
        print(f"[AGENTIC CLI] Rollback to {tag} failed.")


def cmd_set_sleep(args):
    sec = float(args.seconds)
    watchdog = AgenticWatchdog(role="miner")
    watchdog.set_sleep_seconds(sec)
    print(f"[AGENTIC CLI] Watchdog sleep duration successfully updated to {sec:.1f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="CanopyMRV Miner Agentic CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Show miner agent status, budget, and last feedback")
    subparsers.add_parser("step", help="Trigger an immediate perception & reasoning step")
    subparsers.add_parser("history", help="Show round history, scores, and interventions")

    rb = subparsers.add_parser("rollback", help="Rollback miner pipeline to previous snapshot")
    rb.add_argument("tag", nargs="?", default=None, help="Snapshot tag (default: latest)")

    ss = subparsers.add_parser("set-sleep", help="Set watchdog sleep duration in seconds")
    ss.add_argument("seconds", type=float, help="Sleep duration in seconds")

    args = parser.parse_args()
    if args.command == "status":
        cmd_status(args)
    elif args.command == "step":
        cmd_step(args)
    elif args.command == "history":
        cmd_history(args)
    elif args.command == "rollback":
        cmd_rollback(args)
    elif args.command == "set-sleep":
        cmd_set_sleep(args)


if __name__ == "__main__":
    main()
