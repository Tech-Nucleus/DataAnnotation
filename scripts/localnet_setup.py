#!/usr/bin/env python3
"""
Localnet chain setup helper.

Performs the chain operations that btcli 9.x sometimes fails on
(transfer, subnet create, register) using the bittensor Python SDK directly.

Usage:
    source .venv-neurons/bin/activate
    python scripts/localnet_setup.py               # full setup
    python scripts/localnet_setup.py --step fund    # only fund wallets
    python scripts/localnet_setup.py --step subnet  # only create subnet
    python scripts/localnet_setup.py --step register # only register miner/validator
    python scripts/localnet_setup.py --step info    # only print addresses/balances
"""

import argparse
import sys

import bittensor as bt
_orig_wallet = bt.wallet
class PasswordWallet(_orig_wallet):
    @property
    def coldkey(self):
        return self.get_coldkey(password="5121472")
    def unlock_coldkey(self):
        return self.get_coldkey(password="5121472")
bt.wallet = PasswordWallet


def parse_args():
    p = argparse.ArgumentParser(description="Localnet chain setup helper")
    p.add_argument("--chain-endpoint", default="ws://127.0.0.1:9944",
                    help="Subtensor WebSocket endpoint (default: ws://127.0.0.1:9944)")
    p.add_argument("--wallet-path", default=None,
                    help="Wallet directory (default: ~/.bittensor/wallets)")
    p.add_argument("--owner-wallet", default="owner", help="Owner wallet name")
    p.add_argument("--owner-hotkey", default="ownerhk", help="Owner hotkey name")
    p.add_argument("--miner-wallet", default="miner", help="Miner wallet name")
    p.add_argument("--miner-hotkey", default="minerhk", help="Miner hotkey name")
    p.add_argument("--validator-wallet", default="validator", help="Validator wallet name")
    p.add_argument("--validator-hotkey", default="valhk", help="Validator hotkey name")
    p.add_argument("--alice-tao", type=float, default=10000,
                    help="TAO to transfer from Alice to owner (default: 10000)")
    p.add_argument("--validator-tao", type=float, default=1000,
                    help="TAO to transfer from Alice to validator (default: 1000)")
    p.add_argument("--miner-tao", type=float, default=1000,
                    help="TAO to transfer from Alice to miner (default: 1000)")
    p.add_argument("--step", choices=["all", "fund", "subnet", "register", "info"],
                    default="all", help="Which step to run")
    return p.parse_args()


def load_wallet(name, hotkey, path=None):
    kwargs = {"name": name, "hotkey": hotkey}
    if path:
        kwargs["path"] = path
    return bt.wallet(**kwargs)


def print_info(st, wallets):
    print("\n=== Wallet Information ===")
    for label, w in wallets.items():
        balance = st.get_balance(w.coldkeypub.ss58_address)
        print(f"  {label:12s} | coldkey: {w.coldkeypub.ss58_address} | hotkey: {w.hotkey.ss58_address} | balance: {balance}")
    print(f"\n  Subnets: {st.get_subnets()}")


def step_fund(st, args):
    alice = load_wallet("alice", "default", args.wallet_path)
    owner = load_wallet(args.owner_wallet, args.owner_hotkey, args.wallet_path)
    validator = load_wallet(args.validator_wallet, args.validator_hotkey, args.wallet_path)
    miner = load_wallet(args.miner_wallet, args.miner_hotkey, args.wallet_path)

    alice_balance = st.get_balance(alice.coldkeypub.ss58_address)
    print(f"Alice balance: {alice_balance}")
    if alice_balance.tao < 100:
        print("ERROR: Alice has insufficient funds. Is the subtensor running in --dev mode?")
        sys.exit(1)

    for label, wallet, amount in [
        ("owner", owner, args.alice_tao),
        ("validator", validator, args.validator_tao),
        ("miner", miner, args.miner_tao),
    ]:
        current = st.get_balance(wallet.coldkeypub.ss58_address)
        if current.tao >= amount:
            print(f"  {label} already has {current} — skipping transfer")
            continue
        print(f"  Transferring {amount} TAO to {label} ({wallet.coldkeypub.ss58_address})...")
        ok = st.transfer(wallet=alice, dest=wallet.coldkeypub.ss58_address,
                         amount=bt.Balance.from_tao(amount))
        if not ok:
            print(f"  ERROR: Transfer to {label} failed")
            sys.exit(1)
        print(f"  {label} balance: {st.get_balance(wallet.coldkeypub.ss58_address)}")


def step_subnet(st, args):
    owner = load_wallet(args.owner_wallet, args.owner_hotkey, args.wallet_path)
    subnets = st.get_subnets()
    # Only create if we don't already have netuid 2
    if 2 in subnets:
        print(f"Subnet netuid 2 already exists (subnets: {subnets}) — skipping")
        return 2
    print("Creating subnet...")
    ok = st.register_subnet(wallet=owner)
    if not ok:
        print("ERROR: Subnet creation failed")
        sys.exit(1)
    subnets = st.get_subnets()
    netuid = max(subnets)
    print(f"Subnet created: netuid={netuid} (subnets: {subnets})")
    return netuid


def step_register(st, args, netuid=2):
    miner = load_wallet(args.miner_wallet, args.miner_hotkey, args.wallet_path)
    validator = load_wallet(args.validator_wallet, args.validator_hotkey, args.wallet_path)

    mg = st.metagraph(netuid)
    for label, wallet in [("miner", miner), ("validator", validator)]:
        if wallet.hotkey.ss58_address in mg.hotkeys:
            uid = mg.hotkeys.index(wallet.hotkey.ss58_address)
            print(f"  {label} already registered as UID {uid} — skipping")
            continue
        print(f"  Registering {label} on netuid {netuid}...")
        ok = st.burned_register(wallet=wallet, netuid=netuid)
        if not ok:
            print(f"  ERROR: {label} registration failed")
            sys.exit(1)
        print(f"  {label} registered successfully")

    mg = st.metagraph(netuid)
    print(f"\n  Metagraph UIDs: {list(range(mg.n.item()))}")
    print(f"  Hotkeys: {mg.hotkeys}")


def main():
    args = parse_args()
    st = bt.subtensor(network=args.chain_endpoint)
    print(f"Connected to subtensor at {args.chain_endpoint}")

    wallets = {}
    try:
        wallets["alice"] = load_wallet("alice", "default", args.wallet_path)
        wallets["owner"] = load_wallet(args.owner_wallet, args.owner_hotkey, args.wallet_path)
        wallets["miner"] = load_wallet(args.miner_wallet, args.miner_hotkey, args.wallet_path)
        wallets["validator"] = load_wallet(args.validator_wallet, args.validator_hotkey, args.wallet_path)
    except Exception as e:
        print(f"Warning: Could not load all wallets: {e}")

    if args.step in ("all", "fund"):
        print("\n=== Step: Fund wallets ===")
        step_fund(st, args)

    if args.step in ("all", "subnet"):
        print("\n=== Step: Create subnet ===")
        step_subnet(st, args)

    if args.step in ("all", "register"):
        print("\n=== Step: Register miner & validator ===")
        step_register(st, args)

    if args.step in ("all", "info"):
        print_info(st, wallets)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
