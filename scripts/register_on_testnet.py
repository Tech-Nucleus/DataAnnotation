import argparse
import bittensor as bt

def main():
    parser = argparse.ArgumentParser(description="Register a wallet on the subnet")
    parser.add_argument("--wallet.name", dest="wallet_name", required=True, help="Wallet name")
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", required=True, help="Hotkey name")
    parser.add_argument("--subtensor.network", dest="network", default="test", help="Subtensor network (e.g. test, finney)")
    parser.add_argument("--subtensor.chain_endpoint", dest="chain_endpoint", default=None, help="Chain endpoint")
    parser.add_argument("--netuid", type=int, default=498, help="Netuid of the subnet")
    
    args = parser.parse_args()
    
    st = bt.subtensor(network=args.network, chain_endpoint=args.chain_endpoint)
    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    
    print(f"Connecting to network: {args.network}...")
    print(f"Loaded wallet: {wallet}")
    print(f"Subnet Netuid: {args.netuid}")
    
    # Check registration
    mg = st.metagraph(args.netuid)
    if wallet.hotkey.ss58_address in mg.hotkeys:
        uid = mg.hotkeys.index(wallet.hotkey.ss58_address)
        print(f"Wallet is ALREADY registered on subnet {args.netuid} with UID {uid}.")
        return
        
    print(f"Registering wallet on subnet {args.netuid} via burned_register...")
    success = st.burned_register(wallet=wallet, netuid=args.netuid)
    if success:
        print("Registration successful!")
    else:
        print("Registration failed!")

if __name__ == "__main__":
    main()
