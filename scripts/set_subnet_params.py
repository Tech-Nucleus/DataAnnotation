#!/usr/bin/env python3
import sys
import bittensor as bt

def main():
    st = bt.subtensor(network='ws://127.0.0.1:9944')
    print("Connected to subtensor at ws://127.0.0.1:9944")

    # Load owner wallet
    try:
        owner_wallet = bt.wallet(name='owner', hotkey='ownerhk')
        keypair = owner_wallet.coldkey
        print(f"Loaded owner wallet: {owner_wallet.coldkeypub.ss58_address}")
    except Exception as e:
        print(f"Failed to load owner wallet: {e}")
        sys.exit(1)

    # 1. Disable commit-reveal weights
    print("Disabling commit_reveal_weights_enabled...")
    try:
        call = st.substrate.compose_call(
            call_module='AdminUtils',
            call_function='sudo_set_commit_reveal_weights_enabled',
            call_params={'netuid': 2, 'enabled': False}
        )
        extrinsic = st.substrate.create_signed_extrinsic(call=call, keypair=keypair)
        res = st.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True, wait_for_finalization=True)
        print(f"Successfully disabled commit-reveal: {res.is_success}")
    except Exception as e:
        print(f"Failed to disable commit-reveal: {e}")

    # 2. Set weights set rate limit to 0
    print("Setting weights_set_rate_limit to 0...")
    try:
        call = st.substrate.compose_call(
            call_module='AdminUtils',
            call_function='sudo_set_weights_set_rate_limit',
            call_params={'netuid': 2, 'weights_set_rate_limit': 0}
        )
        extrinsic = st.substrate.create_signed_extrinsic(call=call, keypair=keypair)
        res = st.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True, wait_for_finalization=True)
        print(f"Successfully set weights_set_rate_limit: {res.is_success}")
    except Exception as e:
        print(f"Failed to set weights_set_rate_limit: {e}")

    # 3. Set tempo to 1
    print("Setting tempo to 1...")
    try:
        call = st.substrate.compose_call(
            call_module='AdminUtils',
            call_function='sudo_set_tempo',
            call_params={'netuid': 2, 'tempo': 1}
        )
        extrinsic = st.substrate.create_signed_extrinsic(call=call, keypair=keypair)
        res = st.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True, wait_for_finalization=True)
        print(f"Successfully set tempo: {res.is_success}")
    except Exception as e:
        print(f"Failed to set tempo: {e}")

    # Verify hyperparameters
    hp = st.get_subnet_hyperparameters(2)
    print("\nVerified Hyperparameters on Subnet 2:")
    print(f"  tempo: {hp.tempo}")
    print(f"  weights_rate_limit (weights_set_rate_limit): {hp.weights_rate_limit}")
    print(f"  commit_reveal_weights_enabled: {hp.commit_reveal_weights_enabled}")

if __name__ == "__main__":
    main()
