# Localnet End-to-End Testing Guide

This guide walks you through a **complete ground-up localnet deployment** of the
Data Annotation Subnet.  Every command has been verified end-to-end on
btcli **9.7.1** + bittensor SDK **9.7.0** + the official `ghcr.io/opentensor/subtensor:latest`
Docker image.

> **Copy-paste ready** — follow each step in order and you will have a running
> local subnet with a miner and validator exchanging annotation tasks.

---

## Prerequisites

| Requirement | Minimum |
|---|---|
| Python | 3.10+ |
| Docker | 20.x+ (with `docker` available without `sudo`) |
| Disk | ~5 GB free (images, models, chain data) |
| Ports | 8090, 8091, 9944, 30333 must be free |

---

## 1. Environment Clean Slate

**Always start here.** This step stops any previously running processes and
removes stale wallet/chain data so every run is reproducible.

```bash
# Kill any running subnet processes
pkill -f "neurons/miner.py"  || true
pkill -f "neurons/validator.py" || true
pkill -f "server.py" || true

# Force-remove the old subtensor container (stops and removes in one command)
docker rm -f subtensor-devnet-stable 2>/dev/null || true

# Remove wallet data and miner/validator state
rm -rf ~/.bittensor/wallets/
rm -rf ~/.bittensor/miners/

# Remove cached images and exported datasets from previous runs
rm -rf ./artifacts/localnet/self_hosted_image_cache/
rm -rf ./artifacts/localnet/self_hosted_commercial/
rm -rf ./artifacts/miner_annotation/
```

> [!IMPORTANT]
> The `docker rm -f` step is **essential**. Without it, Docker will refuse to
> create a new container with the same name, and you'll get a
> `Conflict. The container name … is already in use` error.

---

## 2. Start the Local Subtensor Node

Run the official subtensor Docker image in development (Alice-funded) mode:

```bash
docker run -d \
  --name subtensor-devnet-stable \
  -p 9944:9944 \
  -p 30333:30333 \
  ghcr.io/opentensor/subtensor:latest \
  --dev \
  --rpc-external \
  --rpc-methods=unsafe \
  --rpc-cors=all \
  --alice \
  --force-authoring
```

**Wait for the node to start producing blocks:**

```bash
echo "Waiting for RPC..."
for i in $(seq 1 30); do
  if curl -sS -m 2 -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","id":1,"method":"chain_getHeader","params":[]}' \
    http://127.0.0.1:9944 2>/dev/null | grep -q '"result"'; then
    echo "RPC is up (attempt $i)"
    break
  fi
  sleep 2
done
```

**Verify:**
```bash
docker ps | grep subtensor-devnet-stable
```

You should see the container running and port 9944 mapped.

> [!NOTE]
> The `--dev` flag creates a pre-funded **Alice** account
> (`5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY`) with 1,000,000 τ.
> We use Alice to fund the owner, miner, and validator wallets.

---

## 3. Create Wallets

Activate the btcli virtual environment and create four wallets.
The `--wallet-path` flag is **required** to suppress an interactive prompt.

```bash
source .venv-btcli/bin/activate

# Alice wallet (maps to the pre-funded devnet account)
btcli wallet create \
  --wallet-name alice \
  --wallet-path ~/.bittensor/wallets \
  --hotkey default \
  --no-use-password \
  --uri Alice

# Owner wallet (creates the subnet)
btcli wallet create \
  --wallet-name owner \
  --wallet-path ~/.bittensor/wallets \
  --hotkey ownerhk \
  --n-words 12 \
  --no-use-password

# Validator wallet
btcli wallet create \
  --wallet-name validator \
  --wallet-path ~/.bittensor/wallets \
  --hotkey valhk \
  --n-words 12 \
  --no-use-password

# Miner wallet
btcli wallet create \
  --wallet-name miner \
  --wallet-path ~/.bittensor/wallets \
  --hotkey minerhk \
  --n-words 12 \
  --no-use-password
```

**Verify:**
```bash
btcli wallet list --wallet-path ~/.bittensor/wallets
```

You should see `alice`, `owner`, `validator`, and `miner` wallets listed.

---

## 4. Fund Wallets, Create Subnet, Register Neurons

> [!IMPORTANT]
> The btcli 9.7.x `wallet transfer`, `subnets create`, and `subnets register`
> commands have known compatibility issues with certain subtensor Docker image
> versions (you may see `'bool' object has no attribute 'metadata'` errors).
>
> We provide a **Python helper script** that uses the bittensor SDK directly,
> which works reliably across all versions.

### Option A: One-command helper (recommended)

```bash
source .venv-neurons/bin/activate

python scripts/localnet_setup.py \
  --chain-endpoint ws://127.0.0.1:9944
```

This script automatically:
1. Transfers TAO from Alice → owner (10,000 τ), validator (1,000 τ), miner (1,000 τ)
2. Creates subnet netuid 2
3. Registers miner and validator on netuid 2
4. Prints all wallet addresses and balances

### Option B: Manual Python commands (if you want to understand each step)

```bash
source .venv-neurons/bin/activate

python -c "
import bittensor as bt

st = bt.subtensor(network='ws://127.0.0.1:9944')

# Load wallets
alice     = bt.wallet(name='alice',     hotkey='default')
owner     = bt.wallet(name='owner',     hotkey='ownerhk')
validator = bt.wallet(name='validator', hotkey='valhk')
miner     = bt.wallet(name='miner',     hotkey='minerhk')

# 1. Fund wallets from Alice
print('Funding owner...')
st.transfer(wallet=alice, dest=owner.coldkeypub.ss58_address,
            amount=bt.Balance.from_tao(10000))

print('Funding validator...')
st.transfer(wallet=alice, dest=validator.coldkeypub.ss58_address,
            amount=bt.Balance.from_tao(1000))

print('Funding miner...')
st.transfer(wallet=alice, dest=miner.coldkeypub.ss58_address,
            amount=bt.Balance.from_tao(1000))

# 2. Create subnet
print('Creating subnet...')
st.register_subnet(wallet=owner)
print('Subnets:', st.get_subnets())

# 3. Register miner and validator on netuid 2
print('Registering miner...')
st.burned_register(wallet=miner, netuid=2)

print('Registering validator...')
st.burned_register(wallet=validator, netuid=2)

# 4. Verify
mg = st.metagraph(2)
print('Miner UID:',     mg.hotkeys.index(miner.hotkey.ss58_address))
print('Validator UID:', mg.hotkeys.index(validator.hotkey.ss58_address))
print('Miner hotkey:',  miner.hotkey.ss58_address)
"
```

---

## 5. Prepare Workspace Directories

```bash
mkdir -p ./artifacts/miner_annotation
mkdir -p ./artifacts/localnet/self_hosted_image_cache
mkdir -p ./artifacts/localnet/self_hosted_commercial
```

---

## 6. Configure Environment

Copy the example environment file and edit your R2 credentials:

```bash
cp .env.example .env
```

At minimum, set the following in `.env`:

```bash
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
R2_BUCKET_NAME=your_bucket_name
R2_PUBLIC_BUCKET_URL=https://pub-<hash>.r2.dev
```

> [!NOTE]
> All other settings (wallet names, network, backend) are passed as command-line
> flags when starting the miner and validator. The `.env` file is only needed for
> R2 credentials and optional tuning parameters.

---

## 7. Get the Miner SS58 Hotkey Address

You need the miner's hotkey SS58 address for the validator's
`LOCALNET_MINER_PORT_BY_SS58` setting:

```bash
source .venv-neurons/bin/activate
python -c "
import bittensor as bt
w = bt.wallet(name='miner', hotkey='minerhk')
print(w.hotkey.ss58_address)
"
```

Save this address — you'll paste it in the validator start command below.

---

## 8. Start the Miner (Terminal 1)

Open a **dedicated terminal** for the miner:

```bash
cd /path/to/bittensor-subnet-template-1
source .venv-neurons/bin/activate
source .env   # loads R2 credentials

env PYTHONPATH=. MINER_ADVERSARIAL=0 \
  python neurons/miner.py \
  --wallet.name miner \
  --wallet.hotkey minerhk \
  --subtensor.network local \
  --subtensor.chain_endpoint ws://127.0.0.1:9944 \
  --netuid 2 \
  --miner.model_backend yolo_local \
  --miner.yolo_pretrained_weights yolov8n.pt \
  --miner.skip_training \
  --axon.port 8091 \
  --logging.debug \
  --miner.dual_flywheel_r2_prefix localnet/miners/miner1
```

**Expected output:**
```
Miner using ModelTrainingAnnotationEngine with backend=yolo_local
Serving miner axon … on network: ws://127.0.0.1:9944 with netuid: 2
Miner running...
```

---

## 9. Start the Validator (Terminal 2)

Open a **second dedicated terminal** for the validator.
Replace `<MINER_SS58_ADDRESS>` with the address from Step 7:

```bash
cd /path/to/bittensor-subnet-template-1
source .venv-neurons/bin/activate
source .env   # loads R2 credentials

PROJECT_ROOT="$(pwd)"

env PYTHONPATH=. \
  FORCE_LOCAL_SET_WEIGHTS=1 \
  DEFAULT_ACCEPT_CONFIDENCE=0.01 \
  DEFAULT_ACCEPT_SEVERITY_CONFIDENCE=0.01 \
  DEFAULT_MIN_VOTERS=1 \
  DEFAULT_MIN_MEAN_IOU_TO_MEDIAN=0.1 \
  LOCALNET_MINER_PORT_BY_SS58="<MINER_SS58_ADDRESS>=8091" \
  FALLBACK_SINGLE_MINER_ENABLED=1 \
  FALLBACK_SINGLE_MINER_MIN_RELIABILITY=0.3 \
  python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey valhk \
  --subtensor.network local \
  --subtensor.chain_endpoint ws://127.0.0.1:9944 \
  --netuid 2 \
  --axon.port 8090 \
  --neuron.sample_size 3 \
  --neuron.forward_step_sleep_seconds 15 \
  --neuron.annotation_timeout 300 \
  --neuron.flywheel_commercial_export_every 1 \
  --neuron.flywheel_commercial_dataset_prefix "file://${PROJECT_ROOT}/artifacts/localnet/self_hosted_commercial" \
  --neuron.flywheel_image_cache_root "${PROJECT_ROOT}/artifacts/localnet/self_hosted_image_cache" \
  --logging.debug
```

> [!WARNING]
> The `--neuron.flywheel_image_cache_root` and
> `--neuron.flywheel_commercial_dataset_prefix` paths **must be absolute**.
> Using relative paths causes a `ValueError: relative path can't be expressed as
> a file URI` error. The `PROJECT_ROOT="$(pwd)"` variable handles this.

**Expected output:**
```
event=validator_init mode=annotation_only
event=image_corpus_load_start ...
step(0) block(...)
```

---

## 10. Verification

### Check processes
```bash
ps aux | grep -E "(miner|validator)" | grep -v grep
```

### Check subnet metagraph
```bash
source .venv-neurons/bin/activate
python -c "
import bittensor as bt
st = bt.subtensor(network='ws://127.0.0.1:9944')
mg = st.metagraph(2)
print('Neurons:', mg.n.item())
for uid in range(mg.n.item()):
    print(f'  UID {uid} | hotkey={mg.hotkeys[uid][:16]}... | axon_port={mg.axons[uid].port}')
"
```

### Check R2 bucket for uploaded annotations
```bash
source .venv-neurons/bin/activate
source .env
python -c "
import boto3, os
s3 = boto3.client('s3',
    endpoint_url=os.environ.get('R2_ENDPOINT_URL') or os.environ.get('R2_S3_ENDPOINT'),
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    region_name='auto',
)
resp = s3.list_objects_v2(Bucket=os.environ['R2_BUCKET_NAME'], MaxKeys=20)
for obj in resp.get('Contents', []):
    print(f'  {obj[\"Key\"]} ({obj[\"Size\"]} bytes)')
if not resp.get('Contents'):
    print('  (bucket is empty — wait for the first annotation cycle)')
"
```

### Check commercial dataset export
```bash
ls -la artifacts/localnet/self_hosted_commercial/
```

---

## 11. Stopping Everything

```bash
pkill -f "neurons/miner.py"    || true
pkill -f "neurons/validator.py" || true
docker stop subtensor-devnet-stable
```

---

## 12. Clear R2 Buckets (Fresh Start)

To remove all uploaded data from R2 and start clean:

```bash
source .venv-neurons/bin/activate
source .env
python -c "
import boto3, os
s3 = boto3.client('s3',
    endpoint_url=os.environ.get('R2_ENDPOINT_URL') or os.environ.get('R2_S3_ENDPOINT'),
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    region_name='auto',
)
bucket = os.environ['R2_BUCKET_NAME']
paginator = s3.get_paginator('list_objects_v2')
deleted = 0
for page in paginator.paginate(Bucket=bucket):
    objects = [{'Key': obj['Key']} for obj in page.get('Contents', [])]
    if objects:
        s3.delete_objects(Bucket=bucket, Delete={'Objects': objects})
        deleted += len(objects)
print(f'Deleted {deleted} objects from bucket \"{bucket}\"')
"
```

---

## Adapting for Testnet / Mainnet

This guide is designed for localnet but the same miner/validator commands
work on **testnet** and **mainnet** with these changes:

| Setting | Localnet | Testnet | Mainnet |
|---|---|---|---|
| `--subtensor.network` | `local` | `test` | `finney` |
| `--subtensor.chain_endpoint` | `ws://127.0.0.1:9944` | `wss://test.finney.opentensor.ai:443` | `wss://entrypoint-finney.opentensor.ai:443` |
| `FORCE_LOCAL_SET_WEIGHTS` | `1` | `0` | `0` |
| `LOCALNET_MINER_PORT_BY_SS58` | Required | Not needed | Not needed |
| Funding | Alice transfer | Faucet / purchase | Purchase TAO |
| `DEFAULT_ACCEPT_CONFIDENCE` | `0.01` (relaxed) | `0.3` (moderate) | `0.5` (strict) |
| `DEFAULT_MIN_VOTERS` | `1` | `2` | `3` |
| `--neuron.sample_size` | `3` | `50` | `50` |

> [!TIP]
> For testnet, you can obtain testnet TAO from the web faucet at:
> **https://taoswap.org/testnet-faucet**
> Or ask in the Bittensor Discord `#testnet-faucet` channel.

---

## Troubleshooting

### Docker: "name is already in use"
```bash
docker stop subtensor-devnet-stable 2>/dev/null; docker rm subtensor-devnet-stable 2>/dev/null
```
Then re-run the `docker run` command from Step 2.

### btcli: "'bool' object has no attribute 'metadata'"
This is a known btcli 9.7.x ↔ subtensor compatibility issue.
Use the Python helper script (`scripts/localnet_setup.py`) or the SDK directly.

### Validator: "relative path can't be expressed as a file URI"
Use **absolute paths** for `--neuron.flywheel_image_cache_root` and
`--neuron.flywheel_commercial_dataset_prefix`. Set `PROJECT_ROOT="$(pwd)"`
and use `"${PROJECT_ROOT}/..."`.

### Miner not receiving tasks
1. Verify miner is registered: check metagraph (Step 10)
2. Verify `LOCALNET_MINER_PORT_BY_SS58` matches the miner's actual SS58 address
3. Verify miner axon port (8091) is reachable: `curl http://127.0.0.1:8091`

### Wallet prompt asking for path
Always pass `--wallet-path ~/.bittensor/wallets` to suppress the interactive prompt.
