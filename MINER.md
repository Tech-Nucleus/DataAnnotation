# Miner Setup Guide — Climate MRV Subnet (Netuid 498, Testnet)

This guide walks you through the **complete miner flow** from a fresh Linux
machine to a running miner on Bittensor testnet.  Every command is copy-paste
ready.  No prior Bittensor experience is assumed.

> [!IMPORTANT]
> **Subnet**: `DataAnnotation` · **Netuid**: `498` · **Network**: `test`
> **Dataset**: Climate MRV — Sentinel-2 satellite imagery (Phase 1 Testnet)
> Miners annotate **satellite images** (deforestation, land-cover change, fire
> scars) rather than the previous construction-safety dataset.

---

## What a miner does

1. Receives unlabeled **Sentinel-2 RGB chips** (256×256 px) from the validator
2. Runs a vision model to classify each chip into one of the Climate MRV land-cover classes:
   `intact_forest`, `degraded_forest`, `deforestation`, `regrowth`, `plantation`,
   `wetland`, `water`, `agriculture`, `urban`, `fire_scar`, `bare_land`
3. Uploads `annotations.json` to the shared Cloudflare R2 bucket
4. Validator scores the miner against hidden golden samples (Hansen + ESA WorldCover) and publishes on-chain weights

---

## Step 0: Install prerequisites

You need **Python 3.10+**, **Git**, and **8 GB+ RAM** (16 GB recommended for local model training).

```bash
# Clone the subnet repository
git clone https://github.com/KomaiX512/DataAnnotation.git bittensor-subnet-template-1
cd bittensor-subnet-template-1

# ---- Neurons virtual environment (for miner/validator scripts) ----
python3 -m venv .venv-neurons
source .venv-neurons/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Install model server dependencies (REQUIRED for Path A — self_hosted)
pip install fastapi uvicorn
```

> [!IMPORTANT]
> **btcli virtual environment** (separate from neurons — needed for wallet
> and registration commands):
> ```bash
> python3 -m venv .venv-btcli
> source .venv-btcli/bin/activate
> pip install bittensor-cli
> ```
> After this, `btcli --version` should print `BTCLI version: 9.7.x`.

---

## Step 1: Create a wallet (coldkey + hotkey)

```bash
# Activate the btcli environment
source .venv-btcli/bin/activate

# Create wallet — answer the prompts (set a password or press Enter for none)
btcli wallet create \
  --wallet-name miner \
  --hotkey minerhk \
  --n-words 12

# Verify the wallet was created
btcli wallet list
```

> [!IMPORTANT]
> **Save your mnemonic phrase** in a secure location.  You cannot recover your
> wallet without it.
>
> **btcli flag syntax**: btcli uses `--wallet-name` (dashes) and `--hotkey`
> (no prefix), **not** `--wallet.name` or `--wallet.hotkey` (dots).
> The dot-notation is only used by neuron scripts (`miner.py`, `validator.py`).

### Check your coldkey address

```bash
source .venv-neurons/bin/activate

python3 -c "
import bittensor as bt
w = bt.wallet(name='miner', hotkey='minerhk')
print('Coldkey SS58:', w.coldkey.ss58_address)
print('Hotkey  SS58:', w.hotkey.ss58_address)
"
```

---

## Step 2: Fund your wallet with testnet TAO

You need **~1 TAO** on the coldkey to pay the registration burn cost.

**Option A — Swap TAO (fastest):**

Visit **https://taoswap.org/testnet-faucet** to request testnet TAO for your coldkey. You can also use:

- **Bittensor Discord** → `#testnet-faucet` channel:
  [https://discord.gg/bittensor](https://discord.gg/bittensor)
  Post your **coldkey** SS58 address and request testnet TAO.

**Option B — Transfer from another funded wallet:**

```bash
source .venv-btcli/bin/activate

btcli wallet transfer \
  --wallet-name <SOURCE_WALLET> \
  --dest <YOUR_MINER_COLDKEY_SS58> \
  --amount 2 \
  --network test \
  -y
```

**Check balance:**

```bash
source .venv-neurons/bin/activate

# btcli balance check may error on some versions; use this Python fallback:
python3 -c "
import bittensor as bt
sub = bt.subtensor(network='test')
w = bt.wallet(name='miner')
balance = sub.get_balance(w.coldkey.ss58_address)
print(f'Coldkey balance: {balance}')
"
```

---

## Step 3: Register on subnet 498 (testnet)

### Option A: btcli (recommended for external miners)

```bash
source .venv-btcli/bin/activate

btcli subnets register \
  --netuid 498 \
  --wallet-name miner \
  --hotkey minerhk \
  --network test \
  -y
```

### Option B: Python script

```bash
source .venv-neurons/bin/activate

python scripts/register_on_testnet.py \
  --wallet.name miner \
  --wallet.hotkey minerhk \
  --subtensor.network test \
  --netuid 498
```

**Verify registration:**

```bash
source .venv-neurons/bin/activate

python3 -c "
import bittensor as bt
sub = bt.subtensor(network='test')
mg = sub.metagraph(netuid=498)
w = bt.wallet(name='miner', hotkey='minerhk')
if w.hotkey.ss58_address in mg.hotkeys:
    uid = mg.hotkeys.index(w.hotkey.ss58_address)
    print(f'✅ Registered on subnet 498 — UID: {uid}')
    print(f'   Stake: {float(mg.S[uid]):.2f} TAO')
else:
    print('❌ NOT registered on subnet 498')
"
```

> [!WARNING]
> **Stake limit**: Do NOT stake more than **4,000 TAO** on your miner hotkey.
> If your stake exceeds `vpermit_tao_limit` (default 4,096 TAO), the validator
> will **skip your miner** during sampling because it classifies you as a
> validator.  Miners should keep their stake minimal (1-10 TAO is sufficient
> for registration).

---

## Step 4: Configure `.env`

```bash
# Copy the example and edit
cp .env.example .env
```

Open `.env` and set these values for **testnet**:

```bash
# ===== SUBNET (TESTNET) =====
NETUID=498
SUBTENSOR_NETWORK=test
SUBTENSOR_CHAIN_ENDPOINT=wss://test.finney.opentensor.ai:443

# ===== WALLET =====
WALLET_NAME=miner
WALLET_HOTKEY=minerhk

# ===== R2 STORAGE (shared bucket — contact subnet owner for credentials) =====
R2_BUCKET_NAME=subnet
R2_ACCOUNT_ID=51abf57b5c6f9b6cf2f91cc87e0b9ffe
R2_S3_ENDPOINT=https://51abf57b5c6f9b6cf2f91cc87e0b9ffe.r2.cloudflarestorage.com
R2_ENDPOINT_URL=https://51abf57b5c6f9b6cf2f91cc87e0b9ffe.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=6db9f1b555e51d83a73b3d6f0c3a5c26
R2_SECRET_ACCESS_KEY=1270b967bbd3cc88c65f6d3216e8cf730ea7954b37cb23f867abd57a7ac2f4ba
# R2_PUBLIC_BUCKET_URL=https://pub-3aa7ed152eb9407cb756c8349a5ef02f.r2.dev

# ===== MINER CONFIG =====
MINER_MODEL_BACKEND=self_hosted       # or: yolo_local, openai_vision
MINER_ANNOTATION_WORKSPACE=./artifacts/miner_annotation
MINER_R2_PREFIX=miners/annotations

# ===== SELF-HOSTED SERVER (Path A) =====
SELF_HOSTED_TRAIN_URL=http://localhost:8081/train
SELF_HOSTED_INFER_URL=http://localhost:8081/infer
```

**Required fields summary:**

| Variable | Description |
|---|---|
| `R2_ACCESS_KEY_ID` | Cloudflare R2 access key |
| `R2_SECRET_ACCESS_KEY` | Cloudflare R2 secret key |
| `R2_ENDPOINT_URL` | `https://<account-id>.r2.cloudflarestorage.com` |
| `R2_BUCKET_NAME` | R2 bucket name (shared: `subnet`) |
| `MINER_MODEL_BACKEND` | `self_hosted` · `yolo_local` · `openai_vision` |
| `MINER_ANNOTATION_WORKSPACE` | Local scratch directory for downloads |

> [!TIP]
> The Climate MRV dataset is served by the **validator** — miners do NOT need
> to download satellite imagery themselves.  The validator sends image URLs
> inside each `AnnotationTask` synapse.  The `.env` file is auto-loaded by the
> miner script — you do NOT need to `source .env` manually.

---

## Step 5: Choose a model backend and configure it

### Path A: `self_hosted` — Local REST API server (Recommended)

This is the most flexible option.  You run a local HTTP server that handles
`/train` and `/infer` requests.  The reference server uses YOLOv8.

**In `.env`:**
```bash
MINER_MODEL_BACKEND=self_hosted
SELF_HOSTED_TRAIN_URL=http://localhost:8081/train
SELF_HOSTED_INFER_URL=http://localhost:8081/infer
```

**Start the reference server** (keep this terminal open):
```bash
source .venv-neurons/bin/activate

env PYTHONPATH=. python server.py \
  --host 127.0.0.1 \
  --port 8081 \
  --checkpoint yolov8n.pt
```

You should see:
```
============================================================
  Reference Self-Hosted Model Server v2.0
============================================================
  Host:       127.0.0.1
  Port:       8081
  Checkpoint: yolov8n.pt
  YOLO avail: True
  PIL avail:  True
```

**Test the server is responding:**
```bash
curl -s http://127.0.0.1:8081/health | python3 -m json.tool
```

Expected output:
```json
{
    "status": "ok",
    "active_jobs": 0,
    "models_registered": 0,
    "ultralytics_available": true
}
```

### Path B: `yolo_local` — GPU fine-tuning (requires NVIDIA GPU)

**In `.env`:**
```bash
MINER_MODEL_BACKEND=yolo_local
YOLO_MODEL_PATH=yolov8n.pt
YOLO_EPOCHS=10
YOLO_IMGSZ=640
YOLO_BATCH=16
```

### Path C: `openai_vision` — OpenAI hosted vision model

**In `.env`:**
```bash
MINER_MODEL_BACKEND=openai_vision
OPENAI_API_KEY=sk-...
OPENAI_BASE_MODEL=gpt-4o-2024-08-06
```

> [!WARNING]
> OpenAI Vision can incur significant API costs.  Monitor your usage in the
> OpenAI dashboard.

### Inference-Only / Zero-Shot Mode (Skip Training)

If you wish to participate as a miner without local training/fine-tuning (e.g. using pre-trained weights for zero-shot inference, or if your local hardware has limited resources), you can disable the training step.

**In `.env`:**
```bash
MINER_SKIP_TRAINING=True
```

Alternatively, you can start the miner with the CLI flag `--miner.skip_training`. When active, the `/train` endpoint is skipped (no-op) and inference runs directly.

---

## Step 6: Run the miner (testnet)

Open a **new terminal** (keep the server terminal running if using Path A):

```bash
source .venv-neurons/bin/activate

env PYTHONPATH=. python neurons/miner.py \
  --wallet.name miner \
  --wallet.hotkey minerhk \
  --subtensor.network test \
  --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 \
  --netuid 498 \
  --miner.model_backend self_hosted \
  --miner.self_hosted_infer_url http://localhost:8081/infer \
  --miner.self_hosted_train_url http://localhost:8081/train \
  --axon.port 8091 \
  --logging.debug
```

> [!TIP]
> **Local Simulation / NAT loopback workaround**:
> If you are running the miner and validator on the **same machine** for testing,
> network NAT loopback restrictions may block the validator from reaching the miner's
> public IP. To fix this, run the validator with the environment variable:
> `LOCALNET_MINER_PORT_BY_SS58=1`. This automatically patches the target IP to `127.0.0.1`.

> [!NOTE]
> You do NOT need to run `source .env` before the miner script.  The miner
> auto-loads `.env` via `python-dotenv`.  Command-line flags override `.env`
> values.

**Expected startup logs:**
```
Running neuron on subnet: 498 with uid <YOUR_UID> using network: wss://test.finney.opentensor.ai:443
Miner using ModelTrainingAnnotationEngine with backend=self_hosted
Serving miner axon ... on network: wss://test.finney.opentensor.ai:443 with netuid: 498
Miner running...
```

### Mainnet (when subnet goes live)

```bash
env PYTHONPATH=. python neurons/miner.py \
  --wallet.name miner \
  --wallet.hotkey minerhk \
  --subtensor.network finney \
  --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
  --netuid <MAINNET_NETUID> \
  --miner.model_backend self_hosted \
  --miner.self_hosted_infer_url http://localhost:8081/infer \
  --miner.self_hosted_train_url http://localhost:8081/train \
  --axon.port 8091 \
  --logging.debug
```

> [!NOTE]
> **Neuron scripts use dot-notation**: `--wallet.name`, `--wallet.hotkey`,
> `--subtensor.network`, `--subtensor.chain_endpoint`.  This is different from
> btcli which uses dash-notation (`--wallet-name`, `--network`).

---

## Step 7: Verify your miner

### Check logs

**Good signs:**
```
event=annotation_engine_infer_done
event=r2_upload_success
Miner running...
```

**Check R2 uploads:**
```bash
source .venv-neurons/bin/activate

python3 -c "
from dotenv import load_dotenv
load_dotenv()
import os, boto3
from botocore.config import Config

s3 = boto3.client('s3',
    endpoint_url=os.getenv('R2_ENDPOINT_URL'),
    aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
    region_name='auto',
    config=Config(signature_version='s3v4'),
)
resp = s3.list_objects_v2(Bucket=os.getenv('R2_BUCKET_NAME'), Prefix='miners/annotations/', MaxKeys=10)
for obj in resp.get('Contents', []):
    print(obj['Key'], obj['LastModified'])
"
```

**Check metagraph status:**
```bash
source .venv-neurons/bin/activate

python3 -c "
import bittensor as bt
sub = bt.subtensor(network='test')
mg = sub.metagraph(netuid=498)
w = bt.wallet(name='miner', hotkey='minerhk')
if w.hotkey.ss58_address in mg.hotkeys:
    uid = mg.hotkeys.index(w.hotkey.ss58_address)
    print(f'UID: {uid}')
    print(f'Serving: {mg.axons[uid].is_serving}')
    print(f'Stake: {float(mg.S[uid]):.2f}')
    print(f'Trust: {float(mg.T[uid]):.6f}')
    print(f'Incentive: {float(mg.I[uid]):.6f}')
    print(f'Validator Permit: {bool(mg.validator_permit[uid])}')
"
```

### Troubleshooting checklist

| Symptom | Fix |
|---|---|
| Nothing uploads to R2 | Check R2 credentials in `.env`, verify `R2_BUCKET_NAME` |
| `Not registered` error | Re-run registration script (Step 3) |
| `Connection refused` (self_hosted) | Start the server in Step 5 |
| `WalletError: no coldkey found` | Run `btcli wallet list` to verify wallet name |
| Model backend crash | Check `--miner.model_backend` matches `.env` `MINER_MODEL_BACKEND` |
| No validator task received | Validator may be offline; also check your stake is < 4096 TAO |
| Validator skips your miner | Your stake may exceed `vpermit_tao_limit` (4096). See Warning in Step 3. |
| `Missing required packages` | Run `pip install fastapi uvicorn` in `.venv-neurons` |

---

## Climate MRV class taxonomy

Miners will receive Sentinel-2 RGB satellite chips and must classify them into
these land-cover classes:

| Class | Description | Severity |
|---|---|---|
| `intact_forest` | Undisturbed primary / secondary forest | None |
| `degraded_forest` | Canopy intact but visibly disturbed | Low |
| `deforestation` | Clear-cut / fresh conversion | **Critical** |
| `regrowth` | Secondary vegetation on cleared land | Low |
| `plantation` | Commercial monoculture (palm, eucalyptus) | Medium |
| `wetland` | Mangrove, peatland, seasonal floodplain | Medium |
| `water` | Rivers, lakes, reservoirs | None |
| `agriculture` | Cropland / smallholder farms | Low |
| `urban` | Built-up / impervious surfaces | Medium |
| `fire_scar` | Post-fire bare / charred area | **High** |
| `bare_land` | Exposed soil / mining / erosion | Medium |
| `cloud` | Cloud mask (do not annotate) | None |

---

## Google Earth Engine (GEE) for miners — optional

GEE is **validator-only** by default.  Miners receive image URLs in each task
and do NOT need a GEE account to participate.  

If you want to run your own data pipeline or download supplementary training
data, follow the GEE setup in the Validator guide.

### Data sources reference

For full details on the satellite imagery and golden-sample datasets used by
this subnet, see the **Climate MRV Data Sources Specification** document in
the repository root.  Key sources include:

- **Raw imagery**: Sentinel-2 Surface Reflectance (10m), Sentinel-1 SAR (10m)
- **Golden samples**: Hansen Global Forest Change, ESA WorldCover, JRC TMF,
  Dynamic World, RADD Alerts

### Retrieving Golden Datasets for Local Training

To fine-tune your local model on the same dataset structure used for validation, you can retrieve the pre-packaged golden sample dataset directly from the shared R2 bucket.

The pre-exported chips and labels are stored under `data/climate_mrv/samples/` in the R2 bucket. You can download them recursively using the AWS CLI or any compatible S3 utility:

```bash
# Configure your AWS/R2 credentials first, then download the golden training dataset:
aws s3 cp --recursive s3://subnet/data/climate_mrv/samples/ data/climate_mrv/samples/ \
  --endpoint-url https://51abf57b5c6f9b6cf2f91cc87e0b9ffe.r2.cloudflarestorage.com
```

This will download both:
- `data/climate_mrv/samples/raw/` containing Sentinel-2 raw imagery chips.
- `data/climate_mrv/samples/golden/` containing labeled ground-truth chips organized by hazard classes.

---

## R2 Bucket path structure

Each miner writes to its own directory inside the shared bucket:

```
subnet/
└── miners/
    └── annotations/
        └── <task_id>/
            ├── annotations.json    ← miner's annotation output
            └── debug_image.jpg     ← optional annotated image
```
