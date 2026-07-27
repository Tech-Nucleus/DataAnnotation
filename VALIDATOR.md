# Validator Setup Guide — Climate MRV Subnet (Netuid 498, Testnet)

This guide covers the **complete validator flow** from a fresh Linux machine to
a running validator on Bittensor testnet.  Every command is copy-paste ready.

> [!IMPORTANT]
> **Subnet**: `DataAnnotation` · **Netuid**: `498` · **Network**: `test`
> **Dataset**: Climate MRV — Sentinel-2 + Hansen/ESA golden samples (Phase 1 Testnet)

---

## What a validator does

1. **Loads the Climate MRV corpus** — downloads Sentinel-2 RGB chips from
   Google Earth Engine (GEE) and labels them against Hansen Global Forest
   Change / ESA WorldCover golden samples
2. **Builds a Golden Set** (validator-only; never served raw to miners)
3. **Dispatches AnnotationTasks** to miners — serves unlabeled Sentinel-2 chips
4. **Scores miner responses** against the Golden Set
5. **Exports commercial annotations** to Cloudflare R2
6. **Publishes on-chain weights** every epoch

---

## Step 0: Install prerequisites

You need Python 3.10+, Git, and at least 16 GB RAM (GEE downloads can be large).

```bash
# Clone the subnet repository
git clone https://github.com/KomaiX512/DataAnnotation.git bittensor-subnet-template-1
cd bittensor-subnet-template-1

# ---- Neurons virtual environment (for validator scripts) ----
python3 -m venv .venv-neurons
source .venv-neurons/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

> [!IMPORTANT]
> **btcli virtual environment** (separate from neurons — needed for wallet
> and registration commands):
> ```bash
> python3 -m venv .venv-btcli
> source .venv-btcli/bin/activate
> pip install bittensor-cli
> ```

> [!TIP]
> For GEE access, also install the Earth Engine Python API:
> ```bash
> pip install earthengine-api
> ```
> (The validator boots without it using pre-exported fallback chips, but live
> GEE streaming is required for full Phase 1 production.)

---

## Step 1: Create a wallet (coldkey + hotkey)

```bash
source .venv-btcli/bin/activate

# Create wallet — answer the prompts (password is optional, press Enter for none)
btcli wallet create \
  --wallet-name validator \
  --hotkey valhk \
  --n-words 12

# Verify
btcli wallet list
```

> [!IMPORTANT]
> **Save your mnemonic phrase** in a secure location.  You cannot recover your
> wallet without it.

### Check your coldkey address

```bash
python3 -c "
import bittensor as bt
w = bt.wallet(name='validator', hotkey='valhk')
print('Coldkey SS58:', w.coldkey.ss58_address)
print('Hotkey  SS58:', w.hotkey.ss58_address)
"
```

---

## Step 2: Fund your wallet and register on subnet 498

### Transfer testnet TAO

You need **~1 TAO** for the registration burn.  If you already have a funded
wallet, transfer to your new validator coldkey:

```bash
source .venv-btcli/bin/activate

btcli wallet transfer \
  --wallet-name <SOURCE_WALLET> \
  --dest <YOUR_VALIDATOR_COLDKEY_SS58> \
  --amount 2 \
  --network test \
  -y
```

**Check balance:**
```bash
btcli wallet balance --wallet-name validator --network test
```

### Register on subnet 498

#### Option A: Python script (most reliable — recommended)

```bash
source .venv-neurons/bin/activate

python scripts/register_on_testnet.py \
  --wallet.name validator \
  --wallet.hotkey valhk \
  --subtensor.network test \
  --netuid 498
```

#### Option B: btcli

```bash
source .venv-btcli/bin/activate

btcli subnets register \
  --netuid 498 \
  --wallet-name validator \
  --hotkey valhk \
  --network test \
  -y
```

**Verify registration:**
```bash
btcli subnets show --netuid 498 --network test
```

---

## Step 3: Configure `.env`

```bash
cp .env.example .env
```

Open `.env` and set these values — the most important fields are highlighted:

```bash
# ===== SUBNET =====
NETUID=498
SUBTENSOR_NETWORK=test

# ===== WALLET =====
WALLET_NAME=validator
WALLET_HOTKEY=valhk

# ===== R2 STORAGE =====
R2_BUCKET_NAME=subnet
R2_ACCOUNT_ID=51abf57b5c6f9b6cf2f91cc87e0b9ffe
R2_S3_ENDPOINT=https://51abf57b5c6f9b6cf2f91cc87e0b9ffe.r2.cloudflarestorage.com
R2_ENDPOINT_URL=https://51abf57b5c6f9b6cf2f91cc87e0b9ffe.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=6db9f1b555e51d83a73b3d6f0c3a5c26
R2_SECRET_ACCESS_KEY=1270b967bbd3cc88c65f6d3216e8cf730ea7954b37cb23f867abd57a7ac2f4ba
# R2_PUBLIC_BUCKET_URL=https://pub-3aa7ed152eb9407cb756c8349a5ef02f.r2.dev

# ===== CLIMATE MRV DATASET =====
VALIDATOR_GOLDEN_DATASET=climate_mrv
VALIDATOR_GOLDEN_RATIO=0.30
VALIDATOR_GOLDEN_SPLIT_SEED=20260601
CLIMATE_MRV_N_RAW_CHIPS=200
CLIMATE_MRV_N_GOLDEN_CHIPS=60

# ===== VALIDATOR INFRA =====
VALIDATOR_IMAGE_CACHE_ROOT=./data/flywheel/image_cache
VALIDATOR_COMMERCIAL_DATASET_PREFIX=file:///home/komail/bittensor-subnet-template-1/artifacts/commercial_dataset
VALIDATOR_COMMERCIAL_EXPORT_EVERY=10
```

**Required fields for a validator:**

| Variable | Description |
|---|---|
| `R2_ACCESS_KEY_ID` | Cloudflare R2 access key |
| `R2_SECRET_ACCESS_KEY` | Cloudflare R2 secret key |
| `R2_ENDPOINT_URL` | `https://<account-id>.r2.cloudflarestorage.com` |
| `R2_BUCKET_NAME` | R2 bucket name (shared: `subnet`) |
| `VALIDATOR_GOLDEN_DATASET` | `climate_mrv` (default, required for Phase 1) |
| `VALIDATOR_IMAGE_CACHE_ROOT` | Local path for cached satellite image chips |
| `VALIDATOR_COMMERCIAL_DATASET_PREFIX` | Where commercial exports are written |

---

## Step 4: Set up Google Earth Engine (GEE)

The Climate MRV validator uses GEE to stream Sentinel-2 imagery and golden
reference data from Hansen/JRC/ESA datasets.

### 4a. Authenticate with GEE

**Personal account (recommended for testnet):**

```bash
source .venv-neurons/bin/activate

# Authenticate in your browser
python3 -c "import ee; ee.Authenticate()"
```

Follow the browser prompt → copy the authentication code → paste it back in
the terminal.

**Verify authentication works:**
```bash
python3 -c "
import ee
ee.Initialize()
print('GEE authenticated OK')
info = ee.Image('UMD/hansen/global_forest_change_2023_v1_11').getInfo()
print('Hansen dataset accessible:', info['id'])
"
```

### 4b. Service account (production / cloud VM)

```bash
# Set GEE_PROJECT in .env if using a cloud project
echo "GEE_PROJECT=my-gcp-project-id" >> .env
```

### 4c. Offline fallback (no GEE)

If you cannot authenticate with GEE, the validator automatically falls back
to pre-exported sample chips in `data/climate_mrv/samples/`.

**Download sample chips for offline use:**
```bash
source .venv-neurons/bin/activate

python3 -m template.hazard.climate_mrv_corpus \
  --output-dir data/climate_mrv/samples \
  --n-chips 50
```

Set in `.env`:
```bash
CLIMATE_MRV_FALLBACK_DIR=data/climate_mrv/samples
```

> [!WARNING]
> The fallback chip set is a minimal placeholder.  For accurate scoring and
> mainnet deployment, live GEE access is required.

---

## Step 5: Run the validator (testnet)

```bash
source .venv-neurons/bin/activate

PROJECT_ROOT="$(pwd)"

env PYTHONPATH=. python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey valhk \
  --subtensor.network test \
  --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 \
  --netuid 498 \
  --axon.port 8090 \
  --neuron.flywheel_golden_dataset_id climate_mrv \
  --neuron.flywheel_golden_ratio 0.30 \
  --neuron.flywheel_image_cache_root "${PROJECT_ROOT}/data/flywheel/image_cache" \
  --neuron.flywheel_commercial_dataset_prefix "file://${PROJECT_ROOT}/artifacts/commercial_dataset" \
  --neuron.flywheel_commercial_export_every 10 \
  --neuron.annotation_timeout 120 \
  --logging.debug
```

> [!IMPORTANT]
> The `.env` file is auto-loaded by the validator via `python-dotenv`.
> You do NOT need to run `source .env`.  The `--neuron.annotation_timeout 120`
> flag gives miners 120 seconds to download images, run inference, and upload
> annotations.  The default (10s) is too short for 30-image batches.
>
> If you are running both the validator and miner on the **same machine** for testing,
> network NAT loopback restrictions may block the validator from reaching the miner's
> public IP. To fix this, prefix the validator command with `LOCALNET_MINER_PORT_BY_SS58=1`.
> This forces the validator to route axon queries to `127.0.0.1`.

### Mainnet (when subnet goes live)

```bash
env PYTHONPATH=. python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey valhk \
  --subtensor.network finney \
  --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
  --netuid <MAINNET_NETUID> \
  --axon.port 8090 \
  --neuron.flywheel_golden_dataset_id climate_mrv \
  --neuron.flywheel_golden_ratio 0.30 \
  --neuron.flywheel_image_cache_root "${PROJECT_ROOT}/data/flywheel/image_cache" \
  --neuron.flywheel_commercial_dataset_prefix "file://${PROJECT_ROOT}/artifacts/commercial_dataset" \
  --neuron.flywheel_commercial_export_every 10 \
  --logging.debug
```

### Localnet (for development / simulation)

First, get the miner's SS58 address:
```bash
source .venv-neurons/bin/activate
python3 -c "import bittensor as bt; print(bt.wallet(name='miner', hotkey='minerhk').hotkey.ss58_address)"
```

Then run with relaxed thresholds for local testing:
```bash
source .venv-neurons/bin/activate
source .env

PROJECT_ROOT="$(pwd)"

env PYTHONPATH=. \
  FORCE_LOCAL_SET_WEIGHTS=1 \
  DEFAULT_ACCEPT_CONFIDENCE=0.01 \
  DEFAULT_ACCEPT_SEVERITY_CONFIDENCE=0.01 \
  DEFAULT_MIN_VOTERS=1 \
  DEFAULT_MIN_MEAN_IOU_TO_MEDIAN=0.1 \
  LOCALNET_MINER_PORT_BY_SS58="<MINER_SS58_ADDRESS>=8091" \
  FALLBACK_SINGLE_MINER_ENABLED=1 \
  python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey valhk \
  --subtensor.network local \
  --subtensor.chain_endpoint ws://127.0.0.1:9944 \
  --netuid 2 \
  --axon.port 8090 \
  --neuron.flywheel_golden_dataset_id climate_mrv \
  --neuron.flywheel_golden_ratio 0.30 \
  --neuron.flywheel_image_cache_root "${PROJECT_ROOT}/data/flywheel/image_cache" \
  --neuron.flywheel_commercial_dataset_prefix "file://${PROJECT_ROOT}/artifacts/localnet/commercial" \
  --neuron.flywheel_commercial_export_every 1 \
  --neuron.forward_step_sleep_seconds 15 \
  --neuron.annotation_timeout 300 \
  --logging.debug
```

> [!WARNING]
> **Absolute paths required!** `--neuron.flywheel_image_cache_root` and
> `--neuron.flywheel_commercial_dataset_prefix` must use absolute paths.
> Use `PROJECT_ROOT="$(pwd)"` as shown above.

---

## Step 6: Verify your validator

### Key log events to watch for

| Event | Meaning |
|---|---|
| `event=validator_init mode=annotation_only` | Validator initialized successfully |
| `event=image_corpus_mode mode=climate_mrv` | Climate MRV dataset loading started |
| `event=climate_mrv_corpus_load_done golden=N annotation=M` | Dataset loaded (N golden, M annotation chips) |
| `event=climate_mrv_gee_unavailable` | GEE offline — using fallback chips |
| `step(N) block(M)` | Main validation loop running |
| `event=evaluator_golden_score_payload` | Miner annotation scored against golden |
| `event=annotation_flywheel_round_done` | Annotation round complete |
| `set_weights on chain successfully!` | Weights published on-chain |

### Verify corpus loaded

```bash
# Check image cache directory
ls -la data/flywheel/image_cache/ | head -20

# Count cached chips
ls data/flywheel/image_cache/*.jpg 2>/dev/null | wc -l
```

### Check commercial export

```bash
# View exported annotation JSONL
ls -la artifacts/commercial_dataset/
head -2 artifacts/commercial_dataset/commercial-dataset-step-0.jsonl | python3 -m json.tool
```

### Verify R2 export

```bash
source .venv-neurons/bin/activate
source .env

python3 -c "
import os, boto3
s3 = boto3.client('s3',
    endpoint_url=os.getenv('R2_ENDPOINT_URL'),
    aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
)
resp = s3.list_objects_v2(Bucket=os.getenv('R2_BUCKET_NAME'), Prefix='commercial/', MaxKeys=20)
for obj in resp.get('Contents', []):
    print(obj['Key'], obj['Size'], 'bytes')
"
```

---

## Step 7: GEE dataset reference (Phase 1 Testnet)

### Raw imagery sources (annotation pool — served to miners)

| Source | GEE Asset ID | Resolution | Bands used |
|---|---|---|---|
| Sentinel-2 L2A | `COPERNICUS/S2_SR_HARMONIZED` | 10 m | B4 (Red), B3 (Green), B2 (Blue) |
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` | 10 m | VV, VH |
| Cloud Score+ | `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` | 10 m | cs_cdf |

### Golden samples (validator-only — never served to miners)

| Source | GEE Asset ID | Used for |
|---|---|---|
| Hansen GFC v1.11 | `UMD/hansen/global_forest_change_2023_v1_11` | Deforestation events |
| JRC TMF (2023) | `projects/JRC/TMF/v1_2023/AnnualChanges` | Tropical moist forest transition |
| ESA WorldCover 10 m | `ESA/WorldCover/v200` | Land-cover baseline label |
| Dynamic World | `GOOGLE/DYNAMICWORLD/V1` | Near-real-time LULC |
| RADD Alerts | `projects/radar-wur/raddalert/v1` | SAR disturbance alerts |
| MapBiomas Amazon | Public GEE collection | Amazon basin 2023 |

### Phase 1 sampling regions (Amazon + Congo + SE Asia)

| Region | Bounding Box |
|---|---|
| Amazon (Pará, Brazil) | `[-54, -5, -48, 0]` |
| Amazon (Mato Grosso) | `[-57, -14, -51, -8]` |
| Amazon (Rondônia) | `[-65, -13, -59, -8]` |
| Congo DRC (North) | `[17, 0, 25, 5]` |
| SE Asia (Borneo) | `[108, -3, 117, 4]` |
| SE Asia (Sumatra) | `[102, -5, 108, 5]` |
| Central Africa (Cameroon) | `[12, 2, 18, 8]` |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `event=climate_mrv_gee_unavailable` | Run `earthengine authenticate` or set `CLIMATE_MRV_FALLBACK_DIR` |
| `golden split is empty` | Check `VALIDATOR_GOLDEN_RATIO` (must be > 0) |
| `event=r2_upload_failed` | Verify `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_BUCKET_NAME` |
| Weights not published | Check `VALIDATOR_DISABLE_SET_WEIGHTS` is not set; verify validator permit |
| Zero miners scoring | Ensure miners are registered and running; wait 1–2 epochs |
| `ValueError: relative path` | Use `PROJECT_ROOT="$(pwd)"` in the run command |

---

## Advanced: Validator permit staking

To unlock a validator permit (required for weight-setting), stake at least
1,000 TAO on testnet:

```bash
source .venv-btcli/bin/activate

btcli stake add \
  --wallet-name validator \
  --hotkey valhk \
  --amount 1000 \
  --network test \
  -y
```

Check permit status:
```bash
python3 -c "
import bittensor as bt
sub = bt.subtensor(network='test')
mg = sub.metagraph(netuid=498)
w = bt.wallet(name='validator', hotkey='valhk')
uid = mg.hotkeys.index(w.hotkey.ss58_address)
print('UID:', uid)
print('Stake:', float(mg.S[uid]))
print('Permit:', bool(mg.validator_permit[uid]))
"
```
