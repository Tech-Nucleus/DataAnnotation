# Decentralized Data Annotation Subnet
A Bittensor subnet where miners compete to annotate images with bounding boxes and hazard classes, and validators assemble the most accurate labels into a commercial dataset. The design is model-agnostic: any vision model can be used for labeling as long as it produces the required outputs.

## How It Works

```text
+------------------+                   +--------------------+
|                  |   Hidden Golden   |                    |
|    Validator     |------------------>|       Miners       |
|                  |       Set         |                    |
+------------------+                   +--------------------+
         ^                                       |
         | Downloads                             | Uploads
         | Fused Consensus                       | Annotations
         v                                       v
+------------------+                   +--------------------+
|  Commercial      |                   |    Cloudflare R2   |
|  JSONL Export    |<------------------|   Object Storage   |
+------------------+                   +--------------------+
```

1. **Validator** holds a secret Golden Set of human-verified labels.
2. **Miners** download unlabeled images, run their own models, and upload annotations to Cloudflare R2.
3. **Validator** scores miners on the Golden Set (hidden from miners), fuses the best annotations, and pays miners based on accuracy and consensus contribution.
4. A **commercial JSONL dataset** is exported containing only high-confidence, non-Golden annotations.

## Incentive Mechanism

Miners earn rewards based on two things:

* **Annotation Fidelity**: How well they label the hidden Golden Set images (IoU + class match).
* **Adoption Bonus**: How often their annotations are selected for the final fused dataset.

**Reward formula**: `Reward = alpha * Fidelity + (1-alpha) * Adoption_Bonus`

Validators set on-chain weights proportionally; honest, high-quality miners earn the most TAO.

## Quick Links

* 📖 [Miner Guide](MINER.md)
* 🔍 [Validator Guide](VALIDATOR.md)
* 📊 [Subnet Architecture](docs/ARCHITECTURE.md)

## Quick Start (3 steps)

```bash
git clone https://github.com/KomaiX512/DataAnnotation.git bittensor-subnet-template-1 && cd bittensor-subnet-template-1
pip install -r requirements.txt
cp .env.testnet.example .env   # pre-filled for testnet (netuid 498) — edit wallet name only
```

## Choosing a role

* To run a miner → read [MINER.md](MINER.md)
* To run a validator → read [VALIDATOR.md](VALIDATOR.md)
