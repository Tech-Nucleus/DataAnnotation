# Technology Nucleus — Canopy MRV (Bittensor Subnet 498)

> **Canopy MRV (CanopyMRV)** is the premier decentralized Earth Observation and climate MRV network powered by **Bittensor Subnet 498 (NetUID 498)**, engineered by **[Technology Nucleus](https://technonucleus.com)**.
> 
> 🌐 **Official Portals**:
> - **Canopy MRV Subnet 498 Platform**: [https://mrv.technonucleus.com](https://mrv.technonucleus.com)
> - **Technology Nucleus Official Portal**: [https://technonucleus.com](https://technonucleus.com)
> - **Bittensor Taostats Subnet 498 Explorer**: [https://test.taostats.io](https://test.taostats.io)

---

The **Technology Nucleus** Data Annotation subnet produces cryptographically verified, high-precision carbon and forestry ecological annotations for planetary Climate MRV (Monitoring, Reporting, and Verification).

Miners run high-accuracy computer vision segmentation models to delineate individual tree canopies, dense vegetation clusters, and mangrove forests from high-resolution aerial and satellite tiles. Validators score submissions against hidden labeled golden samples and adjudicate public imagery using multimodal visual reasoning models.

---

## What's New in Version 2.0.0 (Agentic Subnet Upgrade)

- **AI Agent Autopilot Mode (`--agentic`)**: Miners can operate autonomously in self-improving agent mode. Following each round, the miner agent reads validator diagnostic feedback, reasons with **Gemini 2.5 Flash**, tunes inference parameters or applies scoped code patches (e.g. SAHI sliding-window tiling), validates on a local canary image, and manages sleep cycles via an interruptible watchdog.
- **Multimodal Validator Diagnostic Judge**: Validators evaluate miner predictions alongside golden ground truth on RAM-only side-by-side composite grids ($768 \times (384 \cdot k)$ px), querying Gemini 2.5 Flash to generate 1–2 sentence actionable engineering diagnoses without leaking ground-truth coordinates or filenames.
- **Scoped Leaderboard Receipts (`agentic.feedback.v1`)**: Scoped receipts delivered via Cloudflare R2 and piggybacked on `AnnotationTask` synapses, giving miners private feedback on their rank, percentile, fidelity delta, and visual diagnosis.
- **Canary Verification & Automatic Rollback Guard**: All autonomous interventions are validated against `test_clustered_trees.jpg`. If latency exceeds limits or performance degrades for 2 consecutive rounds, the miner automatically restores its pipeline snapshot.
- **Self-Healing Operator CLI (`scripts/miner_agent_cli.py`)**: Real-time management interface to monitor agent status, inspect history, trigger steps, or adjust sleep intervals.
- **High-Precision Multi-Vertex Polygons**: Transitioned from crude bounding boxes to organic polygon crown contours (`polygon: [[x, y], ...]`). High-precision polygon delineations receive priority scoring from multimodal vision judges.
- **Round-by-Round Dataset Partitioning**: Datasets are segmented into deterministic batches (`batch_1`, `batch_2`, etc.) matching network rounds.
- **Cloudflare R2 Direct Artifact Pipeline**: Scalable, high-throughput payload retrieval pipeline via Cloudflare R2 object storage with isolated namespace prefixes.

---

## Documentation Links

- **[Miner Documentation (v2.0.0)](MINER.md)**: Setup guide, AI Agent Autopilot mode (`--agentic`), high-precision polygon requirements, and batch sync specifications.
- **[Validator Documentation (v2.0.0)](VALIDATOR.md)**: Epoch task scheduling, Multimodal Diagnostic Engine, golden isolation, and scoped receipts.
- **[System Architecture](docs/ARCHITECTURE.md)**: Deep dive into the dual flywheel, incentive mechanisms, and security bounds.

---

## Quickstart

### Clone & Install
```bash
git clone https://github.com/Tech-Nucleus/DataAnnotation.git
cd DataAnnotation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running a Miner

#### Standard Manual Mode
```bash
python neurons/miner.py \
    --netuid 498 \
    --subtensor.network test \
    --wallet.name miner \
    --wallet.hotkey default \
    --axon.port 8091
```

#### AI Agent Autopilot Mode
```bash
python neurons/miner.py \
    --netuid 498 \
    --subtensor.network test \
    --wallet.name miner \
    --wallet.hotkey default \
    --axon.port 8091 \
    --agentic \
    --agentic.api_key <YOUR_GEMINI_API_KEY>
```

---

## License
MIT License. Protected evaluation runtime components are proprietary to the subnet operator.
