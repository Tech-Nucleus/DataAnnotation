# Subnet Architecture: Decentralized Data Annotation

This document details the architectural design, security mechanisms, consensus protocols, and reward systems of the decentralized annotation subnet. 

The subnet is designed to build high-quality, commercial-grade object detection datasets through decentralized human-in-the-loop and model-assisted labeling.

---

## 1. System Overview

The subnet operates as a decentralized, trustless pipeline where:
- **Miners** act as high-throughput annotation engines, running local object detection pipelines (e.g., YOLO) to identify objects, localized via bounding boxes, and classify them.
- **Validators** serve annotation tasks comprising a mix of secret **Golden Set (ground truth)** images and **Unlabeled** production images, evaluate miner accuracy, fuse predictions into a single consensus, and export a high-fidelity commercial dataset.

```mermaid
graph TD
    subgraph Validator
        IC[Image Corpus] -->|Select 10% Golden + 90% Unlabeled (Default)| IP[Injection Plan]
        IP -->|Metadata Stripping & Hashing| CIP[Camouflaged Images]
        CIP -->|Add Network Jitter| Dendrite[Dendrite Client]
    end

    subgraph Miner
        Dendrite -->|Annotation Task Synapse| ME[Vision / YOLO Detector Engine]
        ME -->|PerImageAnnotationItem| R2[S3/R2 Storage Upload]
        R2 -->|Signed URL / URI| Response[Synapse Response]
    end

    subgraph Consensus & Incentive
        Response -->|Evaluation| AE[Fidelity & Consensus Scorer]
        AE -->|Weights| BF[Bayesian Fusion Aggregator]
        BF -->|Fused Dataset| Exp[Commercial Export]
        AE -->|Rewards| Chain[On-Chain Weight Settlement]
    end
```

---

## 2. Participant Roles

### 2.1 Miners
Miners must be highly efficient at spatial localization and classification.
- **Model-Agnostic Engine**: Miners run any vision model to perform annotations. Supported out-of-the-box backends include local YOLOv8 (`yolo_local`), custom API wrappers (`self_hosted`), or API-based VLMs like GPT-4o (`openai_vision`). Regardless of the backend, they output strictly bounding boxes and hazard classes.
- **Zero-Trust Input**: Miners do not supply reasoning chains, confidence scores, or severity levels. They are only scored on what they can locate and classify.
- **Storage Offload**: Miners upload their completed annotation JSON documents to their own S3/R2 bucket and return a signed URI to the validator, minimizing block/network overhead.

### 2.2 Validators
Validators act as the curators, gatekeepers, and aggregators:
- **Corpus Management**: Maintain an `ImageCorpus` consisting of human-annotated Golden Set images (the secret holdout) and Unlabeled images awaiting annotation.
- **Task Generation**: Dynamically package images for each step according to an `InjectionPlan` (typically 10% Golden Set, 90% Unlabeled by default, configurable via `--neuron.flywheel_golden_ratio`).
- **Consensus & Export**: Aggregated miner responses on unlabeled data are compiled, filtered for agreement, mapped to commercial severity guidelines, and written to a secure commercial dataset in JSONL format.

---

## 3. Security Design & Anti-Gaming

To prevent miners from recognizing Golden Set images (which would allow them to perform selectively high-quality work on Goldens while submitting junk on Unlabeled images), the validator enforces a strict camouflage pipeline:

### 3.1 Camouflage Pipeline
1. **Metadata Stripping**: Every image is re-encoded into a standard JPEG format, stripping all original EXIF, color profile, and timestamp metadata.
2. **Opaque Filename Hashing**: Image filenames are replaced with SHA-256 hashes of the image payload, preventing miners from identifying images via directory naming or index patterns.
3. **Network Latency Jitter**: Validators insert random micro-delays (uniform jitter) when serving images to miners. This prevents miners from using response-time side channels to distinguish local cached Golden images from newly retrieved Unlabeled images.

### 3.2 Sybil & Collusion Resistance (Bayesian Fusion)
If miners collude or launch multiple Sybil instances to report incorrect labels, the validator's Bayesian aggregation model neutralizes them:
- **Fidelity-Weighted Votes**: A miner's weight during aggregation on unlabeled images is proportional to their historical accuracy (fidelity) on the secret Golden images.
- **Low-Weight Neutralization**: Sybils submitting low-fidelity results on Golden images receive near-zero reliability weights, making it mathematically impossible for them to influence the consensus on unlabeled images, even in large numbers.
- **Escalation Rules**: If there is spatial disagreement or too few miners participating, the image is flagged for human escalation (`escalation_required = true`) rather than exporting a faulty label.

---

## 4. Evaluation & Rewards (Dual-Flywheel)

The validator scores miners using the **Dual-Flywheel Reward Composer**, which balances raw fidelity against commercial consensus contribution.

```
Total Reward = alpha * Fidelity_Score + (1 - alpha) * Adoption_Bonus
```

Where `alpha` (configured via `--neuron.flywheel_alpha_annotation` or environment variable `VALIDATOR_ALPHA_ANNOTATION`) defaults to `0.7` (70% weight to Annotation Fidelity and 30% weight to Adoption Bonus).

### 4.1 Annotation Fidelity Score
Fidelity is computed strictly on Golden Set images using:
```
Fidelity = w_iou * IoU_Score + w_class * Class_Match_Score
```
- **IoU Match**: Hungarian matching pairs miner bounding boxes with ground truth boxes. Bounding boxes must overlap above a minimum threshold.
- **Class Match**: Binary check on the classification of matched boxes.
- **Hallucination Penalty**: Every miner bounding box that has no overlapping ground truth box triggers a multiplicative penalty (default `0.5`), reducing the final score.

### 4.2 Adoption Bonus (Consensus Contribution)
Unlabeled annotations are aggregated into a consensus. Miners whose submissions closely match the final accepted consensus objects receive an **Adoption Bonus**. This incentivizes miners to continually submit high-quality work on unlabeled production data.
