# CanopyMRV Benchmark Citations & Source Verification

This document records the exact citations, URLs, licenses, and verified incumbent metrics for each benchmark track in accordance with §2, §3, and §10.

---

## Track A — Individual Tree Crown & Canopy (High-Resolution Aerial)

### Dataset
- **Name:** OAM-TCD (OpenAerialMap Tree Cover Dataset)
- **Repository:** [`restor/tcd`](https://huggingface.co/datasets/restor/tcd)
- **Paper:** Restor, *OAM-TCD: A globally diverse dataset of high-resolution tree cover maps*, NeurIPS 2024 Datasets & Benchmarks Track.
  - URL: [NeurIPS 2024 Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/58efdd77196fa8159062afa0408245da-Paper-Datasets_and_Benchmarks_Track.pdf)
- **License:** CC BY 4.0 (permissive for research benchmarking and metric redistribution; CC BY-NC and CC BY-SA subsets partitioned into `restor/tcd-nc` and `restor/tcd-sa`).
- **Resolution & Format:** 5,072 images at 2048×2048 px, GSD = 10 cm/px (0.10 m/px), 3-band RGB. Predefined 5-fold cross-validation plus holdout test split.

### Incumbent & Verified Metrics
- **Model:** `restor/tcd-segformer-mit-b5` (Weights on Hugging Face: [`restor/tcd-segformer-mit-b5`](https://huggingface.co/restor/tcd-segformer-mit-b5))
  - Other baselines: `mit-b0`, `mit-b1`, `mit-b2`, `mit-b3`, `restor/tcd-mask-rcnn-r50`.
- **Target Metrics (Semantic):**
  - **IoU:** **0.887** (Verified against NeurIPS 2024 paper Table 2 / Model Card).
  - **Accuracy:** **0.905** (Verified against NeurIPS 2024 paper Table 2).
  - **F1 Score:** **0.914** (Verified against NeurIPS 2024 paper Table 2).
- **Target Metrics (Instance):**
  - Mask R-CNN ResNet-50: **mAP50 = 42.1** (individual tree crowns).

---

## Track B — Tropical / Agroforestry Crown Detection (MRV Focus)

### Primary Dataset
- **Name:** SelvaBox
- **Repository:** [`CanopyRS/SelvaBox`](https://huggingface.co/datasets/CanopyRS/SelvaBox)
- **Code Repositories:**
  - `CanopyRS`: [https://github.com/CanopyRS/CanopyRS](https://github.com/CanopyRS/CanopyRS) (Apache 2.0)
  - `geodataset`: [https://github.com/CanopyRS/geodataset](https://github.com/CanopyRS/geodataset) (Apache 2.0)
- **Paper:** *SelvaBox: A high-resolution dataset for tropical tree crown detection*, ICLR 2026.
  - URL: [arXiv:2507.00170](https://arxiv.org/pdf/2507.00170)
- **License:** CC BY 4.0 (Code Apache 2.0).
- **Resolution & Format:** 83,000+ manually labeled crowns from UAV imagery across Brazil, Ecuador, Panama. GSD = 1.2–5.1 cm/px (0.012–0.051 m/px).

### Agroforestry OOD & Generalization Suite
- **ReforesTree:**
  - Paper: *ReforesTree: A Dataset for Estimating Tropical Forest Carbon Stock with Airborne LiDAR and High-Resolution RGB Drone Imagery*, AAAI 2022.
  - URL: [arXiv:2201.11192](https://arxiv.org/pdf/2201.11192)
  - Details: 4,600+ individual crown bounding boxes matched to DBH, species, aboveground biomass, and carbon stock at 2 cm/px. License: CC BY 4.0.
- **NeonTreeEvaluation:** ~16k crowns, 10 cm/px GSD. DeepForest supervised box mAP@50 ≈ 49.89.
- **BAMFORESTS:** 27k crowns, 1.6–1.8 cm/px GSD. Mask R-CNN mAP@50 ≈ 69.05, Mask2Former ≈ 68.89.
- **Detectree2:** 3.8k crowns, 10 cm/px GSD.

### Incumbent & Verified Metrics
- **Metric:** **$\text{RF1}_{75}$** — Raster-level F1 score at IoU threshold 0.75, using Algorithm 1 to optimize $\tau_{\text{nms}}$ and $s_{\text{min}}$ on validation.
- **Incumbent Best Model:** DINO 5-scale Swin-L-384 trained multi-resolution.
- **Verified Values (Table 3/4 of SelvaBox paper):**
  - SelvaBox In-domain $\text{RF1}_{75}$: ~0.74–0.78 depending on biome/site.
  - DeepForest on NEON: mAP@50 = 49.89.
  - BAMFORESTS: Mask R-CNN mAP@50 = 69.05, Mask2Former = 68.89.

---

## Track C — Farmland Parcel Delineation

### Dataset
- **Name:** Fields of The World (FTW)
- **Repository:** Source Cooperative [`kerner-lab/fields-of-the-world`](https://beta.source.cooperative/kerner-lab/fields-of-the-world)
- **Mirror:** Hugging Face [`Voxel51/fields-of-the-world`](https://huggingface.co/datasets/Voxel51/fields-of-the-world)
- **Tooling:** [`ftw-baselines`](https://github.com/fieldsoftheworld/ftw-baselines)
- **Paper:** *Fields of The World: A Global Benchmark for Agricultural Field Instance Segmentation*, 2024.
  - URL: [arXiv:2409.16252](https://arxiv.org/abs/2409.16252)
- **License:** CC BY-SA 4.0.
- **Resolution & Format:** 70,462 chips, ~166,293 km², 1.63M labeled field polygons across 24–25 countries. Two-date, 4-band (R, G, B, NIR) Sentinel-2 at 10 m/px GSD.

### Incumbents & Verified Metrics
- **FTW Baseline (U-Net + EfficientNet-B3, 3-class):**
  - Slovenia (SVN): Pixel IoU 0.67, Object Precision 0.60, Object Recall 0.07.
  - France (FRA): Pixel IoU 0.83, Object Precision 0.71, Object Recall 0.45.
  - South Africa (ZAF): Pixel IoU 0.83, Object Precision 0.63, Object Recall 0.35.
- **SOTA Incumbent:** **PTAViT3D** (*Tackling Fluffy Clouds: Robust Field Boundary Delineation*)
  - URL: [arXiv:2409.13568](https://arxiv.org/pdf/2409.13568)
  - Verified per-country pixel mIoU:
    - Austria: 0.87
    - Belgium: 0.89
    - Estonia: 0.90
    - Finland: 0.93
    - France: 0.84
    - Croatia: 0.85
    - Denmark: 0.85
    - Corsica: 0.74
    - Cambodia: 0.67

---

## Track D — Mangrove Segmentation

### Primary Dataset
- **Name:** MANGO
- **Repository:** Hugging Face [`hjh1037/MANGO`](https://huggingface.co/datasets/hjh1037/MANGO)
- **Paper:** *MANGO: A Global Dataset for Mangrove Segmentation from Sentinel-2 Imagery*, 2024.
  - URL: [arXiv:2408.01750](https://arxiv.org/abs/2408.01750)
- **License:** CC BY 4.0 / Open Access.
- **Resolution & Format:** 42,703 Sentinel-2 image-mask pairs across 124 countries, 256×256 px at 10 m/px GSD, country-disjoint 8:1:1 split.

### Secondary Dataset
- **Name:** MagSet-2
- **Repository:** GitHub [`lucasjvds/MangroveAI`](https://github.com/lucasjvds/MangroveAI)
- **Paper:** *A Deep Learning-Based Approach for Mangrove Monitoring*, 2024.
- **License:** MIT.

### Incumbents & Verified Metrics
- **MANGO SOTA:** UNet++: **91.47% IoU** (under country-disjoint holdout split).
- **MagSet-2 SOTA:**
  - Swin-UMamba: **IoU 72.87%**, Accuracy 86.64%, F1 84.27%.
  - SegFormer: IoU 72.32%.
  - U-Net: IoU 61.76%.
