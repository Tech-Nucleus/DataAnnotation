# CanopyMRV SOTA Benchmark Iteration Log (§5)

Record of all iterations across tracks, following the Improvement Ladder:
- Rung 0: Protocol fixes (GSD, tiling, band ordering, normalization, class mapping, thresholds)
- Rung 1: Mask head comparisons (SAM2, SAM3, Grounded-SAM, trained decoders)
- Rung 2: Detector swap (DINO Swin-L, Mask R-CNN, YOLOv8/v11-seg, Mask2Former)
- Rung 3: Fine-tuning on track training split only
- Rung 4: Multi-resolution & test-time augmentation (SAHI, WBF, soft-NMS)
- Rung 5: Cross-dataset joint training
- Rung 6: Subnet-native Bayesian consensus fusion

| Iteration | Rung | Change | Track | Metric Before | After | Delta | Grounded Status | Kept? | Why |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Baseline | Unmodified CanopyMRV stack (SegFormer mit-b2 whole-image) | A | 0.8725 (reproduced mit-b5) | 0.8634 | -0.0091 | 0.8634 ± 0.0000 | Kept | Official baseline on all 439 holdout test images (10 cm/px) |
| 1 | 0 | Tiled inference (1024x1024, stride 768, threshold 0.45) on mit-b2 | A | 0.8634 | 0.8689 | +0.0055 | 0.8689 ± 0.0000 | Kept | Preserves boundary detail on mit-b2 trained on 256px crops |
| 2 | Baseline | Unmodified CanopyMRV stack (SegFormer mit-b2 zero-shot) | D | 0.7287 (Swin-UMamba) | 0.3775 | -0.3512 | 0.3775 ± 0.0000 | Discarded | Cross-domain strawman (10 cm aerial vs 10 m Sentinel-2) |
| 3 | 1 | Backbone swap to SegFormer mit-b5 (whole-image vs tiled) | A | 0.8689 (mit-b2 tiled) | 0.8725 (whole-img)<br>0.8716 (tiled) | +0.0036 (whole-img) | **0.8725 ± 0.0000** | Kept | Whole-image beats tiled inference on mit-b5; tiling claim retracted for SOTA model |
| 4 | 4 | 8-fold Dihedral D4 Test-Time Augmentation (TTA) on mit-b5 | A | 0.8725 (single-pass) | **0.8763** (TTA) | **+0.0038** | **0.8763 mIoU** | Kept | Empirically verifies author's TTA protocol; reaches published 0.8760 holdout score |
| 5 | Baseline | Unmodified CanopyMRV crown detector (YOLO tree detector) | B | 0.7600 (DINO Swin-L) | 0.1712 | -0.5888 | 0.1712 ± 0.0000 | Baseline | Initial baseline; shard 0 (62/1,477 img) marked non-comparable |
| 6 | 0 | Production miner stack (Qwen2.5-VL + YOLO) with Alg 1 calibration | B | 0.1712 | 0.1715 | +0.0003 | **0.1715 ± 0.0001** | Kept | Worst track. Qwen provides zero localization signal; 0.0003 is numerical noise |
| 7 | 3 | Native fine-tuning of SegFormer mit-b2 on MagSet-2 (30 epochs) | D | 0.3775 (zero-shot) | 0.7095 | +0.3320 | 0.7095 ± 0.0053 | Retracted Claim | Gap closure claim retracted: test set ($n=5$) is statistically ungrounded; MANGO is gated 403 |
| 8 | 6 | Subnet-native Bayesian Consensus Fusion (buggy union in assembler) | B | 0.1786 (k=1) | 0.1389 (k=5) | -0.0397 | Regression | Discarded | Bug found in assembler: checked image-level miner count instead of box voters |
| 9 | 6 | Fixed Bayesian Consensus Filtering in dataset_assembler.py (k=2) | B | 0.1786 (k=1) | **0.2113** (k=2) | **+0.0327** | **0.2113 RF1_75** | Kept | Fixed line 896 (`_box_count < min_voters`); eliminates single-miner FP; precision +33.9% |
| 10 | 4 | Single-script 8-fold Dihedral D4 TTA inside `bench/run.py` on all 439 images | A | 0.8725 (single-pass) | **0.8758** (TTA) | **+0.0033** | **0.8758 mIoU** | Kept | Single-script verified on all 439 holdout images; closes gap to published 0.8760 to within -0.0002; arithmetic laundering retracted |
| 11 | 6 | Validation-calibrated Consensus Grid Sweep & Test Evaluation (124 img) | B | 0.1727 (single miner 0) | **0.2121** (k=5, min_voters=5) | **+0.0394** | **0.2121 RF1_75** | Kept | Sweep on true validation split (65 img); resolves self-contradiction; +0.0394 lift over single miner; beats all single miners in pool |
| 12 | 3 | Native YOLO fine-tuning on SelvaBox train shard 0 (18 img, 2,242 boxes, 10 ep) | B | 0.1727 (zero-shot) | **0.1806** (fine-tuned) | **+0.0079** | **0.1806 RF1_75** | Kept | Precision jumps from 0.1279 to 0.2267 (+77.2%); prunes 6,006 false positives; FT consensus reaches 0.3495 precision |
| 13 | 6 | Multi-shard Consensus Evaluation on full 310 test images (shards 0-4) | B | 0.1814 (single miner 0) | **0.2180** (k=5, min_voters=5) | **+0.0366** | **0.2180 RF1_75** | Kept | Evaluated on full >=300 img requirement (310 holdout images); consensus beats every single miner (0.2180 vs 0.2050 best single miner) |
| 14 | 3 | Multi-shard Fine-Tuned Detector Evaluation on full 310 test images | B | 0.1814 (zero-shot) | **0.1685** (fine-tuned) | **-0.0129** | **0.1685 RF1_75** | Discarded | Untiled full-image training caused severe downsampling (crowns shrank to 7.2px) |
| 15 | 3 | Tiled Training Set Expansion (648 tiles) & Native Fine-Tuning (RTX 4090) | B | 0.1814 (zero-shot) | **0.3268** (fine-tuned) | **+0.1454** | **0.3268 RF1_75** | Kept | Slicing into 1024px tiles preserves resolution; precision jumps to 0.3764 (+169.2%); eliminates 14,649 FPs |
| 16 | 6 | Multi-Architecture Diversity Consensus (6-miner pool with YOLOv8n-seg) | B | 0.2023 (k=5) | **0.1909** (k=6) | **-0.0114** | **0.1909 RF1_75** | Kept | Lift shrank: adding 3.2M param nano model aggressively filters recall (0.2196 -> 0.1520) |
| 17 | 0 | Round 6 Protocol Parity Audit & Baseline / Consensus Reconciliation | B | 0.2048 / 0.2023 | **0.1814 / 0.2180** | N/A | **NON-COMPARABLE** | Kept | Reconciled zero-shot (0.1814) and consensus (0.2180); audited DINO-Swin-L (75% overlap 1777px stride 444px raster NMS); marked comparison NON-COMPARABLE; Track B closed |
