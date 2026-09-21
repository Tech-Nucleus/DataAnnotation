# Phase 1 — Reproduction Gate Status (§3)

Mandatory blocking gate: You may not report a single CanopyMRV number until you have reproduced the incumbent.
**Gate criterion:** Reproduced value within **±0.01 absolute** of published.

| Track | Incumbent Model | Dataset & Split | Published Metric | Reproduced Metric | Delta | Gate Status | Command | Commit |
|---|---|---|---|---|---|---|---|---|
| A | `restor/tcd-segformer-mit-b5` | `restor/tcd` (holdout test, 439 images) | mIoU: 0.876 (holdout) / 0.887 (5-fold CV) | mIoU: 0.8725, Acc: 0.9476, F1: 0.8967 (tree IoU: 0.8127, bg IoU: 0.9322) | -0.0035 (vs 0.876) | **PASS** | `.venv-neurons/bin/python3 bench/run.py --track A --mode reproduce` | `3b3e1ec` |
| B | `DINO 5-scale Swin-L-384` | `SelvaBox` (val/test) | $\text{RF1}_{75}$: ~0.76 | PENDING | PENDING | PENDING | `python bench/run.py --track B --mode reproduce` | pending |
| C | `PTAViT3D` / FTW U-Net | `fields-of-the-world` | Pixel IoU: 0.84 (FRA), 0.83 (ZAF) | PENDING | PENDING | PENDING | `python bench/run.py --track C --mode reproduce` | pending |
| D | `UNet++` / `Swin-UMamba` | `MANGO` (country-disjoint) / `MagSet-2` | IoU: 91.47% (MANGO) / 72.87% (MagSet-2) | PENDING | PENDING | PENDING | `python bench/run.py --track D --mode reproduce` | pending |

## Gate Status Legend
- `PASS`: Reproduced within ±0.01 absolute of published.
- `FAIL`: Discrepancy > 0.01 absolute after 3 debug attempts.
- `HARNESS UNVERIFIED`: Reproduction could not be verified; excluded from SOTA claims.
