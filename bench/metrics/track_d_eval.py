#!/usr/bin/env python3
"""
Track D (Mangrove Segmentation) Evaluation Harness (§4, §5).
Evaluates models on MagSet-2 / Sentinel-2 multispectral imagery.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
import torch
import yaml
from PIL import Image

from bench.metrics.iou_metrics import SegmentationEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench.track_d")


def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


class MagSet2Dataset:
    """Reads MagSet-2 dataset from uncompressed directory."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.image_files = sorted(glob.glob(str(data_dir / "satellite-images" / "*.tiff")))
        self.mask_files = sorted(glob.glob(str(data_dir / "masks" / "*.npy")))

        assert len(self.image_files) == len(self.mask_files), (
            f"Image count ({len(self.image_files)}) != Mask count ({len(self.mask_files)})"
        )
        logger.info("Loaded MagSet-2 with %d paired samples from %s", len(self.image_files), data_dir)

    def __len__(self) -> int:
        return len(self.image_files)

    def get_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Returns (multispectral_img [H, W, 9], binary_mask [H, W], sample_id).
        Bands: 0=R, 1=G, 2=B, 3=NIR, 4=Veg NIR, 5=SWIR, 6=NDVI, 7=NDWI, 8=NDMI.
        """
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        sample_id = Path(img_path).stem

        img = tifffile.imread(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.uint8)

        # Ensure binary mask: 1=mangrove, 0=non-mangrove
        target_mask = (mask != 0).astype(np.uint8)
        return img, target_mask, sample_id


def evaluate_track_d(
    config: dict,
    data_dir: Path = Path("bench/data/magset2/dataset"),
    seed: int = 0,
    device: str = "cuda",
    use_spectral_refinement: bool = True,
) -> dict:
    """Execute Track D evaluation on MagSet-2."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = MagSet2Dataset(data_dir)
    n_total = len(dataset)

    from bench.adapters.canopymrv_miner import CanopyMRVMinerAdapter

    adapter = CanopyMRVMinerAdapter(
        checkpoint_path="models/tree_detection.pt",
        segformer_path="models/tcd-segformer-mit-b2",
        device=device,
        load_yolo=False,
        load_segformer=True,
        load_qwen=False,
    )

    evaluator = SegmentationEvaluator(target_class=1)
    start_time = time.time()

    for idx in range(n_total):
        multi_img, target_mask, sample_id = dataset.get_sample(idx)

        # Extract normalized RGB for vision model
        rgb = multi_img[:, :, 0:3]
        rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        pil_rgb = Image.fromarray((rgb_norm * 255).astype(np.uint8))

        # Semantic mask from SegFormer
        seg_mask = adapter.predict_semantic_mask(pil_rgb, threshold=0.45)

        if use_spectral_refinement and multi_img.shape[2] >= 7:
            # Multi-spectral Sentinel-2 refinement: NDVI (band 6) and NDWI (band 7)
            ndvi = multi_img[:, :, 6]
            ndwi = multi_img[:, :, 7]
            # Mangroves typically have high NDVI (> 0.2) and moderate-to-high moisture/NDWI
            spectral_mask = (ndvi > 0.15) & (ndwi > -0.3)
            pred_mask = ((seg_mask == 1) | spectral_mask).astype(np.uint8)
        else:
            pred_mask = seg_mask

        evaluator.add_batch(pred_mask, target_mask)

    wall_s = round(time.time() - start_time, 2)
    metrics = evaluator.compute()

    incumbent_name = config.get("incumbent", {}).get("secondary_model_id", "Swin-UMamba")
    incumbent_iou = config.get("incumbent", {}).get("published_metrics", {}).get("magset2_swin_umamba_iou", 0.7287)
    incumbent_acc = config.get("incumbent", {}).get("published_metrics", {}).get("magset2_swin_umamba_acc", 0.8664)
    incumbent_f1 = config.get("incumbent", {}).get("published_metrics", {}).get("magset2_swin_umamba_f1", 0.8427)

    commit = get_git_commit()

    run_record = {
        "run_id": f"run_trackD_eval_{seed}_{int(time.time())}",
        "utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "track": "D",
        "dataset": "lucasjvds/MangroveAI (MagSet-2)",
        "split": "test",
        "gsd_m": 10.0,
        "tile_px": 128,
        "pipeline": [
            {"component": "models/tcd-segformer-mit-b2", "role": "canopy_mask", "provenance": "segformer_mask"},
            {"component": "spectral_refinement_ndvi_ndwi", "role": "spectral_filtering", "provenance": "sentinel2_bands"},
        ],
        "seed": seed,
        "metric": "IoU",
        "value": metrics["tree_iou"],  # In 2-class mangrove segmentation, target IoU is mangrove class IoU
        "mean_iou": metrics["mean_iou"],
        "tree_iou": metrics["tree_iou"],
        "bg_iou": metrics["bg_iou"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "n_images": metrics["n_images"],
        "wall_s": wall_s,
        "incumbent": incumbent_name,
        "incumbent_value": incumbent_iou,
        "incumbent_acc": incumbent_acc,
        "incumbent_f1": incumbent_f1,
        "delta": round(metrics["tree_iou"] - incumbent_iou, 4),
        "comparable": True,
        "notes": f"use_spectral_refinement={use_spectral_refinement}",
    }

    results_file = Path("bench/results/track_D_runs.jsonl")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "a") as f:
        f.write(json.dumps(run_record) + "\n")

    logger.info("Track D results logged to %s", results_file)
    logger.info("Final Results: %s", json.dumps(run_record, indent=2))
    return run_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track D Evaluation")
    parser.add_argument("--config", default="bench/configs/d_mango.yaml")
    parser.add_argument("--data-dir", default="bench/data/magset2/dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-spectral", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate_track_d(
        config=cfg,
        data_dir=Path(args.data_dir),
        seed=args.seed,
        device=args.device,
        use_spectral_refinement=not args.no_spectral,
    )
