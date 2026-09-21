#!/usr/bin/env python3
"""
Track D (Mangrove Segmentation) Native Fine-Tuning & Evaluation (Rung 3).
Fine-tunes a native segmentation model on MagSet-2 training split
and evaluates on the holdout test split across multiple seeds.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench.train_track_d")


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


class MagSet2SplitDataset(Dataset):
    """Dataset for fine-tuning on MagSet-2 RGB images and binary mangrove masks."""

    def __init__(self, image_paths: List[str], mask_paths: List[str], processor, is_train: bool = True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = tifffile.imread(self.image_paths[idx]).astype(np.float32)
        mask = np.load(self.mask_paths[idx]).astype(np.int64)
        mask = (mask != 0).astype(np.int64)

        # Extract RGB (bands 0, 1, 2) and normalize to [0, 255] uint8
        rgb = img[:, :, 0:3]
        rgb_norm = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255).astype(np.uint8)
        pil_rgb = Image.fromarray(rgb_norm)

        # Process with HuggingFace processor
        encoded = self.processor(images=pil_rgb, segmentation_maps=mask, return_tensors="pt")
        item = {
            "pixel_values": encoded.pixel_values.squeeze(0),
            "labels": encoded.labels.squeeze(0) if "labels" in encoded else torch.tensor(mask, dtype=torch.long),
        }
        return item


def train_and_eval_seed(
    data_dir: Path,
    seed: int,
    n_epochs: int = 30,
    batch_size: int = 6,
    lr: float = 1e-4,
    device: str = "cuda",
) -> Dict[str, float]:
    """Train on train split and evaluate on test split for a single seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    all_images = sorted(glob.glob(str(data_dir / "satellite-images" / "*.tiff")))
    all_masks = sorted(glob.glob(str(data_dir / "masks" / "*.npy")))
    n_total = len(all_images)

    # 80/20 train/test split
    indices = list(range(n_total))
    rng = random.Random(42)  # Fixed split seed for fair comparison
    rng.shuffle(indices)

    n_train = int(0.8 * n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_imgs = [all_images[i] for i in train_idx]
    train_masks = [all_masks[i] for i in train_idx]
    test_imgs = [all_images[i] for i in test_idx]
    test_masks = [all_masks[i] for i in test_idx]

    logger.info("Seed %d | Train samples: %d | Test samples: %d", seed, len(train_imgs), len(test_imgs))

    model_id = "models/tcd-segformer-mit-b2"
    processor = AutoImageProcessor.from_pretrained(model_id, do_resize=False)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_id,
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_dataset = MagSet2SplitDataset(train_imgs, train_masks, processor, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Fine-tuning loop
    model.train()
    start_t = time.time()
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        for batch in train_loader:
            pv = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pv, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

    train_time = round(time.time() - start_t, 2)

    # Evaluation on holdout test split
    model.eval()
    test_ious = []
    test_accs = []
    test_f1s = []

    for img_path, mask_path in zip(test_imgs, test_masks):
        img = tifffile.imread(img_path).astype(np.float32)
        gt_mask = (np.load(mask_path) != 0).astype(np.uint8)
        h, w = gt_mask.shape

        rgb = img[:, :, 0:3]
        rgb_norm = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255).astype(np.uint8)
        pil_rgb = Image.fromarray(rgb_norm)

        inputs = processor(images=pil_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            upsampled = torch.nn.functional.interpolate(
                outputs.logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            pred_mask = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        # Binary metrics for mangrove class
        y_true = gt_mask.flatten()
        y_pred = pred_mask.flatten()

        iou = jaccard_score(y_true, y_pred, average="binary", zero_division=1)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=1)

        test_ious.append(iou)
        test_accs.append(acc)
        test_f1s.append(f1)

    mean_iou = float(np.mean(test_ious))
    mean_acc = float(np.mean(test_accs))
    mean_f1 = float(np.mean(test_f1s))

    logger.info("Seed %d Results: IoU=%.4f | Acc=%.4f | F1=%.4f | Train Time=%.1fs", seed, mean_iou, mean_acc, mean_f1, train_time)

    return {
        "seed": seed,
        "mean_iou": round(mean_iou, 4),
        "mean_acc": round(mean_acc, 4),
        "mean_f1": round(mean_f1, 4),
        "train_time_s": train_time,
        "n_test": len(test_imgs),
        "n_train": len(train_imgs),
    }


def main():
    parser = argparse.ArgumentParser(description="Track D Native Fine-Tuning")
    parser.add_argument("--data-dir", default="bench/data/magset2/dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results = []

    for seed in args.seeds:
        res = train_and_eval_seed(
            data_dir=data_dir,
            seed=seed,
            n_epochs=args.epochs,
            device=args.device,
        )
        results.append(res)

    ious = [r["mean_iou"] for r in results]
    accs = [r["mean_acc"] for r in results]
    f1s = [r["mean_f1"] for r in results]

    mean_iou = float(np.mean(ious))
    std_iou = float(np.std(ious))
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    mean_f1 = float(np.mean(f1s))
    std_f1 = float(np.std(f1s))

    logger.info("=== Track D Native Fine-Tuning (Rung 3) Summary across %d seeds ===", len(args.seeds))
    logger.info("IoU: %.4f ± %.4f", mean_iou, std_iou)
    logger.info("Acc: %.4f ± %.4f", mean_acc, std_acc)
    logger.info("F1:  %.4f ± %.4f", mean_f1, std_f1)

    commit = get_git_commit()
    incumbent_iou = 0.7287

    run_record = {
        "run_id": f"run_trackD_finetuned_multiseed_{int(time.time())}",
        "utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "track": "D",
        "dataset": "lucasjvds/MangroveAI (MagSet-2)",
        "split": "test_holdout",
        "gsd_m": 10.0,
        "tile_px": 128,
        "pipeline": [
            {"component": "SegFormer-mit-b2-finetuned-magset2", "role": "canopy_mask", "provenance": "finetuned_weights"},
        ],
        "seeds": args.seeds,
        "metric": "IoU",
        "value": round(mean_iou, 4),
        "value_std": round(std_iou, 4),
        "accuracy": round(mean_acc, 4),
        "accuracy_std": round(std_acc, 4),
        "f1": round(mean_f1, 4),
        "f1_std": round(std_f1, 4),
        "n_images": results[0]["n_test"],
        "n_train_images": results[0]["n_train"],
        "incumbent": "Swin-UMamba",
        "incumbent_value": incumbent_iou,
        "delta": round(mean_iou - incumbent_iou, 4),
        "comparable": True,
        "rung": 3,
        "notes": f"Native fine-tuning on MagSet-2 train split for {args.epochs} epochs; evaluated on test holdout",
    }

    results_file = Path("bench/results/track_D_runs.jsonl")
    with open(results_file, "a") as f:
        f.write(json.dumps(run_record) + "\n")

    logger.info("Run record logged to %s", results_file)


if __name__ == "__main__":
    main()
