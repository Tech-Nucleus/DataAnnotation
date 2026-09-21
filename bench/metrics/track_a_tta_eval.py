#!/usr/bin/env python3
"""
Track A: Dihedral D4 Test-Time Augmentation (TTA) Verification (§10.2).
Evaluates SegFormer mit-b5 with and without 8-fold dihedral TTA to verify
whether TTA accounts for the published 0.8760 holdout mIoU.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from bench.metrics.iou_metrics import SegmentationEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench.track_a_tta")


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


# Dihedral D4 group transforms (8 symmetries: 4 rotations x 2 reflections)
DIHEDRAL_TRANSFORMS = [
    (lambda x: x, lambda y: y, "identity"),
    (lambda x: torch.flip(x, [-1]), lambda y: torch.flip(y, [-1]), "hflip"),
    (lambda x: torch.flip(x, [-2]), lambda y: torch.flip(y, [-2]), "vflip"),
    (lambda x: torch.rot90(x, 1, [-2, -1]), lambda y: torch.rot90(y, -1, [-2, -1]), "rot90"),
    (lambda x: torch.rot90(x, 2, [-2, -1]), lambda y: torch.rot90(y, -2, [-2, -1]), "rot180"),
    (lambda x: torch.rot90(x, 3, [-2, -1]), lambda y: torch.rot90(y, -3, [-2, -1]), "rot270"),
    (lambda x: torch.flip(torch.rot90(x, 1, [-2, -1]), [-1]), lambda y: torch.rot90(torch.flip(y, [-1]), -1, [-2, -1]), "rot90_hflip"),
    (lambda x: torch.flip(torch.rot90(x, 3, [-2, -1]), [-1]), lambda y: torch.rot90(torch.flip(y, [-1]), -3, [-2, -1]), "rot270_hflip"),
]


def evaluate_with_and_without_tta(
    parquet_path: Path = Path("bench/data/oam_tcd/data/test-00000-of-00001.parquet"),
    weights_path: str = "bench/data/weights/tcd-segformer-mit-b5",
    device: str = "cuda",
    n_images: int = 25,
) -> Dict:
    table = pq.read_table(str(parquet_path))
    total_available = table.num_rows
    n_eval = min(n_images, total_available) if n_images > 0 else total_available

    logger.info("Evaluating Track A on %d images with single-pass and 8-fold TTA...", n_eval)

    model = SegformerForSemanticSegmentation.from_pretrained(weights_path).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(weights_path)

    evaluator_single = SegmentationEvaluator(target_class=1)
    evaluator_tta = SegmentationEvaluator(target_class=1)

    t0 = time.time()
    for idx in range(n_eval):
        row = table.slice(idx, 1).to_pydict()
        img_bytes = row["image"][0]["bytes"]
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = pil_img.size

        ann_bytes = row["annotation"][0]["bytes"]
        gt_raw = np.array(Image.open(io.BytesIO(ann_bytes)))
        if gt_raw.ndim == 3:
            gt_binary = (gt_raw != 0).any(axis=-1).astype(np.uint8)
        else:
            gt_binary = (gt_raw != 0).astype(np.uint8)

        pv = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

        # 1. Single-pass inference (standard)
        with torch.no_grad():
            out_single = model(pixel_values=pv).logits
            upsampled_single = torch.nn.functional.interpolate(
                out_single, size=(h, w), mode="bilinear", align_corners=False
            )
            prob_single = torch.softmax(upsampled_single, dim=1)[0, 1].cpu().numpy()
            pred_single = (prob_single > 0.50).astype(np.uint8)

        evaluator_single.add_batch(pred_single, gt_binary)

        # 2. 8-fold Dihedral TTA
        accumulator = torch.zeros((1, 2, h, w), device=device, dtype=torch.float32)
        for t_fn, inv_fn, _ in DIHEDRAL_TRANSFORMS:
            aug_pv = t_fn(pv)
            with torch.no_grad():
                out_aug = model(pixel_values=aug_pv).logits
                upsampled_aug = torch.nn.functional.interpolate(
                    out_aug, size=(h, w), mode="bilinear", align_corners=False
                )
                probs_aug = torch.softmax(upsampled_aug, dim=1)
                accumulator += inv_fn(probs_aug)
            del out_aug, upsampled_aug, probs_aug, aug_pv

        avg_probs = (accumulator / len(DIHEDRAL_TRANSFORMS))[0, 1].cpu().numpy()
        pred_tta = (avg_probs > 0.50).astype(np.uint8)
        del accumulator

        evaluator_tta.add_batch(pred_tta, gt_binary)

        if (idx + 1) % 5 == 0 or idx == n_eval - 1:
            logger.info("Processed %d/%d images...", idx + 1, n_eval)

    wall_time = round(time.time() - t0, 2)
    res_single = evaluator_single.compute()
    res_tta = evaluator_tta.compute()

    logger.info("=== Track A TTA Verification Results (%d images) ===", n_eval)
    logger.info("Single-pass: mIoU=%.4f (Tree IoU=%.4f, Bg IoU=%.4f, Acc=%.4f, F1=%.4f)",
                res_single["mean_iou"], res_single["tree_iou"], res_single["bg_iou"], res_single["accuracy"], res_single["f1"])
    logger.info("8-fold TTA:  mIoU=%.4f (Tree IoU=%.4f, Bg IoU=%.4f, Acc=%.4f, F1=%.4f)",
                res_tta["mean_iou"], res_tta["tree_iou"], res_tta["bg_iou"], res_tta["accuracy"], res_tta["f1"])
    delta_miou = round(res_tta["mean_iou"] - res_single["mean_iou"], 4)
    logger.info("TTA Delta:   %+0.4f mIoU", delta_miou)

    record = {
        "n_images": n_eval,
        "single_pass_miou": res_single["mean_iou"],
        "single_pass_tree_iou": res_single["tree_iou"],
        "tta_miou": res_tta["mean_iou"],
        "tta_tree_iou": res_tta["tree_iou"],
        "delta_miou": delta_miou,
        "wall_time_s": wall_time,
    }
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track A TTA Verification")
    parser.add_argument("--n-images", type=int, default=25)
    args = parser.parse_args()
    evaluate_with_and_without_tta(n_images=args.n_images)
