#!/usr/bin/env python3
"""
Track B (SelvaBox / ReforesTree Crown Detection) Evaluation Harness (§4, §5).
Evaluates crown detection models under the SelvaBox / CanopyRS protocol (ICLR 2026).
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image

from bench.adapters.canopymrv_miner import CanopyMRVMinerAdapter
from bench.metrics.rf1_metric import compute_rf1, optimize_thresholds_algorithm_1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench.track_b")


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


def evaluate_track_b(
    config: dict,
    parquet_path: Path = Path("bench/data/selvabox/data/test-00000-of-00024.parquet"),
    seed: int = 0,
    device: str = "cuda",
    iou_threshold: float = 0.75,
    optimize_thresholds: bool = True,
    score_threshold: float = 0.30,
    nms_threshold: float = 0.50,
    load_qwen: bool = True,
) -> dict:
    """Execute Track B evaluation on SelvaBox."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not parquet_path.exists():
        raise FileNotFoundError(f"SelvaBox parquet file not found: {parquet_path}")

    table = pq.read_table(str(parquet_path))
    n_samples = table.num_rows
    logger.info("Loaded SelvaBox test shard with %d samples from %s", n_samples, parquet_path)

    # Initialize production miner stack with Qwen2.5-VL and YOLO
    adapter = CanopyMRVMinerAdapter(
        checkpoint_path="models/tree_detection.pt",
        load_yolo=True,
        load_segformer=False,
        load_qwen=load_qwen,
        device=device,
    )

    all_predictions = []
    all_ground_truths = []

    start_time = time.time()
    for idx in range(n_samples):
        row = table.slice(idx, 1).to_pydict()
        img_bytes = row["image"][0]["bytes"]
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Ground truth boxes in [xmin, ymin, width, height] -> convert to [x1, y1, x2, y2]
        raw_gt = row["annotations"][0]["bbox"]
        gt_boxes = []
        for b in raw_gt:
            x, y, w, h = b
            gt_boxes.append([x, y, x + w, y + h])

        gt_arr = np.array(gt_boxes, dtype=np.float32) if len(gt_boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        all_ground_truths.append({"boxes": gt_arr})

        # Run miner crown detector (with Qwen2.5-VL scene reasoning when load_qwen=True)
        preds = adapter.predict_boxes(pil_img)
        p_boxes = []
        p_scores = []
        for box, conf, cls_name in preds:
            p_boxes.append(box)
            p_scores.append(conf)

        p_boxes_arr = np.array(p_boxes, dtype=np.float32) if len(p_boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        p_scores_arr = np.array(p_scores, dtype=np.float32) if len(p_scores) > 0 else np.zeros((0,), dtype=np.float32)
        all_predictions.append({"boxes": p_boxes_arr, "scores": p_scores_arr})

    # Threshold optimization (Algorithm 1) on calibration subset (first half)
    if optimize_thresholds and len(all_predictions) > 10:
        logger.info("Running Algorithm 1 threshold optimization on calibration subset...")
        n_calib = len(all_predictions) // 2
        best_s, best_tau, best_rf1 = optimize_thresholds_algorithm_1(
            all_predictions[:n_calib],
            all_ground_truths[:n_calib],
            iou_threshold=iou_threshold,
        )
        logger.info("Optimal thresholds from calibration: score=%.2f, nms=%.2f (calib RF1_75=%.4f)", best_s, best_tau, best_rf1)
        score_threshold = best_s
        nms_threshold = best_tau

    # Compute final metrics at iou_threshold
    metrics = compute_rf1(
        all_predictions,
        all_ground_truths,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )

    # Also compute at IoU 0.50
    metrics_50 = compute_rf1(
        all_predictions,
        all_ground_truths,
        iou_threshold=0.50,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )

    wall_s = round(time.time() - start_time, 2)
    commit = get_git_commit()

    incumbent_name = config.get("incumbent", {}).get("model_id", "DINO-5scale-Swin-L-384")
    incumbent_rf1 = config.get("incumbent", {}).get("published_metrics", {}).get("rf1_75", 0.76)

    pipeline_components = [
        {"component": "models/tree_detection.pt", "role": "crown_detector", "provenance": "yolo_box"},
    ]
    if load_qwen:
        pipeline_components.insert(0, {
            "component": "models/qwen2.5-vl-3b",
            "role": "macro_landscape_reasoning",
            "provenance": "qwen2.5_vl_bfloat16",
        })

    run_record = {
        "run_id": f"run_trackB_eval_{seed}_{int(time.time())}",
        "utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "track": "B",
        "dataset": "CanopyRS/SelvaBox",
        "split": "test (shard 0 of 24)",
        "gsd_m": 0.045,
        "tile_px": 1777,
        "pipeline": pipeline_components,
        "seed": seed,
        "metric": "RF1_75",
        "value": metrics["rf1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "rf1_50": metrics_50["rf1"],
        "precision_50": metrics_50["precision"],
        "recall_50": metrics_50["recall"],
        "n_images": n_samples,
        "wall_s": wall_s,
        "score_threshold": score_threshold,
        "nms_threshold": nms_threshold,
        "incumbent": incumbent_name,
        "incumbent_value": incumbent_rf1,
        "delta": round(metrics["rf1"] - incumbent_rf1, 4),
        "comparable": False,
        "notes": f"shard=test-00000 (62 of 1477 images); not comparable to full 24-shard test set; load_qwen={load_qwen}; optimize_thresholds={optimize_thresholds}",
    }

    results_file = Path("bench/results/track_B_runs.jsonl")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "a") as f:
        f.write(json.dumps(run_record) + "\n")

    logger.info("Track B results logged to %s", results_file)
    logger.info("Final Results: %s", json.dumps(run_record, indent=2))
    return run_record


def main():
    parser = argparse.ArgumentParser(description="Track B Evaluation")
    parser.add_argument("--config", default="bench/configs/b_selvabox.yaml")
    parser.add_argument("--parquet", default="bench/data/selvabox/data/test-00000-of-00024.parquet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iou-threshold", type=float, default=0.75)
    parser.add_argument("--no-qwen", action="store_true", help="Disable Qwen2.5-VL")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    load_qwen = not args.no_qwen
    seeds = args.seeds if args.seeds is not None else [args.seed]

    results = []
    for s in seeds:
        res = evaluate_track_b(
            config=cfg,
            parquet_path=Path(args.parquet),
            seed=s,
            device=args.device,
            iou_threshold=args.iou_threshold,
            load_qwen=load_qwen,
        )
        results.append(res)

    if len(results) > 1:
        rf1_75s = [r["value"] for r in results]
        rf1_50s = [r["rf1_50"] for r in results]
        logger.info("=== Track B Multi-Seed Summary (%d seeds) ===", len(results))
        logger.info("RF1_75: %.4f ± %.4f", float(np.mean(rf1_75s)), float(np.std(rf1_75s)))
        logger.info("RF1_50: %.4f ± %.4f", float(np.mean(rf1_50s)), float(np.std(rf1_50s)))


if __name__ == "__main__":
    main()
