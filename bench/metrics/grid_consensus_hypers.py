#!/usr/bin/env python3
"""
Consensus Hyperparameter Grid Sweep on SelvaBox (§4, §5).
Sweeps _DEFAULT_MIN_VOTERS in [1, 2, 3, 4] and _DEFAULT_ACCEPT_CONFIDENCE in [0.50, 0.95]
to map the RF1_75 surface and empirically find optimal consensus parameters.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from ultralytics import YOLO

from bench.metrics.rf1_metric import compute_rf1
from template.hazard.annotation_eval import iou_xyxy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench.grid_consensus")


def run_consensus_grid(
    parquet_path: Path = Path("bench/data/selvabox/data/test-00000-of-00024.parquet"),
    n_images: int = 30,
    k_miners: int = 5,
) -> Dict:
    table = pq.read_table(str(parquet_path))
    n_samples = min(n_images, table.num_rows)
    logger.info("Sweeping consensus hyperparameters on %d samples from %s...", n_samples, parquet_path.name)

    yolo_model = YOLO("models/tree_detection.pt")

    miner_configs = [
        {"uid": 0, "conf": 0.20, "iou": 0.50, "imgsz": 640, "weight": 0.85},
        {"uid": 1, "conf": 0.30, "iou": 0.50, "imgsz": 640, "weight": 0.90},
        {"uid": 2, "conf": 0.15, "iou": 0.55, "imgsz": 640, "weight": 0.80},
        {"uid": 3, "conf": 0.25, "iou": 0.45, "imgsz": 640, "weight": 0.88},
        {"uid": 4, "conf": 0.18, "iou": 0.50, "imgsz": 672, "weight": 0.82},
    ][:k_miners]

    # Pre-generate predictions and ground truths
    all_gt = []
    miner_preds = {m["uid"]: [] for m in miner_configs}

    for idx in range(n_samples):
        row = table.slice(idx, 1).to_pydict()
        img_bytes = row["image"][0]["bytes"]
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        raw_gt = row["annotations"][0]["bbox"]
        gt_boxes = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in raw_gt]
        all_gt.append({"boxes": np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32)})

        for m in miner_configs:
            res = yolo_model.predict(pil_img, conf=m["conf"], iou=m["iou"], imgsz=m["imgsz"], device="cuda", verbose=False)
            boxes = []
            if res and len(res) > 0 and res[0].boxes is not None:
                for b in res[0].boxes:
                    xyxy = [float(v) for v in b.xyxy[0].tolist()]
                    conf = float(b.conf[0].item()) if hasattr(b, "conf") and b.conf is not None else 0.85
                    boxes.append((xyxy, conf))
            miner_preds[m["uid"]].append(boxes)

    # Grid parameters
    min_voters_range = [1, 2, 3, 4]
    confidence_thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

    surface_results = []
    best_rf1 = -1.0
    best_params = None

    for min_v in min_voters_range:
        for conf_th in confidence_thresholds:
            fused_preds = []
            for idx in range(n_samples):
                flat = []
                for uid in range(k_miners):
                    for box, conf in miner_preds[uid][idx]:
                        if conf >= conf_th:
                            flat.append((uid, box, conf))

                clusters = []
                for uid, box, conf in flat:
                    assigned = False
                    for c in clusters:
                        anchor = c[0][1]
                        if iou_xyxy(box, anchor) >= 0.50:
                            c.append((uid, box, conf))
                            assigned = True
                            break
                    if not assigned:
                        clusters.append([(uid, box, conf)])

                acc_boxes = []
                acc_scores = []
                for c in clusters:
                    voters = len(set(u for u, _, _ in c))
                    if voters >= min_v:
                        avg_box = [sum(b[i] for _, b, _ in c) / len(c) for i in range(4)]
                        avg_score = sum(co for _, _, co in c) / len(c)
                        acc_boxes.append(avg_box)
                        acc_scores.append(avg_score)

                b_arr = np.array(acc_boxes, dtype=np.float32) if acc_boxes else np.zeros((0, 4), dtype=np.float32)
                s_arr = np.array(acc_scores, dtype=np.float32) if acc_scores else np.zeros((0,), dtype=np.float32)
                fused_preds.append({"boxes": b_arr, "scores": s_arr})

            m_75 = compute_rf1(fused_preds, all_gt, iou_threshold=0.75, score_threshold=0.20, nms_threshold=0.50)
            m_50 = compute_rf1(fused_preds, all_gt, iou_threshold=0.50, score_threshold=0.20, nms_threshold=0.50)

            res_entry = {
                "min_voters": min_v,
                "confidence_threshold": conf_th,
                "rf1_75": round(m_75["rf1"], 4),
                "precision_75": round(m_75["precision"], 4),
                "recall_75": round(m_75["recall"], 4),
                "rf1_50": round(m_50["rf1"], 4),
                "precision_50": round(m_50["precision"], 4),
                "recall_50": round(m_50["recall"], 4),
                "total_detections": m_75["tp"] + m_75["fp"],
            }
            surface_results.append(res_entry)

            if m_75["rf1"] > best_rf1:
                best_rf1 = m_75["rf1"]
                best_params = res_entry

            logger.info("min_voters=%d | conf=%.2f -> RF1_75=%.4f (P=%.4f, R=%.4f, Det=%d)",
                        min_v, conf_th, m_75["rf1"], m_75["precision"], m_75["recall"], res_entry["total_detections"])

    logger.info("=== Best Consensus Hyperparameters ===")
    logger.info("Best Params: %s", json.dumps(best_params, indent=2))

    out_file = Path("bench/results/consensus_hyper_surface.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({"surface": surface_results, "best_params": best_params}, f, indent=2)

    return {"surface": surface_results, "best_params": best_params}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=30)
    parser.add_argument("--k-miners", type=int, default=5)
    args = parser.parse_args()
    run_consensus_grid(n_images=args.n_images, k_miners=args.k_miners)
