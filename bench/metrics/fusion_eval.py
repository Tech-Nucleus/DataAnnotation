#!/usr/bin/env python3
"""
Rung 6: Subnet-Native Bayesian Consensus Fusion Evaluation (§4, §5).
Uses DatasetAssembler from template/hazard/dataset_assembler.py to fuse
1 to 5 miner variants via Bayesian Dawid-Skene aggregation and spatial box fusion,
evaluating the resulting consensus curve on tree crown delineation.
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from bench.metrics.rf1_metric import compute_rf1
from template.hazard.annotation_eval import PerMinerAnnotationScore
from template.hazard.dataset_assembler import AggregatedObject, DatasetAssembler
from template.protocol import PerImageAnnotationItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench.fusion_eval")


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


class MockImageCorpus:
    """Lightweight corpus mock for DatasetAssembler."""

    def __init__(self, image_dims: Dict[str, Tuple[int, int]]):
        self._image_dims = image_dims

    def is_golden(self, image_id: str) -> bool:
        return False

    def golden_lookup(self, image_id: str):
        return None

    def known_image_path(self, image_id: str):
        return None

    def annotation_images(self):
        return []

    def golden_images(self):
        return []


def run_fusion_benchmark(
    parquet_path: Path = Path("bench/data/selvabox/data/test-00000-of-00024.parquet"),
    device: str = "cuda",
    iou_threshold: float = 0.75,
    max_k: int = 5,
    n_images: Optional[int] = None,
) -> List[Dict]:
    """Evaluate Bayesian Dawid-Skene consensus fusion for k = 1..max_k miners."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    table = pq.read_table(str(parquet_path))
    total_samples = table.num_rows
    if n_images is not None and n_images < total_samples:
        total_samples = n_images

    logger.info("Evaluating Rung 6 Consensus Fusion on %d images from %s", total_samples, parquet_path)

    # Initialize vision models / variants
    from ultralytics import YOLO

    yolo_model = YOLO("models/tree_detection.pt")

    # Miner variant definitions (simulating diverse subnet miner pool):
    # Miner 0: Standard YOLO (conf=0.20)
    # Miner 1: Conservative YOLO (conf=0.30)
    # Miner 2: Aggressive YOLO (conf=0.15)
    # Miner 3: Standard YOLO with slight NMS variation (conf=0.25, iou=0.45)
    # Miner 4: Multi-scale/jittered YOLO (conf=0.18, imgsz=672)
    miner_configs = [
        {"uid": 0, "conf": 0.20, "iou": 0.50, "imgsz": 640, "weight": 0.85},
        {"uid": 1, "conf": 0.30, "iou": 0.50, "imgsz": 640, "weight": 0.90},
        {"uid": 2, "conf": 0.15, "iou": 0.55, "imgsz": 640, "weight": 0.80},
        {"uid": 3, "conf": 0.25, "iou": 0.45, "imgsz": 640, "weight": 0.88},
        {"uid": 4, "conf": 0.18, "iou": 0.50, "imgsz": 672, "weight": 0.82},
    ][:max_k]

    # Pre-generate predictions for all miners across all images
    logger.info("Generating predictions for %d miner variants...", len(miner_configs))
    miner_annotations: Dict[int, Dict[str, List[PerImageAnnotationItem]]] = {
        m["uid"]: {} for m in miner_configs
    }
    all_ground_truths = []
    image_dims: Dict[str, Tuple[int, int]] = {}

    for idx in range(total_samples):
        row = table.slice(idx, 1).to_pydict()
        img_bytes = row["image"][0]["bytes"]
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = pil_img.size
        img_id = f"selvabox_{idx:05d}"
        image_dims[img_id] = (w, h)

        # Ground truth
        raw_gt = row["annotations"][0]["bbox"]
        gt_boxes = []
        for b in raw_gt:
            x, y, bw, bh = b
            gt_boxes.append([x, y, x + bw, y + bh])
        gt_arr = np.array(gt_boxes, dtype=np.float32) if len(gt_boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        all_ground_truths.append({"boxes": gt_arr})

        # Generate each miner's prediction
        for m in miner_configs:
            res = yolo_model.predict(
                pil_img,
                conf=m["conf"],
                iou=m["iou"],
                imgsz=m["imgsz"],
                device=device,
                verbose=False,
            )
            items: List[PerImageAnnotationItem] = []
            if res and len(res) > 0 and res[0].boxes is not None:
                for b in res[0].boxes:
                    box = [float(v) for v in b.xyxy[0].tolist()]
                    conf = float(b.conf[0].item()) if hasattr(b, "conf") and b.conf is not None else 0.85
                    items.append(
                        PerImageAnnotationItem(
                            hazard_class="dense_tree",
                            bounding_box=box,
                            confidence=conf,
                            area=(box[2] - box[0]) * (box[3] - box[1]),
                        )
                    )
            miner_annotations[m["uid"]][img_id] = items

    logger.info("Miner predictions generated. Now evaluating Bayesian Consensus Fusion curves for k in 1..%d...", max_k)

    # Setup DatasetAssembler
    corpus = MockImageCorpus(image_dims)
    assembler = DatasetAssembler(corpus=corpus, storage_prefix="file:///tmp/fusion_eval", draw_boxes=False)

    priors = {"dense_tree": 0.85, "_background": 0.15}
    results = []

    for k in range(1, max_k + 1):
        active_miners = miner_configs[:k]
        active_uids = [m["uid"] for m in active_miners]

        per_miner_scores: Dict[int, PerMinerAnnotationScore] = {}
        miner_hotkeys: Dict[int, str] = {}
        for m in active_miners:
            uid = m["uid"]
            score = PerMinerAnnotationScore(
                uid=uid,
                class_weights={"dense_tree": m["weight"], "_background": 0.10},
                fidelity_scores_by_image_id={img_id: m["weight"] for img_id in image_dims},
            )
            per_miner_scores[uid] = score
            miner_hotkeys[uid] = f"hotkey_miner_{uid}"

        # Fuse predictions per image
        fused_predictions = []
        for idx in range(total_samples):
            img_id = f"selvabox_{idx:05d}"
            image_votes = {uid: miner_annotations[uid].get(img_id, []) for uid in active_uids}

            # Run Bayesian Dawid-Skene aggregation
            agg_result = assembler._aggregate_image(
                image_id=img_id,
                image_votes=image_votes,
                per_miner_scores=per_miner_scores,
                miner_hotkeys=miner_hotkeys,
                priors=priors,
            )

            fused_boxes = []
            fused_scores = []
            for obj in agg_result.get("objects", []):
                if obj.accepted_hazard_class and obj.accepted_hazard_class != "_background" and obj.fused_bounding_box:
                    fused_boxes.append(list(obj.fused_bounding_box))
                    fused_scores.append(float(obj.confidence))

            p_boxes_arr = np.array(fused_boxes, dtype=np.float32) if len(fused_boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
            p_scores_arr = np.array(fused_scores, dtype=np.float32) if len(fused_scores) > 0 else np.zeros((0,), dtype=np.float32)
            fused_predictions.append({"boxes": p_boxes_arr, "scores": p_scores_arr})

        # Evaluate fused predictions against ground truth
        m_75 = compute_rf1(
            fused_predictions,
            all_ground_truths,
            iou_threshold=0.75,
            score_threshold=0.20,
            nms_threshold=0.50,
        )
        m_50 = compute_rf1(
            fused_predictions,
            all_ground_truths,
            iou_threshold=0.50,
            score_threshold=0.20,
            nms_threshold=0.50,
        )

        logger.info(
            "k=%d Miners | RF1_75=%.4f (P=%.4f, R=%.4f) | RF1_50=%.4f (P=%.4f, R=%.4f)",
            k,
            m_75["rf1"],
            m_75["precision"],
            m_75["recall"],
            m_50["rf1"],
            m_50["precision"],
            m_50["recall"],
        )

        res_record = {
            "k_miners": k,
            "rf1_75": round(m_75["rf1"], 4),
            "precision_75": round(m_75["precision"], 4),
            "recall_75": round(m_75["recall"], 4),
            "rf1_50": round(m_50["rf1"], 4),
            "precision_50": round(m_50["precision"], 4),
            "recall_50": round(m_50["recall"], 4),
            "tp_75": m_75["tp"],
            "fp_75": m_75["fp"],
            "fn_75": m_75["fn"],
        }
        results.append(res_record)

    commit = get_git_commit()
    fusion_run_record = {
        "run_id": f"run_rung6_consensus_fusion_{int(time.time())}",
        "utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "track": "B",
        "dataset": "CanopyRS/SelvaBox (shard 0)",
        "aggregation_method": "bayesian_dawid_skene_v1",
        "n_images": total_samples,
        "max_k": max_k,
        "fusion_curve": results,
        "rung": 6,
        "comparable": False,
        "notes": "Subnet-native Bayesian Dawid-Skene consensus fusion via DatasetAssembler; evaluated across k=1..5 miners",
    }

    results_file = Path("bench/results/fusion_runs.jsonl")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "a") as f:
        f.write(json.dumps(fusion_run_record) + "\n")

    logger.info("Consensus fusion results logged to %s", results_file)
    return results


def main():
    parser = argparse.ArgumentParser(description="Rung 6 Consensus Fusion Evaluation")
    parser.add_argument("--parquet", default="bench/data/selvabox/data/test-00000-of-00024.parquet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--n-images", type=int, default=None)
    args = parser.parse_args()

    run_fusion_benchmark(
        parquet_path=Path(args.parquet),
        device=args.device,
        max_k=args.max_k,
        n_images=args.n_images,
    )


if __name__ == "__main__":
    main()
