#!/usr/bin/env python3
"""
Single-Script Final Evaluation for Track B (CanopyRS / SelvaBox):
Covers TASK B.4, TASK B.6, and TASK B.7 under non-negotiable compliance rules.

1. Loads all 310 test holdout images (shards test-00000 through test-00004).
2. Evaluates Zero-Shot Base Detector (models/tree_detection.pt) across all 310 images.
3. Evaluates Fine-Tuned Detector (models/tree_detection_finetuned_selvabox.pt, trained on 648 tiles) across all 310 images.
4. Evaluates 5-Miner Pool (variants of YOLOv8s) consensus for k=1..5.
5. Evaluates 6-Miner Pool with Genuinely Distinct Model Variant (Miner 5: YOLOv8n-seg or fine-tuned variant) for k=1..6.
6. Measures whether consensus lift grows, shrinks, or stays flat.
7. Logs all metrics to bench/results/track_b_final_evaluation.json.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from ultralytics import YOLO

from bench.metrics.rf1_metric import compute_rf1
from template.hazard.annotation_eval import (
    PerImageAnnotationItem,
    PerMinerAnnotationScore,
)
from template.hazard.dataset_assembler import DatasetAssembler
from template.hazard.image_corpus import ImageCorpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench.eval_track_b_final")


class MockImageCorpus(ImageCorpus):
    def __init__(self, dims: Dict[str, Tuple[int, int]]):
        self._dims = dims

    def get_image_dimensions(self, image_id: str) -> Tuple[int, int]:
        return self._dims.get(image_id, (1777, 1777))

    def get_all_unannotated_images(self):
        return []

    def get_annotated_images(self):
        return []

    def golden_images(self):
        return []


def load_test_data(parquet_paths: List[Path]) -> Tuple[List[dict], List[dict], Dict[str, Tuple[int, int]]]:
    samples = []
    ground_truths = []
    image_dims = {}

    for p in parquet_paths:
        logger.info("Loading test shard: %s", p.name)
        table = pq.read_table(str(p))
        for idx in range(table.num_rows):
            row = table.slice(idx, 1).to_pydict()
            img_bytes = row["image"][0]["bytes"]
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = pil_img.size
            img_id = f"{p.stem}_{idx:05d}"
            image_dims[img_id] = (w, h)

            raw_gt = row["annotations"][0]["bbox"]
            gt_boxes = []
            for b in raw_gt:
                x, y, bw, bh = b
                gt_boxes.append([x, y, x + bw, y + bh])
            gt_arr = np.array(gt_boxes, dtype=np.float32) if len(gt_boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
            ground_truths.append({"boxes": gt_arr})
            samples.append({"img_id": img_id, "pil_img": pil_img})

    logger.info("Loaded %d total test samples from %d shards", len(samples), len(parquet_paths))
    return samples, ground_truths, image_dims


def evaluate_single_detector(
    model: YOLO,
    samples: List[dict],
    ground_truths: List[dict],
    conf: float = 0.20,
    iou: float = 0.70,
    imgsz: int = 640,
    device: str = "cuda:0",
    name: str = "Detector",
) -> Dict:
    logger.info("Evaluating %s (conf=%.2f, iou=%.2f, imgsz=%d)...", name, conf, iou, imgsz)
    predictions = []
    total_dets = 0
    t0 = time.time()
    for s in samples:
        res = model.predict(s["pil_img"], conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
        boxes = []
        scores = []
        if res and len(res) > 0 and res[0].boxes is not None:
            for b in res[0].boxes:
                boxes.append(b.xyxy[0].tolist())
                sc = float(b.conf[0].item()) if hasattr(b, "conf") and b.conf is not None else 0.85
                scores.append(sc)
        total_dets += len(boxes)
        pred_dict = {
            "boxes": np.array(boxes, dtype=np.float32) if len(boxes) > 0 else np.zeros((0, 4), dtype=np.float32),
            "scores": np.array(scores, dtype=np.float32) if len(scores) > 0 else np.zeros((0,), dtype=np.float32),
        }
        predictions.append(pred_dict)

    eval_75 = compute_rf1(predictions, ground_truths, iou_threshold=0.75, score_threshold=conf, nms_threshold=0.50)
    eval_50 = compute_rf1(predictions, ground_truths, iou_threshold=0.50, score_threshold=conf, nms_threshold=0.50)
    elapsed = time.time() - t0

    result = {
        "name": name,
        "rf1_75": round(float(eval_75["rf1"]), 4),
        "precision_75": round(float(eval_75["precision"]), 4),
        "recall_75": round(float(eval_75["recall"]), 4),
        "rf1_50": round(float(eval_50["rf1"]), 4),
        "precision_50": round(float(eval_50["precision"]), 4),
        "recall_50": round(float(eval_50["recall"]), 4),
        "total_detections": total_dets,
        "eval_time_sec": round(elapsed, 2),
    }
    logger.info(
        "%s Results: RF1_75=%.4f (P=%.4f, R=%.4f) | RF1_50=%.4f | Dets=%d in %.1fs",
        name,
        result["rf1_75"],
        result["precision_75"],
        result["recall_75"],
        result["rf1_50"],
        result["total_detections"],
        result["eval_time_sec"],
    )
    return result


def generate_miner_pool_predictions(
    miner_specs: List[dict],
    samples: List[dict],
    device: str = "cuda:0",
) -> Dict[int, Dict[str, List[PerImageAnnotationItem]]]:
    """Generate annotations for each miner in the pool."""
    logger.info("Generating predictions for %d miners in pool...", len(miner_specs))
    miner_annotations: Dict[int, Dict[str, List[PerImageAnnotationItem]]] = {
        m["uid"]: {} for m in miner_specs
    }

    # Load models once
    loaded_models: Dict[str, YOLO] = {}
    for m in miner_specs:
        ckpt = m["checkpoint"]
        if ckpt not in loaded_models:
            logger.info("Loading model for pool: %s", ckpt)
            loaded_models[ckpt] = YOLO(ckpt)

    for s_idx, s in enumerate(samples):
        img_id = s["img_id"]
        pil_img = s["pil_img"]
        for m in miner_specs:
            model = loaded_models[m["checkpoint"]]
            res = model.predict(
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

        if (s_idx + 1) % 50 == 0 or (s_idx + 1) == len(samples):
            logger.info("Generated predictions for %d / %d samples", s_idx + 1, len(samples))

    return miner_annotations


def evaluate_consensus_curve(
    samples: List[dict],
    ground_truths: List[dict],
    image_dims: Dict[str, Tuple[int, int]],
    miner_annotations: Dict[int, Dict[str, List[PerImageAnnotationItem]]],
    miner_specs: List[dict],
    k_max: int = 6,
) -> List[Dict]:
    """Evaluate Dawid-Skene consensus for k=1..k_max."""
    curve_results = []
    mock_corpus = MockImageCorpus(image_dims)
    assembler = DatasetAssembler(mock_corpus, storage_prefix="file:///tmp/fusion_eval", draw_boxes=False)

    for k in range(1, k_max + 1):
        sub_miners = miner_specs[:k]
        active_uids = [m["uid"] for m in sub_miners]
        # In validation sweep, min_voters = k was optimal for eliminating false positives
        min_voters = k

        per_miner_scores = {
            uid: PerMinerAnnotationScore(
                uid=uid,
                class_weights={"dense_tree": 0.85, "_background": 0.10},
                fidelity_scores_by_image_id={img_id: 0.85 for img_id in image_dims},
            )
            for uid in active_uids
        }
        miner_hotkeys = {uid: f"hotkey_miner_{uid}" for uid in active_uids}
        priors = {"dense_tree": 0.85, "_background": 0.15}

        predictions = []
        total_dets = 0
        t0 = time.time()

        for s in samples:
            img_id = s["img_id"]
            image_votes = {
                uid: miner_annotations[uid].get(img_id, [])
                for uid in active_uids
            }
            # Cluster boxes across active miners
            clusters = assembler._cluster_boxes(image_votes, per_miner_scores)
            fused_boxes = []
            fused_scores = []

            for c_idx, cluster in enumerate(clusters):
                if len(cluster) < min_voters:
                    continue
                agg_obj, _ = assembler._infer_cluster(
                    image_id=img_id,
                    cluster_id=f"c_{c_idx}",
                    cluster_votes=cluster,
                    all_miner_ids=active_uids,
                    per_miner_scores=per_miner_scores,
                    miner_hotkeys=miner_hotkeys,
                    priors=priors,
                )
                if agg_obj.accepted_hazard_class and agg_obj.accepted_hazard_class != "_background" and agg_obj.fused_bounding_box:
                    fused_boxes.append(list(agg_obj.fused_bounding_box))
                    fused_scores.append(float(agg_obj.confidence))

            total_dets += len(fused_boxes)
            pred_dict = {
                "boxes": np.array(fused_boxes, dtype=np.float32) if len(fused_boxes) > 0 else np.zeros((0, 4), dtype=np.float32),
                "scores": np.array(fused_scores, dtype=np.float32) if len(fused_scores) > 0 else np.zeros((0,), dtype=np.float32),
            }
            predictions.append(pred_dict)

        eval_75 = compute_rf1(predictions, ground_truths, iou_threshold=0.75, score_threshold=0.20, nms_threshold=0.50)
        eval_50 = compute_rf1(predictions, ground_truths, iou_threshold=0.50, score_threshold=0.20, nms_threshold=0.50)
        elapsed = time.time() - t0

        res_entry = {
            "k": k,
            "min_voters": min_voters,
            "miners_included": [m["name"] for m in sub_miners],
            "rf1_75": round(float(eval_75["rf1"]), 4),
            "precision_75": round(float(eval_75["precision"]), 4),
            "recall_75": round(float(eval_75["recall"]), 4),
            "rf1_50": round(float(eval_50["rf1"]), 4),
            "precision_50": round(float(eval_50["precision"]), 4),
            "recall_50": round(float(eval_50["recall"]), 4),
            "total_detections": total_dets,
            "eval_time_sec": round(elapsed, 2),
        }
        curve_results.append(res_entry)
        logger.info(
            "Consensus k=%d (min_voters=%d): RF1_75=%.4f (P=%.4f, R=%.4f) | Dets=%d in %.1fs",
            k,
            min_voters,
            res_entry["rf1_75"],
            res_entry["precision_75"],
            res_entry["recall_75"],
            res_entry["total_detections"],
            res_entry["eval_time_sec"],
        )

    return curve_results


def main():
    parser = argparse.ArgumentParser(description="Final Track B Single-Script Evaluation")
    parser.add_argument("--test-dir", default="bench/data/selvabox/data")
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="bench/results/track_b_final_evaluation.json")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    test_shards = sorted(list(test_dir.glob("test-*.parquet")))[: args.num_shards]
    if not test_shards:
        logger.error("No test shards found in %s", test_dir)
        sys.exit(1)

    logger.info("Found %d test shards: %s", len(test_shards), [s.name for s in test_shards])
    samples, ground_truths, image_dims = load_test_data(test_shards)

    # 1. Evaluate Zero-Shot Base Detector
    base_model_path = "models/tree_detection.pt"
    base_model = YOLO(base_model_path)
    base_results = evaluate_single_detector(
        base_model,
        samples,
        ground_truths,
        conf=0.20,
        iou=0.50,
        imgsz=640,
        device=args.device,
        name="Zero-Shot Base YOLOv8s",
    )

    # 2. Evaluate Fine-Tuned Detector (trained on 648 SelvaBox tiles)
    ft_model_path = "models/tree_detection_finetuned_selvabox.pt"
    if Path(ft_model_path).exists():
        ft_model = YOLO(ft_model_path)
        ft_results = evaluate_single_detector(
            ft_model,
            samples,
            ground_truths,
            conf=0.25,
            iou=0.70,
            imgsz=640,
            device=args.device,
            name="Fine-Tuned YOLOv8s (648 tiles)",
        )
    else:
        logger.warning("Fine-tuned model %s not found!", ft_model_path)
        ft_results = None

    # 3. Define 6-Miner Pool
    # Miners 0-4: 5 hyperparameter variants of YOLOv8s base (models/tree_detection.pt)
    # Miner 5: Genuinely distinct model variant (YOLOv8n-seg or fine-tuned model)
    distinct_model_path = "models/tree_detection_yolov8n_selvabox.pt"
    if not Path(distinct_model_path).exists():
        logger.info("%s not ready yet, falling back to fine-tuned YOLOv8s for distinct miner", distinct_model_path)
        distinct_model_path = "models/tree_detection_finetuned_selvabox.pt"

    miner_specs = [
        {"uid": 0, "name": "Miner 0 (YOLOv8s base, conf=0.20)", "checkpoint": "models/tree_detection.pt", "conf": 0.20, "iou": 0.70, "imgsz": 640},
        {"uid": 1, "name": "Miner 1 (YOLOv8s base, conf=0.30)", "checkpoint": "models/tree_detection.pt", "conf": 0.30, "iou": 0.70, "imgsz": 640},
        {"uid": 2, "name": "Miner 2 (YOLOv8s base, conf=0.15)", "checkpoint": "models/tree_detection.pt", "conf": 0.15, "iou": 0.70, "imgsz": 640},
        {"uid": 3, "name": "Miner 3 (YOLOv8s base, nms=0.60)", "checkpoint": "models/tree_detection.pt", "conf": 0.25, "iou": 0.60, "imgsz": 640},
        {"uid": 4, "name": "Miner 4 (YOLOv8s base, imgsz=672)", "checkpoint": "models/tree_detection.pt", "conf": 0.20, "iou": 0.70, "imgsz": 672},
        {"uid": 5, "name": f"Miner 5 (Genuinely Distinct: {Path(distinct_model_path).stem})", "checkpoint": distinct_model_path, "conf": 0.25, "iou": 0.70, "imgsz": 640},
    ]

    # Generate predictions for all 6 miners
    miner_annotations = generate_miner_pool_predictions(miner_specs, samples, device=args.device)

    # Evaluate consensus curve k=1..6
    consensus_curve = evaluate_consensus_curve(
        samples,
        ground_truths,
        image_dims,
        miner_annotations,
        miner_specs,
        k_max=len(miner_specs),
    )

    # Compute consensus lift
    single_miner_rf1 = consensus_curve[0]["rf1_75"]
    k5_rf1 = consensus_curve[4]["rf1_75"] if len(consensus_curve) >= 5 else 0.0
    k6_rf1 = consensus_curve[5]["rf1_75"] if len(consensus_curve) >= 6 else 0.0

    k5_lift = round(k5_rf1 - single_miner_rf1, 4)
    k6_lift = round(k6_rf1 - single_miner_rf1, 4)
    lift_trend = "grew" if k6_rf1 > k5_rf1 else ("shrank" if k6_rf1 < k5_rf1 else "stayed flat")

    final_report = {
        "benchmark": "CanopyRS/SelvaBox",
        "holdout_shards": [s.name for s in test_shards],
        "total_test_images": len(samples),
        "zero_shot_base": base_results,
        "fine_tuned_yolo": ft_results,
        "consensus_curve": consensus_curve,
        "lift_analysis": {
            "single_miner_rf1": single_miner_rf1,
            "k5_rf1": k5_rf1,
            "k5_lift": k5_lift,
            "k6_rf1": k6_rf1,
            "k6_lift": k6_lift,
            "delta_k5_to_k6": round(k6_rf1 - k5_rf1, 4),
            "lift_trend": lift_trend,
        },
    }

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(final_report, f, indent=2)
    logger.info("Final Track B report written to %s", out_p)
    print(json.dumps(final_report["lift_analysis"], indent=2))


if __name__ == "__main__":
    main()
