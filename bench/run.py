#!/usr/bin/env python3
"""
CanopyMRV SOTA Benchmark Runner (§4, §5).
Executes Phase 1 (Reproduction Gate) and Phase 2 (CanopyMRV Evaluation)
across all four benchmark tracks under strict protocol conformity.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from bench.metrics.iou_metrics import SegmentationEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench.run")


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


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class BenchmarkDatasetReader:
    """Reads OAM-TCD dataset from parquet split files."""

    def __init__(self, parquet_path: Path):
        self.parquet_path = parquet_path
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        self.table = pq.read_table(str(parquet_path))
        self.num_rows = self.table.num_rows
        logger.info("Loaded parquet with %d rows from %s", self.num_rows, parquet_path)

    def get_sample(self, idx: int) -> Tuple[Image.Image, np.ndarray, str]:
        """
        Extract image and mask for row idx.
        Returns (PIL.Image, target_mask as np.ndarray [H, W], image_id).
        """
        row = self.table.slice(idx, 1).to_pydict()
        
        # Check column names
        img_col = "image" if "image" in row else "img"
        mask_col = "annotation" if "annotation" in row else ("mask" if "mask" in row else "label")
        id_col = "image_id" if "image_id" in row else ("id" if "id" in row else "image_id")

        # Image bytes
        raw_img = row[img_col][0]
        if isinstance(raw_img, dict) and "bytes" in raw_img:
            img_bytes = raw_img["bytes"]
        elif isinstance(raw_img, bytes):
            img_bytes = raw_img
        else:
            img_bytes = bytes(raw_img)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Mask bytes or array
        raw_mask = row[mask_col][0]
        if isinstance(raw_mask, dict) and "bytes" in raw_mask:
            mask_bytes = raw_mask["bytes"]
            mask_img = Image.open(io.BytesIO(mask_bytes))
            mask_arr = np.array(mask_img, dtype=np.uint8)
        elif isinstance(raw_mask, bytes):
            mask_img = Image.open(io.BytesIO(raw_mask))
            mask_arr = np.array(mask_img, dtype=np.uint8)
        elif isinstance(raw_mask, np.ndarray):
            mask_arr = raw_mask.astype(np.uint8)
        else:
            mask_arr = np.array(raw_mask, dtype=np.uint8)

        # Ensure 2D binary mask: 0=background, 1=tree
        # In OAM-TCD, non-zero pixels across channels represent tree cover
        if len(mask_arr.shape) > 2:
            target_mask = np.any(mask_arr != 0, axis=-1).astype(np.uint8)
        else:
            target_mask = (mask_arr != 0).astype(np.uint8)

        img_id = str(row[id_col][0]) if id_col in row else f"sample_{idx}"
        return pil_img, target_mask, img_id


def evaluate_track_a(
    config: dict,
    mode: str = "reproduce",
    seed: int = 0,
    max_samples: Optional[int] = None,
    device: str = "cuda",
    tiled: bool = False,
    tile_size: int = 1024,
    stride: int = 768,
    threshold: float = 0.50,
    model_path: Optional[str] = None,
    tta: bool = False,
) -> dict:
    """Execute Track A evaluation (reproduction or CanopyMRV evaluation)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    parquet_file = Path("bench/data/oam_tcd/data/test-00000-of-00001.parquet")
    if not parquet_file.exists():
        raise FileNotFoundError(
            f"Official test split not found at {parquet_file}. Run data fetcher first."
        )

    split_sha = compute_sha256(parquet_file)
    reader = BenchmarkDatasetReader(parquet_file)
    n_total = reader.num_rows
    n_eval = min(n_total, max_samples) if max_samples else n_total

    logger.info("Evaluating Track A | Mode: %s | Samples: %d/%d | Tiled: %s", mode, n_eval, n_total, tiled)

    # Initialize model
    if mode == "reproduce":
        # Check if local weights exist, otherwise load from HF
        weights_dir = Path(model_path) if model_path else Path("bench/data/weights/tcd-segformer-mit-b5")
        if weights_dir.exists() and (weights_dir / "model.safetensors").exists():
            model_id = str(weights_dir)
        else:
            # Fallback to local b2 if b5 not fully downloaded yet
            b2_dir = Path("models/tcd-segformer-mit-b2")
            if b2_dir.exists() and (b2_dir / "model.safetensors").exists():
                logger.info("Using local SegFormer mit-b2 for reproduction evaluation")
                model_id = str(b2_dir)
            else:
                model_id = "restor/tcd-segformer-mit-b5"

        logger.info("Loading incumbent SegFormer from %s on %s ...", model_id, device)
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
        model.eval()

        pipeline_info = [
            {"component": model_id, "role": "semantic_segmentation_mask", "provenance": "incumbent"}
        ]
    else:
        # CanopyMRV miner stack evaluation
        from bench.adapters.canopymrv_miner import CanopyMRVMinerAdapter
        seg_path = model_path if model_path else "models/tcd-segformer-mit-b2"
        adapter = CanopyMRVMinerAdapter(segformer_path=seg_path, device=device)
        pipeline_info = [
            {"component": "Qwen2.5-VL-3B-Instruct", "role": "macro_landscape_reasoning", "provenance": "qwen_box"},
            {"component": seg_path, "role": "canopy_mask", "provenance": "segformer_mask"},
            {"component": "models/tree_detection.pt", "role": "crown_detector", "provenance": "yolo_box"},
        ]

    evaluator = SegmentationEvaluator(target_class=1)
    start_time = time.time()

    for idx in range(n_eval):
        pil_img, target_mask, img_id = reader.get_sample(idx)
        img_w, img_h = pil_img.size

        if mode == "reproduce":
            inputs = processor(images=pil_img, return_tensors="pt").to(device)
            if tta:
                # 8-fold dihedral D4 test-time augmentation
                transforms = [
                    (lambda x: x, lambda y: y),
                    (lambda x: torch.flip(x, [-1]), lambda y: torch.flip(y, [-1])),
                    (lambda x: torch.flip(x, [-2]), lambda y: torch.flip(y, [-2])),
                    (lambda x: torch.rot90(x, 1, [-2, -1]), lambda y: torch.rot90(y, -1, [-2, -1])),
                    (lambda x: torch.rot90(x, 2, [-2, -1]), lambda y: torch.rot90(y, -2, [-2, -1])),
                    (lambda x: torch.rot90(x, 3, [-2, -1]), lambda y: torch.rot90(y, -3, [-2, -1])),
                    (lambda x: torch.flip(torch.rot90(x, 1, [-2, -1]), [-1]), lambda y: torch.rot90(torch.flip(y, [-1]), -1, [-2, -1])),
                    (lambda x: torch.flip(torch.rot90(x, 3, [-2, -1]), [-1]), lambda y: torch.rot90(torch.flip(y, [-1]), -3, [-2, -1])),
                ]
                accumulator = torch.zeros((1, 2, img_h, img_w), device=device, dtype=torch.float32)
                for t_fn, inv_fn in transforms:
                    aug_pv = t_fn(inputs["pixel_values"])
                    with torch.no_grad():
                        out = model(pixel_values=aug_pv).logits
                        upsampled = torch.nn.functional.interpolate(
                            out, size=(img_h, img_w), mode="bilinear", align_corners=False
                        )
                        probs = torch.softmax(upsampled, dim=1)
                        accumulator += inv_fn(probs)
                    del out, upsampled, probs, aug_pv
                pred_mask = (accumulator / len(transforms)).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                del accumulator
            else:
                with torch.no_grad():
                    outputs = model(**inputs)
                    upsampled = torch.nn.functional.interpolate(
                        outputs.logits,
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred_mask = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        else:
            pred_mask = adapter.predict_semantic_mask(
                pil_img,
                tiled=tiled,
                tile_size=tile_size,
                stride=stride,
                threshold=threshold,
            )

        evaluator.add_batch(pred_mask, target_mask)

        if (idx + 1) % 10 == 0 or (idx + 1) == n_eval:
            current = evaluator.compute()
            logger.info(
                "Progress [%d/%d]: mIoU: %.4f | Tree IoU: %.4f | Acc: %.4f",
                idx + 1,
                n_eval,
                current["mean_iou"],
                current["tree_iou"],
                current["accuracy"],
            )

    wall_s = round(time.time() - start_time, 2)
    metrics = evaluator.compute()

    commit = get_git_commit()
    incumbent_name = config.get("incumbent", {}).get("model_id", "restor/tcd-segformer-mit-b5")
    incumbent_val = config.get("incumbent", {}).get("published_metrics", {}).get("iou", 0.887)

    run_record = {
        "run_id": f"run_trackA_{mode}_{seed}_{int(time.time())}",
        "utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "track": "A",
        "dataset": config.get("dataset", {}).get("id", "restor/tcd"),
        "split": config.get("dataset", {}).get("split", "test"),
        "split_sha256": split_sha,
        "gsd_m": config.get("dataset", {}).get("gsd_m", 0.10),
        "tile_px": config.get("dataset", {}).get("tile_size_px", 2048),
        "pipeline": pipeline_info,
        "seed": seed,
        "metric": "mIoU",
        "value": metrics["mean_iou"],
        "tree_iou": metrics["tree_iou"],
        "bg_iou": metrics["bg_iou"],
        "macro_mean_iou": metrics["macro_mean_iou"],
        "macro_tree_iou": metrics["macro_tree_iou"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "n_images": metrics["n_images"],
        "wall_s": wall_s,
        "incumbent": incumbent_name,
        "incumbent_value": incumbent_val,
        "delta": round(metrics["mean_iou"] - incumbent_val, 4),
        "comparable": (n_eval == n_total),
        "tiled": tiled,
        "tta": tta,
        "tile_size": tile_size if tiled else None,
        "stride": stride if tiled else None,
        "threshold": threshold,
        "notes": f"mode={mode}; max_samples={max_samples}; tiled={tiled}; tta={tta}; threshold={threshold}",
    }

    # Append to results JSONL
    results_file = Path("bench/results/track_A_runs.jsonl")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "a") as f:
        f.write(json.dumps(run_record) + "\n")

    logger.info("Run record logged to %s", results_file)
    logger.info("Final Results: %s", json.dumps(run_record, indent=2))
    return run_record


def main():
    parser = argparse.ArgumentParser(description="CanopyMRV SOTA Benchmark Harness")
    parser.add_argument("--track", choices=["A", "B", "C", "D"], default="A")
    parser.add_argument("--config", default="bench/configs/a_oam_tcd.yaml")
    parser.add_argument("--mode", choices=["reproduce", "eval"], default="reproduce")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tiled", action="store_true", help="Enable tiled inference with overlap")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--tta", action="store_true", help="Enable 8-fold dihedral D4 TTA")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.track == "A":
        evaluate_track_a(
            config,
            mode=args.mode,
            seed=args.seed,
            max_samples=args.max_samples,
            device=args.device,
            tiled=args.tiled,
            tile_size=args.tile_size,
            stride=args.stride,
            threshold=args.threshold,
            model_path=args.model_path,
            tta=args.tta,
        )
    else:
        logger.warning("Track %s runner is pending implementation.", args.track)


if __name__ == "__main__":
    main()
