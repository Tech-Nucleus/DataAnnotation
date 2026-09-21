#!/usr/bin/env python3
"""
SelvaBox YOLO Dataset Preparation & Fine-Tuning Pipeline (§4, §5).
Converts SelvaBox parquet shards into Ultralytics YOLO format
and fine-tunes a tree crown detector on tropical forest imagery.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq
import yaml
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench.prepare_selvabox")


def convert_parquet_to_yolo(
    parquet_path: Path,
    output_img_dir: Path,
    output_lbl_dir: Path,
    prefix: str = "img",
    max_samples: int = None,
    tile_size: int = None,
    stride: int = None,
) -> int:
    """Convert a SelvaBox parquet shard to YOLO images and label txt files, optionally tiling large orthomosaics."""
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(str(parquet_path))
    n_samples = table.num_rows if max_samples is None else min(max_samples, table.num_rows)
    logger.info("Converting %d samples from %s to YOLO format (tile_size=%s, stride=%s)...", n_samples, parquet_path.name, tile_size, stride)

    converted = 0
    total_boxes_written = 0
    for idx in range(n_samples):
        row = table.slice(idx, 1).to_pydict()
        img_bytes = row["image"][0]["bytes"]
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_w, img_h = pil_img.size
        raw_boxes = row["annotations"][0]["bbox"]

        if tile_size is not None and (img_w > tile_size or img_h > tile_size):
            step = stride if stride is not None else tile_size
            x_steps = list(range(0, img_w - tile_size, step)) + [max(0, img_w - tile_size)]
            y_steps = list(range(0, img_h - tile_size, step)) + [max(0, img_h - tile_size)]
            x_steps = sorted(list(set(x_steps)))
            y_steps = sorted(list(set(y_steps)))

            t_idx = 0
            for y0 in y_steps:
                for x0 in x_steps:
                    x1_t, y1_t, x2_t, y2_t = x0, y0, x0 + tile_size, y0 + tile_size
                    tile_crop = pil_img.crop((x1_t, y1_t, x2_t, y2_t))
                    file_id = f"{prefix}_{idx:05d}_t{t_idx:04d}"
                    t_idx += 1

                    label_lines = []
                    for b in raw_boxes:
                        bx, by, bw, bh = b
                        bx2, by2 = bx + bw, by + bh
                        ix1 = max(bx, x1_t)
                        iy1 = max(by, y1_t)
                        ix2 = min(bx2, x2_t)
                        iy2 = min(by2, y2_t)
                        if ix2 > ix1 and iy2 > iy1:
                            inter_area = (ix2 - ix1) * (iy2 - iy1)
                            orig_area = max(1.0, float(bw * bh))
                            if inter_area >= 0.3 * orig_area and (ix2 - ix1) >= 4 and (iy2 - iy1) >= 4:
                                lx1 = max(0.0, min(1.0, (ix1 - x1_t) / tile_size))
                                ly1 = max(0.0, min(1.0, (iy1 - y1_t) / tile_size))
                                lx2 = max(0.0, min(1.0, (ix2 - x1_t) / tile_size))
                                ly2 = max(0.0, min(1.0, (iy1 - y1_t) / tile_size))
                                lx3 = max(0.0, min(1.0, (ix2 - x1_t) / tile_size))
                                ly3 = max(0.0, min(1.0, (iy2 - y1_t) / tile_size))
                                lx4 = max(0.0, min(1.0, (ix1 - x1_t) / tile_size))
                                ly4 = max(0.0, min(1.0, (iy2 - y1_t) / tile_size))
                                label_lines.append(f"0 {lx1:.6f} {ly1:.6f} {lx2:.6f} {ly2:.6f} {lx3:.6f} {ly3:.6f} {lx4:.6f} {ly4:.6f}")

                    img_file = output_img_dir / f"{file_id}.jpg"
                    tile_crop.save(img_file, quality=95)
                    lbl_file = output_lbl_dir / f"{file_id}.txt"
                    lbl_file.write_text("\n".join(label_lines) + "\n")
                    converted += 1
                    total_boxes_written += len(label_lines)
        else:
            file_id = f"{prefix}_{idx:05d}"
            img_file = output_img_dir / f"{file_id}.jpg"
            pil_img.save(img_file, quality=95)

            label_lines = []
            for b in raw_boxes:
                x, y, w, h = b
                if w <= 2 or h <= 2:
                    continue
                x1 = max(0.0, min(1.0, x / img_w))
                y1 = max(0.0, min(1.0, y / img_h))
                x2 = max(0.0, min(1.0, (x + w) / img_w))
                y2 = max(0.0, min(1.0, y / img_h))
                x3 = max(0.0, min(1.0, (x + w) / img_w))
                y3 = max(0.0, min(1.0, (y + h) / img_h))
                x4 = max(0.0, min(1.0, x / img_w))
                y4 = max(0.0, min(1.0, (y + h) / img_h))

                label_lines.append(f"0 {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}")

            lbl_file = output_lbl_dir / f"{file_id}.txt"
            lbl_file.write_text("\n".join(label_lines) + "\n")
            converted += 1
            total_boxes_written += len(label_lines)

    logger.info("Successfully converted %d images/tiles (%d total boxes) to %s", converted, total_boxes_written, output_img_dir.parent)
    return converted


def create_dataset_yaml(data_root: Path) -> Path:
    """Create Ultralytics dataset YAML file."""
    yaml_dict = {
        "path": str(data_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "tree_crown"},
    }
    yaml_path = data_root / "selvabox.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
    logger.info("Dataset YAML written to %s", yaml_path)
    return yaml_path


def finetune_yolo(
    data_yaml: Path,
    base_checkpoint: str = "models/tree_detection.pt",
    epochs: int = 15,
    imgsz: int = 640,
    batch_size: int = 8,
    device: str = "cuda:0",
    output_model_path: str = "models/tree_detection_finetuned_selvabox.pt",
) -> str:
    """Fine-tune YOLO model on SelvaBox data."""
    from ultralytics import YOLO

    logger.info("Loading base checkpoint: %s", base_checkpoint)
    model = YOLO(base_checkpoint)

    logger.info("Starting fine-tuning on %s for %d epochs...", data_yaml, epochs)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project="bench/runs/train_selvabox",
        name="finetune",
        exist_ok=True,
        workers=4,
        verbose=True,
    )

    best_ckpt = Path("bench/runs/train_selvabox/finetune/weights/best.pt")
    if best_ckpt.exists():
        import shutil
        out_p = Path(output_model_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(best_ckpt), str(out_p))
        logger.info("Fine-tuned model saved to %s", out_p)
        return str(out_p)
    else:
        logger.warning("best.pt not found, returning base checkpoint.")
        return base_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SelvaBox YOLO Dataset and Fine-Tune")
    parser.add_argument("--train-parquet", default="bench/data/selvabox/data/train-00000-of-00034.parquet")
    parser.add_argument("--val-parquet", default="bench/data/selvabox/data/validation-00000-of-00006.parquet")
    parser.add_argument("--dest", default="bench/data/selvabox/yolo_dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size for slicing orthomosaics")
    parser.add_argument("--stride", type=int, default=512, help="Stride for slicing orthomosaics")
    parser.add_argument("--device", default="cuda:0", help="Training device (e.g. cuda:0 or cpu)")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    train_p = Path(args.train_parquet)
    val_p = Path(args.val_parquet)

    if train_p.exists():
        convert_parquet_to_yolo(
            train_p,
            dest / "images" / "train",
            dest / "labels" / "train",
            prefix="train_00",
            tile_size=args.tile_size,
            stride=args.stride,
        )
    if val_p.exists():
        convert_parquet_to_yolo(
            val_p,
            dest / "images" / "val",
            dest / "labels" / "val",
            prefix="val_00",
            tile_size=args.tile_size,
            stride=args.stride,
        )

    data_yaml = create_dataset_yaml(dest)

    if not args.skip_train and train_p.exists():
        finetune_yolo(data_yaml, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
