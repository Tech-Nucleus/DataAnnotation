"""YOLO Local backend — fine-tunes and runs inference using Ultralytics YOLO."""

from __future__ import annotations

import hashlib
import io
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import bittensor as bt

from template.miner.backends.base import (
    BaseModelBackend,
    InferImage,
    TrainImage,
    TrainResult,
)
from template.protocol import PerImageAnnotationItem


def _norm_class_name(name: str) -> str:
    """Normalise a class name for separator/case-insensitive matching.

    COCO uses spaces (``dining table``); datasets often use underscores
    (``dining_table``). Both must resolve to the same pretrained class index so
    fine-tuning keeps the detection head (nc unchanged) instead of resetting it.
    """
    return " ".join(str(name or "").strip().lower().replace("_", " ").split())


class YoloLocalBackend(BaseModelBackend):
    """Fine-tune and run inference with a local Ultralytics YOLO model.

    When training data has labels, those are used directly.  When labels
    are absent, the backend can optionally run *pseudo-labeling*: infer
    with the pretrained checkpoint and treat high-confidence detections
    as ground truth for fine-tuning.  If ``skip_training`` is set in
    the miner config, ``train()`` is a no-op and ``infer()`` uses the
    pretrained weights.
    """

    def __init__(self, config: object):
        miner_cfg = getattr(config, "miner", object())

        # Pretrained weights (starting point)
        raw = str(getattr(miner_cfg, "yolo_pretrained_weights", "") or "yolov8s.pt").strip()
        self.pretrained_weights = Path(raw).expanduser()

        # Workspace for YOLO dataset layout
        ws = str(
            getattr(miner_cfg, "annotation_workspace", "artifacts/miner_annotation")
        ).strip()
        self.workspace = Path(ws) / "yolo_local"
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Class taxonomy
        self.class_taxonomy: Optional[List[str]] = None
        taxonomy_path = str(getattr(miner_cfg, "class_taxonomy_path", "") or "").strip()
        if taxonomy_path:
            import json

            self.class_taxonomy = json.loads(Path(taxonomy_path).read_text())

        # Training hyperparameters (all overridable via CLI)
        self.default_hypers: Dict = {
            "epochs": int(getattr(miner_cfg, "yolo_epochs", 50)),
            "imgsz": int(getattr(miner_cfg, "yolo_imgsz", 640)),
            "batch": int(getattr(miner_cfg, "yolo_batch", 16)),
            "lr0": float(getattr(miner_cfg, "yolo_lr0", 0.01)),
            "lrf": float(getattr(miner_cfg, "yolo_lrf", 0.01)),
            "momentum": float(getattr(miner_cfg, "yolo_momentum", 0.937)),
            "weight_decay": float(getattr(miner_cfg, "yolo_weight_decay", 0.0005)),
            "warmup_epochs": float(getattr(miner_cfg, "yolo_warmup_epochs", 3.0)),
            "optimizer": str(getattr(miner_cfg, "yolo_optimizer", "auto")),
            "augment": bool(getattr(miner_cfg, "yolo_augment", True)),
        }

        self.pseudo_label_conf = float(
            getattr(miner_cfg, "yolo_pseudo_label_conf", 0.5)
        )
        self.skip_training = bool(getattr(miner_cfg, "skip_training", False))

        # Seed labels path (optional)
        self.seed_labels_path: Optional[Path] = None
        seed = str(getattr(miner_cfg, "seed_labels_path", "") or "").strip()
        if seed:
            self.seed_labels_path = Path(seed).expanduser()

        # Cached fine-tuned checkpoint for infer()
        self._fine_tuned_checkpoint: Optional[Path] = None

    # ------------------------------------------------------------------
    # BaseModelBackend interface
    # ------------------------------------------------------------------

    def train(
        self,
        train_images: List[TrainImage],
        config: Dict,
    ) -> TrainResult:
        if self.skip_training or not train_images:
            bt.logging.info("YoloLocalBackend: skipping training (no-op).")
            version = self._model_hash(self.pretrained_weights)
            self._fine_tuned_checkpoint = self.pretrained_weights
            return TrainResult(
                model_version=version,
                metrics={},
                checkpoint_path=self.pretrained_weights,
            )

        from ultralytics import YOLO

        hypers = {**self.default_hypers, **config}

        # Load the pretrained model first so the dataset class map can reuse its COCO
        # class indices — a hazard_class matching a COCO name keeps that index, leaving
        # the detection head intact (genuinely new names extend it).
        model = YOLO(str(self.pretrained_weights))
        base_names = dict(getattr(model, "names", {}) or {})

        # Pseudo-label unlabeled training images using the pretrained model.
        if not any(img.labels for img in train_images):
            bt.logging.info(
                "YoloLocalBackend: no labels on training images — running pseudo-labeling."
            )
            train_images = self._pseudo_label(train_images)

        # Prepare YOLO dataset directory
        dataset_yaml = self._prepare_dataset(train_images, base_names=base_names)

        bt.logging.info(
            f"YoloLocalBackend: starting training — {len(train_images)} images, "
            f"hypers={hypers}"
        )
        started = time.time()

        train_kwargs = {
            "data": str(dataset_yaml),
            "project": str(self.workspace / "runs"),
            "name": "train",
            "exist_ok": True,
            "verbose": True,
        }
        # Pass through supported Ultralytics training args
        for key in (
            "epochs", "imgsz", "batch", "lr0", "lrf", "momentum",
            "weight_decay", "warmup_epochs", "optimizer", "augment",
        ):
            if key in hypers:
                train_kwargs[key] = hypers[key]
        # Per-miner training seed (derived from the miner-specific workspace) so two
        # independent miners produce different boxes instead of byte-identical
        # submissions that the validator would reject as duplicates.
        train_kwargs["seed"] = int(
            hypers.get("seed", int(hashlib.sha256(str(self.workspace).encode()).hexdigest()[:8], 16) % 100000)
        )

        results = model.train(**train_kwargs)

        elapsed = time.time() - started
        bt.logging.info(f"YoloLocalBackend: training complete in {elapsed:.1f}s")

        # 5. Find best checkpoint — trust the trainer's reported paths (Ultralytics may
        # relocate outputs), then fall back to the expected location.
        best_path: Optional[Path] = None
        trainer = getattr(model, "trainer", None)
        for attr in ("best", "last"):
            cand = getattr(trainer, attr, None) if trainer is not None else None
            if cand and Path(cand).is_file():
                best_path = Path(cand)
                break
        if best_path is None:
            for cand in (
                self.workspace / "runs" / "train" / "weights" / "best.pt",
                self.workspace / "runs" / "train" / "weights" / "last.pt",
            ):
                if cand.is_file():
                    best_path = cand
                    break

        self._fine_tuned_checkpoint = best_path if best_path is not None else self.pretrained_weights
        version = self._model_hash(self._fine_tuned_checkpoint)

        metrics: Dict[str, float] = {}
        if results and hasattr(results, "results_dict"):
            metrics = {k: float(v) for k, v in results.results_dict.items()}
        metrics["training_seconds"] = elapsed

        bt.logging.info(
            f"YoloLocalBackend: best checkpoint={self._fine_tuned_checkpoint}, "
            f"version={version[:16]}…"
        )

        return TrainResult(
            model_version=version,
            metrics=metrics,
            checkpoint_path=self._fine_tuned_checkpoint,
        )

    def infer(
        self,
        inference_images: List[InferImage],
        model_version: str,
    ) -> Dict[str, List[PerImageAnnotationItem]]:
        import os
        import random

        if os.environ.get("MINER_ADVERSARIAL") == "1":
            bt.logging.warning("MINER_ADVERSARIAL is active: returning synthetic garbage boxes!")
            results_map: Dict[str, List[PerImageAnnotationItem]] = {}
            rng = random.Random(1337)
            for img in inference_images:
                from PIL import Image as PILImage
                try:
                    pil_img = PILImage.open(str(img.image_path))
                    width, height = pil_img.size
                except Exception:
                    width, height = 640, 640
                x1 = rng.uniform(0, max(1.0, width * 0.6))
                y1 = rng.uniform(0, max(1.0, height * 0.6))
                x2 = min(float(width), x1 + rng.uniform(width * 0.1, width * 0.35))
                y2 = min(float(height), y1 + rng.uniform(height * 0.1, height * 0.35))
                results_map[img.image_id] = [
                    PerImageAnnotationItem(
                        hazard_class="adversarial_garbage",
                        bounding_box=[x1, y1, x2, y2],
                    )
                ]
            return results_map

        from ultralytics import YOLO

        checkpoint = self._fine_tuned_checkpoint or self.pretrained_weights
        bt.logging.info(
            f"YoloLocalBackend: running inference on {len(inference_images)} images "
            f"with checkpoint={checkpoint}"
        )

        model = YOLO(str(checkpoint))
        results_map: Dict[str, List[PerImageAnnotationItem]] = {}

        for img in inference_images:
            from PIL import Image as PILImage

            pil_img = PILImage.open(str(img.image_path))
            results = model(pil_img, verbose=False)

            annotations: List[PerImageAnnotationItem] = []
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        cls_idx = int(box.cls[0].item())
                        cls_name = model.names.get(cls_idx, f"class_{cls_idx}")
                        annotations.append(
                            PerImageAnnotationItem(
                                hazard_class=cls_name,
                                bounding_box=xyxy,
                            )
                        )
            results_map[img.image_id] = annotations

        return results_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pseudo_label(
        self,
        images: List[TrainImage],
    ) -> List[TrainImage]:
        """Generate pseudo-labels using the pretrained model."""
        from ultralytics import YOLO
        from PIL import Image as PILImage

        model = YOLO(str(self.pretrained_weights))
        labeled: List[TrainImage] = []

        for img in images:
            pil_img = PILImage.open(str(img.image_path))
            results = model(pil_img, verbose=False)

            labels: List[PerImageAnnotationItem] = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf[0].item())
                        if conf >= self.pseudo_label_conf:
                            xyxy = box.xyxy[0].tolist()
                            cls_idx = int(box.cls[0].item())
                            cls_name = model.names.get(cls_idx, f"class_{cls_idx}")
                            labels.append(
                                PerImageAnnotationItem(
                                    hazard_class=cls_name,
                                    bounding_box=xyxy,
                                )
                            )
            labeled.append(
                TrainImage(
                    image_id=img.image_id,
                    image_path=img.image_path,
                    labels=labels if labels else None,
                )
            )

        labeled_count = sum(1 for im in labeled if im.labels)
        bt.logging.info(
            f"YoloLocalBackend: pseudo-labeled {labeled_count}/{len(images)} images "
            f"(conf≥{self.pseudo_label_conf})"
        )
        return labeled

    def _prepare_dataset(
        self,
        images: List[TrainImage],
        base_names: Optional[Dict[int, str]] = None,
    ) -> Path:
        """Create a YOLO-format dataset directory and return the path to dataset.yaml."""
        import yaml

        ds_root = self.workspace / "dataset"
        img_dir = ds_root / "images" / "train"
        lbl_dir = ds_root / "labels" / "train"

        # Clean previous run
        if ds_root.exists():
            shutil.rmtree(ds_root)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Seed the class map from the base model's COCO classes (or an explicit
        # taxonomy) so matching names reuse their original index and the detection
        # head is preserved. The miner's exact spelling is kept as the display name.
        name_to_id: Dict[str, int] = {}
        id_to_display: Dict[int, str] = {}
        if self.class_taxonomy:
            for i, n in enumerate(self.class_taxonomy):
                name_to_id[_norm_class_name(n)] = i
                id_to_display[i] = n
        elif base_names:
            for idx in sorted(base_names):
                name_to_id[_norm_class_name(base_names[idx])] = int(idx)
                id_to_display[int(idx)] = str(base_names[idx])

        def resolve(raw: str) -> int:
            key = _norm_class_name(raw)
            if key not in name_to_id:
                name_to_id[key] = len(name_to_id)
            cid = name_to_id[key]
            id_to_display[cid] = raw
            return cid

        for img in images:
            # Symlink or copy image
            dest_img = img_dir / f"{img.image_id}.jpg"
            if not dest_img.exists():
                try:
                    dest_img.symlink_to(img.image_path.resolve())
                except OSError:
                    shutil.copy2(str(img.image_path), str(dest_img))

            # Write labels
            dest_lbl = lbl_dir / f"{img.image_id}.txt"
            if img.labels:
                from PIL import Image as PILImage

                pil_img = PILImage.open(str(img.image_path))
                w, h = pil_img.size

                lines = []
                for item in img.labels:
                    cls_idx = resolve(item.hazard_class)
                    x1, y1, x2, y2 = item.bounding_box
                    # Convert xyxy to YOLO format: cx, cy, bw, bh (normalized)
                    cx = ((x1 + x2) / 2.0) / w
                    cy = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                dest_lbl.write_text("\n".join(lines) + "\n")
            else:
                # Empty label file (background image)
                dest_lbl.write_text("")

        if not id_to_display:
            id_to_display = {0: "object"}
        nc = max(id_to_display) + 1
        names = {i: id_to_display.get(i, f"class_{i}") for i in range(nc)}

        # Write dataset.yaml
        dataset_yaml = ds_root / "dataset.yaml"
        yaml_data = {
            "path": str(ds_root.resolve()),
            "train": "images/train",
            "val": "images/train",  # Use same split for val (single-split fine-tune)
            "names": names,
        }
        dataset_yaml.write_text(yaml.dump(yaml_data, default_flow_style=False))

        bt.logging.info(
            f"YoloLocalBackend: prepared dataset at {ds_root} — "
            f"{len(images)} images, {nc} classes"
        )
        return dataset_yaml

    @staticmethod
    def _model_hash(checkpoint: Path) -> str:
        """Compute a short hash of a model checkpoint file for versioning."""
        if not checkpoint.is_file():
            return hashlib.sha256(str(checkpoint).encode()).hexdigest()
        h = hashlib.sha256()
        with open(checkpoint, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
