"""
CanopyMRV Miner Adapter.
Wraps the CanopyMRV miner models and inference engine to output standardized
predictions for each benchmark track.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class CanopyMRVMinerAdapter:
    """Adapter wrapping the CanopyMRV miner stack for benchmark evaluation."""

    def __init__(
        self,
        checkpoint_path: str = "models/tree_detection.pt",
        segformer_path: str = "models/tcd-segformer-mit-b2",
        qwen_path: str = "models/qwen2.5-vl-3b",
        device: Optional[str] = None,
        conf_threshold: float = 0.20,
        load_yolo: bool = False,
        load_segformer: bool = True,
        load_qwen: bool = False,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.segformer_path = segformer_path
        self.qwen_path = qwen_path
        self.conf_threshold = conf_threshold
        self.load_yolo = load_yolo
        self.load_segformer = load_segformer
        self.load_qwen = load_qwen

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._engine = None
        self._init_engine()

    def _init_engine(self) -> None:
        """Initialize the underlying EcologicalVisionEngine."""
        try:
            from template.miner.ecological_reasoning import EcologicalVisionEngine

            self._engine = EcologicalVisionEngine(
                checkpoint_path=self.checkpoint_path,
                segformer_path=self.segformer_path,
                qwen_path=self.qwen_path,
                device=self.device,
                conf_threshold=self.conf_threshold,
                load_yolo=self.load_yolo,
                load_segformer=self.load_segformer,
                load_qwen=self.load_qwen,
            )
            logger.info("CanopyMRVMinerAdapter initialized with EcologicalVisionEngine on %s", self.device)
        except Exception as exc:
            logger.error("Failed to initialize EcologicalVisionEngine: %s", exc)
            self._engine = None

    def predict_semantic_mask(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        tiled: bool = False,
        tile_size: int = 1024,
        stride: int = 768,
        threshold: float = 0.50,
    ) -> np.ndarray:
        """
        Run semantic tree canopy segmentation (Track A / Track D).
        Returns binary mask (H, W) where 1 = canopy/tree, 0 = background.
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        img_w, img_h = pil_img.size

        if self._engine is not None and self._engine._segformer_model is not None:
            processor = self._engine._segformer_processor
            model = self._engine._segformer_model

            if not tiled or (img_w <= tile_size and img_h <= tile_size):
                inputs = processor(images=pil_img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    upsampled = torch.nn.functional.interpolate(
                        outputs.logits,
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    probs = torch.softmax(upsampled, dim=1)[0, 1].cpu().numpy()
                    return (probs > threshold).astype(np.uint8)
            else:
                # Tiled inference with overlap blending
                img_np = np.array(pil_img)
                prob_map = np.zeros((img_h, img_w), dtype=np.float32)
                count_map = np.zeros((img_h, img_w), dtype=np.float32)

                xs = list(range(0, img_w - tile_size + 1, stride))
                if len(xs) == 0 or xs[-1] + tile_size < img_w:
                    xs.append(max(0, img_w - tile_size))
                ys = list(range(0, img_h - tile_size + 1, stride))
                if len(ys) == 0 or ys[-1] + tile_size < img_h:
                    ys.append(max(0, img_h - tile_size))

                tiles = [Image.fromarray(img_np[y:y+tile_size, x:x+tile_size]) for y in ys for x in xs]
                coords = [(y, x) for y in ys for x in xs]

                batch_size = 4
                for i in range(0, len(tiles), batch_size):
                    b_tiles = tiles[i:i+batch_size]
                    b_coords = coords[i:i+batch_size]
                    inputs = processor(images=b_tiles, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        upsampled = torch.nn.functional.interpolate(
                            outputs.logits,
                            size=(tile_size, tile_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        batch_probs = torch.softmax(upsampled, dim=1)[:, 1].cpu().numpy()

                    for p, (y, x) in zip(batch_probs, b_coords):
                        prob_map[y:y+tile_size, x:x+tile_size] += p
                        count_map[y:y+tile_size, x:x+tile_size] += 1.0

                prob_map /= np.maximum(count_map, 1.0)
                return (prob_map > threshold).astype(np.uint8)

        # Fallback: simple green excess index thresholding
        np_img = np.array(pil_img)
        r = np_img[:, :, 0].astype(np.float32)
        g = np_img[:, :, 1].astype(np.float32)
        b = np_img[:, :, 2].astype(np.float32)
        exg = 2.0 * g - r - b
        return (exg > 15.0).astype(np.uint8)

    def predict_boxes(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> List[Tuple[List[float], float, str]]:
        """
        Run tree crown detection (Track B).
        Returns list of (box_xyxy, confidence, class_name).
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        boxes = []
        if self._engine is not None:
            anns, _ = self._engine.reason_and_annotate(pil_img)
            for a in anns:
                boxes.append((a.bounding_box, a.confidence or 0.85, a.hazard_class))
        return boxes
