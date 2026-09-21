"""
Ecological Vision Reasoning Engine for Climate MRV Carbon Credits.
==================================================================

Performs high-reasoning vision and bio-spectral analysis on satellite/aerial
imagery to classify and quantify vegetation for carbon credits:

Taxonomy & Carbon Weight Multipliers:
- mangrove: 3.5x (coastal wetland, ultra-high blue carbon sequestration)
- dense_tree: 1.8x (mature dense forest canopy, high biomass)
- ordinary_tree: 1.0x (standard terrestrial tree crown)
- farm: 0.7x (agricultural cropland, agroforestry, managed canopy)
- plant: 0.4x (woody perennial shrubs, small plants, secondary regrowth)

Computes:
- Flexible Oriented Bounding Boxes (OBB) & polygon contours
- Per-object pixel area, coverage ratio, and weighted carbon contribution
- Image-level net_weight (sum of box carbon weights), tree_coverage_percentage,
  tree_count, dominant_class, and class_breakdown.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from template.miner.geometry import (
    CARBON_WEIGHT_MULTIPLIERS,
    canonical_carbon_class,
    canonical_image_name,
)
from template.protocol import PerImageAnnotationItem

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None


class EcologicalVisionEngine:
    """High-reasoning multimodal vision engine for Carbon MRV data annotation."""

    def __init__(
        self,
        checkpoint_path: str = "models/tree_detection.pt",
        segformer_path: str = "models/tcd-segformer-mit-b2",
        qwen_path: str = "models/qwen2.5-vl-3b",
        device: Optional[str] = None,
        conf_threshold: float = 0.20,
        load_yolo: bool = True,
        load_segformer: bool = True,
        load_qwen: bool = True,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.segformer_path = segformer_path
        self.qwen_path = qwen_path
        self.conf_threshold = conf_threshold
        if device is None:
            try:
                import torch
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device

        self._model = None
        self._segformer_model = None
        self._segformer_processor = None
        self._qwen_model = None
        self._qwen_processor = None

        if load_yolo:
            self._init_model()
        if load_segformer:
            self._init_segformer()
        if load_qwen:
            self._init_qwen()

    def _init_model(self) -> None:
        """Initialize the underlying neural vision detector."""
        try:
            from ultralytics import YOLO

            p = Path(self.checkpoint_path)
            if not p.exists():
                logger.warning(
                    "Checkpoint %s not found. Falling back to default YOLO.",
                    self.checkpoint_path,
                )
                self.checkpoint_path = "yolov8s-worldv2.pt" if Path("yolov8s-worldv2.pt").exists() else "yolov8n.pt"

            self._model = YOLO(self.checkpoint_path)
            logger.info("EcologicalVisionEngine detector initialized with %s on %s", self.checkpoint_path, self.device)
        except Exception as exc:
            logger.error("Failed to initialize vision model: %s", exc)
            self._model = None

    def _init_segformer(self) -> None:
        """Initialize SOTA SegFormer tree canopy delineation model from Hugging Face / local."""
        try:
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
            p = Path(self.segformer_path)
            model_id = str(p) if p.exists() else "restor/tcd-segformer-mit-b2"
            self._segformer_processor = AutoImageProcessor.from_pretrained(model_id)
            self._segformer_model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(self.device)
            self._segformer_model.eval()
            logger.info("SegFormer tree canopy delineation loaded on %s from %s", self.device, model_id)
        except Exception as exc:
            logger.warning("Could not initialize SegFormer model: %s", exc)
            self._segformer_model = None
            self._segformer_processor = None

    def _init_qwen(self) -> None:
        """Initialize Qwen2.5-VL-3B multimodal vision reasoning model."""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            import torch
            p = Path(self.qwen_path)
            if p.exists() and (p / "model-00001-of-00002.safetensors").exists():
                self._qwen_processor = AutoProcessor.from_pretrained(str(p))
                self._qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    str(p),
                    dtype=torch.bfloat16,
                    device_map="auto",
                )
                self._qwen_model.eval()
                logger.info("Qwen2.5-VL-3B multimodal vision reasoning loaded on %s", self.device)
            else:
                self._qwen_model = None
                self._qwen_processor = None
        except Exception as exc:
            logger.warning("Could not initialize Qwen2.5-VL model: %s", exc)
            self._qwen_model = None
            self._qwen_processor = None

    def _analyze_scene_with_qwen(self, pil_img: Image.Image) -> Dict[str, Any]:
        """Use Qwen2.5-VL to perform macro-landscape ecological reasoning."""
        if self._qwen_model is None or self._qwen_processor is None:
            return {"landscape": "unknown", "has_mangroves": False, "has_fields": True, "has_trees": True}

        try:
            import json, re, torch
            prompt = (
                'Respond in compact JSON only: {"landscape": "urban"|"agricultural"|"forest"|"coastal_wetland", '
                '"has_mangroves": true|false, "has_fields": true|false, "has_trees": true|false}'
            )
            messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt}]}]
            text = self._qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._qwen_processor(text=[text], images=[pil_img], padding=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self._qwen_model.generate(**inputs, max_new_tokens=64)
                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                raw_out = self._qwen_processor.batch_decode(trimmed, skip_special_tokens=True)[0]

            match = re.search(r"\{.*\}", raw_out, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return {
                    "landscape": str(data.get("landscape", "unknown")).lower(),
                    "has_mangroves": bool(data.get("has_mangroves", False)),
                    "has_fields": bool(data.get("has_fields", False)),
                    "has_trees": bool(data.get("has_trees", True)),
                }
        except Exception as exc:
            logger.warning("Qwen2.5-VL scene analysis error: %s", exc)

        return {"landscape": "unknown", "has_mangroves": False, "has_fields": True, "has_trees": True}

    def reason_and_annotate(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        image_id: str = "",
    ) -> Tuple[List[PerImageAnnotationItem], Dict[str, Any]]:
        """Run deep vision reasoning and annotate all vegetation features.

        Returns:
            (annotations_list, image_level_metrics)
        """
        # Load / normalize PIL image
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        img_w, img_h = pil_img.size
        img_area = float(max(1, img_w * img_h))
        np_img = np.array(pil_img)

        # 1. Run Qwen2.5-VL Macro-Landscape Ecological Reasoning
        scene_meta = self._analyze_scene_with_qwen(pil_img)
        has_mangroves = scene_meta.get("has_mangroves", False)
        has_fields = scene_meta.get("has_fields", False) or scene_meta.get("landscape") == "agricultural"

        # 2. Run SOTA SegFormer tree canopy delineation
        seg_mask = None
        if self._segformer_model is not None and self._segformer_processor is not None:
            try:
                import torch
                inputs = self._segformer_processor(images=pil_img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self._segformer_model(**inputs)
                    upsampled = torch.nn.functional.interpolate(
                        outputs.logits,
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    seg_mask = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            except Exception as exc:
                logger.warning("SegFormer inference error: %s", exc)
                seg_mask = None

        # 3. Run neural detection for canopy crown / vegetation proposals
        raw_boxes = []
        if self._model is not None:
            try:
                results = self._model.predict(
                    pil_img,
                    conf=self.conf_threshold,
                    device=self.device,
                    verbose=False,
                )
                if results and len(results) > 0 and results[0].boxes is not None:
                    for b in results[0].boxes:
                        xyxy = [float(v) for v in b.xyxy[0].tolist()]
                        conf = float(b.conf[0].item()) if hasattr(b, "conf") and b.conf is not None else 0.85
                        raw_boxes.append((xyxy, conf))
            except Exception as exc:
                logger.warning("Neural detection error: %s", exc)

        # 4. If no detector boxes but SegFormer found canopies, extract SegFormer proposals
        if not raw_boxes and seg_mask is not None and cv2 is not None:
            seg_cnts, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in seg_cnts:
                if cv2.contourArea(cnt) >= 25:
                    x, y, w, h = cv2.boundingRect(cnt)
                    raw_boxes.append(([float(x), float(y), float(x + w), float(y + h)], 0.88))

        # Fallback if no model or zero detections: grid-spectral scan
        if not raw_boxes and cv2 is not None:
            raw_boxes = self._spectral_proposal_scan(np_img, img_w, img_h)

        annotations: List[PerImageAnnotationItem] = []
        total_physical_area = 0.0
        total_carbon_weight = 0.0
        class_stats: Dict[str, Dict[str, float]] = {}

        for box_xyxy, conf in raw_boxes:
            x1, y1, x2, y2 = [int(v) for v in box_xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw < 4 or bh < 4:
                continue

            crop = np_img[y1:y2, x1:x2]
            # 5. Extract Bio-Spectral & Morphological Reasoning with Qwen landscape context
            eco_class, reasoning_conf = self._classify_crop(crop, bw, bh, img_w, img_h, has_mangroves=has_mangroves)
            
            # Combine detector confidence and ecological classification confidence
            final_conf = round(min(1.0, conf * 0.5 + reasoning_conf * 0.5), 4)

            # 5. Extract Flexible Contour Polygon / OBB (SegFormer mask-guided if available)
            poly = None
            poly_area = 0.0
            if seg_mask is not None and cv2 is not None:
                crop_mask = seg_mask[y1:y2, x1:x2]
                if np.any(crop_mask):
                    cnts, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        best_cnt = max(cnts, key=cv2.contourArea)
                        c_area = float(cv2.contourArea(best_cnt))
                        if c_area >= 10:
                            epsilon = 0.015 * cv2.arcLength(best_cnt, True)
                            approx = cv2.approxPolyDP(best_cnt, max(1.0, epsilon), True)
                            poly = [[round(float(pt[0][0] + x1), 2), round(float(pt[0][1] + y1), 2)] for pt in approx]
                            poly_area = c_area

            if poly is None or poly_area <= 0:
                poly, poly_area = self._extract_polygon(crop, x1, y1, x2, y2)
            if poly_area <= 0:
                poly_area = float(bw * bh)

            # 4. Compute Ecological Carbon Weight
            area_ratio = poly_area / img_area
            multiplier = CARBON_WEIGHT_MULTIPLIERS[eco_class]
            item_weight = round(area_ratio * multiplier, 6)

            total_physical_area += poly_area
            total_carbon_weight += item_weight

            if eco_class not in class_stats:
                class_stats[eco_class] = {"count": 0, "area": 0.0, "weight": 0.0}
            class_stats[eco_class]["count"] += 1
            class_stats[eco_class]["area"] += poly_area
            class_stats[eco_class]["weight"] += item_weight

            annotations.append(
                PerImageAnnotationItem(
                    hazard_class=eco_class,
                    bounding_box=[round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)],
                    polygon=poly,
                    area=round(poly_area, 2),
                    weight=item_weight,
                    confidence=final_conf,
                )
            )

        # 6. Extract Agricultural Field Parcels (cropland, agroforestry, cultivated plots)
        if cv2 is not None:
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            r = np_img[:, :, 0].astype(np.float32)
            g = np_img[:, :, 1].astype(np.float32)
            b = np_img[:, :, 2].astype(np.float32)
            exg = 2.0 * g - r - b

            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            local_var = cv2.absdiff(gray, blur)
            field_candidate = (exg > 10.0) & (local_var < 15) & (gray > 40) & (gray < 210)
            if seg_mask is not None:
                field_candidate = field_candidate & (seg_mask == 0)

            field_mask = field_candidate.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel)
            field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)

            field_cnts, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in field_cnts:
                c_area = float(cv2.contourArea(c))
                if 1500 <= c_area <= 350000:
                    x, y, bw, bh = cv2.boundingRect(c)
                    epsilon = 0.02 * cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, max(2.0, epsilon), True)
                    poly = [[round(float(pt[0][0]), 2), round(float(pt[0][1]), 2)] for pt in approx]

                    area_ratio = c_area / img_area
                    multiplier = CARBON_WEIGHT_MULTIPLIERS["field"]
                    item_weight = round(area_ratio * multiplier, 6)

                    total_physical_area += c_area
                    total_carbon_weight += item_weight

                    if "field" not in class_stats:
                        class_stats["field"] = {"count": 0, "area": 0.0, "weight": 0.0}
                    class_stats["field"]["count"] += 1
                    class_stats["field"]["area"] += c_area
                    class_stats["field"]["weight"] += item_weight

                    annotations.append(
                        PerImageAnnotationItem(
                            hazard_class="field",
                            bounding_box=[round(float(x), 2), round(float(y), 2), round(float(x + bw), 2), round(float(y + bh), 2)],
                            polygon=poly,
                            area=round(c_area, 2),
                            weight=item_weight,
                            confidence=0.86,
                        )
                    )

        # 7. Whole-Image Net Metrics
        tree_coverage_ratio = min(1.0, max(0.0, total_physical_area / img_area))
        tree_coverage_percentage = round(tree_coverage_ratio * 100.0, 2)
        net_weight = round(total_carbon_weight, 6)
        tree_count = len(annotations)

        # Dominant class (highest carbon weight contribution)
        dominant_class = "ordinary_tree"
        if class_stats:
            dominant_class = max(class_stats.keys(), key=lambda c: class_stats[c]["weight"])

        metrics = {
            "image_id": image_id,
            "image_name": canonical_image_name(image_id),
            "net_weight": net_weight,
            "tree_coverage_percentage": tree_coverage_percentage,
            "tree_coverage_ratio": round(tree_coverage_ratio, 6),
            "tree_count": tree_count,
            "dominant_class": dominant_class,
            "class_breakdown": {
                c: {
                    "count": int(d["count"]),
                    "area_ratio": round(d["area"] / img_area, 6),
                    "carbon_weight": round(d["weight"], 6),
                }
                for c, d in class_stats.items()
            },
        }

        return annotations, metrics

    def _classify_crop(
        self,
        crop: np.ndarray,
        bw: int,
        bh: int,
        img_w: int,
        img_h: int,
        has_mangroves: bool = False,
    ) -> Tuple[str, float]:
        """Perform bio-spectral and structural reasoning on a detected crop."""
        if crop.ndim != 3 or crop.shape[2] < 3:
            return "ordinary_tree", 0.70

        r = crop[:, :, 0].astype(np.float32)
        g = crop[:, :, 1].astype(np.float32)
        b = crop[:, :, 2].astype(np.float32)

        # Spectral Indices
        exg = 2.0 * g - r - b  # Excess Green Index (Vegetation vigor)
        mean_exg = float(np.mean(exg))
        
        # Moisture / Blue-Green reflectance (identifies tidal/wetland/saline hydrology)
        moisture = float(np.mean(g - b))
        
        # VARI (Visible Atmospherically Resistant Index)
        denom = g + r - b + 1e-5
        vari = float(np.mean((g - r) / denom))

        area = float(bw * bh)
        aspect = bw / max(1.0, float(bh))
        elongation = max(aspect, 1.0 / max(0.01, aspect))

        # Texture variance (crown complexity)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        texture_var = float(np.var(gray))

        # 1. Mangrove: STRICTLY restricted to verified coastal wetland environments!
        if has_mangroves and moisture > 12.0 and mean_exg > 20.0 and area > 120 and texture_var > 150:
            return "mangrove", 0.92

        # 2. Dense Tree / Intact Forest: Large area, high green vigor, high crown texture
        if area > 1100 or (area > 450 and mean_exg > 25.0 and texture_var > 200):
            return "dense_tree", 0.88

        # 3. Field / Cropland / Agroforestry: Regular geometric plot, high elongation or uniform texture
        if (elongation > 1.9 or texture_var < 100) and area > 100 and mean_exg > 15.0:
            return "field", 0.84

        # 4. Plant / Woody Shrub / Understory: Small area, low spread, moderate greenness
        if area < 75 or (area < 150 and mean_exg < 18.0):
            return "plant", 0.80

        # 5. Ordinary Tree: Standard terrestrial tree crown (honest baseline)
        return "ordinary_tree", 0.85

    def _extract_polygon(
        self,
        crop: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> Tuple[List[List[float]], float]:
        """Extract flexible contour polygon or oriented bounding box."""
        if cv2 is None or crop.ndim != 3 or crop.shape[2] < 3:
            # Fallback to 4-point bounding box polygon
            poly = [
                [round(float(x1), 2), round(float(y1), 2)],
                [round(float(x2), 2), round(float(y1), 2)],
                [round(float(x2), 2), round(float(y2), 2)],
                [round(float(x1), 2), round(float(y2), 2)],
            ]
            area = float(max(0, (x2 - x1) * (y2 - y1)))
            return poly, area

        try:
            r = crop[:, :, 0].astype(np.float32)
            g = crop[:, :, 1].astype(np.float32)
            b = crop[:, :, 2].astype(np.float32)
            exg = 2.0 * g - r - b
            exg_norm = np.clip((exg - exg.min()) / (exg.max() - exg.min() + 1e-5) * 255.0, 0, 255).astype(np.uint8)
            _, thresh = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                c_area = float(cv2.contourArea(cnt))
                if c_area > 8:
                    rect = cv2.minAreaRect(cnt)
                    box_pts = cv2.boxPoints(rect)
                    poly = [[round(float(p[0] + x1), 2), round(float(p[1] + y1), 2)] for p in box_pts]
                    return poly, c_area
        except Exception:
            pass

        # Default 4-point polygon matching bounding box
        poly = [
            [round(float(x1), 2), round(float(y1), 2)],
            [round(float(x2), 2), round(float(y1), 2)],
            [round(float(x2), 2), round(float(y2), 2)],
            [round(float(x1), 2), round(float(y2), 2)],
        ]
        area = float(max(0, (x2 - x1) * (y2 - y1)))
        return poly, area

    def _spectral_proposal_scan(
        self,
        np_img: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> List[Tuple[List[float], float]]:
        """Fallback spectral vegetation detector when no neural model is available."""
        if cv2 is None or np_img.ndim != 3:
            return []

        r = np_img[:, :, 0].astype(np.float32)
        g = np_img[:, :, 1].astype(np.float32)
        b = np_img[:, :, 2].astype(np.float32)
        exg = 2.0 * g - r - b
        exg_norm = np.clip((exg - exg.min()) / (exg.max() - exg.min() + 1e-5) * 255.0, 0, 255).astype(np.uint8)
        _, thresh = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        proposals = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 16:
                x, y, w, h = cv2.boundingRect(cnt)
                proposals.append(([float(x), float(y), float(x + w), float(y + h)], 0.80))
        return proposals
