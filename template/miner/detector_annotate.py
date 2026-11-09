from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from template.protocol import ImageAnnotationDocument, PerImageAnnotationItem


def annotate_image_detector_only(
    *,
    checkpoint: Path,
    image_bytes: bytes,
    image_id: str,
    image_url: str,
    model_version: str,
    miner_uid: str,
    conf_thres: Optional[float] = None,
    iou_thres: Optional[float] = None,
    imgsz: Optional[int] = None,
    epsilon_factor: Optional[float] = None,
    max_area_ratio: Optional[float] = None,
    force_coarse_box: Optional[bool] = None,
) -> ImageAnnotationDocument:
    """Run YOLO-only detector on the provided image and return annotations."""
    try:
        from PIL import Image
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "pillow and ultralytics are required to run YOLO annotations."
        ) from exc

    from template.miner.agentic_controller import load_miner_params
    params = load_miner_params()

    eff_conf = conf_thres if conf_thres is not None else float(params.get("conf_thres", 0.25))
    eff_iou = iou_thres if iou_thres is not None else float(params.get("iou_thres", 0.45))
    eff_imgsz = imgsz if imgsz is not None else int(params.get("imgsz", 640))
    eff_eps = epsilon_factor if epsilon_factor is not None else float(params.get("epsilon_factor", 0.008))
    eff_max_area = max_area_ratio if max_area_ratio is not None else float(params.get("max_area_ratio", 0.60))
    eff_coarse = force_coarse_box if force_coarse_box is not None else bool(params.get("force_coarse_box", False))

    # Load YOLO model
    model = YOLO(str(checkpoint))

    # Read image
    img = Image.open(io.BytesIO(image_bytes))

    # Run Tier 2 SAHI 2x2 Overlapping Sliding-Window Tile Inference + NMS Merge
    annotations: List[PerImageAnnotationItem] = []
    from template.miner.geometry import extract_canopy_geometry
    from template.hazard.annotation_eval import iou_xyxy

    img_w, img_h = img.size if hasattr(img, "size") else (1024, 1024)
    raw_candidates: List[PerImageAnnotationItem] = []

    # If high-res aerial/satellite chip (>= 1024px), run 2x2 tiles with 15% overlap
    if img_w >= 1024 and img_h >= 1024:
        tw = int(img_w * 0.575)
        th = int(img_h * 0.575)
        tiles = [
            (0, 0, tw, th),
            (img_w - tw, 0, img_w, th),
            (0, img_h - th, tw, img_h),
            (img_w - tw, img_h - th, img_w, img_h),
        ]
        for tx0, ty0, tx1, ty1 in tiles:
            tile_crop = img.crop((tx0, ty0, tx1, ty1))
            t_res = model(tile_crop, conf=eff_conf, iou=eff_iou, imgsz=eff_imgsz, verbose=False)
            if t_res and len(t_res) > 0 and t_res[0].boxes is not None:
                for b_idx, box in enumerate(t_res[0].boxes):
                    c_xyxy = box.xyxy[0].tolist()
                    g_xyxy = [c_xyxy[0] + tx0, c_xyxy[1] + ty0, c_xyxy[2] + tx0, c_xyxy[3] + ty0]
                    cls_idx = int(box.cls[0].item())
                    cls_name = model.names[cls_idx]
                    conf = float(box.conf[0].item()) if hasattr(box, "conf") and box.conf is not None else 1.0
                    ann_item = extract_canopy_geometry(
                        pil_img=img,
                        box_xyxy=g_xyxy,
                        hazard_class=cls_name,
                        confidence=conf,
                        epsilon_factor=eff_eps,
                        force_coarse_box=eff_coarse,
                    )
                    raw_candidates.append(ann_item)

    # Whole-image pass to capture macroscopic canopies
    full_res = model(img, conf=eff_conf, iou=eff_iou, imgsz=eff_imgsz, verbose=False)
    if full_res and len(full_res) > 0 and full_res[0].boxes is not None:
        for b_idx, box in enumerate(full_res[0].boxes):
            xyxy = box.xyxy[0].tolist()
            cls_idx = int(box.cls[0].item())
            cls_name = model.names[cls_idx]
            conf = float(box.conf[0].item()) if hasattr(box, "conf") and box.conf is not None else 1.0
            mask_xy = None
            if hasattr(full_res[0], "masks") and full_res[0].masks is not None and len(full_res[0].masks) > b_idx:
                mask_xy = full_res[0].masks.xy[b_idx].tolist()
            ann_item = extract_canopy_geometry(
                pil_img=img,
                box_xyxy=xyxy,
                hazard_class=cls_name,
                confidence=conf,
                mask_xy=mask_xy,
                epsilon_factor=eff_eps,
                force_coarse_box=eff_coarse,
            )
            raw_candidates.append(ann_item)

    # Polygon & Bounding Box NMS Merge across all crops
    if raw_candidates:
        sorted_cands = sorted(raw_candidates, key=lambda a: float(getattr(a, "confidence", 1.0) or 1.0), reverse=True)
        kept: List[PerImageAnnotationItem] = []
        for cand in sorted_cands:
            suppressed = False
            for k in kept:
                if iou_xyxy(cand.bounding_box, k.bounding_box) >= 0.45:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(cand)
        annotations = kept

    from template.miner.geometry import compute_image_net_metrics, canonical_image_name
    img_w, img_h = img.size if hasattr(img, "size") else (1024, 1024)
    total_area = float(img_w * img_h)

    # Filter oversized boxes, clamp coordinates, and skip degenerate boxes
    clean_annotations = []
    for a in annotations:
        if a.bounding_box and len(a.bounding_box) == 4:
            x1, y1, x2, y2 = a.bounding_box
            x1 = max(0.0, min(float(img_w), float(x1)))
            y1 = max(0.0, min(float(img_h), float(y1)))
            x2 = max(0.0, min(float(img_w), float(x2)))
            y2 = max(0.0, min(float(img_h), float(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            box_area = (x2 - x1) * (y2 - y1)
            if box_area / max(1.0, total_area) >= eff_max_area:
                continue
            a = a.model_copy(update={"bounding_box": [x1, y1, x2, y2]})
        clean_annotations.append(a)

    if len(clean_annotations) > 512:
        clean_annotations = sorted(
            clean_annotations,
            key=lambda a: float(getattr(a, "confidence", 1.0) if getattr(a, "confidence", 1.0) is not None else 1.0),
            reverse=True,
        )[:512]
    annotations = clean_annotations

    net_weight, coverage_pct, tree_count = compute_image_net_metrics(annotations, img_w, img_h)
    canonical_name = canonical_image_name(image_id, image_url)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ImageAnnotationDocument(
        image_id=image_id,
        image_url=image_url,
        miner_uid=miner_uid,
        timestamp=ts,
        annotations=annotations,
        model_version=model_version,
        image_name=canonical_name,
        net_weight=net_weight,
        tree_coverage_ratio=net_weight,
        tree_coverage_percentage=coverage_pct,
        tree_count=tree_count,
    )

