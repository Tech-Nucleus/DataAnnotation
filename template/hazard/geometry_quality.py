"""First-principles geometric and quality audit for ecological canopy annotations.

Provides mathematically rigorous, objective metrics to detect and penalize:
1. Crude rectangular boxes masquerading as segmentation polygons (box-gaming).
2. Spiky, fractal, saw-tooth, or lightning-bolt boundary artifacts (low precision).
3. Non-vegetative and ocean/water hallucinations (using spectral Excess Green ExG).
4. True raster mask Intersection over Union (IoU) for organic ground-truth parity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


@dataclass(frozen=True)
class AnnotationQualityAudit:
    total_count: int
    box_count: int
    box_ratio: float
    spiky_count: int
    spiky_ratio: float
    water_hallucination_count: int
    mean_compactness: float
    mean_convexity: float
    quality_multiplier: float
    is_acceptable: bool
    rejection_reason: Optional[str] = None


def shoelace_area(poly: Sequence[Sequence[float]]) -> float:
    """Calculate the signed area of a 2D polygon using the Shoelace formula."""
    n = len(poly)
    if n < 3:
        return 0.0
    try:
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += float(poly[i][0]) * float(poly[j][1]) - float(poly[j][0]) * float(poly[i][1])
        return abs(area) / 2.0
    except (TypeError, ValueError, IndexError):
        return 0.0


def polygon_perimeter(poly: Sequence[Sequence[float]]) -> float:
    """Calculate the perimeter length of a closed 2D polygon."""
    n = len(poly)
    if n < 3:
        return 0.0
    try:
        peri = 0.0
        for i in range(n):
            j = (i + 1) % n
            dx = float(poly[j][0]) - float(poly[i][0])
            dy = float(poly[j][1]) - float(poly[i][1])
            peri += math.hypot(dx, dy)
        return peri
    except (TypeError, ValueError, IndexError):
        return 0.0


def is_crude_box(
    poly: Optional[Sequence[Sequence[float]]],
    bbox: Optional[Sequence[float]] = None,
    box_area_threshold: float = 0.88,
) -> bool:
    """Determine if a polygon is a crude rectangular box masquerading as a contour.

    A polygon is classified as a crude box if:
    1. It is missing or degenerate (< 3 vertices).
    2. It has 4 or 5 vertices and its area occupies >= 85% of its bounding box.
    3. It has up to 8 vertices and its area occupies >= 90% of its bounding box.
    4. All points share <= 2 distinct X coordinates and <= 2 distinct Y coordinates.
    """
    if not poly or len(poly) < 3:
        return True  # Missing or degenerate polygon defaults to box-like

    cleaned = [poly[0]]
    for p in poly[1:]:
        if p[0] != cleaned[-1][0] or p[1] != cleaned[-1][1]:
            cleaned.append(p)
    if len(cleaned) > 2 and cleaned[0][0] == cleaned[-1][0] and cleaned[0][1] == cleaned[-1][1]:
        cleaned.pop()

    n = len(cleaned)
    if n < 3:
        return True

    poly_area = shoelace_area(cleaned)
    if poly_area <= 1.0:
        return True

    # Check distinct X and Y coordinates (e.g. axis-aligned boxes with extra vertices)
    xs = {round(float(p[0]), 1) for p in cleaned}
    ys = {round(float(p[1]), 1) for p in cleaned}
    if len(xs) <= 2 and len(ys) <= 2:
        return True

    # Check axis-aligned bounding box of the points themselves
    min_x = min(float(p[0]) for p in cleaned)
    max_x = max(float(p[0]) for p in cleaned)
    min_y = min(float(p[1]) for p in cleaned)
    max_y = max(float(p[1]) for p in cleaned)
    inferred_box_area = max(1.0, (max_x - min_x) * (max_y - min_y))

    if bbox is not None and len(bbox) == 4:
        bx1, by1, bx2, by2 = [float(v) for v in bbox]
        box_w = max(1.0, bx2 - bx1)
        box_h = max(1.0, by2 - by1)
        b_area = box_w * box_h
    else:
        b_area = inferred_box_area

    area_ratio = poly_area / b_area
    inferred_ratio = poly_area / inferred_box_area

    # Literal 4-corner rectangle
    if n == 4 and (area_ratio >= 0.85 or inferred_ratio >= 0.85):
        return True

    # 5-corner rectangle (e.g. 1 clipped corner or 1 midpoint)
    if n == 5 and (area_ratio >= 0.85 or inferred_ratio >= 0.85):
        return True

    # 6 to 8 corner near-rectangle (axis-aligned contour gaming)
    if n <= 8 and (area_ratio >= 0.90 or inferred_ratio >= 0.90):
        return True

    return False



def isoperimetric_quotient(poly: Sequence[Sequence[float]]) -> float:
    """Compute the Isoperimetric Quotient (Compactness / Circularity) Q in [0, 1].

    Q = (4 * pi * Area) / (Perimeter^2)
    - Circle: Q = 1.0
    - Square: Q = pi / 4 ~= 0.785
    - Organic tree crown / stand: Q in [0.20, 0.85]
    - Spiky, starburst, or lightning-bolt contour: Q < 0.15
    """
    area = shoelace_area(poly)
    if area <= 1.0:
        return 0.0
    peri = polygon_perimeter(poly)
    if peri <= 0.0:
        return 0.0
    q = (4.0 * math.pi * area) / (peri * peri)
    return float(max(0.0, min(1.0, q)))


def convexity_ratio(poly: Sequence[Sequence[float]]) -> float:
    """Compute the ratio of polygon area to its convex hull area in [0, 1].

    - Natural tree crowns and stands: C in [0.50, 0.95]
    - Erratic jagged, starburst, or lightning-bolt shapes: C < 0.35
    """
    area = shoelace_area(poly)
    if area <= 1.0:
        return 0.0

    if _CV2_AVAILABLE:
        try:
            pts = np.array(poly, dtype=np.float32)
            hull = cv2.convexHull(pts)
            hull_area = float(cv2.contourArea(hull))
            if hull_area <= 1.0:
                return 1.0
            return float(max(0.0, min(1.0, area / hull_area)))
        except Exception:
            pass

    # Fallback Monotone Chain Convex Hull
    points = sorted({(float(p[0]), float(p[1])) for p in poly})
    if len(points) < 3:
        return 1.0

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull_pts = lower[:-1] + upper[:-1]
    hull_area = shoelace_area(hull_pts)
    if hull_area <= 1.0:
        return 1.0
    return float(max(0.0, min(1.0, area / hull_area)))


def is_spiky_polygon(poly: Sequence[Sequence[float]]) -> bool:
    """Identify if a polygon exhibits unphysical, erratic, or lightning-bolt spikes."""
    if not poly or len(poly) < 6:
        return False  # Small polygons are checked by is_crude_box

    q = isoperimetric_quotient(poly)
    c = convexity_ratio(poly)

    # Extreme boundary perimeter collapse with low compactness
    if q < 0.15:
        return True
    if c < 0.35 and q < 0.30:
        return True
    return False


def verify_crop_vegetation(
    crop_rgb: np.ndarray,
    poly_pts_relative: Optional[Sequence[Sequence[float]]] = None,
) -> bool:
    """Verify living vegetation presence using chlorophyll absorption physics.

    In living vegetation:
    1. Chlorophyll strongly absorbs blue light (430-450nm) and red light (640-660nm)
       while reflecting green (550nm). Thus G >> B and G > R.
    2. Water, ocean reefs, and breaking surf reflect blue/cyan light (G ≈ B).
    3. Asphalt, roads, and concrete have neutral gray/brown reflectance (R ≈ G ≈ B).

    Returns False if the region is ocean water, breaking surf, asphalt road, or bare sand.
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return True
    if crop_rgb.ndim != 3 or crop_rgb.shape[2] < 3:
        return True

    h, w = crop_rgb.shape[:2]
    # If polygon is supplied and cv2 is available, mask to polygon interior
    if poly_pts_relative and _CV2_AVAILABLE:
        try:
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(poly_pts_relative, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
            valid = mask == 1
            if np.sum(valid) > 50:
                r = crop_rgb[:, :, 0][valid].astype(np.float32)
                g = crop_rgb[:, :, 1][valid].astype(np.float32)
                b = crop_rgb[:, :, 2][valid].astype(np.float32)
            else:
                r = crop_rgb[:, :, 0].astype(np.float32).ravel()
                g = crop_rgb[:, :, 1].astype(np.float32).ravel()
                b = crop_rgb[:, :, 2].astype(np.float32).ravel()
        except Exception:
            r = crop_rgb[:, :, 0].astype(np.float32).ravel()
            g = crop_rgb[:, :, 1].astype(np.float32).ravel()
            b = crop_rgb[:, :, 2].astype(np.float32).ravel()
    else:
        r = crop_rgb[:, :, 0].astype(np.float32).ravel()
        g = crop_rgb[:, :, 1].astype(np.float32).ravel()
        b = crop_rgb[:, :, 2].astype(np.float32).ravel()

    # Excess Green: ExG = 2G - R - B
    exg = 2.0 * g - r - b
    mean_exg = float(np.mean(exg))

    # Optical physics of water vs terrestrial vegetation/soil:
    # Water strongly absorbs red wavelengths (640-700nm), leaving Blue dominating Red: B >> R.
    # Terrestrial surfaces (soil, sand, dry grass, trees) have R >= B.
    # Ocean water also has negative Excess Green (ExG = 2G - R - B < 0).
    mean_b_minus_r = float(np.mean(b - r))
    if mean_b_minus_r > 20.0 and mean_exg < 0.0:
        return False

    # Deep marine water or breaking reef surf: B strongly dominates both G and R
    mean_b = float(np.mean(b))
    mean_r = float(np.mean(r))
    if mean_b > mean_r + 30.0:
        return False

    return True


def audit_annotation_geometry(
    annotations: Sequence[Any],
    image_np: Optional[np.ndarray] = None,
) -> AnnotationQualityAudit:
    """Comprehensive, first-principles geometric audit of an image's annotations.

    Evaluates:
    - Box-gaming ratio (4-point rectangular boxes masquerading as polygons)
    - Boundary spikiness ratio (erratic lightning-bolt and saw-tooth artifacts)
    - Ocean water / non-vegetation hallucinations
    """
    total = len(annotations)
    if total == 0:
        return AnnotationQualityAudit(
            total_count=0,
            box_count=0,
            box_ratio=0.0,
            spiky_count=0,
            spiky_ratio=0.0,
            water_hallucination_count=0,
            mean_compactness=1.0,
            mean_convexity=1.0,
            quality_multiplier=1.0,
            is_acceptable=True,
            rejection_reason=None,
        )

    box_count = 0
    spiky_count = 0
    water_count = 0
    q_list = []
    c_list = []

    img_h, img_w = (image_np.shape[:2]) if image_np is not None else (2048, 2048)

    for item in annotations:
        if isinstance(item, dict):
            poly = item.get("polygon") or item.get("fused_polygon")
            bbox = item.get("bounding_box") or item.get("bbox") or item.get("fused_bounding_box")
            area = item.get("area")
        else:
            poly = getattr(item, "polygon", None) or getattr(item, "fused_polygon", None)
            bbox = getattr(item, "bounding_box", None) or getattr(item, "fused_bounding_box", None)
            area = getattr(item, "area", None)

        if is_crude_box(poly, bbox):
            box_count += 1
        elif poly and len(poly) >= 3:
            q = isoperimetric_quotient(poly)
            c = convexity_ratio(poly)
            q_list.append(q)
            c_list.append(c)

            if is_spiky_polygon(poly):
                spiky_count += 1

        # Check ocean / water hallucination on large objects
        if image_np is not None and bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            ix1, iy1 = max(0, x1), max(0, y1)
            ix2, iy2 = min(img_w, x2), min(img_h, y2)
            box_area = (ix2 - ix1) * (iy2 - iy1)
            # Only test large stands (> 5,000 px or > 0.1% of 2048x2048)
            if box_area > 5000 and ix2 > ix1 and iy2 > iy1:
                crop = image_np[iy1:iy2, ix1:ix2]
                rel_poly = (
                    [[p[0] - ix1, p[1] - iy1] for p in poly]
                    if poly else None
                )
                if not verify_crop_vegetation(crop, rel_poly):
                    water_count += 1

    box_ratio = float(box_count / total)
    spiky_ratio = float(spiky_count / total)
    mean_q = float(np.mean(q_list)) if q_list else 0.70
    mean_c = float(np.mean(c_list)) if c_list else 0.85

    rejection_reasons = []

    # 1. Box penalty: Exponentially penalize based on box frequency
    if box_count > 0:
        if box_ratio > 0.08 or (total <= 10 and box_count >= 2) or box_count >= 5:
            box_multiplier = 0.0
            rejection_reasons.append(f"box_gaming_detected: excessive_box_polygons ({box_count}/{total} boxes, {box_ratio*100:.1f}%)")
        else:
            box_multiplier = max(0.0, float((1.0 - 5.0 * box_ratio) ** 2))
    else:
        box_multiplier = 1.0

    # 2. Spiky penalty: Reject saw-tooth, starburst, or lightning-bolt boundaries
    if spiky_count > 0:
        if spiky_ratio > 0.08 or (total <= 10 and spiky_count >= 2) or spiky_count >= 5:
            spiky_multiplier = 0.0
            rejection_reasons.append(f"spiky_erratic_polygons ({spiky_count}/{total}, {spiky_ratio*100:.1f}%)")
        else:
            spiky_multiplier = max(0.0, float((1.0 - 5.0 * spiky_ratio) ** 2))
    else:
        spiky_multiplier = 1.0

    # 3. Ocean / water false positive penalty
    if water_count > 0:
        water_multiplier = 0.0
        rejection_reasons.append(f"ocean_water_hallucinations ({water_count} stands)")
    else:
        water_multiplier = 1.0

    quality_multiplier = float(round(box_multiplier * spiky_multiplier * water_multiplier, 4))
    is_acceptable = quality_multiplier >= 0.70 and not rejection_reasons

    return AnnotationQualityAudit(
        total_count=total,
        box_count=box_count,
        box_ratio=round(box_ratio, 4),
        spiky_count=spiky_count,
        spiky_ratio=round(spiky_ratio, 4),
        water_hallucination_count=water_count,
        mean_compactness=round(mean_q, 4),
        mean_convexity=round(mean_c, 4),
        quality_multiplier=quality_multiplier,
        is_acceptable=is_acceptable,
        rejection_reason="; ".join(rejection_reasons) if rejection_reasons else None,
    )


def raster_polygon_iou(
    poly1: Sequence[Sequence[float]],
    poly2: Sequence[Sequence[float]],
    bbox1: Sequence[float],
    bbox2: Sequence[float],
) -> float:
    """Compute high-speed true raster mask IoU between two polygons.

    If cv2 is available, rasterizes onto the bounding union subgrid.
    Accurately penalizes box-vs-polygon geometric mismatch.
    """
    if not _CV2_AVAILABLE:
        # Fallback to bounding box IoU if cv2 is not available
        x1 = max(float(bbox1[0]), float(bbox2[0]))
        y1 = max(float(bbox1[1]), float(bbox2[1]))
        x2 = min(float(bbox1[2]), float(bbox2[2]))
        y2 = min(float(bbox1[3]), float(bbox2[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = max(0.0, float(bbox1[2] - bbox1[0])) * max(0.0, float(bbox1[3] - bbox1[1]))
        a2 = max(0.0, float(bbox2[2] - bbox2[0])) * max(0.0, float(bbox2[3] - bbox2[1]))
        union = a1 + a2 - inter
        return float(inter / union if union > 0.0 else 0.0)

    try:
        ux1 = min(float(bbox1[0]), float(bbox2[0]))
        uy1 = min(float(bbox1[1]), float(bbox2[1]))
        ux2 = max(float(bbox1[2]), float(bbox2[2]))
        uy2 = max(float(bbox1[3]), float(bbox2[3]))

        uw = int(math.ceil(ux2 - ux1))
        uh = int(math.ceil(uy2 - uy1))
        if uw <= 0 or uh <= 0:
            return 0.0

        # Scale down large stands to max 256x256 subgrid for microsecond rasterization
        max_dim = max(uw, uh)
        scale = 256.0 / max_dim if max_dim > 256 else 1.0
        rw = max(1, int(round(uw * scale)))
        rh = max(1, int(round(uh * scale)))

        p1_scaled = np.array(
            [[round((p[0] - ux1) * scale), round((p[1] - uy1) * scale)] for p in poly1],
            dtype=np.int32,
        )
        p2_scaled = np.array(
            [[round((p[0] - ux1) * scale), round((p[1] - uy1) * scale)] for p in poly2],
            dtype=np.int32,
        )

        m1 = np.zeros((rh, rw), dtype=np.uint8)
        m2 = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(m1, [p1_scaled], 1)
        cv2.fillPoly(m2, [p2_scaled], 1)

        inter = int(np.logical_and(m1, m2).sum())
        union = int(np.logical_or(m1, m2).sum())
        return float(inter / union if union > 0 else 0.0)
    except Exception:
        return 0.0


def compute_image_fidelity(
    miner_fidelity: float,
    annotations: Sequence[Any],
    audit: Optional[AnnotationQualityAudit] = None,
    image_np: Optional[np.ndarray] = None,
) -> float:
    """Compute an authentic, mathematically grounded per-image fidelity score.

    Combines:
    1. Miner Base Empirical Fidelity (anchor on hidden golden ground-truth benchmark)
    2. Area-weighted Detection & Classification Confidence on this specific image
    3. Geometric Morphology Fidelity (organic crown compactness, convexity, contour smoothness)
    4. Optical / Spectral Vegetative Concordance (Excess Green in polygon pixels)
    5. Penalty multipliers (box gaming, spikiness, water hallucinations)
    """
    total = len(annotations)
    if total == 0:
        return 0.0

    if audit is None:
        audit = audit_annotation_geometry(annotations, image_np)

    # 1. Detection Confidence
    confs: list[float] = []
    areas: list[float] = []
    for item in annotations:
        if isinstance(item, dict):
            c = float(item.get("confidence", 0.95))
            a = float(item.get("area", 100.0))
        else:
            c = float(getattr(item, "confidence", None) or 0.95)
            a = float(getattr(item, "area", None) or 100.0)
        confs.append(max(0.0, min(1.0, c)))
        areas.append(max(1.0, a))

    total_area = sum(areas)
    if total_area > 0 and len(confs) == len(areas):
        mean_conf = sum(c * a for c, a in zip(confs, areas)) / total_area
    else:
        mean_conf = sum(confs) / max(1, len(confs)) if confs else 0.5

    # 2. Geometric Morphology
    mean_q = audit.mean_compactness
    mean_c = audit.mean_convexity
    geom_q = 0.45 * mean_q + 0.55 * mean_c

    # 3. Spectral Vegetative Concordance
    spectral = 0.90
    if image_np is not None and _CV2_AVAILABLE:
        try:
            mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            sampled = 0
            for item in annotations[:30]:
                if isinstance(item, dict):
                    poly = item.get("polygon") or item.get("fused_polygon")
                else:
                    poly = getattr(item, "polygon", None) or getattr(item, "fused_polygon", None)
                if poly and len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                    sampled += 1
            if np.sum(mask) > 100:
                crop_pixels = image_np[mask == 1]
                ch0 = crop_pixels[:, 0].astype(np.float32)
                ch1 = crop_pixels[:, 1].astype(np.float32)
                ch2 = crop_pixels[:, 2].astype(np.float32)
                # Channel invariant check: green component must exceed min of others
                exg = 2.0 * ch1 - ch0 - ch2
                spectral = float(np.clip(0.5 + 0.5 * np.mean(exg > -5.0), 0.0, 1.0))
        except Exception:
            spectral = 0.90

    base_miner = max(0.0, min(1.0, float(miner_fidelity)))
    composite = (
        0.35 * base_miner
        + 0.25 * mean_conf
        + 0.25 * geom_q
        + 0.15 * spectral
    )
    final_score = composite * audit.quality_multiplier
    return round(float(max(0.0, min(1.0, final_score))), 4)

