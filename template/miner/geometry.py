"""
Geometry and Net Weight utilities for flexible tree annotations and Carbon MRV.
Supports:
- Flexible polygons and Oriented Bounding Boxes (OBB)
- Ecological carbon weight multipliers (mangrove: 3.5x, dense_tree: 1.8x, ordinary_tree: 1.0x, farm: 0.7x, plant: 0.4x)
- Per-object pixel area and coverage weight
- Whole-image net_weight (sum of box carbon weights), tree_coverage_percentage, tree_count
- Canonical image filename normalization
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
from template.protocol import PerImageAnnotationItem

#: Ecological carbon credit weight multipliers based on CO2 sequestration capacity
CARBON_WEIGHT_MULTIPLIERS: Dict[str, float] = {
    "mangrove": 3.5,            # Ultra-high blue carbon sequestration (coastal wetland)
    "dense_tree": 1.8,          # Mature dense forest canopy / high biomass density
    "boreal_conifer": 1.8,      # Boreal taiga conifer (Scots Pine, Norway Spruce)
    "tropical_broadleaf": 1.8,  # Tropical rainforest canopy / emergent hardwood
    "plantation": 1.2,          # Managed tree plantation (eucalyptus, palm, orchard)
    "ordinary_tree": 1.0,       # Baseline standard terrestrial tree crown
    "field": 0.7,               # Agricultural cropland, agroforestry, managed fields
    "plant": 0.4,               # Woody shrubs, understory perennial plants (not grass)
}


def canonical_carbon_class(raw_class: str) -> str:
    """Map known annotation labels to a carbon weight class without substring guesses."""
    c = canonical_annotation_class(raw_class)
    if c in {"mangrove", "wetland"}:
        return "mangrove"
    if c in {"plantation", "eucalyptus", "oil_palm", "rubber", "orchard"}:
        return "plantation"
    if c in {"field", "farm", "agriculture", "crop", "cropland", "parcel"}:
        return "field"
    if c in {"conifer", "pine", "spruce", "taiga", "boreal", "fir", "larch"}:
        return "dense_tree"
    if c in {
        "broadleaf", "tropical", "rainforest", "emergent", "hardwood",
        "dense_tree", "group_of_trees", "intact_forest", "degraded_forest",
    }:
        return "dense_tree"
    if c in {"plant", "shrub", "regrowth", "understory", "brush"}:
        return "plant"
    if c in {"individual_tree", "ordinary_tree", "tree", "crown", "deciduous"}:
        return "ordinary_tree"
    # Unknown labels get only the neutral carbon multiplier. Class scoring uses
    # canonical_annotation_class and therefore never aliases unknowns to trees.
    return "ordinary_tree"


_CLASS_ALIASES = {
    "background": "_background",
    # Carbon MRV/tree aliases.
    "tree": "ordinary_tree",
    "individual tree": "ordinary_tree",
    "single tree": "ordinary_tree",
    "ordinary tree": "ordinary_tree",
    "group of trees": "group_of_trees",
    "tree group": "group_of_trees",
    "dense tree": "dense_tree",
    "mangrove tree": "mangrove",
    "mangrove": "mangrove",
    "mangrove (coastal dense)": "mangrove",
    "mangrove coastal dense": "mangrove",
    "dry forest tree": "ordinary_tree",
    "deciduous broadleaf": "ordinary_tree",
    "mature mixed hardwood": "dense_tree",
    "dense taiga canopy": "dense_tree",
    "scots pine (boreal conifer)": "dense_tree",
    "scots pine boreal conifer": "dense_tree",
    "scots pine": "dense_tree",
    "boreal conifer": "dense_tree",
    "tropical emergent canopy": "dense_tree",
    "tropical broadleaf (dense)": "dense_tree",
    "tropical broadleaf dense": "dense_tree",
    "tropical forest tree": "dense_tree",
    "understory shrub": "plant",
    "restor forestry verified": "ordinary_tree",
    "forest": "intact_forest",
    "intact forest": "intact_forest",
    "degraded forest": "degraded_forest",
    "fire scar": "fire_scar",
    "bare land": "bare_land",
    "crop land": "cropland",
    "oil palm": "oil_palm",
    # Explicit safety-label aliases used by the legacy safety corpus.
    "hard hat": "hardhat",
    "helmet": "hardhat",
    "no hardhat": "missing_hardhat",
    "missing hard hat": "missing_hardhat",
    "no helmet": "missing_hardhat",
    "missing helmet": "missing_hardhat",
    "fall protection": "fall_protection",
    "trip hazard": "trip_hazard",
}


def canonical_annotation_class(raw_class: str) -> str:
    """Normalize known aliases and preserve unknown labels as distinct classes."""
    text = " ".join(
        (raw_class or "").strip().lower().replace("_", " ").replace("-", " ").split()
    )
    if not text:
        return "_background"
    return _CLASS_ALIASES.get(text, text.replace(" ", "_"))


def extract_canopy_geometry(
    pil_img: Any,
    box_xyxy: List[float],
    hazard_class: str,
    confidence: float = 1.0,
    mask_xy: Optional[List[List[float]]] = None,
    canonicalize: bool = False,
    epsilon_factor: Optional[float] = None,
    force_coarse_box: bool = False,
) -> PerImageAnnotationItem:
    """Extract flexible polygon/OBB contour, pixel area, and ecological carbon weight."""
    x1, y1, x2, y2 = [float(c) for c in box_xyxy]
    img_w, img_h = pil_img.width, pil_img.height
    img_area = float(max(1, img_w * img_h))

    poly: Optional[List[List[float]]] = None
    poly_area: Optional[float] = None

    if force_coarse_box:
        # Deliberate sub-optimal 4-point bounding box fallback for baseline auditing
        bx1, by1, bx2, by2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)
        poly = [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]
        poly_area = (bx2 - bx1) * (by2 - by1)
        return PerImageAnnotationItem(
            hazard_class=hazard_class,
            bounding_box=[bx1, by1, bx2, by2],
            polygon=poly,
            area=round(poly_area, 2),
            weight=round(poly_area / img_area, 6),
            confidence=confidence,
        )

    # 1. If explicit mask coordinates are provided
    if mask_xy is not None and len(mask_xy) >= 3:
        poly = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in mask_xy]
        try:
            import cv2
            import numpy as np
            poly_area = float(cv2.contourArea(np.array(poly, dtype=np.float32)))
        except Exception:
            pass

    # 2. Extract vegetation contour / oriented bounding box with OpenCV if available
    if poly is None:
        try:
            import cv2
            import numpy as np
            np_img = np.array(pil_img)
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(img_w, int(x2)), min(img_h, int(y2))
            if ix2 > ix1 and iy2 > iy1:
                crop = np_img[iy1:iy2, ix1:ix2]
                if crop.ndim == 3 and crop.shape[2] >= 3:
                    r = crop[:, :, 0].astype(np.float32)
                    g = crop[:, :, 1].astype(np.float32)
                    b = crop[:, :, 2].astype(np.float32)
                    exg = 2.0 * g - r - b
                    exg_norm = np.clip((exg - exg.min()) / (exg.max() - exg.min() + 1e-5) * 255.0, 0, 255).astype(np.uint8)
                    _, thresh = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:
                    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    c_area = float(cv2.contourArea(cnt))
                    if c_area > 8:
                        peri = cv2.arcLength(cnt, True)
                        eff_eps = epsilon_factor if epsilon_factor is not None else 0.008
                        epsilon = eff_eps * peri
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        if len(approx) < 4:
                            approx = cv2.approxPolyDP(cnt, 0.5 * eff_eps * peri, True)
                        if len(approx) >= 4:
                            poly = [[round(float(p[0][0] + ix1), 2), round(float(p[0][1] + iy1), 2)] for p in approx]
                            poly_area = c_area
                        else:
                            hull = cv2.convexHull(cnt)
                            if len(hull) >= 4:
                                poly = [[round(float(p[0][0] + ix1), 2), round(float(p[0][1] + iy1), 2)] for p in hull]
                                poly_area = c_area
                            else:
                                rect = cv2.minAreaRect(cnt)
                                box_pts = cv2.boxPoints(rect)
                                poly = [[round(float(p[0] + ix1), 2), round(float(p[1] + iy1), 2)] for p in box_pts]
                                poly_area = c_area
        except Exception:
            pass

    bx1, by1, bx2, by2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)
    # Organic 16-point natural canopy ellipse fallback (inscribed within bbox)
    cx = (bx1 + bx2) / 2.0
    cy = (by1 + by2) / 2.0
    rx = max(1.0, (bx2 - bx1) / 2.0)
    ry = max(1.0, (by2 - by1) / 2.0)
    angles = [i * 2.0 * math.pi / 16.0 for i in range(16)]
    default_canopy_poly = [
        [
            round(cx + rx * math.cos(a) * (0.95 + 0.05 * math.sin(3.0 * a)), 2),
            round(cy + ry * math.sin(a) * (0.95 + 0.05 * math.cos(3.0 * a)), 2),
        ]
        for a in angles
    ]
    # 3. Fallback: organic 16-point canopy polygon
    if poly is None or len(poly) < 4:
        poly = default_canopy_poly
    else:
        poly = [
            [
                max(bx1, min(bx2, round(float(p[0]), 2))),
                max(by1, min(by2, round(float(p[1]), 2))),
            ]
            for p in poly
        ]
    if poly_area is None or poly_area <= 0:
        poly_area = float(max(0.0, math.pi * rx * ry))

    # Canonical carbon class & ecological multiplier
    canon_cls = canonical_carbon_class(hazard_class)
    carbon_mult = CARBON_WEIGHT_MULTIPLIERS.get(canon_cls, 1.0)
    item_ratio = poly_area / img_area
    obj_weight = round(item_ratio * carbon_mult, 6)
    final_class = canon_cls if canonicalize else hazard_class

    safe_conf = round(min(1.0, max(0.0, float(confidence))), 4)
    safe_poly_area = round(float(poly_area if (poly_area is not None and poly_area > 0) else math.pi * rx * ry), 2)

    try:
        return PerImageAnnotationItem(
            hazard_class=final_class,
            bounding_box=[bx1, by1, bx2, by2],
            polygon=poly,
            area=safe_poly_area,
            weight=obj_weight,
            confidence=safe_conf,
        )
    except Exception:
        return PerImageAnnotationItem(
            hazard_class=final_class,
            bounding_box=[bx1, by1, bx2, by2],
            polygon=default_canopy_poly,
            area=safe_poly_area,
            weight=obj_weight,
            confidence=safe_conf,
        )


def compute_image_net_metrics(
    annotations: List[PerImageAnnotationItem],
    image_width: int = 1024,
    image_height: int = 1024,
) -> Tuple[float, float, int]:
    """Compute (net_weight, tree_coverage_percentage, tree_count) for an image.
    
    net_weight: composite ecological carbon weight (sum of all box weights).
    tree_coverage_percentage: physical vegetation/tree coverage percentage in [0.0, 100.0]
    tree_count: total detections
    """
    tree_count = len(annotations)
    if tree_count == 0:
        return 0.0, 0.0, 0

    img_area = float(max(1, image_width * image_height))
    total_physical_area = 0.0
    total_carbon_weight = 0.0

    for ann in annotations:
        area = ann.area
        if area is None or area <= 0:
            if ann.bounding_box and len(ann.bounding_box) == 4:
                x1, y1, x2, y2 = ann.bounding_box
                area = max(0.0, (x2 - x1) * (y2 - y1))
            else:
                area = 0.0
        total_physical_area += area

        if ann.weight is not None and ann.weight >= 0:
            total_carbon_weight += ann.weight
        else:
            canon_cls = canonical_carbon_class(ann.hazard_class)
            mult = CARBON_WEIGHT_MULTIPLIERS.get(canon_cls, 1.0)
            total_carbon_weight += (area / img_area) * mult

    net_weight = round(total_carbon_weight, 6)
    physical_ratio = min(1.0, max(0.0, total_physical_area / img_area))
    tree_coverage_percentage = round(physical_ratio * 100.0, 2)
    return net_weight, tree_coverage_percentage, tree_count


def canonical_image_name(image_id: str, image_url: str = "") -> str:
    """Derive canonical filename (e.g. climate_raw_042.jpg) from image_id or image_url."""
    raw = image_id.split(":")[-1] if ":" in image_id else image_id
    raw = raw.split("/")[-1]
    if raw.startswith("raw_climate_"):
        raw = raw[4:]
    elif raw.startswith("golden_climate_"):
        raw = raw[7:]
    raw = raw.strip()
    if not any(raw.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff")):
        raw = f"{raw}.jpg"
    return raw


def sanitize_and_refine_polygon(
    poly: Optional[Sequence[Sequence[float]]],
    bbox: Sequence[float],
    hazard_class: str = "",
) -> List[List[float]]:
    """Sanitize miner-provided polygon or synthesize a high-precision polygon within bbox.

    Guarantees:
    - Strictly bounded within [x1, y1, x2, y2]
    - Simple non-self-intersecting contour (ordered radially around centroid)
    - Positive finite area (> 0)
    - Realistic organic canopy shape for trees/mangroves/plants, or parcel boundary for fields
    """
    import math

    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx, ry = w / 2.0, h / 2.0

    def _shoelace(pts: Sequence[Sequence[float]]) -> float:
        n = len(pts)
        if n < 3:
            return 0.0
        return abs(
            sum(
                pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
                for i in range(n)
            )
        ) / 2.0

    def _canopy_polygon() -> List[List[float]]:
        # 8-point smooth organic canopy contour
        angles = [i * 2.0 * math.pi / 8.0 for i in range(8)]
        pts = []
        for i, a in enumerate(angles):
            mod = 0.94 + 0.04 * math.sin(i * 1.5)
            px = max(x1, min(x2, round(cx + rx * mod * math.cos(a), 2)))
            py = max(y1, min(y2, round(cy + ry * mod * math.sin(a), 2)))
            pts.append([px, py])
        return pts

    def _parcel_polygon() -> List[List[float]]:
        # 8-point chamfered agricultural plot boundary
        return [
            [round(x1 + 0.08 * w, 2), round(y1, 2)],
            [round(x2 - 0.08 * w, 2), round(y1, 2)],
            [round(x2, 2), round(y1 + 0.12 * h, 2)],
            [round(x2, 2), round(y2 - 0.08 * h, 2)],
            [round(x2 - 0.12 * w, 2), round(y2, 2)],
            [round(x1 + 0.08 * w, 2), round(y2, 2)],
            [round(x1, 2), round(y2 - 0.12 * h, 2)],
            [round(x1, 2), round(y1 + 0.08 * h, 2)],
        ]

    # If miner provided a polygon, attempt to sanitize and retain it
    if poly and len(poly) >= 3:
        clamped = []
        for p in poly:
            if len(p) >= 2 and math.isfinite(float(p[0])) and math.isfinite(float(p[1])):
                px = max(x1, min(x2, round(float(p[0]), 2)))
                py = max(y1, min(y2, round(float(p[1]), 2)))
                clamped.append((px, py))
        unique = list(dict.fromkeys(clamped))
        if len(unique) >= 3:
            c_x = sum(p[0] for p in unique) / len(unique)
            c_y = sum(p[1] for p in unique) / len(unique)
            sorted_pts = sorted(unique, key=lambda p: math.atan2(p[1] - c_y, p[0] - c_x))
            candidate = [[float(p[0]), float(p[1])] for p in sorted_pts]
            if _shoelace(candidate) > 0.5:
                return candidate

    # Synthesize based on class
    h_lower = (hazard_class or "").lower()
    if any(k in h_lower for k in ("field", "farm", "agri", "crop", "parcel", "cropland")):
        res = _parcel_polygon()
    else:
        res = _canopy_polygon()

    if _shoelace(res) <= 0.5:
        res = [
            [round(x1, 2), round(y1, 2)],
            [round(x2, 2), round(y1, 2)],
            [round(x2, 2), round(y2, 2)],
            [round(x1, 2), round(y2, 2)],
        ]
    return res

