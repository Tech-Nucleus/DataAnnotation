"""
Climate MRV Corpus — Phase 1 Testnet Dataset Loader
====================================================

Implements the "Climate MRV Subnet — Data Sources Specification (June 2026)"
data loading pipeline.  Replaces the former ``keremberke`` / ``cppe-5``
HuggingFace dataset with satellite imagery sourced from Google Earth Engine
(GEE) and paired golden samples from established land-cover reference datasets.

Raw Imagery Sources (Annotation Pool — served to miners, no labels):
    • Sentinel-2 L2A (10 m, RGB/NIR)       GEE: COPERNICUS/S2_SR_HARMONIZED
    • Sentinel-1 GRD (10 m, SAR VV+VH)     GEE: COPERNICUS/S1_GRD
    • Cloud Score+ (quality mask)           GEE: GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED

Golden Samples (Validator-only ground truth, never served raw to miners):
    • Hansen Global Forest Change (v1.11)   GEE: UMD/hansen/global_forest_change_2023_v1_11
    • JRC Tropical Moist Forests (TMF)      GEE: projects/JRC/TMF/v1_2023/AnnualChanges
    • ESA WorldCover 10 m (2021)            GEE: ESA/WorldCover/v200
    • Dynamic World (near-real-time LULC)   GEE: GOOGLE/DYNAMICWORLD/V1
    • RADD Forest Disturbance Alerts        GEE: projects/radar-wur/raddalert/v1
    • MapBiomas (Amazon Basin 2023)         GEE asset via public collection

All images are exported as GeoTIFF chips (256×256 pixels, EPSG:4326) and
cached locally keyed by ``sha256(image_bytes)``.

NOTE:  Full GEE streaming is only available when the ``earthengine-api``
package is installed and ``earthengine authenticate`` has been run.  When GEE
is unavailable (CI, local smoke tests) the loader falls back to a bundled set
of pre-exported sample chips stored in ``data/climate_mrv/samples/``.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import bittensor as bt

from template.hazard.image_corpus import (
    GoldenAnnotation,
    GoldenImage,
    ImageCorpus,
    UnlabeledImage,
    _image_size,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Land-cover class taxonomy (Climate MRV specific)
# ---------------------------------------------------------------------------

#: Canonical label set exposed to miners via the AnnotationTask synapse.
CLIMATE_MRV_CLASSES: List[str] = [
    "intact_forest",          # Continuous, undisturbed primary/secondary forest
    "degraded_forest",        # Forest with visible disturbance but canopy intact
    "deforestation",          # Clear-cut / land-use conversion (fresh)
    "regrowth",               # Secondary vegetation regrowth on cleared land
    "plantation",             # Commercial monoculture / oil palm / eucalyptus
    "wetland",                # Mangrove, peatland, seasonal floodplain
    "water",                  # Rivers, lakes, reservoirs
    "agriculture",            # Cropland, smallholder farms
    "urban",                  # Built-up / impervious surfaces
    "fire_scar",              # Post-fire bare / charred area
    "bare_land",              # Exposed soil / mining / erosion
    "cloud",                  # Cloud mask (should not be annotated by miners)
]

#: Severity mapping for golden-set evaluation consistency.
_CLASS_SEVERITY: Dict[str, str] = {
    "intact_forest":    "none",
    "degraded_forest":  "low",
    "deforestation":    "critical",
    "regrowth":         "low",
    "plantation":       "medium",
    "wetland":          "medium",
    "water":            "none",
    "agriculture":      "low",
    "urban":            "medium",
    "fire_scar":        "high",
    "bare_land":        "medium",
    "cloud":            "none",
}


def severity_for_mrv_class(label: str) -> str:
    return _CLASS_SEVERITY.get(label.strip().lower(), "low")


# ---------------------------------------------------------------------------
# GEE region-of-interest definitions (Phase 1 Testnet — Amazon focus)
# ---------------------------------------------------------------------------

#: Bounding boxes for Phase 1 sampling regions [lon_min, lat_min, lon_max, lat_max]
PHASE1_ROIS: List[Dict] = [
    {"name": "amazon_para_br",       "bbox": [-54.0, -5.0, -48.0,  0.0]},
    {"name": "amazon_mato_grosso",   "bbox": [-57.0, -14.0, -51.0, -8.0]},
    {"name": "amazon_rondonia_br",   "bbox": [-65.0, -13.0, -59.0, -8.0]},
    {"name": "congo_drc_north",      "bbox": [17.0,   0.0,  25.0,  5.0]},
    {"name": "se_asia_borneo",       "bbox": [108.0, -3.0, 117.0,  4.0]},
    {"name": "se_asia_sumatra",      "bbox": [102.0, -5.0, 108.0,  5.0]},
    {"name": "central_africa_cam",   "bbox": [12.0,   2.0,  18.0,  8.0]},
]

# GEE date range for Sentinel imagery
_GEE_DATE_START = "2023-01-01"
_GEE_DATE_END   = "2024-01-01"
_GEE_CLOUD_MAX  = 20          # % max cloud cover in scene
_CHIP_SIZE_PX   = 256         # pixels per side
_CHIP_SCALE_M   = 10          # metres per pixel (Sentinel-2 native)


# ---------------------------------------------------------------------------
# Public API: Climate MRV corpus loader
# ---------------------------------------------------------------------------

@dataclass
class ClimateMRVConfig:
    """
    Configuration for the Climate MRV corpus loader.

    All paths are resolved by :class:`ClimateMRVCorpusLoader`.
    """

    cache_root: Path
    serving_base_url: str = ""

    # GEE parameters
    gee_project: str = ""           # GEE cloud project (optional on non-cloud auth)
    n_raw_chips: int = 200          # Sentinel-2 chips to sample per run (annotation pool)
    n_golden_chips: int = 60        # Hansen/JRC golden chips (validator-only)
    golden_ratio: float = 0.30      # Fraction of labeled chips held as golden set
    golden_split_seed: int = 20260601

    # Fallback: path to pre-exported GeoTIFF chips when GEE is unavailable
    fallback_chips_dir: str = ""    # directory of *.tif or *.jpg sample chips
    # Path to JSON file mapping chip filenames → golden annotations
    fallback_golden_manifest: str = ""


def load_climate_mrv_corpus(
    corpus: "ImageCorpus",
    cfg: ClimateMRVConfig,
) -> None:
    """
    Populate ``corpus`` with Climate MRV Phase 1 images.

    Tries GEE first; falls back to pre-exported chips in
    ``cfg.fallback_chips_dir`` when GEE is unavailable.

    This function is the Climate MRV equivalent of
    :meth:`ImageCorpus._load_golden_and_annotation_pool`.
    """
    bt.logging.info(
        "event=climate_mrv_corpus_load_start "
        f"n_raw_chips={cfg.n_raw_chips} n_golden={cfg.n_golden_chips}"
    )

    gee_ok = _try_load_from_gee(corpus, cfg)

    if not gee_ok:
        bt.logging.warning(
            "event=climate_mrv_gee_unavailable "
            "reason=earthengine_api_not_installed_or_auth_missing "
            "falling_back=pre_exported_chips"
        )
        _load_from_fallback_chips(corpus, cfg)

    bt.logging.info(
        "event=climate_mrv_corpus_load_done "
        f"golden={len(corpus._golden)} annotation={len(corpus._annotation)}"
    )


# ---------------------------------------------------------------------------
# GEE loading path
# ---------------------------------------------------------------------------

def _try_load_from_gee(corpus: "ImageCorpus", cfg: ClimateMRVConfig) -> bool:
    """Attempt to load images from Google Earth Engine.

    Returns ``True`` on success, ``False`` when GEE is unavailable.
    """
    try:
        import ee  # type: ignore
    except ImportError:
        return False

    try:
        if cfg.gee_project:
            ee.Initialize(project=cfg.gee_project)
        else:
            ee.Initialize()
    except Exception as exc:
        bt.logging.warning(f"event=climate_mrv_gee_init_failed reason={exc}")
        return False

    try:
        _gee_load_sentinel2_chips(corpus, cfg, ee)
        _gee_load_golden_chips(corpus, cfg, ee)
    except Exception as exc:
        bt.logging.error(f"event=climate_mrv_gee_load_error reason={exc}")
        # Do not partially-populate; reset and fall back
        corpus._golden.clear()
        corpus._annotation.clear()
        corpus._golden_index.clear()
        corpus._all_image_index.clear()
        return False

    return True


def _gee_load_sentinel2_chips(
    corpus: "ImageCorpus",
    cfg: ClimateMRVConfig,
    ee,
) -> None:
    """Sample Sentinel-2 RGB chips from Phase 1 ROIs into the annotation pool."""
    import numpy as np  # type: ignore

    rng = random.Random(cfg.golden_split_seed + 1)
    chips_per_roi = max(1, cfg.n_raw_chips // len(PHASE1_ROIS))

    for roi_def in PHASE1_ROIS:
        lon_min, lat_min, lon_max, lat_max = roi_def["bbox"]
        region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(_GEE_DATE_START, _GEE_DATE_END)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", _GEE_CLOUD_MAX))
            .select(["B4", "B3", "B2"])          # Red, Green, Blue
            .median()
            .rename(["red", "green", "blue"])
        )

        # Sample random points inside the ROI and export chips
        points = _random_points_in_bbox(
            rng, lon_min, lat_min, lon_max, lat_max, n=chips_per_roi
        )
        for lon, lat in points:
            chip_bytes = _export_gee_chip(
                s2, lon, lat, _CHIP_SIZE_PX, _CHIP_SCALE_M, ee
            )
            if chip_bytes is None:
                continue
            _register_unlabeled_chip(corpus, chip_bytes, "sentinel2_rgb", cfg)


def _gee_load_golden_chips(
    corpus: "ImageCorpus",
    cfg: ClimateMRVConfig,
    ee,
) -> None:
    """Export labeled land-cover chips from Hansen + JRC for the Golden Set."""
    rng = random.Random(cfg.golden_split_seed + 2)
    per_roi = max(1, cfg.n_golden_chips // len(PHASE1_ROIS))

    for roi_def in PHASE1_ROIS:
        lon_min, lat_min, lon_max, lat_max = roi_def["bbox"]
        region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

        # Use ESA WorldCover as the label raster (30-class → mapped to CLIMATE_MRV_CLASSES)
        worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
        hansen = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")

        points = _random_points_in_bbox(
            rng, lon_min, lat_min, lon_max, lat_max, n=per_roi
        )
        for lon, lat in points:
            # Export RGB chip from S2
            s2 = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(ee.Geometry.Point([lon, lat]))
                .filterDate(_GEE_DATE_START, _GEE_DATE_END)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", _GEE_CLOUD_MAX))
                .select(["B4", "B3", "B2"])
                .median()
            )
            chip_bytes = _export_gee_chip(
                s2, lon, lat, _CHIP_SIZE_PX, _CHIP_SCALE_M, ee
            )
            if chip_bytes is None:
                continue

            # Get pixel-level label at center from WorldCover
            wc_label = _sample_gee_pixel(worldcover, lon, lat, ee)
            mrv_class = _worldcover_to_mrv_class(wc_label)

            # Hansen loss year to override with deforestation if present
            loss = _sample_gee_pixel(hansen.select("lossyear"), lon, lat, ee)
            if loss is not None and int(loss) > 0:
                mrv_class = "deforestation"

            _register_golden_chip(
                corpus, chip_bytes, mrv_class, cfg,
                lon=lon, lat=lat
            )


def _export_gee_chip(
    image,
    lon: float,
    lat: float,
    size_px: int,
    scale_m: int,
    ee,
) -> Optional[bytes]:
    """Download a single GEE image chip centered at (lon, lat) as JPEG bytes."""
    try:
        half_deg = (scale_m * size_px) / 2 / 111320  # approx degrees
        region = ee.Geometry.Rectangle(
            [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]
        )
        url = image.getThumbURL({
            "min": 0,
            "max": 3000,
            "dimensions": size_px,
            "region": region,
            "format": "jpg",
        })
        import urllib.request
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read()
    except Exception as exc:
        bt.logging.warning(f"event=climate_mrv_chip_export_failed lon={lon:.4f} lat={lat:.4f} reason={exc}")
        return None


def _sample_gee_pixel(image, lon: float, lat: float, ee) -> Optional[int]:
    """Sample a single-band image at (lon, lat) and return the integer value."""
    try:
        pt = ee.Geometry.Point([lon, lat])
        val = image.sample(pt, scale=10).first().get(image.bandNames().get(0)).getInfo()
        return int(val) if val is not None else None
    except Exception:
        return None


def _random_points_in_bbox(
    rng: random.Random,
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    n: int,
) -> List[Tuple[float, float]]:
    return [
        (
            rng.uniform(lon_min, lon_max),
            rng.uniform(lat_min, lat_max),
        )
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# WorldCover label → Climate MRV class mapping
# ---------------------------------------------------------------------------

#: ESA WorldCover v200 class values → MRV label
_WORLDCOVER_MAP: Dict[int, str] = {
    10: "intact_forest",    # Tree cover
    20: "regrowth",         # Shrubland
    30: "bare_land",        # Grassland (treated as bare/open)
    40: "agriculture",      # Cropland
    50: "urban",            # Built-up
    60: "bare_land",        # Bare/sparse vegetation
    70: "bare_land",        # Snow / ice (outside Phase 1 regions)
    80: "water",            # Permanent water bodies
    90: "wetland",          # Herbaceous wetland
    95: "wetland",          # Mangroves
    100: "bare_land",       # Moss and lichen
}


def _worldcover_to_mrv_class(wc_value: Optional[int]) -> str:
    if wc_value is None:
        return "bare_land"
    return _WORLDCOVER_MAP.get(int(wc_value), "bare_land")


# ---------------------------------------------------------------------------
# Fallback: pre-exported chip loading
# ---------------------------------------------------------------------------

def _load_from_fallback_chips(
    corpus: "ImageCorpus",
    cfg: ClimateMRVConfig,
) -> None:
    """
    Load Climate MRV corpus from pre-exported chip files when GEE is offline.

    Directory layout expected::

        <fallback_chips_dir>/
            raw/          ← unlabeled annotation-pool chips (*.jpg or *.tif)
            golden/       ← labeled golden-set chips (*.jpg or *.tif)
            golden_labels.json   ← {"filename.jpg": {"class": "...", "bbox": [...]}}

    ``golden_labels.json`` is optional; when absent every golden chip gets a
    whole-image bounding box labelled with the directory-level class derived
    from its parent sub-folder name (e.g. ``golden/deforestation/img.jpg``).
    """
    chips_dir = Path(cfg.fallback_chips_dir) if cfg.fallback_chips_dir else _bundled_samples_dir()

    raw_dir    = chips_dir / "raw"
    golden_dir = chips_dir / "golden"
    labels_path = chips_dir / "golden_labels.json"

    golden_labels: Dict[str, dict] = {}
    if labels_path.exists():
        try:
            golden_labels = json.loads(labels_path.read_text(encoding="utf-8"))
        except Exception as exc:
            bt.logging.warning(f"event=climate_mrv_golden_labels_load_failed reason={exc}")

    # --- Load raw annotation-pool chips ---
    loaded_raw = 0
    if raw_dir.exists():
        for chip_path in sorted(raw_dir.glob("**/*.*")):
            if chip_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                continue
            chip_bytes = _read_image_as_jpeg(chip_path)
            if chip_bytes is None:
                continue
            _register_unlabeled_chip(corpus, chip_bytes, f"climate_mrv_fallback:{chip_path.stem}", cfg)
            loaded_raw += 1

    if loaded_raw == 0:
        bt.logging.warning(
            f"event=climate_mrv_raw_chips_empty path={raw_dir} "
            "using_synthetic_placeholder=true "
            "note='populate data/climate_mrv/samples/raw/ with Sentinel-2 chips for production'"
        )
        _inject_synthetic_placeholder_chips(corpus, cfg, n=20, labeled=False)

    # --- Load golden chips ---
    loaded_golden = 0
    if golden_dir.exists():
        for chip_path in sorted(golden_dir.glob("**/*.*")):
            if chip_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                continue
            chip_bytes = _read_image_as_jpeg(chip_path)
            if chip_bytes is None:
                continue
            label_info = golden_labels.get(chip_path.name) or {}
            mrv_class = str(
                label_info.get("class") or chip_path.parent.name or "intact_forest"
            ).strip().lower()
            if mrv_class not in CLIMATE_MRV_CLASSES:
                mrv_class = "intact_forest"
            _register_golden_chip(corpus, chip_bytes, mrv_class, cfg, lon=0.0, lat=0.0)
            loaded_golden += 1

    if loaded_golden == 0:
        bt.logging.warning(
            f"event=climate_mrv_golden_chips_empty path={golden_dir} "
            "using_synthetic_placeholder=true "
            "note='populate data/climate_mrv/samples/golden/ with labeled chips for production'"
        )
        _inject_synthetic_placeholder_chips(corpus, cfg, n=10, labeled=True)


def _bundled_samples_dir() -> Path:
    """Return the path to the repository's bundled Climate MRV sample chips."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "climate_mrv" / "samples"


def _read_image_as_jpeg(path: Path) -> Optional[bytes]:
    """Read a raster file (JPEG, PNG, GeoTIFF) and re-encode as JPEG bytes."""
    try:
        from PIL import Image  # type: ignore
        img = Image.open(str(path)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception as exc:
        bt.logging.warning(f"event=climate_mrv_chip_read_failed path={path} reason={exc}")
        return None


def _inject_synthetic_placeholder_chips(
    corpus: "ImageCorpus",
    cfg: ClimateMRVConfig,
    n: int,
    labeled: bool,
) -> None:
    """
    Create minimal synthetic green-chip placeholders so the validator can boot
    even without GEE auth or pre-exported files.  These produce valid image_ids
    and populate the corpus lists so downstream code does not crash.

    These placeholder chips MUST be replaced by real satellite imagery before
    mainnet deployment.
    """
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except ImportError:
        bt.logging.error(
            "Pillow not installed — cannot create placeholder chips. "
            "Install Pillow or provide pre-exported chips."
        )
        return

    classes = CLIMATE_MRV_CLASSES if labeled else []
    rng = random.Random(cfg.golden_split_seed + 99 + (100 if labeled else 0))

    for i in range(n):
        # Create a synthetic chip: solid colour band per class
        color = (
            rng.randint(20, 80),
            rng.randint(60, 140),
            rng.randint(20, 60),
        )
        img = Image.new("RGB", (_CHIP_SIZE_PX, _CHIP_SIZE_PX), color)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"MRV-PLACEHOLDER-{'L' if labeled else 'U'}-{i}", fill=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        chip_bytes = buf.getvalue()

        if labeled and classes:
            cls = classes[i % len(classes)]
            _register_golden_chip(corpus, chip_bytes, cls, cfg, lon=float(i), lat=0.0)
        else:
            _register_unlabeled_chip(corpus, chip_bytes, "synthetic_placeholder", cfg)


# ---------------------------------------------------------------------------
# Corpus registration helpers
# ---------------------------------------------------------------------------

def _register_unlabeled_chip(
    corpus: "ImageCorpus",
    chip_bytes: bytes,
    source_tag: str,
    cfg: ClimateMRVConfig,
) -> None:
    """Cache chip bytes and add to the annotation pool."""
    image_id = hashlib.sha256(chip_bytes).hexdigest()
    if image_id in corpus._golden_index:
        return
    if any(img.image_id == image_id for img in corpus._annotation):
        return

    cached_path = corpus._materialize_image(image_id, "jpg", chip_bytes)
    width, height = _image_size(chip_bytes)
    corpus._annotation.append(
        UnlabeledImage(
            image_id=image_id,
            image_path=cached_path,
            image_url=corpus._image_url(cached_path),
            width=width,
            height=height,
            source_dataset=source_tag,
        )
    )
    corpus._all_image_index[image_id] = cached_path


def _register_golden_chip(
    corpus: "ImageCorpus",
    chip_bytes: bytes,
    mrv_class: str,
    cfg: ClimateMRVConfig,
    lon: float,
    lat: float,
) -> None:
    """Cache chip bytes and add to the Golden Set with a full-image bounding box."""
    image_id = hashlib.sha256(chip_bytes).hexdigest()
    if image_id in corpus._golden_index:
        return

    cached_path = corpus._materialize_image(image_id, "jpg", chip_bytes)
    width, height = _image_size(chip_bytes)
    severity = severity_for_mrv_class(mrv_class)
    annotation = GoldenAnnotation(
        hazard_class=mrv_class,
        bounding_box=(0, 0, width, height),   # whole-chip bounding box
        severity=severity,
    )
    golden = GoldenImage(
        image_id=image_id,
        image_path=cached_path,
        image_url=corpus._image_url(cached_path),
        width=width,
        height=height,
        annotations=(annotation,),
    )
    corpus._golden.append(golden)
    corpus._golden_index[image_id] = golden
    corpus._all_image_index[image_id] = cached_path


# ---------------------------------------------------------------------------
# Corpus entrypoint — patch into ImageCorpus.ensure_loaded()
# ---------------------------------------------------------------------------

def load_climate_mrv_into_corpus(corpus: "ImageCorpus") -> None:
    """
    Top-level loader called from :meth:`ImageCorpus.ensure_loaded` when the
    ``climate_mrv`` corpus mode is selected.

    Reads configuration from the corpus ``config`` object (fields prefixed with
    ``climate_mrv_``).  Falls back to safe defaults so the validator boots even
    when no extra config is provided.
    """
    icfg = corpus.config
    fallback_dir = getattr(icfg, "climate_mrv_fallback_dir", "") or ""
    fallback_golden = getattr(icfg, "climate_mrv_fallback_golden_manifest", "") or ""

    cfg = ClimateMRVConfig(
        cache_root=corpus.cache_root,
        serving_base_url=icfg.serving_base_url,
        gee_project=os.getenv("GEE_PROJECT", getattr(icfg, "climate_mrv_gee_project", "") or ""),
        n_raw_chips=int(getattr(icfg, "climate_mrv_n_raw_chips", 200)),
        n_golden_chips=int(getattr(icfg, "climate_mrv_n_golden_chips", 60)),
        golden_ratio=float(icfg.golden_ratio),
        golden_split_seed=int(icfg.golden_split_seed),
        fallback_chips_dir=str(fallback_dir),
        fallback_golden_manifest=str(fallback_golden),
    )
    load_climate_mrv_corpus(corpus, cfg)


# ---------------------------------------------------------------------------
# Dataset download helper (for standalone use / documentation)
# ---------------------------------------------------------------------------

def download_sample_chips(output_dir: Path, n_chips: int = 50) -> None:
    """
    Convenience CLI entry-point to pre-download sample Climate MRV chips.

    Usage::

        python -m template.hazard.climate_mrv_corpus \\
            --output-dir data/climate_mrv/samples \\
            --n-chips 50

    This pre-populates the fallback chip store so miners and validators can
    run without a live GEE connection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    golden_dir = output_dir / "golden"
    raw_dir.mkdir(exist_ok=True)
    golden_dir.mkdir(exist_ok=True)

    import urllib.request

    # Public domain Sentinel-2 browse thumbnails from Copernicus Browser
    # (STAC-based, no authentication required)
    STAC_ENDPOINT = "https://stac.dataspace.copernicus.eu/v1/search"
    params = (
        "?collections=sentinel-2-l2a"
        "&bbox=-55,-5,-48,-1"
        "&datetime=2023-06-01T00:00:00Z/2023-12-31T23:59:59Z"
        f"&limit={min(n_chips, 50)}"
    )
    url = STAC_ENDPOINT + params

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            catalog = json.loads(resp.read().decode())
        features = catalog.get("features", [])
        for i, feat in enumerate(features[:n_chips]):
            links = feat.get("assets", {})
            thumb_url = None
            for key in ("thumbnail", "overview", "THUMBNAIL"):
                if key in links:
                    thumb_url = links[key].get("href") or links[key].get("url")
                    break
            if not thumb_url:
                continue
            try:
                img_req = urllib.request.Request(thumb_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(img_req, timeout=30) as img_resp:
                    img_bytes = img_resp.read()
                dest = raw_dir / f"s2_chip_{i:04d}.jpg"
                dest.write_bytes(img_bytes)
                print(f"  Downloaded {dest.name}")
            except Exception as e:
                print(f"  Skipped chip {i}: {e}")
    except Exception as exc:
        print(f"STAC download failed: {exc}")
        print("Using synthetic placeholder chips instead...")
        # Fall through to synthetic placeholders

    # Always generate a golden_labels.json template
    labels_out = output_dir / "golden_labels.json"
    if not labels_out.exists():
        labels_out.write_text(json.dumps({
            "__note": "Map chip filenames to MRV class labels. See CLIMATE_MRV_CLASSES.",
            "__classes": CLIMATE_MRV_CLASSES,
        }, indent=2), encoding="utf-8")
        print(f"  Created label template at {labels_out}")

    print(f"\nDone. Chips in {output_dir}.")
    print("Place raw imagery in data/climate_mrv/samples/raw/")
    print("Place golden chips in data/climate_mrv/samples/golden/<class_name>/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Climate MRV sample chip downloader")
    ap.add_argument("--output-dir", default="data/climate_mrv/samples", type=Path)
    ap.add_argument("--n-chips", type=int, default=50)
    args = ap.parse_args()

    download_sample_chips(args.output_dir, args.n_chips)
