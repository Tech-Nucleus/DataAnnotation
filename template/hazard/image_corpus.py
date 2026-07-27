"""
Validator-side image corpus for the dual-flywheel subnet.

This module owns the four image populations described in the subnet
specification:

  1. Golden Set         (validator-only ground truth, 30% of the labeled
                         construction-safety dataset, never shared raw).
  2. Miner Annotation Pool (the remaining shared images, served to miners as
                            URLs only, no labels).
  3. Optional Extra Annotation Pool (additional unlabeled datasets, if configured).
  4. Cross-Domain Benchmark (validator-only, used for offline evaluation only).

Every image receives a permanent ``image_id`` derived from
``sha256(image_bytes)``, which makes per-image-id traceability across rounds
and miners trivial.

The implementation is real and dependency-grounded: it relies on the
HuggingFace ``datasets`` library and ``Pillow`` to materialize image bytes,
and it caches every image to disk under a configurable root keyed by
``image_id``. There is intentionally no fallback path — if a dataset cannot
be loaded the validator hard-fails so operators notice and fix the
configuration.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import bittensor as bt

from template.protocol import SeverityTier


_MISSING_PPE_NEEDLES = ("no", "missing", "without", "lack", "absent")
_PPE_TOKENS = ("hardhat", "helmet", "vest", "mask", "harness", "goggles", "gloves")
_FALL_TOKENS = ("fall", "edge", "scaffold", "ladder", "height")
_FIRE_TOKENS = ("fire", "spark", "explosion", "electrical", "shock")
_LOW_RISK_TOKENS = ("trip", "slip", "spill", "minor")


def _normalize_label(label: str) -> str:
    """Lowercase and split into ascii tokens for keyword matching."""
    if not label:
        return ""
    cleaned = label.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return cleaned.strip()


def _severity_for_label(label: str) -> SeverityTier:
    """Map a hazard class label to a severity tier using deterministic rules.

    The rules mirror the subnet spec ("missing hardhat -> high") and provide a
    consistent severity assignment across the Golden ground-truth dataset.
    """

    normalized = _normalize_label(label)
    if not normalized:
        return "none"
    tokens = normalized.split()
    has_missing_marker = any(t in _MISSING_PPE_NEEDLES for t in tokens)
    has_ppe_token = any(t in _PPE_TOKENS for t in tokens)
    if has_missing_marker and has_ppe_token:
        return "high"
    if any(t in _FALL_TOKENS for t in tokens):
        return "high"
    if any(t in _FIRE_TOKENS for t in tokens):
        return "critical"
    if has_ppe_token:
        return "medium"
    if any(t in _LOW_RISK_TOKENS for t in tokens):
        return "low"
    return "medium"


@dataclass(frozen=True)
class GoldenAnnotation:
    """One ground-truth instance for a Golden Set image."""

    hazard_class: str
    bounding_box: Tuple[int, int, int, int]  # pixel coords [x_min, y_min, x_max, y_max]
    severity: SeverityTier


@dataclass(frozen=True)
class GoldenImage:
    """A fully-labeled Golden Set image (validator-only)."""

    image_id: str
    image_path: Path
    image_url: str
    width: int
    height: int
    annotations: Tuple[GoldenAnnotation, ...]


@dataclass(frozen=True)
class UnlabeledImage:
    """An unlabeled annotation-pool image served to miners."""

    image_id: str
    image_path: Path
    image_url: str
    width: int
    height: int
    source_dataset: str


@dataclass(frozen=True)
class BenchmarkImage:
    """A cross-domain benchmark image (validator-only)."""

    image_id: str
    image_path: Path
    width: int
    height: int
    annotations: Tuple[GoldenAnnotation, ...]
    source_dataset: str


@dataclass
class ImageCorpusConfig:
    """User-configurable settings for the dual-flywheel image corpus.

    **Default dataset (Phase 1 Testnet): Climate MRV**
    When ``golden_dataset_id`` is ``'climate_mrv'`` (the default), the
    :mod:`template.hazard.climate_mrv_corpus` loader is used to pull
    Sentinel-2 imagery and Hansen/ESA golden samples (with automatic GEE
    fallback to pre-exported chips).  Set ``golden_dataset_id`` to any
    HuggingFace dataset ID to revert to the legacy HF loading path.
    """

    cache_root: Path
    serving_base_url: str = ""
    # ---------------------------------------------------------------
    # DATASET SOURCE — default is Climate MRV (satellite imagery).
    # Set to a HuggingFace dataset ID (e.g.
    # "keremberke/construction-safety-object-detection") to use the
    # legacy HF loading path instead.
    # ---------------------------------------------------------------
    golden_dataset_id: str = "climate_mrv"
    golden_split: str = "train"
    golden_ratio: float = 0.3
    golden_split_seed: int = 20260601
    annotation_dataset_ids: Sequence[str] = field(default_factory=tuple)
    annotation_split: str = "train"
    annotation_max_per_dataset: int = 512
    benchmark_dataset_id: str = "rishitdagli/cppe-5"
    benchmark_split: str = "test"
    benchmark_max_samples: int = 64
    # Modern ``datasets`` rejects legacy Hub dataset scripts; the parquet
    # snapshot branch is the supported load path for these repos.
    hf_revision: str = "refs/convert/parquet"
    # Ratio of annotation pool images to expose as labeled Training Pool for miners.
    # These images get ground-truth annotations shared with miners for fine-tuning.
    training_pool_ratio: float = 0.15
    training_pool_max: int = 30
    # When set, load Golden + pool from ``scripts/localnet/prepare_coco_val2017_subset.py``
    # manifest instead of HuggingFace (localnet COCO acceptance tests).
    coco_manifest_path: str = ""

    # ---------------------------------------------------------------
    # Climate MRV specific fields (ignored when golden_dataset_id != 'climate_mrv')
    # ---------------------------------------------------------------
    climate_mrv_gee_project: str = ""           # GEE cloud project ID (optional)
    climate_mrv_n_raw_chips: int = 200          # Sentinel-2 annotation-pool chips per run
    climate_mrv_n_golden_chips: int = 60        # Hansen/ESA golden chips (validator-only)
    climate_mrv_fallback_dir: str = ""          # Pre-exported chip directory (offline mode)
    climate_mrv_fallback_golden_manifest: str = ""  # JSON label manifest for fallback chips

    def normalized_annotation_entries(self) -> List[Tuple[str, str]]:
        """Return ``(hub_dataset_id, split)`` for each annotation source.

        Comma-separated ``annotation_dataset_ids`` entries may use
        ``org/name@split``; otherwise ``annotation_split`` is used.
        """

        if isinstance(self.annotation_dataset_ids, str):
            raw = self.annotation_dataset_ids
        else:
            raw = ",".join(self.annotation_dataset_ids)
        default_split = (self.annotation_split or "train").strip()
        entries: List[Tuple[str, str]] = []
        for piece in raw.split(","):
            item = piece.strip()
            if not item:
                continue
            if "@" in item:
                ds_id, sp = item.split("@", 1)
                ds_id, sp = ds_id.strip(), sp.strip()
                if ds_id:
                    entries.append((ds_id, sp or default_split))
            else:
                entries.append((item, default_split))
        return entries


class ImageCorpus:
    """Validator-owned image corpus with per-image_id traceability.

    Loads three populations (Golden, Annotation, Benchmark) once,
    caches image bytes locally keyed by ``sha256(image_bytes)``, and exposes
    typed accessors plus a quick lookup table of Golden ground-truth records
    (used by :class:`AnnotationFidelityScorer`).
    """

    def __init__(self, config: ImageCorpusConfig):
        self.config = config
        self.cache_root = Path(config.cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._loaded = False
        self._golden: List[GoldenImage] = []
        self._annotation: List[UnlabeledImage] = []
        self._training_pool: List[GoldenImage] = []  # labeled subset of annotation pool
        self._benchmark: List[BenchmarkImage] = []
        self._golden_index: Dict[str, GoldenImage] = {}
        self._all_image_index: Dict[str, Path] = {}

    # ------------------------------------------------------------------ API
    def ensure_loaded(self) -> None:
        """Idempotently materialize all four populations to local disk.

        Loading priority (first match wins):
          1. COCO local manifest   (``coco_manifest_path`` is set)
          2. Climate MRV dataset   (``golden_dataset_id == 'climate_mrv'``)
          3. HuggingFace datasets  (legacy — any other dataset ID)
        """
        with self._lock:
            if self._loaded:
                return
            bt.logging.info("event=image_corpus_load_start cache_root=%s" % self.cache_root)
            manifest = (self.config.coco_manifest_path or "").strip()
            if manifest:
                from template.hazard.coco_corpus import load_coco_manifest_into_corpus
                load_coco_manifest_into_corpus(self, Path(manifest))
            elif (self.config.golden_dataset_id or "").strip().lower() == "climate_mrv":
                from template.hazard.climate_mrv_corpus import load_climate_mrv_into_corpus
                bt.logging.info(
                    "event=image_corpus_mode mode=climate_mrv "
                    "spec='Climate MRV Subnet Data Sources Specification June 2026'"
                )
                load_climate_mrv_into_corpus(self)
            else:
                # Legacy HuggingFace path (retained for backward compatibility)
                bt.logging.info(
                    f"event=image_corpus_mode mode=huggingface "
                    f"dataset={self.config.golden_dataset_id}"
                )
                self._load_golden_and_annotation_pool()
                self._load_annotation_pool()
                self._load_benchmark()
            self._loaded = True
            bt.logging.info(
                "event=image_corpus_load_done golden=%d annotation=%d benchmark=%d"
                % (
                    len(self._golden),
                    len(self._annotation),
                    len(self._benchmark),
                )
            )

    def golden_images(self) -> List[GoldenImage]:
        self.ensure_loaded()
        return list(self._golden)

    def annotation_images(self) -> List[UnlabeledImage]:
        self.ensure_loaded()
        return list(self._annotation)

    def training_pool_images(self) -> List[GoldenImage]:
        """Return the public Training Pool — labeled images shared with miners."""
        self.ensure_loaded()
        return list(self._training_pool)

    def benchmark_images(self) -> List[BenchmarkImage]:
        self.ensure_loaded()
        return list(self._benchmark)

    def golden_lookup(self, image_id: str) -> Optional[GoldenImage]:
        """Return the Golden record for ``image_id`` if known."""
        self.ensure_loaded()
        return self._golden_index.get(image_id)

    def is_golden(self, image_id: str) -> bool:
        self.ensure_loaded()
        return image_id in self._golden_index

    def known_image_path(self, image_id: str) -> Optional[Path]:
        """Local filesystem path for a previously-cached image, if known.

        Used by validator-side assembly / serving helpers.
        """
        self.ensure_loaded()
        return self._all_image_index.get(image_id)

    # -------------------------------------------------------- internal load
    def _load_golden_and_annotation_pool(self) -> None:
        """Load the labeled construction-safety dataset and apply the
        deterministic Golden vs annotation-pool split."""

        records = list(
            _iter_hf_image_records(
                self.config.golden_dataset_id,
                self.config.golden_split,
                expect_labels=True,
                limit=0,
                hf_revision=self.config.hf_revision,
            )
        )
        if not records:
            raise RuntimeError(
                f"Golden dataset {self.config.golden_dataset_id} returned 0 labeled images."
            )

        # Deterministic split: hash(image_id + seed) -> bucket. This guarantees
        # the same image always lands in the same partition across restarts.
        golden_ratio = max(0.0, min(1.0, float(self.config.golden_ratio)))
        seed_str = str(self.config.golden_split_seed)

        for image_bytes, image_format, label_payloads in records:
            image_id = hashlib.sha256(image_bytes).hexdigest()
            cached_path = self._materialize_image(image_id, image_format, image_bytes)
            self._all_image_index[image_id] = cached_path
            width, height = _image_size(image_bytes)
            annotations = tuple(
                _golden_annotations_from_payload(item, width, height)
                for item in label_payloads
            )
            annotations = tuple(item for item in annotations if item is not None)
            if not annotations:
                # Hard-fail mode: skip images with zero usable labels rather than
                # silently inflating Golden counts. We log the discard for ops.
                bt.logging.warning(
                    f"event=image_corpus_drop_unlabeled_record dataset={self.config.golden_dataset_id} "
                    f"image_id={image_id} reason=no_valid_annotations"
                )
                continue

            partition_score = int(
                hashlib.sha256(f"{image_id}:{seed_str}".encode("utf-8")).hexdigest()[:8],
                16,
            ) / 0xFFFFFFFF
            if partition_score < golden_ratio:
                golden = GoldenImage(
                    image_id=image_id,
                    image_path=cached_path,
                    image_url=self._image_url(cached_path),
                    width=width,
                    height=height,
                    annotations=annotations,
                )
                self._golden.append(golden)
                self._golden_index[image_id] = golden
            else:
                self._annotation.append(
                    UnlabeledImage(
                        image_id=image_id,
                        image_path=cached_path,
                        image_url=self._image_url(cached_path),
                        width=width,
                        height=height,
                        source_dataset=self.config.golden_dataset_id,
                    )
                )

        if not self._golden:
            raise RuntimeError(
                "Golden split is empty after partitioning; check golden_ratio/seed."
            )
        if not self._annotation:
            raise RuntimeError(
                "Annotation split is empty after partitioning; check golden_ratio/seed."
            )

    def _load_annotation_pool(self) -> None:
        """Load every configured unlabeled annotation dataset and cache bytes."""

        entries = self.config.normalized_annotation_entries()
        if not entries:
            return
        per_cap = max(0, int(self.config.annotation_max_per_dataset))
        for dataset_id, ann_split in entries:
            count = 0
            for image_bytes, image_format, _payload in _iter_hf_image_records(
                dataset_id,
                ann_split,
                expect_labels=False,
                limit=per_cap,
                hf_revision=self.config.hf_revision,
            ):
                image_id = hashlib.sha256(image_bytes).hexdigest()
                if image_id in self._golden_index:
                    continue
                if any(existing.image_id == image_id for existing in self._annotation):
                    continue
                cached_path = self._materialize_image(image_id, image_format, image_bytes)
                self._all_image_index[image_id] = cached_path
                width, height = _image_size(image_bytes)
                self._annotation.append(
                    UnlabeledImage(
                        image_id=image_id,
                        image_path=cached_path,
                        image_url=self._image_url(cached_path),
                        width=width,
                        height=height,
                        source_dataset=dataset_id,
                    )
                )
                count += 1
            if count == 0:
                raise RuntimeError(
                    f"Annotation dataset {dataset_id} returned 0 images. Check the HF id."
                )

    def _load_benchmark(self) -> None:
        cap = max(0, int(self.config.benchmark_max_samples))
        records = list(
            _iter_hf_image_records(
                self.config.benchmark_dataset_id,
                self.config.benchmark_split,
                expect_labels=True,
                limit=cap,
                hf_revision=self.config.hf_revision,
            )
        )
        if not records:
            raise RuntimeError(
                f"Benchmark dataset {self.config.benchmark_dataset_id} returned 0 images."
            )
        for image_bytes, image_format, label_payloads in records:
            image_id = hashlib.sha256(image_bytes).hexdigest()
            cached_path = self._materialize_image(image_id, image_format, image_bytes)
            self._all_image_index[image_id] = cached_path
            width, height = _image_size(image_bytes)
            annotations = tuple(
                _golden_annotations_from_payload(item, width, height)
                for item in label_payloads
            )
            annotations = tuple(item for item in annotations if item is not None)
            if not annotations:
                continue
            self._benchmark.append(
                BenchmarkImage(
                    image_id=image_id,
                    image_path=cached_path,
                    width=width,
                    height=height,
                    annotations=annotations,
                    source_dataset=self.config.benchmark_dataset_id,
                )
            )

    # ----------------------------------------------------------- IO helpers
    def _materialize_image(self, image_id: str, image_format: str, payload: bytes) -> Path:
        ext = (image_format or "jpg").lower().strip()
        if ext not in {"jpg", "jpeg", "png", "webp", "bmp"}:
            ext = "jpg"
        target = self.cache_root / f"{image_id}.{ext}"
        if not target.exists():
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(payload)
            tmp.replace(target)
        # Always (re)record the path so callers can dereference deterministically.
        self._all_image_index[image_id] = target
        return target

    def _image_url(self, local_path: Path) -> str:
        base = (self.config.serving_base_url or "").strip()
        if not base:
            return local_path.as_uri()
        if not base.endswith("/"):
            base = base + "/"
        return base + local_path.name


# ---------------------------------------------------------------------------
# HuggingFace dataset adapters
# ---------------------------------------------------------------------------

def _iter_hf_image_records(
    dataset_id: str,
    split: str,
    *,
    expect_labels: bool,
    limit: int,
    hf_revision: str = "",
) -> Iterable[Tuple[bytes, str, List[dict]]]:
    """Stream ``(image_bytes, image_format, label_payloads)`` tuples from HF.

    ``label_payloads`` is a list of normalized dicts with keys
    ``hazard_class``, ``bounding_box`` (pixel xyxy) and ``severity``. For
    unlabeled datasets the list is always empty.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment must install it
        raise ImportError(
            "huggingface 'datasets' library is required for the dual-flywheel image corpus."
        ) from exc

    bt.logging.info(
        f"event=image_corpus_hf_load dataset={dataset_id} split={split} "
        f"revision={(hf_revision or '').strip() or 'default'}"
    )
    rev = (hf_revision or "").strip()
    ds = (
        load_dataset(dataset_id, split=split, revision=rev)
        if rev
        else load_dataset(dataset_id, split=split)
    )
    total = len(ds) if hasattr(ds, "__len__") else 0
    bt.logging.info(
        f"event=image_corpus_hf_loaded dataset={dataset_id} split={split} samples={total}"
    )

    image_field = _detect_image_field(ds)
    if image_field is None:
        raise RuntimeError(
            f"Dataset {dataset_id} has no image-typed column; cannot harvest images."
        )

    label_field = _detect_label_field(ds) if expect_labels else None
    class_names = _detect_class_names(ds, label_field) if label_field else []

    iterator: Iterable = ds
    if limit and limit > 0 and total > limit:
        iterator = ds.select(range(limit))

    yielded = 0
    for sample in iterator:
        try:
            image_bytes, image_format = _encode_pil_image(sample[image_field])
        except Exception as exc:  # bad rows are skipped but the dataset itself must work
            bt.logging.warning(
                f"event=image_corpus_skip_sample dataset={dataset_id} reason={exc}"
            )
            continue
        labels: List[dict] = []
        if label_field is not None:
            try:
                labels = _normalize_label_payload(sample[label_field], class_names)
            except Exception as exc:
                bt.logging.warning(
                    f"event=image_corpus_skip_label dataset={dataset_id} reason={exc}"
                )
                labels = []
        yield image_bytes, image_format, labels
        yielded += 1
        if limit and limit > 0 and yielded >= limit:
            break


def _detect_image_field(ds) -> Optional[str]:
    features = getattr(ds, "features", None)
    if not features:
        return None
    for name, feature in features.items():
        feature_type = type(feature).__name__
        if feature_type == "Image":
            return name
    # Fallback to common conventional names.
    for candidate in ("image", "img", "picture"):
        if candidate in features:
            return candidate
    return None


def _detect_label_field(ds) -> Optional[str]:
    features = getattr(ds, "features", None)
    if not features:
        return None
    for candidate in ("objects", "annotations", "labels", "label"):
        if candidate in features:
            return candidate
    return None


def _detect_class_names(ds, label_field: Optional[str]) -> List[str]:
    if label_field is None:
        return []
    features = getattr(ds, "features", None)
    if not features or label_field not in features:
        return []
    feature = features[label_field]
    # Sequence of dicts with category as ClassLabel
    inner = getattr(feature, "feature", None)
    if isinstance(inner, dict):
        category = inner.get("category") or inner.get("class") or inner.get("label")
        if category is not None and hasattr(category, "names"):
            return list(category.names)
    # Top-level dict with ClassLabel sequence
    if isinstance(feature, dict):
        category = feature.get("category") or feature.get("class") or feature.get("label")
        if category is not None and hasattr(category, "feature") and hasattr(category.feature, "names"):
            return list(category.feature.names)
    return []


def _encode_pil_image(image_obj) -> Tuple[bytes, str]:
    """Serialize a PIL/HF image record to raw bytes (and a usable format hint)."""

    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pillow is required to materialize HF images.") from exc

    if hasattr(image_obj, "save"):
        # Already a PIL Image. Re-encode to JPEG so byte-for-byte hashes are stable.
        buf = io.BytesIO()
        rgb = image_obj.convert("RGB") if image_obj.mode != "RGB" else image_obj
        rgb.save(buf, format="JPEG", quality=92)
        return buf.getvalue(), "jpg"
    if isinstance(image_obj, dict) and "bytes" in image_obj and image_obj["bytes"]:
        raw = image_obj["bytes"]
        path = image_obj.get("path") or ""
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else "jpg"
        # Re-encode to JPEG to normalize hashes regardless of source format.
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue(), ext if ext in {"jpg", "jpeg", "png", "webp"} else "jpg"
    if isinstance(image_obj, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(image_obj))).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue(), "jpg"
    if isinstance(image_obj, str):
        with open(image_obj, "rb") as handle:
            raw = handle.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue(), "jpg"
    raise TypeError(f"Unsupported image object type: {type(image_obj).__name__}")


def _image_size(image_bytes: bytes) -> Tuple[int, int]:
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pillow is required for image dimension lookup.") from exc
    with Image.open(io.BytesIO(image_bytes)) as img:
        return int(img.width), int(img.height)


def _normalize_label_payload(payload, class_names: Sequence[str]) -> List[dict]:
    """Normalize HF object-detection style payloads to a list of dicts.

    Supports both keremberke-style ``{"bbox": [...], "category": [...]}`` and
    cppe-5 style ``[{"bbox": [...], "category_id": ...}, ...]``.
    """

    items: List[dict] = []
    if payload is None:
        return items

    if isinstance(payload, dict) and "bbox" in payload and "category" in payload:
        bboxes = payload.get("bbox") or []
        categories = payload.get("category") or []
        for bbox, cat in zip(bboxes, categories):
            class_label = _resolve_class_name(cat, class_names)
            items.append({"bbox": list(bbox), "category": class_label})
        return items

    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            bbox = entry.get("bbox") or entry.get("box")
            if not bbox:
                continue
            cat = entry.get("category") or entry.get("category_id") or entry.get("label")
            class_label = _resolve_class_name(cat, class_names)
            items.append({"bbox": list(bbox), "category": class_label})
        return items

    return items


def _resolve_class_name(value, class_names: Sequence[str]) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, bool)):
        idx = int(value)
        if 0 <= idx < len(class_names):
            return str(class_names[idx])
        return str(idx)
    return str(value)


def _golden_annotations_from_payload(
    payload: dict, image_width: int, image_height: int
) -> Optional[GoldenAnnotation]:
    """Convert a normalized label dict to a :class:`GoldenAnnotation`.

    The bbox may be in COCO ``[x, y, w, h]`` or pascal ``[x_min, y_min, x_max, y_max]``
    form; we detect by checking whether ``[2]`` and ``[3]`` look like extents
    or sizes relative to the image dimensions.
    """

    bbox = payload.get("bbox") or []
    if len(bbox) != 4:
        return None
    x1, y1, x2_or_w, y2_or_h = [float(v) for v in bbox]
    # COCO if (x + w) <= image_width and (y + h) <= image_height and we don't already
    # exceed bounds when interpreted as xyxy.
    looks_like_xywh = (
        x2_or_w + x1 <= image_width + 1
        and y2_or_h + y1 <= image_height + 1
        and x2_or_w <= image_width
        and y2_or_h <= image_height
        and x2_or_w >= 0
        and y2_or_h >= 0
    )
    if looks_like_xywh and x2_or_w < image_width and y2_or_h < image_height and (x2_or_w + y2_or_h) > 0:
        x_min, y_min = int(round(x1)), int(round(y1))
        x_max = int(round(x1 + x2_or_w))
        y_max = int(round(y1 + y2_or_h))
    else:
        x_min, y_min = int(round(x1)), int(round(y1))
        x_max, y_max = int(round(x2_or_w)), int(round(y2_or_h))

    x_min = max(0, min(image_width, x_min))
    y_min = max(0, min(image_height, y_min))
    x_max = max(0, min(image_width, x_max))
    y_max = max(0, min(image_height, y_max))
    if x_max <= x_min or y_max <= y_min:
        return None

    hazard_class = str(payload.get("category") or "hazard").strip().lower().replace(" ", "_")
    severity = _severity_for_label(hazard_class)
    return GoldenAnnotation(
        hazard_class=hazard_class,
        bounding_box=(x_min, y_min, x_max, y_max),
        severity=severity,
    )


def golden_annotation_to_jsonable(record: GoldenAnnotation) -> dict:
    return {
        "hazard_class": record.hazard_class,
        "bounding_box": list(record.bounding_box),
        "severity": record.severity,
    }


def golden_image_to_jsonable(record: GoldenImage) -> dict:
    return {
        "image_id": record.image_id,
        "image_path": str(record.image_path),
        "image_url": record.image_url,
        "width": record.width,
        "height": record.height,
        "annotations": [golden_annotation_to_jsonable(a) for a in record.annotations],
    }


def dump_golden_manifest(corpus: ImageCorpus, target: Path) -> Path:
    """Persist the Golden Set manifest (validator-only) to disk for auditing."""
    payload = {
        "schema_version": "golden_manifest.v1",
        "image_count": len(corpus.golden_images()),
        "images": [golden_image_to_jsonable(item) for item in corpus.golden_images()],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target
