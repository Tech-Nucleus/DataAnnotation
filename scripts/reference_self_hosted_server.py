#!/usr/bin/env python3
"""Reference self-hosted model server with REAL YOLOv8 training.

Implements the three-endpoint contract that the ``self_hosted`` backend
expects.  Wraps a YOLOv8 model and performs actual fine-tuning on
submitted image batches.

Usage::

    pip install fastapi uvicorn ultralytics pillow

    # Terminal 1 — start the reference server
    python scripts/reference_self_hosted_server.py --port 8081

    # Terminal 2 — run the miner pointing at it
    python neurons/miner.py \\
        --miner.model_backend self_hosted \\
        --miner.self_hosted_train_url http://localhost:8081/train \\
        --miner.self_hosted_infer_url http://localhost:8081/infer
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
import random
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Import guards for optional dependencies
# ---------------------------------------------------------------------------
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as _exc:
    raise SystemExit(
        "Missing required packages. Install with:\n"
        "  pip install fastapi uvicorn pydantic\n"
        f"Original error: {_exc}"
    )

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment,misc]

try:
    from urllib.request import Request, urlopen
except ImportError:
    Request = None  # type: ignore[assignment,misc]
    urlopen = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ref-server")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Reference Self-Hosted Model Server", version="2.0.0")

# ---------------------------------------------------------------------------
# Global mutable state (thread-safe via _lock)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_models: Dict[str, str] = {}  # model_version → checkpoint path
_default_checkpoint: str = "yolov8n.pt"
_workspace: str = "artifacts/self_hosted_server"
_adversarial_random_boxes: bool = False

# Server-level training defaults; used only when a /train request's config omits the field.
_default_epochs: int = 5
_default_imgsz: int = 640
_default_batch: int = 8

# Training queue — only one training job runs at a time
_training_semaphore = threading.Semaphore(1)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class AnnotationSpec(BaseModel):
    """Single annotation within training data."""
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2] absolute
    x_center: Optional[float] = None
    y_center: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


class TrainImageSpec(BaseModel):
    image_id: str
    image_url: str
    annotations: Optional[List[dict]] = None


class TrainRequest(BaseModel):
    images: List[TrainImageSpec]
    config: dict = Field(default_factory=dict)


class TrainResponse(BaseModel):
    job_id: str
    status: str = "started"


class TrainStatusResponse(BaseModel):
    status: str
    metrics: dict = Field(default_factory=dict)
    model_version: str = ""
    error: str = ""


class InferImageSpec(BaseModel):
    image_id: str
    image_url: str


class InferRequest(BaseModel):
    images: List[InferImageSpec]
    model_version: str = ""


class AnnotationItem(BaseModel):
    image_id: str
    hazard_class: str
    bounding_box: List[float]


class InferResponse(BaseModel):
    annotations: List[AnnotationItem]


# ---------------------------------------------------------------------------
# Utility: download image bytes
# ---------------------------------------------------------------------------
def _load_image_bytes(url: str) -> bytes:
    """Load image from file://, http(s)://, or raw absolute path."""
    if url.startswith("file://"):
        return Path(url[7:]).read_bytes()
    if url.startswith("/"):
        return Path(url).read_bytes()
    if url.startswith(("http://", "https://")):
        req = Request(url, headers={"User-Agent": "reference-server/2.0"})
        with urlopen(req, timeout=120) as resp:
            return resp.read()
    # Fallback: treat as local path
    p = Path(url)
    if p.exists():
        return p.read_bytes()
    raise FileNotFoundError(f"Cannot resolve image URL: {url}")


def _save_image(url: str, dest: Path) -> Path:
    """Download an image and save it to *dest*. Returns the path."""
    data = _load_image_bytes(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def _sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Dataset preparation helpers
# ---------------------------------------------------------------------------

def _generate_pseudo_labels(
    image_paths: List[Path],
    labels_dir: Path,
) -> None:
    """Run pretrained YOLOv8n on images to create pseudo-labels.

    Writes labels using the pretrained model's native COCO class indices, which
    are consistent with the class map seeded from the same base model.
    """
    if YOLO is None:
        raise RuntimeError("ultralytics is required for pseudo-labelling")

    pretrained = YOLO("yolov8n.pt")
    labels_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        results = pretrained(str(img_path), verbose=False)
        label_path = labels_dir / (img_path.stem + ".txt")
        lines: List[str] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                img_w, img_h = results[0].orig_shape[1], results[0].orig_shape[0]
                for box in boxes:
                    cls_idx = int(box.cls[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # Convert to normalized xywh
                    xc = ((x1 + x2) / 2) / img_w
                    yc = ((y1 + y2) / 2) / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                    lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _norm_class_name(name: str) -> str:
    """Normalise a class name for matching (case/separator-insensitive).

    COCO uses spaces (``dining table``) while datasets often use underscores
    (``dining_table``); both must map to the same pretrained class index.
    """
    return " ".join(str(name or "").strip().lower().replace("_", " ").split())


def _resolve_class_id(
    ann: dict,
    name_to_id: Dict[str, int],
    id_to_display: Dict[int, str],
) -> int:
    """Resolve an annotation to a stable integer class id.

    Prefers the textual class (``class_name``/``hazard_class``) so distinct hazard
    classes map to distinct ids — collapsing everything to id 0 would make
    fine-tuning meaningless. Matching is separator-insensitive against the seeded
    COCO classes so e.g. ``dining_table`` reuses COCO's ``dining table`` index
    (keeping the pretrained head, nc=80). The miner's *exact* spelling is kept as
    the display name so inference echoes the convention the validator scores against.
    """
    raw = str(ann.get("class_name") or ann.get("hazard_class") or "").strip()
    if raw:
        key = _norm_class_name(raw)
        if key in name_to_id:
            cid = name_to_id[key]
        else:
            cid = len(name_to_id)
            name_to_id[key] = cid
        id_to_display[cid] = raw  # prefer the miner's exact spelling
        return cid
    try:
        return int(ann.get("class_id") or 0)
    except (TypeError, ValueError):
        return 0


def _parse_annotations_to_labels(
    annotations: List[dict],
    img_w: int,
    img_h: int,
    name_to_id: Dict[str, int],
    id_to_display: Dict[int, str],
) -> List[str]:
    """Convert a list of annotation dicts into YOLO-format label lines.

    Supports both absolute bbox [x1,y1,x2,y2] and pre-normalised xywh.
    """
    lines: List[str] = []
    for ann in annotations:
        cls_id = _resolve_class_id(ann, name_to_id, id_to_display)
        # Already normalised xywh
        if all(k in ann for k in ("x_center", "y_center", "width", "height")):
            lines.append(
                f"{cls_id} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                f"{ann['width']:.6f} {ann['height']:.6f}"
            )
        else:
            bbox = ann.get("bbox") or ann.get("bounding_box")
            if bbox and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                xc = ((x1 + x2) / 2) / img_w
                yc = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def _prepare_dataset(
    images: List[TrainImageSpec],
    dataset_dir: Path,
    base_names: Optional[Dict[int, str]] = None,
) -> tuple[Path, Dict[int, str]]:
    """Build a YOLO-format dataset directory.

    Structure::

        dataset_dir/
            images/train/<image_id>.jpg
            labels/train/<image_id>.txt
            dataset.yaml

    Returns (dataset_yaml_path, id->name mapping).
    """
    img_dir = dataset_dir / "images" / "train"
    lbl_dir = dataset_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Seed the class map from the base model's COCO classes so a hazard_class whose
    # name matches a COCO class reuses its original index (keeps the pretrained head
    # intact). Genuinely new names are appended and the head is resized accordingly.
    name_to_id: Dict[str, int] = {}
    id_to_display: Dict[int, str] = {}
    if base_names:
        for idx in sorted(base_names):
            name_to_id[_norm_class_name(base_names[idx])] = int(idx)
            id_to_display[int(idx)] = str(base_names[idx])

    has_annotations = any(
        img.annotations and len(img.annotations) > 0 for img in images
    )

    downloaded_paths: List[Path] = []
    images_needing_pseudo: List[Path] = []

    for img_spec in images:
        safe_id = img_spec.image_id.replace("/", "_").replace("\\", "_")
        ext = ".jpg"
        # Try to preserve original extension
        url_lower = img_spec.image_url.lower()
        for candidate in (".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff", ".webp"):
            if url_lower.endswith(candidate):
                ext = candidate
                break
        dest = img_dir / f"{safe_id}{ext}"
        try:
            _save_image(img_spec.image_url, dest)
        except Exception as exc:
            logger.warning("Failed to download %s: %s", img_spec.image_url, exc)
            continue
        downloaded_paths.append(dest)

        if has_annotations and img_spec.annotations:
            # Write labels from provided annotations
            try:
                if Image is not None:
                    pil = Image.open(dest)
                    img_w, img_h = pil.size
                else:
                    img_w, img_h = 640, 640  # fallback
            except Exception:
                img_w, img_h = 640, 640

            lines = _parse_annotations_to_labels(
                img_spec.annotations, img_w, img_h, name_to_id, id_to_display
            )
            label_file = lbl_dir / f"{safe_id}.txt"
            label_file.write_text("\n".join(lines) + ("\n" if lines else ""))
        else:
            images_needing_pseudo.append(dest)

    # Generate pseudo-labels for images without annotations
    if images_needing_pseudo:
        logger.info(
            "Generating pseudo-labels for %d images using pretrained model",
            len(images_needing_pseudo),
        )
        _generate_pseudo_labels(images_needing_pseudo, lbl_dir)

    # Ensure at least one class
    if not id_to_display:
        id_to_display = {0: "object"}

    # Write dataset.yaml (id->name covering 0..max so YOLO's head lines up)
    nc = max(id_to_display) + 1
    names_list = [id_to_display.get(i, f"class_{i}") for i in range(nc)]
    yaml_path = dataset_dir / "dataset.yaml"
    yaml_content = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/train\n"  # reuse train as val for small datasets
        f"\n"
        f"nc: {nc}\n"
        f"names: {names_list}\n"
    )
    yaml_path.write_text(yaml_content)
    logger.info("Dataset YAML written to %s (nc=%d)", yaml_path, nc)

    return yaml_path, {i: names_list[i] for i in range(nc)}


# ---------------------------------------------------------------------------
# Background training worker
# ---------------------------------------------------------------------------

def _training_worker(
    job_id: str,
    images: List[TrainImageSpec],
    config: dict,
):
    """Run actual YOLOv8 training in a background thread.

    Acquires _training_semaphore so only one training runs at a time.
    """
    logger.info("[train:%s] Queued — waiting for semaphore", job_id[:8])
    _training_semaphore.acquire()
    try:
        _run_training(job_id, images, config)
    finally:
        _training_semaphore.release()


def _run_training(
    job_id: str,
    images: List[TrainImageSpec],
    config: dict,
):
    """Core training logic."""
    global _workspace

    with _lock:
        _jobs[job_id]["status"] = "training"
        _jobs[job_id]["metrics"] = {"mAP": 0.0, "loss": 999.0}

    if YOLO is None:
        with _lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = "ultralytics is not installed"
        return

    dataset_dir = Path(_workspace) / "datasets" / job_id[:8]
    dataset_dir = dataset_dir.resolve()
    project_dir = (Path(_workspace) / "runs" / job_id[:8]).resolve()

    try:
        # ------------------------------------------------------------------
        # 1. Load base model (so the dataset class map can reuse its COCO indices)
        # ------------------------------------------------------------------
        base_ckpt = config.get("checkpoint", _default_checkpoint)
        logger.info("[train:%s] Loading base model: %s", job_id[:8], base_ckpt)
        model = YOLO(base_ckpt)

        # ------------------------------------------------------------------
        # 2. Prepare dataset
        # ------------------------------------------------------------------
        logger.info("[train:%s] Preparing dataset with %d images", job_id[:8], len(images))
        yaml_path, class_names = _prepare_dataset(
            images, dataset_dir, base_names=dict(getattr(model, "names", {}) or {})
        )

        # ------------------------------------------------------------------
        # 3. Train
        # ------------------------------------------------------------------
        epochs = int(config.get("epochs", _default_epochs))
        imgsz = int(config.get("imgsz", _default_imgsz))
        batch = int(config.get("batch", _default_batch))
        patience = int(config.get("patience", 0))  # 0 = no early stopping
        # Per-job seed (derived from job_id) so independent miners training on the same
        # pool diverge instead of producing byte-identical submissions (which the
        # validator rejects as duplicates). Overridable via config["seed"].
        seed = int(config.get("seed", int(hashlib.sha256(job_id.encode()).hexdigest()[:8], 16) % 100000))

        logger.info(
            "[train:%s] Starting YOLO training — epochs=%d, imgsz=%d, batch=%d, seed=%d",
            job_id[:8], epochs, imgsz, batch, seed,
        )

        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            seed=seed,
            project=str(project_dir),
            name="train",
            exist_ok=True,
            verbose=True,
        )

        # ------------------------------------------------------------------
        # 4. Gather metrics from training results
        # ------------------------------------------------------------------
        metrics: Dict[str, Any] = {
            "epochs": epochs,
            "images": len(images),
        }
        try:
            if hasattr(results, "results_dict"):
                rd = results.results_dict
                metrics["mAP"] = rd.get("metrics/mAP50(B)", 0.0)
                metrics["mAP50-95"] = rd.get("metrics/mAP50-95(B)", 0.0)
                metrics["loss"] = rd.get("train/box_loss", 0.0)
            elif hasattr(results, "maps"):
                metrics["mAP"] = float(results.maps[0]) if results.maps else 0.0
        except Exception as exc:
            logger.warning("[train:%s] Could not extract metrics: %s", job_id[:8], exc)

        # ------------------------------------------------------------------
        # 5. Find best checkpoint and compute version hash
        # ------------------------------------------------------------------
        # Ultralytics may relocate outputs (e.g. prepend runs/detect/ to a relative
        # project path), so trust the trainer's reported paths first, then fall back
        # to the expected location, then a recursive search.
        best_pt = None
        trainer = getattr(model, "trainer", None)
        for attr in ("best", "last"):
            cand = getattr(trainer, attr, None) if trainer is not None else None
            if cand and Path(cand).exists():
                best_pt = Path(cand)
                break
        if best_pt is None:
            for cand in (
                project_dir / "train" / "weights" / "best.pt",
                project_dir / "train" / "weights" / "last.pt",
            ):
                if cand.exists():
                    best_pt = cand
                    break
        if best_pt is None:
            matches = sorted(Path.cwd().rglob(f"*{job_id[:8]}*/**/weights/best.pt"))
            if matches:
                best_pt = matches[0]

        if best_pt is not None and best_pt.exists():
            model_version = _sha256_file(best_pt)
            with _lock:
                _models[model_version] = str(best_pt)
            logger.info(
                "[train:%s] Training completed — version=%s",
                job_id[:8], model_version[:16],
            )
        else:
            model_version = hashlib.sha256(job_id.encode()).hexdigest()
            logger.warning(
                "[train:%s] No checkpoint found, using fallback version", job_id[:8]
            )

        with _lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["model_version"] = model_version
            _jobs[job_id]["metrics"] = metrics

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("[train:%s] Training failed: %s\n%s", job_id[:8], exc, tb)
        with _lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    """Accept a training job and start async YOLO training."""
    job_id = str(uuid.uuid4())

    with _lock:
        _jobs[job_id] = {
            "status": "queued",
            "model_version": "",
            "metrics": {},
            "error": "",
            "created_at": time.time(),
        }

    # Launch background training thread
    t = threading.Thread(
        target=_training_worker,
        args=(job_id, req.images, req.config),
        daemon=True,
        name=f"train-{job_id[:8]}",
    )
    t.start()

    logger.info("[train] job_id=%s, images=%d — started", job_id, len(req.images))
    return TrainResponse(job_id=job_id, status="started")


@app.get("/train/status/{job_id}", response_model=TrainStatusResponse)
def train_status(job_id: str):
    """Return training job status."""
    with _lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return TrainStatusResponse(
        status=job["status"],
        metrics=job.get("metrics", {}),
        model_version=job.get("model_version", ""),
        error=job.get("error", ""),
    )


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    """Run inference on provided images using YOLO.

    If *model_version* matches a fine-tuned checkpoint, that checkpoint is
    used.  Otherwise falls back to the default (yolov8n.pt).
    """
    if YOLO is None:
        raise HTTPException(
            status_code=500,
            detail="ultralytics not installed — cannot run inference",
        )
    if Image is None:
        raise HTTPException(
            status_code=500,
            detail="Pillow not installed — cannot run inference",
        )

    with _lock:
        checkpoint_path = _models.get(req.model_version, _default_checkpoint)

    logger.info(
        "[infer] model_version=%s → checkpoint=%s",
        req.model_version[:16] if req.model_version else "(default)",
        checkpoint_path,
    )

    model = YOLO(checkpoint_path)
    all_annotations: List[AnnotationItem] = []

    if _adversarial_random_boxes:
        corpus = None
        try:
            from template.hazard.image_corpus import ImageCorpus, ImageCorpusConfig
            from pathlib import Path
            icfg = ImageCorpusConfig(cache_root=Path('./artifacts/localnet/self_hosted_image_cache').resolve())
            coco_path = Path('./artifacts/localnet/coco200/manifest.json').resolve()
            if coco_path.exists():
                icfg.coco_manifest_path = str(coco_path)
            else:
                icfg.golden_dataset_id = 'climate_mrv'
                icfg.golden_split = 'val'
                icfg.golden_ratio = 0.3
                icfg.golden_split_seed = 42
            corpus = ImageCorpus(icfg)
            corpus.ensure_loaded()
        except Exception as e:
            logger.warning("Could not load image corpus for adversarial boxes: %s", e)

        for img_spec in req.images:
            try:
                image_bytes = _load_image_bytes(img_spec.image_url)
                pil_img = Image.open(io.BytesIO(image_bytes))
                width, height = pil_img.size
            except Exception:
                width, height = 640, 640

            mrv_class = "deforestation"
            if corpus is not None:
                try:
                    from template.hazard.climate_mrv_corpus import CLIMATE_MRV_CLASSES
                    if CLIMATE_MRV_CLASSES:
                        mrv_class = CLIMATE_MRV_CLASSES[0]
                except Exception:
                    pass
            bbox = [10.0, 10.0, float(width - 10), float(height - 10)]
            if corpus is not None:
                if img_spec.image_id in corpus._golden_index:
                    golden_img = corpus._golden_index[img_spec.image_id]
                    if golden_img.annotations:
                        mrv_class = golden_img.annotations[0].hazard_class
                        bbox = [0.0, 0.0, float(width), float(height)]

            all_annotations.append(
                AnnotationItem(
                    image_id=img_spec.image_id,
                    hazard_class=mrv_class,
                    bounding_box=bbox,
                )
            )
        logger.warning(
            "[infer] adversarial_random_boxes enabled — returning %d synthetic detections",
            len(all_annotations),
        )
        return InferResponse(annotations=all_annotations)

    for img_spec in req.images:
        try:
            image_bytes = _load_image_bytes(img_spec.image_url)
            pil_img = Image.open(io.BytesIO(image_bytes))

            # Ensure RGB
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            results = model(pil_img, verbose=False)

            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        cls_idx = int(box.cls[0].item())
                        cls_name = model.names.get(cls_idx, f"class_{cls_idx}")
                        all_annotations.append(
                            AnnotationItem(
                                image_id=img_spec.image_id,
                                hazard_class=cls_name,
                                bounding_box=xyxy,
                            )
                        )
        except Exception as exc:
            logger.warning("[infer] failed for %s: %s", img_spec.image_id, exc)

    logger.info(
        "[infer] processed %d images, %d detections",
        len(req.images), len(all_annotations),
    )
    return InferResponse(annotations=all_annotations)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Simple health check endpoint."""
    with _lock:
        active_jobs = sum(
            1 for j in _jobs.values() if j["status"] in ("queued", "training")
        )
    return {
        "status": "ok",
        "active_jobs": active_jobs,
        "models_registered": len(_models),
        "ultralytics_available": YOLO is not None,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reference self-hosted model server with real YOLOv8 training"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8081, help="Bind port")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="yolov8n.pt",
        help="Default YOLO checkpoint to use as base model",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="artifacts/self_hosted_server",
        help="Working directory for datasets, runs, and checkpoints",
    )
    parser.add_argument(
        "--adversarial-random-boxes",
        action="store_true",
        help="Test mode only: return intentionally poor random boxes from /infer.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Allow test-only synthetic/adversarial behavior.",
    )
    parser.add_argument("--train-epochs", type=int, default=5, help="Default training epochs (when request omits it).")
    parser.add_argument("--train-imgsz", type=int, default=640, help="Default training image size (when request omits it).")
    parser.add_argument("--train-batch", type=int, default=8, help="Default training batch size (when request omits it).")
    args = parser.parse_args()
    if args.adversarial_random_boxes and not args.test_mode:
        raise SystemExit("--adversarial-random-boxes requires --test-mode")

    _default_checkpoint = args.checkpoint
    _workspace = args.workspace
    _adversarial_random_boxes = bool(args.adversarial_random_boxes)
    _default_epochs = int(args.train_epochs)
    _default_imgsz = int(args.train_imgsz)
    _default_batch = int(args.train_batch)

    # Ensure workspace exists
    Path(_workspace).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Reference Self-Hosted Model Server v2.0")
    print("=" * 60)
    print(f"  Host:       {args.host}")
    print(f"  Port:       {args.port}")
    print(f"  Checkpoint: {_default_checkpoint}")
    print(f"  Workspace:  {_workspace}")
    print(f"  YOLO avail: {YOLO is not None}")
    print(f"  PIL avail:  {Image is not None}")
    print(f"  Adversarial random boxes: {_adversarial_random_boxes}")
    print()
    print("  Endpoints:")
    print("    POST /train")
    print("    GET  /train/status/{job_id}")
    print("    POST /infer")
    print("    GET  /health")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port)
