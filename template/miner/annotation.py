from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

import bittensor as bt

from template.hazard.r2_storage import (
    load_r2_credentials_from_env,
    upload_bytes_to_r2,
)
from template.protocol import (
    AnnotationTask,
    AnnotationsFilePayload,
    ImageAnnotationDocument,
)


def fetch_url_bytes(url: str, *, timeout: float = 120.0) -> bytes:
    """Download image bytes from ``http(s)`` or ``file`` URLs."""
    req = Request(url, headers={"User-Agent": "hazard-subnet-miner/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def build_synthetic_labeled_png(width: int = 64, height: int = 64) -> bytes:
    """Test helper: tiny RGB PNG bytes."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("pillow is required.") from exc

    img = Image.new("RGB", (width, height), color=(120, 140, 160))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class AnnotationEngine:
    """
    Annotation-only miner: receive unlabeled image URLs, run YOLO
    detector locally, upload ``annotations.json`` to R2.
    """

    def __init__(self, config=None):
        miner_cfg = getattr(config, "miner", object())
        workspace = Path(
            getattr(miner_cfg, "annotation_workspace", "artifacts/miner_annotation")
        )
        workspace.mkdir(parents=True, exist_ok=True)
        self.workspace = workspace

        raw_backend = str(getattr(miner_cfg, "annotation_backend", "yolo")).strip()
        if raw_backend.lower() == "deterministic":
            raise ValueError(
                "miner.annotation_backend='deterministic' is removed. "
                "Use 'yolo' (YOLO-only detector pipeline)."
            )
        self.annotation_backend = raw_backend.lower() if raw_backend else "yolo"
        if self.annotation_backend != "yolo":
            raise ValueError(
                f"miner.annotation_backend={self.annotation_backend!r} is removed from production. "
                "Use 'yolo' or select a training backend with miner.model_backend."
            )
        self.r2_prefix = str(
            getattr(miner_cfg, "dual_flywheel_r2_prefix", "miners/annotations")
        ).strip()

        self.detector_checkpoint: Path | None = None
        if self.annotation_backend == "yolo":
            detector = str(
                getattr(miner_cfg, "detector_checkpoint", "")
                or "yolov8s.pt"
            ).strip()
            self.detector_checkpoint = Path(detector).expanduser()
            if not self.detector_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Detector checkpoint not found: {self.detector_checkpoint}. "
                    "Set --miner.detector_checkpoint to a local YOLO weights file."
                )

        self._fetch_image: Callable[[str], bytes] = fetch_url_bytes
        self.model_version = self._compute_model_version(miner_cfg)

    @staticmethod
    def _compute_model_version(miner_cfg: object) -> str:
        parts = [
            str(getattr(miner_cfg, "annotation_backend", "yolo")),
            str(getattr(miner_cfg, "detector_checkpoint", "") or ""),
        ]
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        return digest

    def run(
        self,
        synapse: AnnotationTask,
        *,
        miner_hotkey: str,
    ) -> AnnotationTask:
        started = time.time()
        try:
            self._validate_request(synapse)
            creds = load_r2_credentials_from_env()
            remote_base = f"{self.r2_prefix.strip().rstrip('/')}/{synapse.task_id}/"

            records: list[ImageAnnotationDocument] = []
            for spec in synapse.annotation_images:
                img_bytes = self._fetch_image(spec.image_url)
                if self.annotation_backend == "yolo":
                    assert self.detector_checkpoint is not None
                    from template.miner.detector_annotate import annotate_image_detector_only

                    doc = annotate_image_detector_only(
                        checkpoint=self.detector_checkpoint,
                        image_bytes=img_bytes,
                        image_id=spec.image_id,
                        image_url=spec.image_url,
                        model_version=self.model_version,
                        miner_uid=miner_hotkey,
                    )
                else:
                    raise ValueError(
                        f"Unknown annotation_backend: {self.annotation_backend!r}; "
                        "use yolo."
                    )
                records.append(doc)

            payload = AnnotationsFilePayload(
                schema_version="annotations.v1",
                task_id=synapse.task_id,
                records=records,
            )
            raw = json.dumps(payload.model_dump(), indent=2, sort_keys=True).encode("utf-8")
            annotations_key = f"{remote_base}annotations.json"
            annotations_uri = upload_bytes_to_r2(
                raw,
                object_key=annotations_key,
                creds=creds,
                content_type="application/json",
            )

            synapse.annotations_uri = annotations_uri
            synapse.miner_r2_credentials = creds
            synapse.error_message = None
        except (URLError, OSError, ValueError, RuntimeError, ImportError) as exc:
            bt.logging.error(f"Annotation task failed {synapse.task_id}: {exc}")
            synapse.error_message = str(exc)
        synapse.duration_ms = int((time.time() - started) * 1000)
        return synapse

    @staticmethod
    def _validate_request(synapse: AnnotationTask) -> None:
        if not synapse.task_id:
            raise ValueError("task_id is required.")
        if not synapse.annotation_images:
            raise ValueError("annotation_images must be non-empty.")
