"""ModelTrainingAnnotationEngine — the new training-and-inference miner orchestrator.

Replaces the legacy ``AnnotationEngine`` for miners using multi-backend
training.  Orchestrates:

  1. Download images from synapse
  2. Deterministic dataset splitting
  3. Backend-driven train → infer cycle (or auto-research)
  4. Model caching (train-once-and-cache)
  5. annotations.json assembly and R2 upload
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

import bittensor as bt

from template.hazard.r2_storage import (
    load_r2_credentials_from_env,
    upload_bytes_to_r2,
)
from template.miner.backends.base import InferImage, TrainImage, TrainResult
from template.miner.backends.factory import get_backend
from template.miner.model_cache import ModelCache
from template.protocol import (
    AnnotationTask,
    AnnotationsFilePayload,
    ImageAnnotationDocument,
    PerImageAnnotationItem,
)


def _fetch_url_bytes(url: str, *, timeout: float = 120.0) -> bytes:
    """Download image bytes from ``http(s)`` or ``file`` URLs."""
    req = Request(url, headers={"User-Agent": "hazard-subnet-miner/2.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


class ModelTrainingAnnotationEngine:
    """Training-and-inference miner for multi-backend operation.

    Supports three model backends (``yolo_local``, ``self_hosted``,
    ``openai_vision``), deterministic dataset splitting, an optional
    auto-research loop, and train-once-and-cache model persistence.
    """

    def __init__(self, config=None):
        miner_cfg = getattr(config, "miner", object())
        self._config = config

        # Workspace
        workspace = Path(
            getattr(miner_cfg, "annotation_workspace", "artifacts/miner_annotation")
        )
        workspace.mkdir(parents=True, exist_ok=True)
        self.workspace = workspace

        # Image download directory
        self.image_dir = workspace / "downloaded_images"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # Backend selection
        self.backend_name = str(
            getattr(miner_cfg, "model_backend", "yolo_local")
        ).strip()
        self.backend = get_backend(self.backend_name, config)

        # Dataset splitting
        self.split_seed = int(getattr(miner_cfg, "split_seed", 42))
        self.train_split_pct = int(getattr(miner_cfg, "train_split_pct", 70))

        # R2 upload
        self.r2_prefix = str(
            getattr(miner_cfg, "dual_flywheel_r2_prefix", "miners/annotations")
        ).strip()

        # Model cache. The CLI default is an empty string, and Path("") resolves to the
        # current working directory — which would scatter hash-named cache dirs into the
        # repo root. Fall back to a workspace-relative location when unset.
        cache_dir = str(getattr(miner_cfg, "model_cache_dir", "") or "").strip()
        cache_root = Path(cache_dir) if cache_dir else (workspace / "model_cache")
        self.model_cache = ModelCache(cache_root)

        # Caching flags
        self.force_retrain = bool(getattr(miner_cfg, "force_retrain", False))
        self.skip_training = bool(getattr(miner_cfg, "skip_training", False))

        # Auto-research
        self.enable_autoresearch = bool(
            getattr(miner_cfg, "enable_autoresearch", False)
        )
        self.autoresearch_config_path = str(
            getattr(miner_cfg, "autoresearch_config_path", "") or ""
        ).strip()
        self.autoresearch_max_trials = int(
            getattr(miner_cfg, "autoresearch_max_trials", 0)
        )

        # Model version / state
        self._cached_training_pool_hash: Optional[str] = None
        self._cached_model_version: Optional[str] = None
        self._cached_checkpoint_path: Optional[Path] = None

        bt.logging.info(
            f"ModelTrainingAnnotationEngine initialized — "
            f"backend={self.backend_name}, "
            f"split_seed={self.split_seed}, "
            f"train_split_pct={self.train_split_pct}, "
            f"autoresearch={self.enable_autoresearch}"
        )

    def run(
        self,
        synapse: AnnotationTask,
        *,
        miner_hotkey: str,
    ) -> AnnotationTask:
        """Process an ``AnnotationTask`` synapse end-to-end."""
        started = time.time()
        try:
            self._validate_request(synapse)

            # 1. Download all images
            image_paths = self._download_images(synapse)

            # 2. Check if we need to (re)train
            training_pool_hash = synapse.training_pool_hash or self._compute_pool_hash(
                synapse.training_pool
            )
            needs_training = self._should_train(training_pool_hash)

            if needs_training and not self.skip_training:
                # 3. Prepare training data from training pool
                train_images = self._prepare_training_images(synapse)

                if self.enable_autoresearch:
                    # Auto-research path
                    model_version, checkpoint = self._run_autoresearch(
                        train_images, training_pool_hash
                    )
                else:
                    # Standard training path
                    model_version, checkpoint = self._run_training(
                        train_images, training_pool_hash
                    )

                self._cached_model_version = model_version
                self._cached_checkpoint_path = checkpoint
                self._cached_training_pool_hash = training_pool_hash
            else:
                bt.logging.info(
                    "ModelTrainingAnnotationEngine: using cached model "
                    f"(hash={training_pool_hash[:16]}…)"
                )
                if not self._cached_model_version:
                    # First run with skip_training — use pretrained
                    self._cached_model_version = self._compute_model_version()

            # 4. Inference covers EVERY requested annotation image.
            # The validator injects golden + pool images and needs an annotation for
            # each one (golden images drive scoring; pool images feed the commercial
            # dataset). Fine-tuning data comes from the separate labeled training_pool,
            # so the unlabeled annotation images must never be dropped here.
            inference_ids = [spec.image_id for spec in synapse.annotation_images]

            # 5. Run inference
            inference_images = [
                InferImage(
                    image_id=iid,
                    image_path=image_paths[iid],
                )
                for iid in inference_ids
                if iid in image_paths
            ]

            bt.logging.info(
                f"ModelTrainingAnnotationEngine: running inference on "
                f"{len(inference_images)} images"
            )

            annotations_map = self.backend.infer(
                inference_images,
                self._cached_model_version or "pretrained",
            )

            # 6. Assemble annotations.json
            # IMPORTANT: Always emit a record for EVERY image, even when the
            # model found zero objects.  An empty annotations=[] is a valid
            # signal — it means "this image has no hazards."  If this image is
            # a Golden image with zero ground-truth hazards, the validator
            # must reward the miner for the correct "nothing here" call.
            # Dropping empty detections silently would cause golden_missing
            # penalties and distort Bayesian fusion voter counts.
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            image_url_map = {spec.image_id: spec.image_url for spec in synapse.annotation_images}
            records: List[ImageAnnotationDocument] = []
            for image_id, anns in annotations_map.items():
                records.append(
                    ImageAnnotationDocument(
                        image_id=image_id,
                        image_url=image_url_map.get(image_id, ""),
                        model_version=self._cached_model_version or "pretrained0",
                        miner_uid=miner_hotkey,
                        timestamp=ts,
                        annotations=anns,
                    )
                )

            # Also emit empty records for any annotation images that the
            # backend's infer() call didn't return at all (e.g. model crash
            # on a single image).  The validator MUST see every image_id.
            for spec in synapse.annotation_images:
                if spec.image_id not in annotations_map:
                    records.append(
                        ImageAnnotationDocument(
                            image_id=spec.image_id,
                            image_url=spec.image_url,
                            model_version=self._cached_model_version or "pretrained0",
                            miner_uid=miner_hotkey,
                            timestamp=ts,
                            annotations=[],
                        )
                    )

            payload = AnnotationsFilePayload(
                schema_version="annotations.v1",
                task_id=synapse.task_id,
                records=records,
            )

            # 7. Upload to R2
            creds = load_r2_credentials_from_env()
            remote_base = f"{self.r2_prefix.rstrip('/')}/{synapse.task_id}/"
            raw = json.dumps(
                payload.model_dump(), indent=2, sort_keys=True
            ).encode("utf-8")
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

            bt.logging.info(
                f"ModelTrainingAnnotationEngine: task {synapse.task_id} complete — "
                f"{len(records)} images annotated, uri={annotations_uri}"
            )

        except (URLError, OSError, ValueError, RuntimeError, ImportError) as exc:
            bt.logging.error(
                f"ModelTrainingAnnotationEngine: task {synapse.task_id} failed: {exc}"
            )
            synapse.error_message = str(exc)

        synapse.duration_ms = int((time.time() - started) * 1000)
        return synapse

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_request(synapse: AnnotationTask) -> None:
        if not synapse.task_id:
            raise ValueError("task_id is required.")
        if not synapse.annotation_images:
            raise ValueError("annotation_images must be non-empty.")

    def _download_images(self, synapse: AnnotationTask) -> Dict[str, Path]:
        """Download all annotation images to local disk.  Returns id→path mapping."""
        paths: Dict[str, Path] = {}

        all_images = list(synapse.annotation_images)
        # Also include training pool images
        for tp in synapse.training_pool:
            all_images.append(tp)

        for spec in all_images:
            image_id = spec.image_id
            dest = self.image_dir / f"{image_id}.jpg"
            if not dest.exists():
                try:
                    data = _fetch_url_bytes(spec.image_url)
                    dest.write_bytes(data)
                except Exception as exc:
                    bt.logging.warning(
                        f"Failed to download image {image_id}: {exc}"
                    )
                    continue
            paths[image_id] = dest

        bt.logging.info(
            f"ModelTrainingAnnotationEngine: downloaded {len(paths)} images"
        )
        return paths

    def _prepare_training_images(
        self, synapse: AnnotationTask
    ) -> List[TrainImage]:
        """Build TrainImage list from the validator's training pool."""
        train_images: List[TrainImage] = []

        for tp_img in synapse.training_pool:
            img_path = self.image_dir / f"{tp_img.image_id}.jpg"
            if not img_path.exists():
                continue
            labels = list(tp_img.annotations) if tp_img.annotations else None
            train_images.append(
                TrainImage(
                    image_id=tp_img.image_id,
                    image_path=img_path,
                    labels=labels,
                )
            )

        bt.logging.info(
            f"ModelTrainingAnnotationEngine: prepared {len(train_images)} training images "
            f"from training pool"
        )
        return train_images

    def _should_train(self, training_pool_hash: str) -> bool:
        """Decide whether to train or reuse cached model."""
        if self.force_retrain:
            bt.logging.info("ModelTrainingAnnotationEngine: force_retrain=True")
            return True

        if self._cached_training_pool_hash is None:
            # Check persistent cache
            if self.model_cache.has_cached_model(training_pool_hash):
                self._cached_model_version = self.model_cache.get_cached_model_version(
                    training_pool_hash
                )
                self._cached_checkpoint_path = self.model_cache.get_cached_checkpoint_path(
                    training_pool_hash
                )
                self._cached_training_pool_hash = training_pool_hash
                bt.logging.info(
                    "ModelTrainingAnnotationEngine: loaded model from persistent cache"
                )
                return False
            # First run, no cache — need to train
            return True

        if training_pool_hash != self._cached_training_pool_hash:
            bt.logging.info(
                "ModelTrainingAnnotationEngine: training pool changed — retraining"
            )
            return True

        # Same hash, model already trained — skip
        return False

    def _run_training(
        self,
        train_images: List[TrainImage],
        training_pool_hash: str,
    ) -> tuple:
        """Run a single training pass and cache the result."""
        bt.logging.info(
            f"ModelTrainingAnnotationEngine: training with {len(train_images)} images"
        )

        result = self.backend.train(train_images, {})

        # Persist to cache
        self.model_cache.save(
            training_pool_hash=training_pool_hash,
            model_version=result.model_version,
            checkpoint_path=result.checkpoint_path,
        )

        return result.model_version, result.checkpoint_path

    def _run_autoresearch(
        self,
        train_images: List[TrainImage],
        training_pool_hash: str,
    ) -> tuple:
        """Run the auto-research loop and cache the best result."""
        from template.miner.autoresearch import load_search_space, run_autoresearch

        # Check if we have a cached best config
        cached_config = self.model_cache.get_cached_best_config(training_pool_hash)
        if cached_config and not self.force_retrain:
            bt.logging.info(
                "ModelTrainingAnnotationEngine: using cached auto-research config"
            )
            result = self.backend.train(train_images, cached_config)
            return result.model_version, result.checkpoint_path

        # Run full auto-research
        if not self.autoresearch_config_path:
            raise ValueError(
                "--miner.autoresearch_config_path is required when "
                "--miner.enable_autoresearch is set."
            )

        search_space = load_search_space(self.autoresearch_config_path)
        inference_images_out: List[InferImage] = []

        result, best_config = run_autoresearch(
            backend=self.backend,
            all_images=train_images,
            inference_images_out=inference_images_out,
            search_space=search_space,
            max_trials=self.autoresearch_max_trials,
            random_seed=self.split_seed,
            workspace=self.workspace / "autoresearch",
        )

        # Cache the result and best config
        self.model_cache.save(
            training_pool_hash=training_pool_hash,
            model_version=result.model_version,
            checkpoint_path=result.checkpoint_path,
            best_config=best_config,
        )

        return result.model_version, result.checkpoint_path

    def _compute_model_version(self) -> str:
        """Compute a model version hash from config."""
        parts = [
            self.backend_name,
            str(self.split_seed),
            str(self.train_split_pct),
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_pool_hash(training_pool) -> str:
        """Compute hash from training pool when validator doesn't provide one."""
        if not training_pool:
            return hashlib.sha256(b"empty_pool").hexdigest()
        data = [
            {
                "image_id": tp.image_id,
                "annotations": [
                    {"hazard_class": a.hazard_class, "bounding_box": a.bounding_box}
                    for a in tp.annotations
                ],
            }
            for tp in training_pool
        ]
        return ModelCache.compute_training_pool_hash(data)
