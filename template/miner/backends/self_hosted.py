"""Self-hosted backend — delegates training and inference to an external REST API.

This is the universal adapter.  A miner can wrap *any* model (YOLO, DETR,
GPT-4o, Gemini, a custom ensemble) behind three REST endpoints and the
subnet works identically.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import bittensor as bt
import requests

from template.miner.backends.base import (
    BaseModelBackend,
    InferImage,
    TrainImage,
    TrainResult,
)
from template.protocol import PerImageAnnotationItem


class SelfHostedBackend(BaseModelBackend):
    """Delegates to user-provided ``/train``, ``/train/status``, ``/infer`` endpoints."""

    def __init__(self, config: object):
        miner_cfg = getattr(config, "miner", object())

        self.train_url = str(
            getattr(miner_cfg, "self_hosted_train_url", "") or ""
        ).strip().rstrip("/")
        self.infer_url = str(
            getattr(miner_cfg, "self_hosted_infer_url", "") or ""
        ).strip().rstrip("/")
        self.api_key = str(
            getattr(miner_cfg, "self_hosted_api_key", "") or ""
        ).strip()
        self.poll_interval = int(
            getattr(miner_cfg, "self_hosted_poll_interval_seconds", 30)
        )
        self.skip_training = bool(getattr(miner_cfg, "skip_training", False))

        # Validation
        if not self.infer_url:
            raise ValueError(
                "--miner.self_hosted_infer_url is required for the self_hosted backend."
            )

        self._timeout = 600  # HTTP request timeout (seconds)

    # ------------------------------------------------------------------
    # BaseModelBackend interface
    # ------------------------------------------------------------------

    def train(
        self,
        train_images: List[TrainImage],
        config: Dict,
    ) -> TrainResult:
        if self.skip_training or not train_images or not self.train_url:
            bt.logging.info("SelfHostedBackend: skipping training (no-op).")
            return TrainResult(model_version="pretrained", metrics={})

        images_payload = []
        for img in train_images:
            entry: Dict = {
                "image_id": img.image_id,
                "image_url": str(img.image_path),
            }
            if img.labels:
                entry["annotations"] = [
                    {
                        "hazard_class": lbl.hazard_class,
                        "bounding_box": lbl.bounding_box,
                    }
                    for lbl in img.labels
                ]
            images_payload.append(entry)

        body = {
            "images": images_payload,
            "config": config,
        }

        bt.logging.info(
            f"SelfHostedBackend: POST {self.train_url} — {len(train_images)} images"
        )
        resp = self._post(self.train_url, body)
        job_id = resp.get("job_id", "")
        if not job_id:
            raise RuntimeError(
                f"Self-hosted /train did not return a job_id: {resp}"
            )

        bt.logging.info(f"SelfHostedBackend: training job started — job_id={job_id}")

        # Poll for completion
        model_version, metrics = self._poll_training(job_id)

        return TrainResult(
            model_version=model_version,
            metrics=metrics,
        )

    def infer(
        self,
        inference_images: List[InferImage],
        model_version: str,
    ) -> Dict[str, List[PerImageAnnotationItem]]:
        images_payload = [
            {
                "image_id": img.image_id,
                "image_url": str(img.image_path),
            }
            for img in inference_images
        ]

        body = {
            "images": images_payload,
            "model_version": model_version,
        }

        bt.logging.info(
            f"SelfHostedBackend: POST {self.infer_url} — {len(inference_images)} images, "
            f"model_version={model_version[:16]}…"
        )
        resp = self._post(self.infer_url, body)

        raw_annotations = resp.get("annotations", [])
        results: Dict[str, List[PerImageAnnotationItem]] = {}
        for entry in raw_annotations:
            image_id = entry.get("image_id", "")
            hazard_class = entry.get("hazard_class", "")
            bbox = entry.get("bounding_box", [])
            if image_id and hazard_class and len(bbox) == 4:
                results.setdefault(image_id, []).append(
                    PerImageAnnotationItem(
                        hazard_class=hazard_class,
                        bounding_box=[float(v) for v in bbox],
                    )
                )

        # Ensure all requested image_ids have an entry (even if empty)
        for img in inference_images:
            results.setdefault(img.image_id, [])

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, url: str, body: dict) -> dict:
        """POST with retry logic."""
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    url,
                    json=body,
                    headers=self._headers(),
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                wait = 2 ** attempt
                bt.logging.warning(
                    f"SelfHostedBackend: POST {url} attempt {attempt + 1} failed: {exc}. "
                    f"Retrying in {wait}s…"
                )
                time.sleep(wait)
        raise RuntimeError(
            f"SelfHostedBackend: POST {url} failed after 3 attempts: {last_exc}"
        )

    def _poll_training(self, job_id: str) -> tuple:
        """Poll /train/status/{job_id} until completed or failed."""
        status_base = self.train_url.rstrip("/")
        # Derive status URL: /train → /train/status/{job_id}
        status_url = f"{status_base}/status/{job_id}"

        max_polls = 1000  # Safety limit
        for i in range(max_polls):
            try:
                resp = requests.get(
                    status_url,
                    headers=self._headers(),
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                bt.logging.warning(
                    f"SelfHostedBackend: poll {status_url} failed: {exc}"
                )
                time.sleep(self.poll_interval)
                continue

            status = data.get("status", "unknown")
            metrics = data.get("metrics", {})

            if status == "completed":
                model_version = data.get("model_version", "")
                if not model_version:
                    raise RuntimeError(
                        "Self-hosted /train/status returned completed but no model_version."
                    )
                bt.logging.info(
                    f"SelfHostedBackend: training completed — model_version={model_version[:16]}…, "
                    f"metrics={metrics}"
                )
                return model_version, metrics

            if status == "failed":
                error = data.get("error", "unknown error")
                raise RuntimeError(
                    f"Self-hosted training job {job_id} failed: {error}"
                )

            bt.logging.debug(
                f"SelfHostedBackend: training status={status}, "
                f"metrics={metrics}, poll {i + 1}/{max_polls}"
            )
            time.sleep(self.poll_interval)

        raise RuntimeError(
            f"SelfHostedBackend: training job {job_id} did not complete within "
            f"{max_polls * self.poll_interval}s."
        )
