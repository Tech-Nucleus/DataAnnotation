"""OpenAI Vision backend — fine-tunes and runs inference via the OpenAI API.

Uses presigned URLs from the miner's own R2 bucket for image access.
"""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import bittensor as bt

from template.miner.backends.base import (
    BaseModelBackend,
    InferImage,
    TrainImage,
    TrainResult,
)
from template.protocol import PerImageAnnotationItem


class OpenAIVisionBackend(BaseModelBackend):
    """Fine-tune and run inference via OpenAI Chat Completions / Fine-Tuning API.

    Images are provided to OpenAI as presigned URLs (if available) or
    base64 data URIs.  The miner operator must supply an API key via
    ``--miner.openai_api_key``.
    """

    def __init__(self, config: object):
        miner_cfg = getattr(config, "miner", object())

        self.api_key = str(
            getattr(miner_cfg, "openai_api_key", "") or ""
        ).strip()
        if not self.api_key:
            raise ValueError(
                "--miner.openai_api_key is required for the openai_vision backend."
            )

        self.base_model = str(
            getattr(miner_cfg, "openai_base_model", "gpt-4o-2024-08-06")
        ).strip()
        self.base_url = str(getattr(miner_cfg, "openai_base_url", "") or "").strip()
        self.n_epochs = int(getattr(miner_cfg, "openai_n_epochs", 3))
        self.batch_size = int(getattr(miner_cfg, "openai_batch_size", 1))
        self.lr_multiplier = float(
            getattr(miner_cfg, "openai_learning_rate_multiplier", 1.8)
        )
        self.skip_training = bool(getattr(miner_cfg, "skip_training", False))

        # Workspace for temp files
        ws = str(
            getattr(miner_cfg, "annotation_workspace", "artifacts/miner_annotation")
        ).strip()
        self.workspace = Path(ws) / "openai_vision"
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Will hold the fine-tuned model ID after training
        self._fine_tuned_model_id: Optional[str] = None

        # R2 helpers for presigned URLs (optional)
        self._r2_creds = None
        try:
            from template.hazard.r2_storage import load_r2_credentials_from_env

            self._r2_creds = load_r2_credentials_from_env()
        except Exception:
            bt.logging.debug(
                "OpenAIVisionBackend: R2 credentials not available; using base64 for images."
            )

    # ------------------------------------------------------------------
    # BaseModelBackend interface
    # ------------------------------------------------------------------

    def train(
        self,
        train_images: List[TrainImage],
        config: Dict,
    ) -> TrainResult:
        if self.skip_training or not train_images:
            bt.logging.info("OpenAIVisionBackend: skipping training (no-op).")
            self._fine_tuned_model_id = self.base_model
            return TrainResult(model_version=self.base_model, metrics={})

        import openai

        client = self._client(openai)

        # 1. Build JSONL training file
        jsonl_path = self._build_training_jsonl(train_images)

        # 2. Upload file
        bt.logging.info("OpenAIVisionBackend: uploading training file to OpenAI…")
        with open(jsonl_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="fine-tune")
        file_id = file_obj.id
        bt.logging.info(f"OpenAIVisionBackend: uploaded file_id={file_id}")

        # 3. Create fine-tuning job
        hypers = {
            "n_epochs": config.get("n_epochs", self.n_epochs),
            "batch_size": config.get("batch_size", self.batch_size),
            "learning_rate_multiplier": config.get(
                "learning_rate_multiplier", self.lr_multiplier
            ),
        }

        bt.logging.info(
            f"OpenAIVisionBackend: creating fine-tuning job — "
            f"model={self.base_model}, hypers={hypers}"
        )
        job = client.fine_tuning.jobs.create(
            model=self.base_model,
            training_file=file_id,
            hyperparameters=hypers,
        )
        job_id = job.id
        bt.logging.info(f"OpenAIVisionBackend: fine-tuning job created — job_id={job_id}")

        # 4. Poll until complete
        metrics: Dict[str, float] = {}
        while True:
            job = client.fine_tuning.jobs.retrieve(job_id)
            status = job.status

            if status == "succeeded":
                self._fine_tuned_model_id = job.fine_tuned_model
                bt.logging.info(
                    f"OpenAIVisionBackend: fine-tuning succeeded — "
                    f"model_id={self._fine_tuned_model_id}"
                )
                # Log training events
                try:
                    events = client.fine_tuning.jobs.list_events(
                        fine_tuning_job_id=job_id, limit=100
                    )
                    for event in events.data:
                        if hasattr(event, "data") and event.data:
                            for k, v in event.data.items():
                                if isinstance(v, (int, float)):
                                    metrics[k] = float(v)
                except Exception as exc:
                    bt.logging.debug(f"OpenAIVisionBackend: could not fetch events: {exc}")
                break

            if status in ("failed", "cancelled"):
                error = getattr(job, "error", None)
                raise RuntimeError(
                    f"OpenAI fine-tuning job {job_id} {status}: {error}"
                )

            bt.logging.debug(
                f"OpenAIVisionBackend: fine-tuning status={status}, waiting…"
            )
            time.sleep(30)

        return TrainResult(
            model_version=self._fine_tuned_model_id or self.base_model,
            metrics=metrics,
        )

    def infer(
        self,
        inference_images: List[InferImage],
        model_version: str,
    ) -> Dict[str, List[PerImageAnnotationItem]]:
        import openai

        client = self._client(openai)
        model_id = self._fine_tuned_model_id or model_version or self.base_model

        bt.logging.info(
            f"OpenAIVisionBackend: running inference on {len(inference_images)} images "
            f"with model={model_id}"
        )

        results: Dict[str, List[PerImageAnnotationItem]] = {}

        for img in inference_images:
            annotations = self._infer_single(client, model_id, img)
            results[img.image_id] = annotations

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client(self, openai_module):
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return openai_module.OpenAI(**kwargs)

    def _build_training_jsonl(self, images: List[TrainImage]) -> Path:
        """Build an OpenAI vision fine-tuning JSONL file."""
        jsonl_path = self.workspace / "training_data.jsonl"
        lines = []

        for img in images:
            if not img.labels:
                continue

            image_content = self._image_content_block(img.image_path)
            expected_output = json.dumps(
                [
                    {
                        "hazard_class": lbl.hazard_class,
                        "bounding_box": lbl.bounding_box,
                    }
                    for lbl in img.labels
                ]
            )

            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an object detection model. For each image, return "
                            "a JSON array of objects with 'hazard_class' (string) and "
                            "'bounding_box' ([x1, y1, x2, y2] in pixel coordinates)."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            image_content,
                            {"type": "text", "text": "Detect all hazards in this image."},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": expected_output,
                    },
                ]
            }
            lines.append(json.dumps(example))

        jsonl_path.write_text("\n".join(lines) + "\n")
        bt.logging.info(
            f"OpenAIVisionBackend: built training JSONL — {len(lines)} examples"
        )
        return jsonl_path

    def _image_content_block(self, image_path: Path) -> dict:
        """Build an image content block for OpenAI messages.

        Prefers presigned URLs from R2; falls back to base64 data URI.
        """
        # Try presigned URL first
        if self._r2_creds:
            try:
                from template.hazard.r2_storage import (
                    generate_presigned_get_url,
                    upload_bytes_to_r2,
                )

                # Upload to miner's R2 and generate presigned URL
                object_key = f"openai_training_images/{image_path.name}"
                image_data = image_path.read_bytes()
                upload_bytes_to_r2(
                    image_data,
                    object_key=object_key,
                    creds=self._r2_creds,
                    content_type="image/jpeg",
                )
                url = generate_presigned_get_url(
                    creds=self._r2_creds,
                    bucket=self._r2_creds.bucket_name,
                    object_key=object_key,
                    expires_in=3600,
                )
                return {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
            except Exception as exc:
                bt.logging.debug(
                    f"OpenAIVisionBackend: presigned URL failed, falling back to base64: {exc}"
                )

        # Fallback: base64 data URI
        image_data = image_path.read_bytes()
        b64 = base64.b64encode(image_data).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        }

    def _infer_single(
        self,
        client,
        model_id: str,
        img: InferImage,
        max_retries: int = 3,
    ) -> List[PerImageAnnotationItem]:
        """Run inference on a single image with retry logic."""
        image_content = self._image_content_block(img.image_path)

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an object detection model. Return ONLY a JSON array "
                                "of objects, each with 'hazard_class' (string) and "
                                "'bounding_box' ([x1, y1, x2, y2] in pixel coordinates). "
                                "No other text."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                image_content,
                                {
                                    "type": "text",
                                    "text": "Detect all hazards in this image.",
                                },
                            ],
                        },
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                )

                content = response.choices[0].message.content or ""
                return self._parse_detection_response(content)

            except Exception as exc:
                bt.logging.warning(
                    f"OpenAIVisionBackend: inference attempt {attempt + 1} failed "
                    f"for {img.image_id}: {exc}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        bt.logging.error(
            f"OpenAIVisionBackend: all inference attempts failed for {img.image_id}"
        )
        return []

    @staticmethod
    def _parse_detection_response(text: str) -> List[PerImageAnnotationItem]:
        """Parse bounding box detections from OpenAI text response.

        Handles both clean JSON arrays and responses with markdown code
        fences or extra text.
        """
        text = text.strip()

        # Try direct JSON parse first
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return _validate_detections(data)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        code_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if code_match:
            try:
                data = json.loads(code_match.group(1).strip())
                if isinstance(data, list):
                    return _validate_detections(data)
            except json.JSONDecodeError:
                pass

        # Try finding any JSON array in the text
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket_match:
            try:
                data = json.loads(bracket_match.group(0))
                if isinstance(data, list):
                    return _validate_detections(data)
            except json.JSONDecodeError:
                pass

        bt.logging.warning(
            f"OpenAIVisionBackend: could not parse detection response: {text[:200]}…"
        )
        return []


def _validate_detections(data: list) -> List[PerImageAnnotationItem]:
    """Convert raw dicts to validated PerImageAnnotationItem list."""
    items: List[PerImageAnnotationItem] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        hazard_class = entry.get("hazard_class", "")
        bbox = entry.get("bounding_box", [])
        if not hazard_class or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            items.append(
                PerImageAnnotationItem(
                    hazard_class=str(hazard_class),
                    bounding_box=[float(v) for v in bbox],
                )
            )
        except (ValueError, TypeError):
            continue
    return items
