"""Probabilistic, auditable annotation aggregation for commercial export."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import bittensor as bt

from template.hazard.annotation_eval import PerMinerAnnotationScore, iou_xyxy
from template.hazard.image_corpus import ImageCorpus
from template.protocol import PerImageAnnotationItem, R2AccessCredentials

_BACKGROUND_CLASS = "_background"
_DEFAULT_ACCEPT_CONFIDENCE = float(os.getenv("DEFAULT_ACCEPT_CONFIDENCE", "0.9"))
_DEFAULT_ACCEPT_SEVERITY_CONFIDENCE = float(os.getenv("DEFAULT_ACCEPT_SEVERITY_CONFIDENCE", "0.8"))
_DEFAULT_MIN_VOTERS = int(os.getenv("DEFAULT_MIN_VOTERS", "2"))
_DEFAULT_MIN_MEAN_IOU_TO_MEDIAN = float(os.getenv("DEFAULT_MIN_MEAN_IOU_TO_MEDIAN", "0.7"))
_EPS = 1e-9

# Single-miner fallback — adopt annotations from a lone reliable miner
# rather than escalating (which makes the subnet look broken for early adopters).
_FALLBACK_SINGLE_MINER_ENABLED = os.getenv(
    "FALLBACK_SINGLE_MINER_ENABLED", "1"
).strip().lower() in ("1", "true", "yes")
_FALLBACK_SINGLE_MINER_MIN_RELIABILITY = float(
    os.getenv("FALLBACK_SINGLE_MINER_MIN_RELIABILITY", "0.3")
)
_FALLBACK_SINGLE_MINER_AGGREGATION_LABEL = os.getenv(
    "FALLBACK_SINGLE_MINER_AGGREGATION_LABEL", "single_miner_fallback_v1"
).strip()


@dataclass(frozen=True)
class MinerVote:
    miner_uid: int
    miner_hotkey: str
    class_voted: str
    severity_voted: str
    confidence: float
    bounding_box: Optional[Tuple[float, float, float, float]]
    reliability_weight_at_aggregation: float

    def to_jsonable(self) -> dict:
        return {
            "miner_uid": int(self.miner_uid),
            "miner_hotkey": self.miner_hotkey,
            "class_voted": self.class_voted,
            "severity_voted": self.severity_voted,
            "confidence": float(self.confidence),
            "bounding_box": list(self.bounding_box) if self.bounding_box is not None else None,
            "reliability_weight_at_aggregation": float(self.reliability_weight_at_aggregation),
        }


@dataclass(frozen=True)
class AggregatedObject:
    object_cluster_id: str
    accepted_hazard_class: Optional[str]
    accepted_severity: Optional[str]
    confidence: float
    severity_confidence: float
    class_posterior_distribution: Dict[str, float]
    severity_posterior_distribution: Dict[str, float]
    fused_bounding_box: Optional[Tuple[float, float, float, float]]
    spatial_mean_iou_to_median: float
    miner_votes: Sequence[MinerVote]
    escalation_reason: Optional[str]
    aggregation_method: str = "bayesian_dawid_skene_v1"

    def to_jsonable(self) -> dict:
        return {
            "aggregation_method": self.aggregation_method,
            "object_cluster_id": self.object_cluster_id,
            "accepted_hazard_class": self.accepted_hazard_class,
            "accepted_severity": self.accepted_severity,
            "confidence": float(self.confidence),
            "severity_confidence": float(self.severity_confidence),
            "class_posterior_distribution": self.class_posterior_distribution,
            "severity_posterior_distribution": self.severity_posterior_distribution,
            "fused_bounding_box": list(self.fused_bounding_box) if self.fused_bounding_box is not None else None,
            "spatial_mean_iou_to_median": float(self.spatial_mean_iou_to_median),
            "miner_votes": [v.to_jsonable() for v in self.miner_votes],
            "escalation_reason": self.escalation_reason,
        }


@dataclass(frozen=True)
class WinningAnnotation:
    """One image_id aggregation result (accepted or escalated)."""

    image_id: str
    score: float
    chosen_uid: int
    is_golden: bool
    aggregation_method: str
    image_url: str
    width: int
    height: int
    escalation_required: bool
    escalation_reason: Optional[str]
    accepted_objects: Sequence[AggregatedObject]
    miner_contribution_scores: Dict[int, float]
    reliability_window: str
    acceptance_thresholds: Dict[str, float]
    validator_version: str
    timestamp: str
    annotated_image_url: Optional[str] = None

    def to_jsonable(self) -> dict:
        payload = {
            "image_id": self.image_id,
            "score": float(self.score),
            "chosen_uid": int(self.chosen_uid),
            "image_url": self.image_url,
            "annotated_image_url": self.annotated_image_url,
            "width": int(self.width),
            "height": int(self.height),
            "is_golden": bool(self.is_golden),
            "aggregation_method": self.aggregation_method,
            "reliability_window": self.reliability_window,
            "acceptance_thresholds": self.acceptance_thresholds,
            "escalation_required": bool(self.escalation_required),
            "escalation_reason": self.escalation_reason,
            "validator_version": self.validator_version,
            "timestamp": self.timestamp,
            "miner_contribution_scores": {
                str(uid): float(value) for uid, value in self.miner_contribution_scores.items()
            },
            "objects": [obj.to_jsonable() for obj in self.accepted_objects],
        }
        payload["audit_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return payload


@dataclass
class AdoptionLedger:
    """Tracks per-uid adoption counts and contribution-based credits."""

    adoption_counts: Dict[int, int] = field(default_factory=dict)
    last_round_counts: Dict[int, int] = field(default_factory=dict)
    adoption_contributions: Dict[int, float] = field(default_factory=dict)
    last_round_contributions: Dict[int, float] = field(default_factory=dict)
    rounds_observed: int = 0

    def record_round(self, winners: Sequence[WinningAnnotation]) -> None:
        last_counts: Dict[int, int] = {}
        last_contrib: Dict[int, float] = {}
        for winner in winners:
            if winner.escalation_required:
                continue
            self.adoption_counts[winner.chosen_uid] = self.adoption_counts.get(winner.chosen_uid, 0) + 1
            last_counts[winner.chosen_uid] = last_counts.get(winner.chosen_uid, 0) + 1
            for uid, value in winner.miner_contribution_scores.items():
                self.adoption_contributions[uid] = self.adoption_contributions.get(uid, 0.0) + float(value)
                last_contrib[uid] = last_contrib.get(uid, 0.0) + float(value)
        self.last_round_counts = last_counts
        self.last_round_contributions = last_contrib
        self.rounds_observed += 1

    def adoption_share(self) -> Dict[int, float]:
        total = float(sum(self.adoption_counts.values())) or 1.0
        return {uid: count / total for uid, count in self.adoption_counts.items()}

    def round_share(self) -> Dict[int, float]:
        total = float(sum(self.last_round_counts.values())) or 1.0
        return {uid: count / total for uid, count in self.last_round_counts.items()}

    def round_contribution_share(self) -> Dict[int, float]:
        total = float(sum(self.last_round_contributions.values())) or 1.0
        return {uid: value / total for uid, value in self.last_round_contributions.items()}

    def to_jsonable(self) -> dict:
        return {
            "adoption_counts": {str(k): int(v) for k, v in self.adoption_counts.items()},
            "last_round_counts": {str(k): int(v) for k, v in self.last_round_counts.items()},
            "adoption_contributions": {
                str(k): float(v) for k, v in self.adoption_contributions.items()
            },
            "last_round_contributions": {
                str(k): float(v) for k, v in self.last_round_contributions.items()
            },
            "rounds_observed": int(self.rounds_observed),
        }

    @classmethod
    def from_jsonable(cls, payload: dict) -> "AdoptionLedger":
        ledger = cls()
        ledger.adoption_counts = {int(k): int(v) for k, v in payload.get("adoption_counts", {}).items()}
        ledger.last_round_counts = {
            int(k): int(v) for k, v in payload.get("last_round_counts", {}).items()
        }
        ledger.adoption_contributions = {
            int(k): float(v) for k, v in payload.get("adoption_contributions", {}).items()
        }
        ledger.last_round_contributions = {
            int(k): float(v) for k, v in payload.get("last_round_contributions", {}).items()
        }
        ledger.rounds_observed = int(payload.get("rounds_observed", 0))
        return ledger


@dataclass
class DatasetAssembler:
    """Fuses miner annotations probabilistically and emits auditable records."""

    corpus: ImageCorpus
    storage_prefix: str  # file://..., r2://bucket/prefix/, s3://bucket/prefix/
    ledger: AdoptionLedger = field(default_factory=AdoptionLedger)
    draw_boxes: bool = True
    annotated_prefix: str = "commercial/annotated-images/"

    def assemble(
        self,
        *,
        per_miner_scores: Mapping[int, PerMinerAnnotationScore],
        annotations_by_uid: Mapping[int, Mapping[str, Sequence[PerImageAnnotationItem]]],
        miner_hotkeys: Mapping[int, str],
        model_versions: Mapping[int, str],
        timestamps: Mapping[int, str],
    ) -> List[WinningAnnotation]:
        """Aggregate annotations per image with uncertainty gating."""
        all_image_ids: set[str] = set()
        for by_image in annotations_by_uid.values():
            all_image_ids.update(by_image.keys())

        priors = self._class_priors()
        winners: List[WinningAnnotation] = []
        for image_id in sorted(all_image_ids):
            is_golden = self.corpus.is_golden(image_id)
            image_votes = {
                uid: list(by_image.get(image_id, []))
                for uid, by_image in annotations_by_uid.items()
                if by_image.get(image_id) is not None
            }
            width, height = self._image_dims(image_id)
            image_url = self._image_url(image_id)
            if is_golden:
                # Golden rows are scoring-only; keep compact lane.
                best_uid = -1
                best_score = -1.0
                for uid, miner_score in per_miner_scores.items():
                    score = miner_score.fidelity_scores_by_image_id.get(image_id, 0.0)
                    if score > best_score:
                        best_uid = uid
                        best_score = score
                if best_uid >= 0:
                    winners.append(
                        WinningAnnotation(
                            image_id=image_id,
                            score=float(max(0.0, best_score)),
                            chosen_uid=int(best_uid),
                            is_golden=True,
                            aggregation_method="golden_fidelity_v1",
                            image_url=image_url,
                            width=int(width),
                            height=int(height),
                            escalation_required=False,
                            escalation_reason=None,
                            accepted_objects=[],
                            miner_contribution_scores={int(best_uid): 1.0},
                            reliability_window=self._reliability_window(timestamps),
                            acceptance_thresholds=self._acceptance_thresholds(),
                            validator_version=os.getenv("VALIDATOR_VERSION", "1.2.0"),
                            timestamp=str(timestamps.get(best_uid, "")),
                        )
                    )
                continue

            aggregated = self._aggregate_image(
                image_id=image_id,
                image_votes=image_votes,
                per_miner_scores=per_miner_scores,
                miner_hotkeys=miner_hotkeys,
                priors=priors,
            )
            winners.append(
                WinningAnnotation(
                    image_id=image_id,
                    score=float(aggregated["score"]),
                    chosen_uid=int(aggregated["chosen_uid"]),
                    is_golden=False,
                    aggregation_method="bayesian_dawid_skene_v1",
                    image_url=image_url,
                    width=int(width),
                    height=int(height),
                    escalation_required=bool(aggregated["escalation_required"]),
                    escalation_reason=aggregated["escalation_reason"],
                    accepted_objects=aggregated["objects"],
                    miner_contribution_scores=aggregated["miner_contribution_scores"],
                    reliability_window=self._reliability_window(timestamps),
                    acceptance_thresholds=self._acceptance_thresholds(),
                    validator_version=os.getenv("VALIDATOR_VERSION", "1.2.0"),
                    timestamp=str(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                )
            )

        self.ledger.record_round(winners)
        bt.logging.info(
            f"event=dataset_assembled images={len(winners)} "
            f"unique_winners={len({w.chosen_uid for w in winners})}"
        )
        return winners

    def export(
        self,
        winners: Sequence[WinningAnnotation],
        *,
        round_id: str,
        commercial_r2_credentials: Optional[R2AccessCredentials] = None,
    ) -> str:
        """Append the round's winners to the commercial dataset and return its URI.

        Golden-track rows (``is_golden``) are used for scoring and adoption only;
        they are never written to the commercial JSONL so the hidden Golden Set
        cannot leak to customers.

        **Self-contained export**: for each commercial row, the image is uploaded
        to R2 and ``image_url`` in the JSONL is replaced with a permanent,
        publicly-reachable HTTP(S) URL.
        """
        if not winners:
            bt.logging.info("event=dataset_export_skip reason=no_winners")
            return ""

        commercial = [w for w in winners if (not w.is_golden and not w.escalation_required)]
        skipped = len(winners) - len(commercial)
        if skipped:
            bt.logging.info(
                "event=dataset_export_golden_filtered count=%d commercial=%d"
                % (skipped, len(commercial))
            )
        if not commercial:
            bt.logging.info("event=dataset_export_skip reason=no_commercial_rows")
            return ""

        # --- Upload images and rewrite image_url to public HTTP(S) ---
        creds = commercial_r2_credentials
        if creds is None:
            try:
                from template.hazard.r2_storage import load_r2_credentials_from_env
                creds = load_r2_credentials_from_env()
            except RuntimeError:
                creds = None

        rewritten: List[dict] = []
        for w in commercial:
            row_image_url = w.image_url
            row_annotated_image_url = None
            image_path = self.corpus.known_image_path(w.image_id)
            bt.logging.info(
                f"DEBUG DRAW: image_id={w.image_id[:16]} "
                f"draw_boxes={self.draw_boxes} "
                f"image_path={image_path} "
                f"exists={image_path.exists() if image_path else False}"
            )

            # Try to upload the clean image to R2 for a self-contained dataset
            if creds is not None:
                public_url = self._upload_commercial_image(
                    image_id=w.image_id,
                    round_id=round_id,
                    creds=creds,
                )
                if public_url:
                    row_image_url = public_url

            # Try to draw bounding boxes and labels and upload/save annotated version
            if self.draw_boxes and image_path is not None and image_path.exists():
                try:
                    temp_annotated = self._draw_annotations(image_path, w.accepted_objects)

                    # 1. Local copy if local storage prefix (file://) is active
                    parsed_prefix = urlparse(self.storage_prefix or "")
                    if parsed_prefix.scheme == "file":
                        local_annotated_dir = Path(parsed_prefix.path) / self.annotated_prefix
                        local_annotated_dir.mkdir(parents=True, exist_ok=True)
                        local_annotated_path = local_annotated_dir / f"{w.image_id}{image_path.suffix}"
                        import shutil
                        shutil.copy(str(temp_annotated), str(local_annotated_path))
                        row_annotated_image_url = local_annotated_path.as_uri()

                    # 2. Upload to R2 if credentials are provided
                    if creds is not None:
                        try:
                            from template.hazard.r2_storage import upload_image_to_r2
                            object_key = f"{self.annotated_prefix}{w.image_id}{image_path.suffix}"
                            r2_url = upload_image_to_r2(
                                temp_annotated, object_key=object_key, creds=creds
                            )
                            if r2_url:
                                row_annotated_image_url = r2_url
                        except Exception as e:
                            bt.logging.error(f"Failed to upload annotated image to R2: {e}")

                    # Cleanup temporary file
                    if temp_annotated.exists():
                        temp_annotated.unlink()
                except Exception as e:
                    bt.logging.error(f"Error drawing annotations on image {w.image_id}: {e}")

            w_updated = replace(
                w,
                image_url=row_image_url,
                annotated_image_url=row_annotated_image_url,
            )
            rewritten.append(w_updated.to_jsonable())

        payload_lines = "\n".join(
            json.dumps(row, sort_keys=True) for row in rewritten
        )
        body = (payload_lines + "\n").encode("utf-8")

        parsed = urlparse(self.storage_prefix or "")
        if parsed.scheme == "file":
            return self._export_local(body, parsed, round_id)
        if parsed.scheme in ("r2", "s3"):
            if creds is None:
                raise ValueError(
                    "commercial_r2_credentials are required to export to "
                    f"{parsed.scheme}:// storage."
                )
            return self._export_object_storage(
                body, parsed, round_id, creds
            )
        raise ValueError(
            f"Unsupported commercial dataset storage scheme: {self.storage_prefix!r}"
        )

    def _draw_annotations(
        self,
        image_path: Path,
        objects: Sequence[AggregatedObject],
    ) -> Path:
        """Draw bounding boxes and class labels onto a copy of the image, returning its temp path."""
        from PIL import Image as PILImage, ImageDraw, ImageFont
        import tempfile

        with PILImage.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            draw = ImageDraw.Draw(img)
            w_img, h_img = img.size

            try:
                # Use standard system sans-serif font
                font = ImageFont.truetype("DejaVuSans.ttf", 16)
            except Exception:
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except Exception:
                    font = ImageFont.load_default()

            for obj in objects:
                if not obj.accepted_hazard_class or obj.accepted_hazard_class == "_background":
                    continue
                if not obj.fused_bounding_box:
                    continue

                xmin, ymin, xmax, ymax = obj.fused_bounding_box
                # Handle normalized coordinates
                if all(0.0 <= c <= 1.0 for c in (xmin, ymin, xmax, ymax)):
                    xmin *= w_img
                    ymin *= h_img
                    xmax *= w_img
                    ymax *= h_img

                xmin = max(0.0, min(float(w_img), xmin))
                ymin = max(0.0, min(float(h_img), ymin))
                xmax = max(0.0, min(float(w_img), xmax))
                ymax = max(0.0, min(float(h_img), ymax))

                if xmax <= xmin or ymax <= ymin:
                    continue

                color = self._get_color_for_class(obj.accepted_hazard_class)

                # Draw thick bounding box
                for offset in range(3):
                    draw.rectangle(
                        [xmin + offset, ymin + offset, xmax - offset, ymax - offset],
                        outline=color,
                    )

                label = f"{obj.accepted_hazard_class} ({obj.confidence:.2f})"

                try:
                    # In Pillow >= 10.0.0
                    text_bbox = draw.textbbox((0, 0), label, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                except AttributeError:
                    # Legacy Pillow
                    text_w, text_h = draw.textsize(label, font=font)

                text_x1 = xmin
                text_y1 = max(0.0, ymin - text_h - 4)
                text_x2 = min(float(w_img), xmin + text_w + 6)
                text_y2 = min(float(h_img), ymin)

                # Draw filled label background
                draw.rectangle(
                    [text_x1, text_y1, text_x2, text_y2],
                    fill=color,
                )

                brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
                text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)

                draw.text(
                    (text_x1 + 3, text_y1 + 2),
                    label,
                    fill=text_color,
                    font=font,
                )

            temp_dir = tempfile.gettempdir()
            temp_path = Path(temp_dir) / f"annotated_{image_path.name}"
            img.save(temp_path)
            return temp_path

    @staticmethod
    def _get_color_for_class(cls_name: str) -> Tuple[int, int, int]:
        """Generate a deterministic vibrant RGB color for a given class name."""
        h = hashlib.md5(cls_name.encode('utf-8')).digest()
        r = (h[0] * 7 + 13) % 200 + 55
        g = (h[1] * 7 + 13) % 200 + 55
        b = (h[2] * 7 + 13) % 200 + 55
        return (r, g, b)
    # ------------------------------------------------------------------ helpers

    def _upload_commercial_image(
        self,
        *,
        image_id: str,
        round_id: str,
        creds: R2AccessCredentials,
    ) -> str:
        """Upload a single image to R2 for inclusion in the commercial dataset.

        Returns a public HTTP(S) URL, or empty string if the image can't be found.
        Images are uploaded under ``commercial-images/<image_id>.<ext>`` and are
        idempotent — re-uploading the same image_id is a no-op at the R2 level
        (same key overwrites with identical content).
        """
        image_path = self.corpus.known_image_path(image_id)
        if image_path is None or not image_path.exists():
            bt.logging.warning(
                "event=commercial_image_upload_skip image_id=%s reason=not_found",
                image_id[:16],
            )
            return ""
        try:
            from template.hazard.r2_storage import upload_image_to_r2

            object_key = f"commercial-images/{image_id}{image_path.suffix}"
            url = upload_image_to_r2(
                image_path, object_key=object_key, creds=creds
            )
            bt.logging.debug(
                "event=commercial_image_uploaded image_id=%s url=%s",
                image_id[:16], url[:80],
            )
            return url
        except Exception as exc:
            bt.logging.warning(
                "event=commercial_image_upload_error image_id=%s error=%s",
                image_id[:16], exc,
            )
            return ""

    # ------------------------------------------------------------------ helpers (continued)
    def _class_priors(self) -> Dict[str, float]:
        counts: Dict[str, float] = {}
        alpha = 1.1
        for image in self.corpus.golden_images():
            for ann in image.annotations:
                cls = (ann.hazard_class or "").lower().strip()
                if not cls:
                    continue
                counts[cls] = counts.get(cls, 0.0) + 1.0
        classes = sorted(set(counts.keys()) | {_BACKGROUND_CLASS})
        if not classes:
            return {_BACKGROUND_CLASS: 1.0}
        total = sum(counts.get(c, 0.0) + alpha for c in classes)
        return {c: (counts.get(c, 0.0) + alpha) / total for c in classes}

    def _acceptance_thresholds(self) -> Dict[str, float]:
        return {
            "confidence": _DEFAULT_ACCEPT_CONFIDENCE,
            "severity_confidence": _DEFAULT_ACCEPT_SEVERITY_CONFIDENCE,
            "min_voters": float(_DEFAULT_MIN_VOTERS),
            "min_mean_iou_to_median": _DEFAULT_MIN_MEAN_IOU_TO_MEDIAN,
        }

    def _reliability_window(self, timestamps: Mapping[int, str]) -> str:
        values = sorted(v for v in timestamps.values() if v)
        if not values:
            return ""
        return f"{values[0]}/{values[-1]}"

    def _aggregate_image(
        self,
        *,
        image_id: str,
        image_votes: Mapping[int, Sequence[PerImageAnnotationItem]],
        per_miner_scores: Mapping[int, PerMinerAnnotationScore],
        miner_hotkeys: Mapping[int, str],
        priors: Mapping[str, float],
    ) -> dict:
        miner_ids = sorted(image_votes.keys())
        if len(miner_ids) < 2:
            sole_uid = miner_ids[0] if miner_ids else -1
            # --- Single-miner fallback policy ---
            if (
                _FALLBACK_SINGLE_MINER_ENABLED
                and sole_uid >= 0
                and image_votes.get(sole_uid)
            ):
                sole_score = per_miner_scores.get(sole_uid)
                sole_reliability = (
                    sole_score.average_score() if sole_score else 0.0
                )
                if sole_reliability >= _FALLBACK_SINGLE_MINER_MIN_RELIABILITY:
                    # Adopt the lone miner's annotations directly.
                    sole_items = image_votes[sole_uid]
                    fallback_objects: List[AggregatedObject] = []
                    for idx, item in enumerate(sole_items):
                        cls = _safe_class(item.hazard_class)
                        from template.hazard.image_corpus import _severity_for_label
                        sev = _severity_for_label(cls)
                        box = tuple(float(v) for v in item.bounding_box)
                        vote = MinerVote(
                            miner_uid=sole_uid,
                            miner_hotkey=str(miner_hotkeys.get(sole_uid, "")),
                            class_voted=cls,
                            severity_voted=sev,
                            confidence=float(
                                sole_score.weight_for_class(cls)
                                if sole_score else 1e-4
                            ),
                            bounding_box=box,
                            reliability_weight_at_aggregation=float(
                                sole_score.weight_for_class(cls)
                                if sole_score else 1e-4
                            ),
                        )
                        fallback_objects.append(
                            AggregatedObject(
                                object_cluster_id=f"{image_id}-fb-{idx}",
                                accepted_hazard_class=cls,
                                accepted_severity=sev,
                                confidence=sole_reliability,
                                severity_confidence=sole_reliability,
                                class_posterior_distribution={cls: sole_reliability},
                                severity_posterior_distribution={sev: 1.0},
                                fused_bounding_box=box,
                                spatial_mean_iou_to_median=1.0,
                                miner_votes=[vote],
                                escalation_reason=None,
                                aggregation_method=_FALLBACK_SINGLE_MINER_AGGREGATION_LABEL,
                            )
                        )
                    return {
                        "score": float(sole_reliability),
                        "chosen_uid": sole_uid,
                        "objects": fallback_objects,
                        "escalation_required": False,
                        "escalation_reason": None,
                        "miner_contribution_scores": {sole_uid: 1.0},
                    }
            # Fallback disabled or miner below threshold — escalate.
            return {
                "score": 0.0,
                "chosen_uid": sole_uid,
                "objects": [],
                "escalation_required": True,
                "escalation_reason": "only_one_miner",
                "miner_contribution_scores": {},
            }
        clusters = self._cluster_boxes(image_votes, per_miner_scores)
        if not clusters:
            return {
                "score": 0.0,
                "chosen_uid": -1,
                "objects": [],
                "escalation_required": True,
                "escalation_reason": "no_clusters",
                "miner_contribution_scores": {},
            }

        objects: List[AggregatedObject] = []
        contributions: Dict[int, float] = {}
        escalations: List[str] = []
        accepted_confidences: List[float] = []
        for idx, cluster in enumerate(clusters):
            obj, impacts = self._infer_cluster(
                image_id=image_id,
                cluster_id=f"{image_id}-cluster-{idx}",
                cluster_votes=cluster,
                all_miner_ids=miner_ids,
                per_miner_scores=per_miner_scores,
                miner_hotkeys=miner_hotkeys,
                priors=priors,
            )
            objects.append(obj)
            if obj.escalation_reason:
                escalations.append(obj.escalation_reason)
            else:
                accepted_confidences.append(obj.confidence)
                for uid, impact in impacts.items():
                    contributions[uid] = contributions.get(uid, 0.0) + float(impact)
        if escalations:
            return {
                "score": 0.0,
                "chosen_uid": max(contributions.items(), key=lambda x: x[1])[0] if contributions else -1,
                "objects": objects,
                "escalation_required": True,
                "escalation_reason": ";".join(sorted(set(escalations))),
                "miner_contribution_scores": contributions,
            }
        chosen_uid = max(contributions.items(), key=lambda x: x[1])[0] if contributions else -1
        score = float(sum(accepted_confidences) / max(1, len(accepted_confidences)))
        return {
            "score": score,
            "chosen_uid": chosen_uid,
            "objects": objects,
            "escalation_required": False,
            "escalation_reason": None,
            "miner_contribution_scores": contributions,
        }

    def _cluster_boxes(
        self,
        image_votes: Mapping[int, Sequence[PerImageAnnotationItem]],
        per_miner_scores: Mapping[int, PerMinerAnnotationScore],
    ) -> List[List[Tuple[int, PerImageAnnotationItem]]]:
        flat: List[Tuple[int, PerImageAnnotationItem]] = []
        for uid, items in image_votes.items():
            for item in items:
                flat.append((uid, item))

        def _sort_key(pair: tuple[int, PerImageAnnotationItem]) -> float:
            uid, item = pair
            sc = per_miner_scores.get(uid)
            cls = _safe_class(item.hazard_class)
            return float(sc.weight_for_class(cls)) if sc is not None else 1e-4

        flat.sort(key=_sort_key, reverse=True)
        clusters: List[List[Tuple[int, PerImageAnnotationItem]]] = []
        for uid, item in flat:
            assigned = False
            for cluster in clusters:
                anchor = cluster[0][1]
                if iou_xyxy(item.bounding_box, anchor.bounding_box) >= 0.5:
                    cluster.append((uid, item))
                    assigned = True
                    break
            if not assigned:
                clusters.append([(uid, item)])
        return clusters

    def _infer_cluster(
        self,
        *,
        image_id: str,
        cluster_id: str,
        cluster_votes: Sequence[Tuple[int, PerImageAnnotationItem]],
        all_miner_ids: Sequence[int],
        per_miner_scores: Mapping[int, PerMinerAnnotationScore],
        miner_hotkeys: Mapping[int, str],
        priors: Mapping[str, float],
    ) -> tuple[AggregatedObject, Dict[int, float]]:
        vote_by_miner: Dict[int, PerImageAnnotationItem] = {uid: item for uid, item in cluster_votes}
        class_labels = sorted(set(priors.keys()) | {(_safe_class(vote.hazard_class)) for _, vote in cluster_votes} | {_BACKGROUND_CLASS})
        severity_labels = ["none", "low", "medium", "high", "critical"]
        log_probs = {c: math.log(max(_EPS, priors.get(c, _EPS))) for c in class_labels}
        per_miner_votes: List[MinerVote] = []
        for uid in all_miner_ids:
            score = per_miner_scores.get(uid)
            item = vote_by_miner.get(uid)
            if item is None:
                observed_class = _BACKGROUND_CLASS
                observed_severity = "none"
                box = None
            else:
                observed_class = _safe_class(item.hazard_class)
                from template.hazard.image_corpus import _severity_for_label
                observed_severity = _severity_for_label(observed_class)
                box = tuple(float(v) for v in item.bounding_box)
            cls_weight = score.weight_for_class(observed_class) if score is not None else 1e-4
            obs_conf = float(cls_weight)
            per_miner_votes.append(
                MinerVote(
                    miner_uid=uid,
                    miner_hotkey=str(miner_hotkeys.get(uid, "")),
                    class_voted=observed_class,
                    severity_voted=observed_severity,
                    confidence=obs_conf,
                    bounding_box=box,
                    reliability_weight_at_aggregation=cls_weight,
                )
            )
            for true_class in class_labels:
                p = self._p_observed_given_true(observed_class, true_class, len(class_labels), cls_weight)
                log_probs[true_class] += max(1e-4, cls_weight) * math.log(max(_EPS, p))

        class_post = _softmax_dict(log_probs)
        accepted_class, conf = max(class_post.items(), key=lambda kv: kv[1])

        if accepted_class is not None and accepted_class != _BACKGROUND_CLASS:
            from template.hazard.image_corpus import _severity_for_label
            accepted_sev = _severity_for_label(accepted_class)
        else:
            accepted_sev = "none"
        sev_post = {sev: (1.0 if sev == accepted_sev else 0.0) for sev in severity_labels}
        sev_conf = 1.0

        fused_box, mean_iou_to_median, _box_count = self._fuse_box(per_miner_votes)
        escalation_reason = None
        if conf < _DEFAULT_ACCEPT_CONFIDENCE:
            escalation_reason = "low_class_confidence"
        elif len(all_miner_ids) < _DEFAULT_MIN_VOTERS:
            escalation_reason = "insufficient_miners_on_image"
        elif mean_iou_to_median < _DEFAULT_MIN_MEAN_IOU_TO_MEDIAN:
            escalation_reason = "high_spatial_disagreement"

        if escalation_reason is not None:
            accepted_class = None
            accepted_sev = None
            fused_box = None
            conf = 0.0
            sev_conf = 0.0

        impacts: Dict[int, float] = {}
        if escalation_reason is None and accepted_class is not None:
            full_conf = float(class_post.get(accepted_class, 0.0))
            for uid in all_miner_ids:
                reduced = self._posterior_without_uid(
                    uid_to_remove=uid,
                    all_miner_ids=all_miner_ids,
                    vote_by_miner=vote_by_miner,
                    per_miner_scores=per_miner_scores,
                    class_labels=class_labels,
                    priors=priors,
                )
                impacts[uid] = max(0.0, full_conf - float(reduced.get(accepted_class, 0.0)))

        return AggregatedObject(
            object_cluster_id=cluster_id,
            accepted_hazard_class=accepted_class,
            accepted_severity=accepted_sev,
            confidence=float(conf),
            severity_confidence=float(sev_conf),
            class_posterior_distribution=class_post,
            severity_posterior_distribution=sev_post,
            fused_bounding_box=fused_box,
            spatial_mean_iou_to_median=float(mean_iou_to_median),
            miner_votes=per_miner_votes,
            escalation_reason=escalation_reason,
        ), impacts

    @staticmethod
    def _p_observed_given_true(
        observed: str,
        true_label: str,
        class_count: int,
        reliability_weight: float,
    ) -> float:
        r = max(1e-4, min(1.0, reliability_weight))
        p_match = 0.5 + 0.5 * r
        if observed == true_label:
            return p_match
        denom = max(1, class_count - 1)
        return (1.0 - p_match) / denom

    def _posterior_without_uid(
        self,
        *,
        uid_to_remove: int,
        all_miner_ids: Sequence[int],
        vote_by_miner: Mapping[int, PerImageAnnotationItem],
        per_miner_scores: Mapping[int, PerMinerAnnotationScore],
        class_labels: Sequence[str],
        priors: Mapping[str, float],
    ) -> Dict[str, float]:
        log_probs = {c: math.log(max(_EPS, priors.get(c, _EPS))) for c in class_labels}
        for uid in all_miner_ids:
            if uid == uid_to_remove:
                continue
            score = per_miner_scores.get(uid)
            item = vote_by_miner.get(uid)
            if item is None:
                observed_class = _BACKGROUND_CLASS
            else:
                observed_class = _safe_class(item.hazard_class)
            cls_weight = score.weight_for_class(observed_class) if score is not None else 1e-4
            for true_class in class_labels:
                p = self._p_observed_given_true(observed_class, true_class, len(class_labels), cls_weight)
                log_probs[true_class] += max(1e-4, cls_weight) * math.log(max(_EPS, p))
        return _softmax_dict(log_probs)

    @staticmethod
    def _fuse_box(
        votes: Sequence[MinerVote],
    ) -> Tuple[Optional[Tuple[float, float, float, float]], float, int]:
        boxes = [v for v in votes if v.bounding_box is not None]
        if not boxes:
            return None, 0.0, 0
        voters = len(boxes)
        weighted = []
        for v in boxes:
            w = max(1e-4, v.reliability_weight_at_aggregation)
            weighted.append((w, v.bounding_box))
        total_w = sum(w for w, _ in weighted)
        fused = tuple(
            sum(w * box[i] for w, box in weighted) / total_w for i in range(4)
        )
        med = tuple(
            sorted(box[i] for _, box in weighted)[len(weighted) // 2] for i in range(4)
        )
        mean_iou = sum(iou_xyxy(box, med) for _, box in weighted) / max(1, len(weighted))
        return fused, float(mean_iou), voters

    def _image_dims(self, image_id: str) -> tuple[int, int]:
        record = self.corpus.golden_lookup(image_id)
        if record is not None:
            return record.width, record.height
        for unl in self.corpus.annotation_images():
            if unl.image_id == image_id:
                return unl.width, unl.height
        return 0, 0

    def _image_url(self, image_id: str) -> str:
        record = self.corpus.golden_lookup(image_id)
        if record is not None:
            return record.image_url
        for unl in self.corpus.annotation_images():
            if unl.image_id == image_id:
                return unl.image_url
        local = self.corpus.known_image_path(image_id)
        return local.as_uri() if local is not None else ""

    def _export_local(self, body: bytes, parsed, round_id: str) -> str:
        directory = Path(parsed.path)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"commercial-dataset-{round_id}.jsonl"
        target.write_bytes(body)
        master = directory / "commercial-dataset.jsonl"
        with master.open("ab") as handle:
            handle.write(body)
        bt.logging.info(
            f"event=dataset_export_local round={round_id} target={target} "
            f"master={master} bytes={len(body)}"
        )
        return target.as_uri()

    def _export_object_storage(
        self,
        body: bytes,
        parsed,
        round_id: str,
        creds: R2AccessCredentials,
    ) -> str:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise ImportError("boto3 is required for commercial dataset export.") from exc
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        if not bucket:
            raise ValueError(f"Storage prefix missing bucket: {self.storage_prefix}")
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        object_key = f"{prefix}commercial-dataset-{round_id}.jsonl"
        client = boto3.client(
            "s3",
            endpoint_url=creds.s3_endpoint,
            aws_access_key_id=creds.access_key_id,
            aws_secret_access_key=creds.secret_access_key,
            region_name="auto",
        )
        client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=body,
            ContentType="application/x-ndjson",
        )
        uri = f"{parsed.scheme}://{bucket}/{object_key}"
        bt.logging.info(
            f"event=dataset_export_remote round={round_id} uri={uri} bytes={len(body)}"
        )
        return uri


def _safe_class(value: str) -> str:
    return (value or "").lower().strip() or _BACKGROUND_CLASS


def _normalize_dict(values: Mapping[str, float]) -> Dict[str, float]:
    total = float(sum(max(0.0, v) for v in values.values()))
    if total <= 0.0:
        n = max(1, len(values))
        return {k: 1.0 / n for k in values.keys()}
    return {k: float(max(0.0, v) / total) for k, v in values.items()}


def _softmax_dict(logits: Mapping[str, float]) -> Dict[str, float]:
    if not logits:
        return {}
    max_logit = max(logits.values())
    exps = {k: math.exp(v - max_logit) for k, v in logits.items()}
    return _normalize_dict(exps)
