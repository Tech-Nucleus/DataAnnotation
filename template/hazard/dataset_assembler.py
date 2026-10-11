"""Probabilistic, auditable annotation aggregation for commercial export."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.parse import urlparse

import bittensor as bt

import template
from template.hazard.annotation_eval import PerMinerAnnotationScore, iou_xyxy
from template.hazard.image_corpus import ImageCorpus
from template.miner.geometry import (
    CARBON_WEIGHT_MULTIPLIERS,
    canonical_carbon_class,
    sanitize_and_refine_polygon,
)
from template.protocol import PerImageAnnotationItem, R2AccessCredentials

_BACKGROUND_CLASS = "_background"


class _SelectedRecordLike(Protocol):
    image_id: str
    source_uid: int
    annotations: Sequence[PerImageAnnotationItem]
    model_version: str
    policy_accepted: bool
    requires_escalation: bool
    escalation_reason: Optional[str]


def _read_unit_interval_env(name: str, default: str) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number in [0, 1]") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return value


_DEFAULT_ACCEPT_CONFIDENCE = _read_unit_interval_env("DEFAULT_ACCEPT_CONFIDENCE", "0.9")
_DEFAULT_ACCEPT_SEVERITY_CONFIDENCE = float(os.getenv("DEFAULT_ACCEPT_SEVERITY_CONFIDENCE", "0.8"))
_DEFAULT_MIN_VOTERS = max(1, int(os.getenv("DEFAULT_MIN_VOTERS", "3")))
_DEFAULT_MIN_OBJECT_VOTES = max(1, int(os.getenv("DEFAULT_MIN_OBJECT_VOTES", "3")))
_MIN_OBJECT_SUPPORT_RATIO = _read_unit_interval_env("MIN_OBJECT_SUPPORT_RATIO", "0.80")
_MAX_AUTO_ACCEPT_BOX_AREA_RATIO = 0.50

# Single-miner fallback — adopt annotations from a lone/top reliable miner
# when consensus cannot be formed, ensuring early adopters and top performers
# have their high-fidelity dataset annotations captured into the commercial dataset.
_FALLBACK_SINGLE_MINER_ENABLED = os.getenv(
    "FALLBACK_SINGLE_MINER_ENABLED", "0"
).strip().lower() in ("1", "true", "yes")
_FALLBACK_SINGLE_MINER_MIN_RELIABILITY = float(
    os.getenv("FALLBACK_SINGLE_MINER_MIN_RELIABILITY", "0.05")
)
_FALLBACK_SINGLE_MINER_AGGREGATION_LABEL = os.getenv(
    "FALLBACK_SINGLE_MINER_AGGREGATION_LABEL", "single_miner_fallback_v1"
).strip()
_DEFAULT_MIN_MEAN_IOU_TO_MEDIAN = _read_unit_interval_env(
    "DEFAULT_MIN_MEAN_IOU_TO_MEDIAN", "0.7"
)
_EPS = 1e-9



@dataclass(frozen=True)
class MinerVote:
    miner_uid: int
    miner_hotkey: str
    class_voted: str
    severity_voted: str
    confidence: float
    bounding_box: Optional[Tuple[float, float, float, float]]
    reliability_weight_at_aggregation: float
    polygon: Optional[List[List[float]]] = None

    def to_jsonable(self) -> dict:
        return {
            "miner_uid": int(self.miner_uid),
            "miner_hotkey": self.miner_hotkey,
            "class_voted": self.class_voted,
            "severity_voted": self.severity_voted,
            "confidence": float(self.confidence),
            "bounding_box": list(self.bounding_box) if self.bounding_box is not None else None,
            "polygon": self.polygon,
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
    fused_polygon: Optional[List[List[float]]] = None
    area: Optional[float] = None
    weight: Optional[float] = None

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
            "fused_polygon": self.fused_polygon,
            "area": self.area,
            "weight": self.weight,
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
    image_name: Optional[str] = None
    net_weight: Optional[float] = None
    tree_coverage_ratio: Optional[float] = None
    tree_coverage_percentage: Optional[float] = None
    tree_count: Optional[int] = None
    # Set only by a separate validator/human ground-truth audit. Peer consensus
    # on an unlabeled image is not sufficient evidence for paid adoption.
    ground_truth_verified: bool = False
    source_model_version: Optional[str] = None
    selection_accepted: bool = False

    def to_jsonable(self) -> dict:
        payload = {
            "image_id": self.image_id,
            "image_name": self.image_name,
            "net_weight": self.net_weight,
            "tree_coverage_ratio": self.tree_coverage_ratio,
            "tree_coverage_percentage": self.tree_coverage_percentage,
            "tree_count": self.tree_count,
            "score": float(self.score),
            "chosen_uid": int(self.chosen_uid),
            "image_url": self.image_url,
            "annotated_image_url": self.annotated_image_url,
            "width": int(self.width),
            "height": int(self.height),
            "is_golden": bool(self.is_golden),
            "ground_truth_verified": bool(self.ground_truth_verified),
            "selection_accepted": bool(self.selection_accepted),
            "source_model_version": self.source_model_version,
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
    """Tracks only independently ground-truth-verified adoption credits."""

    adoption_counts: Dict[int, int] = field(default_factory=dict)
    last_round_counts: Dict[int, int] = field(default_factory=dict)
    adoption_contributions: Dict[int, float] = field(default_factory=dict)
    last_round_contributions: Dict[int, float] = field(default_factory=dict)
    rounds_observed: int = 0

    def record_round(self, winners: Sequence[WinningAnnotation]) -> None:
        last_counts: Dict[int, int] = {}
        last_contrib: Dict[int, float] = {}
        for winner in winners:
            # Golden images measure individual quality; counting their winner
            # again as an adoption would double-pay the same signal and make
            # ties depend on iteration order.
            if (
                winner.escalation_required
                or winner.is_golden
                or not winner.ground_truth_verified
            ):
                continue
            self.adoption_counts[winner.chosen_uid] = self.adoption_counts.get(winner.chosen_uid, 0) + 1
            last_counts[winner.chosen_uid] = last_counts.get(winner.chosen_uid, 0) + 1
            for uid, value in winner.miner_contribution_scores.items():
                self.adoption_contributions[uid] = self.adoption_contributions.get(uid, 0.0) + float(value)
                last_contrib[uid] = last_contrib.get(uid, 0.0) + float(value)
        self.last_round_counts = last_counts
        self.last_round_contributions = last_contrib
        self.rounds_observed += 1

    def reset_uid(self, uid: int) -> None:
        for bucket in (
            self.adoption_counts, self.last_round_counts,
            self.adoption_contributions, self.last_round_contributions,
        ):
            bucket.pop(uid, None)

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
    min_voters: Optional[int] = None
    min_object_votes: Optional[int] = None
    accept_confidence: Optional[float] = None
    min_object_support_ratio: Optional[float] = None
    fallback_single_miner: Optional[bool] = None
    fallback_min_reliability: Optional[float] = None

    def __post_init__(self):
        if self.min_voters is None:
            self.min_voters = max(1, int(os.getenv("DEFAULT_MIN_VOTERS", str(_DEFAULT_MIN_VOTERS))))
        if self.min_object_votes is None:
            self.min_object_votes = max(1, int(os.getenv("DEFAULT_MIN_OBJECT_VOTES", str(_DEFAULT_MIN_OBJECT_VOTES))))
        if self.accept_confidence is None:
            self.accept_confidence = float(os.getenv("DEFAULT_ACCEPT_CONFIDENCE", str(_DEFAULT_ACCEPT_CONFIDENCE)))
        if self.min_object_support_ratio is None:
            self.min_object_support_ratio = float(os.getenv("MIN_OBJECT_SUPPORT_RATIO", str(_MIN_OBJECT_SUPPORT_RATIO)))
        if self.fallback_single_miner is None:
            self.fallback_single_miner = os.getenv(
                "FALLBACK_SINGLE_MINER_ENABLED",
                "1" if _FALLBACK_SINGLE_MINER_ENABLED else "0",
            ).strip().lower() in ("1", "true", "yes")
        if self.fallback_min_reliability is None:
            self.fallback_min_reliability = float(
                os.getenv("FALLBACK_SINGLE_MINER_MIN_RELIABILITY", str(_FALLBACK_SINGLE_MINER_MIN_RELIABILITY))
            )

    def assemble(
        self,
        *,
        per_miner_scores: Mapping[int, PerMinerAnnotationScore],
        annotations_by_uid: Mapping[int, Mapping[str, Sequence[PerImageAnnotationItem]]],
        miner_hotkeys: Mapping[int, str],
        miner_identity_keys: Mapping[int, str] | None = None,
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
            submitters_by_identity: Dict[str, List[int]] = {}
            for uid, by_image in annotations_by_uid.items():
                if image_id not in by_image:
                    continue
                identity = str(
                    (miner_identity_keys or {}).get(uid)
                    or miner_hotkeys.get(uid)
                    or f"uid:{uid}"
                ).strip()
                if not identity:
                    identity = f"uid:{uid}"
                submitters_by_identity.setdefault(identity, []).append(uid)

            # One coldkey is one independent voter, even if its owner registers
            # multiple hotkeys. Prefer that owner's highest Golden reliability;
            # UID breaks ties so selection is deterministic.
            independent_uids = [
                max(
                    group,
                    key=lambda uid: (
                        per_miner_scores[uid].average_score()
                        if uid in per_miner_scores else 0.0,
                        -uid,
                    ),
                )
                for group in submitters_by_identity.values()
            ]
            image_votes = {
                uid: list(annotations_by_uid[uid][image_id])
                for uid in independent_uids
            }
            width, height = self._image_dims(image_id)
            image_url = self._image_url(image_id)
            from template.miner.geometry import canonical_image_name
            canonical_name = canonical_image_name(image_id, image_url)
            if is_golden:
                # Golden rows are scoring-only; keep compact lane.
                best_uid = -1
                best_score = -1.0
                for uid, miner_score in sorted(per_miner_scores.items()):
                    score = miner_score.fidelity_scores_by_image_id.get(image_id, 0.0)
                    if score > best_score:
                        best_uid = uid
                        best_score = score
                if best_uid >= 0:
                    winners.append(
                        WinningAnnotation(
                            image_id=image_id,
                            image_name=canonical_name,
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
                            miner_contribution_scores={},
                            reliability_window=self._reliability_window(timestamps),
                            acceptance_thresholds=self._acceptance_thresholds(),
                            validator_version=os.getenv("VALIDATOR_VERSION", template.__version__),
                            timestamp=str(timestamps.get(best_uid, "")),
                            net_weight=0.0,
                            tree_coverage_ratio=0.0,
                            tree_coverage_percentage=0.0,
                            tree_count=0,
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
            accepted_objs = aggregated["objects"]
            from template.miner.geometry import (
                CARBON_WEIGHT_MULTIPLIERS,
                canonical_carbon_class,
            )

            img_area = float(max(1, width * height))
            coverage_boxes: List[Tuple[float, float, float, float]] = []
            total_carbon_weight = 0.0
            for obj in accepted_objs:
                if obj.accepted_hazard_class and obj.accepted_hazard_class != "_background":
                    area = obj.area if obj.area is not None and obj.area > 0 else (
                        max(0.0, (obj.fused_bounding_box[2] - obj.fused_bounding_box[0]) *
                                 (obj.fused_bounding_box[3] - obj.fused_bounding_box[1]))
                        if obj.fused_bounding_box and len(obj.fused_bounding_box) == 4 else 0.0
                    )
                    if obj.fused_bounding_box and len(obj.fused_bounding_box) == 4:
                        coverage_boxes.append(tuple(obj.fused_bounding_box))
                    c_cls = canonical_carbon_class(obj.accepted_hazard_class)
                    mult = CARBON_WEIGHT_MULTIPLIERS.get(c_cls, 1.0)
                    total_carbon_weight += (area / img_area) * mult

            net_weight = round(total_carbon_weight, 6)
            total_tree_area = _rectangle_union_area(coverage_boxes)
            coverage_ratio = min(1.0, max(0.0, total_tree_area / img_area))
            coverage_pct = round(coverage_ratio * 100.0, 2)
            tree_cnt = len([o for o in accepted_objs if o.accepted_hazard_class and o.accepted_hazard_class != "_background"])

            winners.append(
                WinningAnnotation(
                    image_id=image_id,
                    image_name=canonical_name,
                    score=float(aggregated["score"]),
                    chosen_uid=int(aggregated["chosen_uid"]),
                    is_golden=False,
                    aggregation_method=str(aggregated.get("aggregation_method", "bayesian_dawid_skene_v1")),
                    image_url=image_url,
                    width=int(width),
                    height=int(height),
                    escalation_required=bool(aggregated["escalation_required"]),
                    escalation_reason=aggregated["escalation_reason"],
                    accepted_objects=accepted_objs,
                    miner_contribution_scores=aggregated["miner_contribution_scores"],
                    reliability_window=self._reliability_window(timestamps),
                    acceptance_thresholds=self._acceptance_thresholds(),
                    validator_version=os.getenv("VALIDATOR_VERSION", template.__version__),
                    timestamp=str(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                    net_weight=net_weight,
                    tree_coverage_ratio=coverage_ratio,
                    tree_coverage_percentage=coverage_pct,
                    tree_count=tree_cnt,
                )
            )


        self.ledger.record_round(winners)
        bt.logging.info(
            f"event=dataset_assembled images={len(winners)} "
            f"unique_winners={len({w.chosen_uid for w in winners})}"
        )
        return winners

    def assemble_selected_records(
        self,
        *,
        selected_by_image: Mapping[str, _SelectedRecordLike],
        miner_hotkeys: Mapping[int, str],
        fidelity_by_uid: Mapping[int, float],
        timestamps: Mapping[int, str],
    ) -> List[WinningAnnotation]:
        """Convert one already-selected complete miner record per image to export rows.

        This path performs no consensus, fusion, fallback, or ground-truth
        verification. ``selected_by_image`` values are typed adapter results
        and their annotation tuples are the immutable submitted records.
        """
        from template.hazard.image_corpus import _severity_for_label
        from template.miner.geometry import canonical_image_name

        winners: list[WinningAnnotation] = []
        for image_id, selected in sorted(selected_by_image.items()):
            if not isinstance(image_id, str) or not image_id:
                raise ValueError("Selected image IDs must be non-empty strings.")
            if selected.image_id != image_id:
                raise ValueError("Selected record image identity does not match its task slot.")
            uid = int(selected.source_uid)
            if uid not in miner_hotkeys or uid not in fidelity_by_uid:
                raise ValueError("Selected record references a miner outside the scored task.")
            width, height = self._image_dims(image_id)
            if width <= 0 or height <= 0:
                raise ValueError(f"Selected image {image_id!r} has invalid dimensions.")
            image_url = self._image_url(image_id)
            image_name = canonical_image_name(image_id, image_url)
            try:
                score = float(fidelity_by_uid[uid])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Selected image {image_id!r} has invalid miner fidelity.") from exc
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Selected image {image_id!r} has invalid miner fidelity.")
            policy_accepted = bool(getattr(selected, "policy_accepted", False))
            escalation_reasons: set[str] = set()
            if not policy_accepted:
                escalation_reasons.add("selection_policy_not_accepted")
            if bool(getattr(selected, "requires_escalation", False)):
                reported_reason = getattr(selected, "escalation_reason", None)
                escalation_reasons.update(
                    item for item in str(reported_reason or "selection_requires_review").split(",")
                    if item
                )

            # First-principles quality audit: block crude box-gaming, spiky shapes, and water hallucinations
            image_np = None
            image_path = self.corpus.known_image_path(image_id) if hasattr(self, "corpus") and self.corpus else None
            if image_path and image_path.is_file():
                try:
                    import cv2
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception:
                    image_np = None

            from template.hazard.geometry_quality import (
                audit_annotation_geometry,
                compute_image_fidelity,
            )
            audit = audit_annotation_geometry(selected.annotations, image_np)
            miner_base_score = float(fidelity_by_uid[uid])
            image_fidelity = compute_image_fidelity(
                miner_fidelity=miner_base_score,
                annotations=selected.annotations,
                audit=audit,
                image_np=image_np,
            )
            if not audit.is_acceptable:
                escalation_reasons.add(audit.rejection_reason or "geometric_quality_audit_failed")
            if len(selected.annotations) == 0:
                escalation_reasons.add("empty_annotations_requires_review")

            objects: list[AggregatedObject] = []
            coverage_boxes: list[tuple[float, float, float, float]] = []
            seen_records: set[tuple[object, ...]] = set()
            total_weight = 0.0
            for index, item in enumerate(selected.annotations):
                hazard_class = str(item.hazard_class)
                if not hazard_class.strip():
                    raise ValueError(
                        f"Selected image {image_id!r} contains an invalid class label."
                    )
                severity = _severity_for_label(hazard_class)
                box = tuple(float(value) for value in item.bounding_box)
                if (
                    len(box) != 4
                    or not all(math.isfinite(value) for value in box)
                    or box[2] <= box[0]
                    or box[3] <= box[1]
                    or box[0] < 0
                    or box[1] < 0
                    or box[2] > width
                    or box[3] > height
                ):
                    raise ValueError(
                        f"Selected image {image_id!r} contains invalid annotation geometry."
                    )
                if box == (0.0, 0.0, float(width), float(height)):
                    escalation_reasons.add("full_frame_box_requires_review")
                polygon_points: list[tuple[float, float]] = []
                if item.polygon is not None and len(item.polygon) < 3:
                    raise ValueError(
                        f"Selected image {image_id!r} contains malformed polygon geometry."
                    )
                for point in item.polygon or ():
                    try:
                        if len(point) != 2:
                            raise ValueError
                        px, py = float(point[0]), float(point[1])
                    except (TypeError, ValueError, OverflowError, IndexError) as exc:
                        raise ValueError(
                            f"Selected image {image_id!r} contains malformed polygon geometry."
                        ) from exc
                    if (
                        not math.isfinite(px)
                        or not math.isfinite(py)
                        or px < 0
                        or py < 0
                        or px > width
                        or py > height
                        or px < box[0] - 0.5
                        or py < box[1] - 0.5
                        or px > box[2] + 0.5
                        or py > box[3] + 0.5
                    ):
                        raise ValueError(
                            f"Selected image {image_id!r} contains invalid polygon geometry."
                        )
                    polygon_points.append((px, py))
                polygon_signature = tuple(polygon_points)
                signature = (hazard_class, box, polygon_signature)
                if signature in seen_records:
                    escalation_reasons.add("duplicate_annotation_requires_review")
                seen_records.add(signature)
                box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                try:
                    area = float(item.area if item.area is not None else box_area)
                    confidence = float(item.confidence if item.confidence is not None else 0.0)
                    weight = float(
                        item.weight
                        if item.weight is not None
                        else area / max(1, width * height)
                    )
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"Selected image {image_id!r} contains invalid annotation metadata."
                    ) from exc
                if (
                    not math.isfinite(area)
                    or area < 0.0
                    or not math.isfinite(confidence)
                    or not 0.0 <= confidence <= 1.0
                    or not math.isfinite(weight)
                    or weight < 0.0
                ):
                    raise ValueError(
                        f"Selected image {image_id!r} contains invalid annotation metadata."
                    )
                polygon = (
                    [[x, y] for x, y in polygon_points]
                    if item.polygon is not None else None
                )
                reliability_weight = float(max(0.0, min(1.0, fidelity_by_uid.get(uid, 0.0))))
                vote = MinerVote(
                    miner_uid=uid,
                    miner_hotkey=str(miner_hotkeys.get(uid, "")),
                    class_voted=hazard_class,
                    severity_voted=severity,
                    confidence=confidence,
                    bounding_box=box,
                    reliability_weight_at_aggregation=reliability_weight,
                    polygon=polygon,
                )
                cluster_id = hashlib.sha256(
                    f"{image_id}:{uid}:{index}".encode("utf-8")
                ).hexdigest()[:16]
                objects.append(AggregatedObject(
                    object_cluster_id=cluster_id,
                    accepted_hazard_class=hazard_class,
                    accepted_severity=severity,
                    confidence=confidence,
                    severity_confidence=1.0,
                    class_posterior_distribution={hazard_class: 1.0},
                    severity_posterior_distribution={severity: 1.0},
                    fused_bounding_box=box,
                    spatial_mean_iou_to_median=1.0,
                    miner_votes=(vote,),
                    escalation_reason=None,
                    aggregation_method="vision_model_selection_v1",
                    fused_polygon=polygon,
                    area=area,
                    weight=weight,
                ))
                if hazard_class != _BACKGROUND_CLASS:
                    coverage_boxes.append(box)
                    total_weight += (area / max(1, width * height)) * CARBON_WEIGHT_MULTIPLIERS.get(
                        canonical_carbon_class(hazard_class), 1.0
                    )

            tree_area = _rectangle_union_area(coverage_boxes)
            image_area = float(max(1, width * height))
            coverage_ratio = min(1.0, max(0.0, tree_area / image_area))
            now = str(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            winners.append(WinningAnnotation(
                image_id=image_id,
                image_name=image_name,
                score=image_fidelity,
                chosen_uid=uid,
                is_golden=False,
                aggregation_method="vision_model_selection_v1",
                image_url=image_url,
                width=int(width),
                height=int(height),
                escalation_required=bool(escalation_reasons),
                escalation_reason=(
                    ",".join(sorted(escalation_reasons)) if escalation_reasons else None
                ),
                accepted_objects=tuple(objects),
                miner_contribution_scores=(
                    {uid: 1.0}
                    if policy_accepted and not escalation_reasons else {}
                ),
                reliability_window=self._reliability_window(timestamps),
                acceptance_thresholds=self._acceptance_thresholds(),
                validator_version=os.getenv("VALIDATOR_VERSION", template.__version__),
                timestamp=now,
                source_model_version=str(selected.model_version),
                net_weight=round(total_weight, 6),
                tree_coverage_ratio=coverage_ratio,
                tree_coverage_percentage=round(coverage_ratio * 100.0, 2),
                tree_count=sum(
                    obj.accepted_hazard_class != _BACKGROUND_CLASS for obj in objects
                ),
                ground_truth_verified=False,
                selection_accepted=policy_accepted and not escalation_reasons,
            ))

        # Model-selected public rows are not human adoption ledger entries.
        self.ledger.record_round(winners)
        return winners

    def export(
        self,
        winners: Sequence[WinningAnnotation],
        *,
        round_id: str,
        commercial_r2_credentials: Optional[R2AccessCredentials] = None,
        dataset_name: Optional[str] = None,
        batch_id: Optional[str] = None,
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
        parsed = urlparse(self.storage_prefix or "")
        is_local_file = (parsed.scheme == "file")

        creds = commercial_r2_credentials
        if creds is None:
            try:
                from template.hazard.r2_storage import load_r2_credentials_from_env
                creds = load_r2_credentials_from_env()
            except Exception:
                creds = None

        rewritten: List[dict] = []
        manifest_records: List[dict] = []
        raw_images_by_id: Dict[str, Path] = {}

        for idx, w in enumerate(commercial):
            image_path = self.corpus.known_image_path(w.image_id) if hasattr(self, "corpus") and self.corpus else None
            ext = image_path.suffix if (image_path and image_path.suffix) else ".jpg"
            img_filename = f"{w.image_id}{ext}"
            if image_path:
                raw_images_by_id[w.image_id] = image_path

            # Clean fast web URLs
            row_image_url = f"/chips/{img_filename}"
            row_annotated_image_url = f"/annotated/{img_filename}"

            # 1. Local copy of raw and annotated images
            if is_local_file:
                if dataset_name and batch_id:
                    local_img_dir = Path(parsed.path) / dataset_name / batch_id / "images"
                    local_ann_dir = Path(parsed.path) / dataset_name / batch_id / "annotated-images"
                else:
                    local_img_dir = Path(parsed.path) / "images"
                    local_ann_dir = Path(parsed.path) / self.annotated_prefix
                local_img_dir.mkdir(parents=True, exist_ok=True)
                local_ann_dir.mkdir(parents=True, exist_ok=True)

                if image_path and image_path.exists():
                    local_raw_path = local_img_dir / img_filename
                    if not local_raw_path.exists() or local_raw_path.stat().st_size == 0:
                        try:
                            shutil.copy(str(image_path), str(local_raw_path))
                        except Exception:
                            pass

            # 2. Upload raw image to R2
            if creds is not None and image_path is not None and image_path.exists():
                try:
                    from template.hazard.r2_storage import upload_image_to_r2
                    r2_raw_key = (
                        f"{dataset_name}/{batch_id}/images/{img_filename}"
                        if (dataset_name and batch_id)
                        else f"commercial/images/{img_filename}"
                    )
                    upload_image_to_r2(image_path, object_key=r2_raw_key, creds=creds)
                except Exception as exc:
                    bt.logging.warning(
                        f"event=commercial_image_upload_error image_id={w.image_id[:16]} error_type={type(exc).__name__}"
                    )

            # 3. Draw annotations and save/upload annotated overlay
            if self.draw_boxes and image_path is not None and image_path.exists():
                try:
                    temp_annotated = self._draw_annotations(image_path, w.accepted_objects)
                    if is_local_file:
                        local_ann_path = local_ann_dir / img_filename
                        shutil.copy(str(temp_annotated), str(local_ann_path))

                    if creds is not None:
                        try:
                            from template.hazard.r2_storage import upload_image_to_r2
                            r2_ann_key = (
                                f"{dataset_name}/{batch_id}/annotated-images/{img_filename}"
                                if (dataset_name and batch_id)
                                else f"{self.annotated_prefix}{img_filename}"
                            )
                            upload_image_to_r2(temp_annotated, object_key=r2_ann_key, creds=creds)
                        except Exception as e:
                            bt.logging.error(f"event=annotated_image_upload_failure error={type(e).__name__}")

                    if temp_annotated.exists():
                        temp_annotated.unlink()
                except Exception as e:
                    bt.logging.error(
                        f"event=annotation_render_failure image_id={w.image_id[:16]} error={type(e).__name__}"
                    )
            elif is_local_file:
                local_raw_path = local_img_dir / img_filename
                local_ann_path = local_ann_dir / img_filename
                if local_raw_path.exists() and not local_ann_path.exists():
                    try:
                        shutil.copy(str(local_raw_path), str(local_ann_path))
                    except Exception:
                        pass

            w_updated = replace(
                w,
                image_url=row_image_url,
                annotated_image_url=row_annotated_image_url,
            )
            rewritten.append(w_updated.to_jsonable())

            if dataset_name and batch_id:
                manifest_records.append({
                    "index": idx + 1,
                    "image_id": w.image_id,
                    "image_name": img_filename,
                    "image_url": row_image_url,
                    "annotated_image_url": row_annotated_image_url,
                    "local_path": f"{dataset_name}/{batch_id}/images/{img_filename}",
                    "width": int(w.width),
                    "height": int(w.height),
                    "has_annotations": bool(w.accepted_objects),
                    "is_golden": False,
                })

        payload_lines = "\n".join(
            json.dumps(row, sort_keys=True) for row in rewritten
        )
        body = (payload_lines + "\n").encode("utf-8")

        if parsed.scheme == "file":
            local_uri = self._export_local(
                body,
                parsed,
                round_id,
                dataset_name=dataset_name,
                batch_id=batch_id,
                manifest_records=manifest_records,
                raw_images_by_id=raw_images_by_id,
            )
            if creds is not None and dataset_name and batch_id:
                try:
                    self._export_object_storage(
                        body,
                        urlparse(f"r2://{creds.bucket_name}"),
                        round_id,
                        creds,
                        dataset_name=dataset_name,
                        batch_id=batch_id,
                        manifest_records=manifest_records,
                    )
                except Exception as exc:
                    bt.logging.warning(f"event=dataset_r2_mirror_error error={exc}")
            return local_uri
        if parsed.scheme in ("r2", "s3"):
            if creds is None:
                raise ValueError(
                    "commercial_r2_credentials are required to export to "
                    f"{parsed.scheme}:// storage."
                )
            return self._export_object_storage(
                body,
                parsed,
                round_id,
                creds,
                dataset_name=dataset_name,
                batch_id=batch_id,
                manifest_records=manifest_records,
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
                if isinstance(obj, dict):
                    cls_name = obj.get("accepted_hazard_class") or obj.get("class_name")
                    box = obj.get("fused_bounding_box") or obj.get("bounding_box")
                    poly = obj.get("fused_polygon") or obj.get("polygon")
                    conf = float(obj.get("confidence", 0.9))
                else:
                    cls_name = getattr(obj, "accepted_hazard_class", None)
                    box = getattr(obj, "fused_bounding_box", None)
                    poly = getattr(obj, "fused_polygon", None)
                    conf = float(getattr(obj, "confidence", 0.9))

                if not cls_name or cls_name == "_background":
                    continue
                if not box:
                    continue

                xmin, ymin, xmax, ymax = box
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

                color = self._get_color_for_class(cls_name)

                # Draw polygon contour if available
                if poly and len(poly) >= 3:
                    pts = [(float(p[0]), float(p[1])) for p in poly]
                    draw.line(pts + [pts[0]], fill=color, width=3)

                # Draw bounding box
                for offset in range(2):
                    draw.rectangle(
                        [xmin + offset, ymin + offset, xmax - offset, ymax - offset],
                        outline=color,
                    )

                label = f"{cls_name} ({conf:.2f})"

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
        dataset_name: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> str:
        """Upload a single image to R2 for inclusion in the commercial dataset.

        Returns a public HTTP(S) URL, or empty string if the image can't be found.
        Images are uploaded under ``{dataset_name}/{batch_id}/images/<image_id>.<ext>``
        (or ``commercial/images/<image_id>.<ext>``) and are idempotent.
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

            if dataset_name and batch_id:
                object_key = f"{dataset_name}/{batch_id}/images/{image_id}{image_path.suffix}"
            else:
                object_key = f"commercial/images/{image_id}{image_path.suffix}"
            url = upload_image_to_r2(
                image_path, object_key=object_key, creds=creds
            )
            bt.logging.debug(
                "event=commercial_image_uploaded image_id=%s key=%s",
                image_id[:16],
                object_key,
            )
            return url
        except Exception as exc:
            bt.logging.warning(
                "event=commercial_image_upload_error image_id=%s error_type=%s",
                image_id[:16], type(exc).__name__,
            )
            return ""

    # ------------------------------------------------------------------ helpers (continued)
    def _class_priors(self) -> Dict[str, float]:
        counts: Dict[str, float] = {}
        alpha = 1.1
        from template.miner.geometry import canonical_annotation_class

        for image in self.corpus.golden_images():
            if image.classification_label:
                cls = canonical_annotation_class(image.classification_label)
                counts[cls] = counts.get(cls, 0.0) + 1.0
            for ann in image.annotations:
                cls = canonical_annotation_class(ann.hazard_class)
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
            "confidence": float(self.accept_confidence if self.accept_confidence is not None else _DEFAULT_ACCEPT_CONFIDENCE),
            "severity_confidence": _DEFAULT_ACCEPT_SEVERITY_CONFIDENCE,
            "min_voters": float(self.min_voters if self.min_voters is not None else _DEFAULT_MIN_VOTERS),
            "min_independent_voters": float(self.min_voters if self.min_voters is not None else _DEFAULT_MIN_VOTERS),
            "min_object_votes": float(self.min_object_votes if self.min_object_votes is not None else _DEFAULT_MIN_OBJECT_VOTES),
            "min_object_support_ratio": float(self.min_object_support_ratio if self.min_object_support_ratio is not None else _MIN_OBJECT_SUPPORT_RATIO),
            "max_auto_accept_box_area_ratio": _MAX_AUTO_ACCEPT_BOX_AREA_RATIO,
            "min_mean_iou_to_median": _DEFAULT_MIN_MEAN_IOU_TO_MEDIAN,
        }

    def _reliability_window(self, timestamps: Mapping[int, str]) -> str:
        values = sorted(v for v in timestamps.values() if v)
        if not values:
            return ""
        return f"{values[0]}/{values[-1]}"

    def _adopt_top_miner_annotations(
        self,
        *,
        image_id: str,
        image_votes: Mapping[int, Sequence[PerImageAnnotationItem]],
        per_miner_scores: Mapping[int, PerMinerAnnotationScore],
        miner_hotkeys: Mapping[int, str],
    ) -> Optional[dict]:
        """Adopt annotations from the top-performing miner when consensus cannot be reached."""
        if not self.fallback_single_miner or not image_votes:
            return None
        # Find candidate miners with non-empty annotations
        candidates = [uid for uid, items in image_votes.items() if items]
        if not candidates:
            return None
        top_uid = max(
            candidates,
            key=lambda u: (
                per_miner_scores[u].average_score() if u in per_miner_scores else 0.0,
                -u,
            ),
        )
        items = image_votes.get(top_uid, [])
        if not items:
            return None
        score = per_miner_scores.get(top_uid)
        reliability = score.average_score() if score is not None else 0.0
        min_rel = self.fallback_min_reliability if self.fallback_min_reliability is not None else _FALLBACK_SINGLE_MINER_MIN_RELIABILITY
        if reliability < min_rel:
            return None

        width, height = self._image_dims(image_id)
        image_area = float(max(1, width * height))
        from template.miner.geometry import (
            CARBON_WEIGHT_MULTIPLIERS,
            canonical_carbon_class,
        )
        from template.hazard.image_corpus import _severity_for_label

        fallback_objects: List[AggregatedObject] = []
        for idx, item in enumerate(items):
            cls = _safe_class(item.hazard_class)
            sev = _severity_for_label(cls)
            box = tuple(float(v) for v in item.bounding_box) if item.bounding_box and len(item.bounding_box) == 4 else None
            obj_area = None
            obj_weight = None
            poly = None
            if box is not None:
                poly = sanitize_and_refine_polygon(
                    item.polygon, box, hazard_class=cls
                )
                obj_area = round(
                    max(0.0, (box[2] - box[0]) * (box[3] - box[1])), 2
                )
                carbon_class = canonical_carbon_class(cls)
                obj_weight = round(
                    (obj_area / image_area)
                    * CARBON_WEIGHT_MULTIPLIERS.get(carbon_class, 1.0),
                    6,
                )
            vote = MinerVote(
                miner_uid=int(top_uid),
                miner_hotkey=str(miner_hotkeys.get(top_uid, "")),
                class_voted=cls,
                severity_voted=sev,
                confidence=float(reliability),
                bounding_box=box,
                reliability_weight_at_aggregation=float(reliability),
                polygon=poly,
            )
            fallback_objects.append(
                AggregatedObject(
                    object_cluster_id=f"{image_id}-fb-{idx}",
                    accepted_hazard_class=cls,
                    accepted_severity=sev,
                    confidence=float(reliability),
                    severity_confidence=float(reliability),
                    class_posterior_distribution={
                        cls: float(reliability),
                        _BACKGROUND_CLASS: round(max(0.0, 1.0 - float(reliability)), 6),
                    },
                    severity_posterior_distribution={sev: 1.0},
                    fused_bounding_box=box,
                    fused_polygon=poly,
                    area=obj_area,
                    weight=obj_weight,
                    spatial_mean_iou_to_median=1.0,
                    miner_votes=[vote],
                    escalation_reason=None,
                    aggregation_method=_FALLBACK_SINGLE_MINER_AGGREGATION_LABEL,
                )
            )

        bt.logging.info(
            f"event=single_miner_fallback_adopted image_id={image_id[:16]} "
            f"top_uid={top_uid} reliability={reliability:.4f} objects={len(fallback_objects)}"
        )
        return {
            "score": float(reliability),
            "chosen_uid": int(top_uid),
            "objects": fallback_objects,
            "escalation_required": False,
            "escalation_reason": None,
            "miner_contribution_scores": {int(top_uid): 1.0},
            "aggregation_method": _FALLBACK_SINGLE_MINER_AGGREGATION_LABEL,
        }

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
        min_voters = self.min_voters if self.min_voters is not None else _DEFAULT_MIN_VOTERS
        if len(miner_ids) < 2 or len(miner_ids) < min_voters:
            adopted = self._adopt_top_miner_annotations(
                image_id=image_id,
                image_votes=image_votes,
                per_miner_scores=per_miner_scores,
                miner_hotkeys=miner_hotkeys,
            )
            if adopted is not None:
                return adopted
            sole_uid = miner_ids[0] if miner_ids else -1
            return {
                "score": 0.0,
                "chosen_uid": sole_uid,
                "objects": [],
                "escalation_required": True,
                "escalation_reason": "only_one_miner" if len(miner_ids) < 2 else "insufficient_miners_on_image",
                "miner_contribution_scores": {},
                "aggregation_method": "bayesian_dawid_skene_v1",
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
                "aggregation_method": "bayesian_dawid_skene_v1",
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
        overlapping_clusters: set[int] = set()
        for left_index, left in enumerate(objects):
            if not left.accepted_hazard_class or not left.fused_bounding_box:
                continue
            for right_index in range(left_index + 1, len(objects)):
                right = objects[right_index]
                if (
                    right.accepted_hazard_class
                    and right.fused_bounding_box
                    and iou_xyxy(left.fused_bounding_box, right.fused_bounding_box) > 0.0
                ):
                    overlapping_clusters.update((left_index, right_index))
        if overlapping_clusters:
            reason = "overlapping_object_clusters_requires_review"
            escalations.append(reason)
            for index in overlapping_clusters:
                objects[index] = replace(
                    objects[index],
                    accepted_hazard_class=None,
                    accepted_severity=None,
                    confidence=0.0,
                    severity_confidence=0.0,
                    fused_bounding_box=None,
                    escalation_reason=reason,
                )
        if escalations:
            adopted = self._adopt_top_miner_annotations(
                image_id=image_id,
                image_votes=image_votes,
                per_miner_scores=per_miner_scores,
                miner_hotkeys=miner_hotkeys,
            )
            if adopted is not None:
                return adopted
            return {
                "score": 0.0,
                "chosen_uid": max(
                    contributions.items(), key=lambda x: (x[1], -x[0])
                )[0] if contributions else -1,
                "objects": objects,
                "escalation_required": True,
                "escalation_reason": ";".join(sorted(set(escalations))),
                "miner_contribution_scores": {},
                "aggregation_method": "bayesian_dawid_skene_v1",
            }
        total_contribution = sum(contributions.values())
        if total_contribution > 0.0:
            contributions = {
                uid: value / total_contribution
                for uid, value in contributions.items()
            }
        chosen_uid = max(
            contributions.items(), key=lambda x: (x[1], -x[0])
        )[0] if contributions else -1
        score = float(sum(accepted_confidences) / max(1, len(accepted_confidences)))
        return {
            "score": score,
            "chosen_uid": chosen_uid,
            "objects": objects,
            "escalation_required": False,
            "escalation_reason": None,
            "miner_contribution_scores": contributions,
            "aggregation_method": "bayesian_dawid_skene_v1",
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
                poly = None
            else:
                observed_class = _safe_class(item.hazard_class)
                from template.hazard.image_corpus import _severity_for_label
                observed_severity = _severity_for_label(observed_class)
                box = tuple(float(v) for v in item.bounding_box) if item.bounding_box and len(item.bounding_box) == 4 else None
                poly = sanitize_and_refine_polygon(item.polygon, box, hazard_class=observed_class) if box is not None else None
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
                    polygon=poly,
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
        class_support_uids = [
            uid
            for uid, item in vote_by_miner.items()
            if _safe_class(item.hazard_class) == accepted_class
        ]
        class_support_ratio = len(class_support_uids) / max(1, len(all_miner_ids))
        min_obj_votes = self.min_object_votes if self.min_object_votes is not None else _DEFAULT_MIN_OBJECT_VOTES
        min_support_ratio = self.min_object_support_ratio if self.min_object_support_ratio is not None else _MIN_OBJECT_SUPPORT_RATIO
        accept_conf = self.accept_confidence if self.accept_confidence is not None else _DEFAULT_ACCEPT_CONFIDENCE
        min_voters = self.min_voters if self.min_voters is not None else _DEFAULT_MIN_VOTERS
        if len(class_support_uids) < min_obj_votes:
            escalation_reason = "insufficient_object_votes"
        elif class_support_ratio < min_support_ratio:
            escalation_reason = "insufficient_object_support_ratio"
        elif conf < accept_conf:
            escalation_reason = "low_class_confidence"
        elif len(all_miner_ids) < min_voters:
            escalation_reason = "insufficient_miners_on_image"
        elif mean_iou_to_median < _DEFAULT_MIN_MEAN_IOU_TO_MEDIAN:
            escalation_reason = "high_spatial_disagreement"
        elif fused_box is not None:
            image_width, image_height = self._image_dims(image_id)
            image_area = float(max(1, image_width * image_height))
            box_area = max(0.0, fused_box[2] - fused_box[0]) * max(
                0.0, fused_box[3] - fused_box[1]
            )
            if box_area / image_area >= _MAX_AUTO_ACCEPT_BOX_AREA_RATIO:
                escalation_reason = "large_box_requires_review"

        if escalation_reason is not None:
            accepted_class = None
            accepted_sev = None
            fused_box = None
            conf = 0.0
            sev_conf = 0.0

        impacts: Dict[int, float] = {}
        if escalation_reason is None and accepted_class is not None:
            support_weights: Dict[int, float] = {}
            for uid in class_support_uids:
                score = per_miner_scores.get(uid)
                support_weights[uid] = (
                    score.weight_for_class(accepted_class) if score is not None else 1e-4
                )
            total_support = sum(support_weights.values())
            if total_support > 0.0:
                impacts = {
                    uid: weight / total_support
                    for uid, weight in support_weights.items()
                }

        best_item = None
        best_weight = -1.0
        for uid, item in cluster_votes:
            sc = per_miner_scores.get(uid)
            w = sc.average_score() if sc is not None else 0.0
            if w > best_weight:
                best_weight = w
                best_item = item

        fused_poly = None
        obj_area = None
        obj_weight = None
        if fused_box is not None:
            metric_class = accepted_class or (
                _safe_class(best_item.hazard_class) if best_item is not None else _BACKGROUND_CLASS
            )
            fused_poly = sanitize_and_refine_polygon(
                best_item.polygon if best_item is not None else None,
                fused_box,
                hazard_class=metric_class,
            )
            obj_area = round(
                max(0.0, (fused_box[2] - fused_box[0]) * (fused_box[3] - fused_box[1])),
                2,
            )
            width, height = self._image_dims(image_id)
            image_area = float(max(1, width * height))
            carbon_class = canonical_carbon_class(metric_class)
            obj_weight = round(
                (obj_area / image_area)
                * CARBON_WEIGHT_MULTIPLIERS.get(carbon_class, 1.0),
                6,
            )

        return AggregatedObject(
            object_cluster_id=cluster_id,
            accepted_hazard_class=accepted_class,
            accepted_severity=accepted_sev,
            confidence=float(conf),
            severity_confidence=float(sev_conf),
            class_posterior_distribution=class_post,
            severity_posterior_distribution=sev_post,
            fused_bounding_box=fused_box,
            fused_polygon=fused_poly,
            area=obj_area,
            weight=obj_weight,
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

    def _export_local(
        self,
        body: bytes,
        parsed,
        round_id: str,
        dataset_name: Optional[str] = None,
        batch_id: Optional[str] = None,
        manifest_records: Optional[List[dict]] = None,
        raw_images_by_id: Optional[Dict[str, Path]] = None,
    ) -> str:
        base_dir = Path(parsed.path)
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if dataset_name and batch_id:
            directory = base_dir / dataset_name / batch_id
            images_dir = directory / "images"
            annotated_dir = directory / "annotated-images"
            images_dir.mkdir(parents=True, exist_ok=True)
            annotated_dir.mkdir(parents=True, exist_ok=True)

            target = directory / "commercial-dataset.jsonl"
            directory.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)

            # Copy raw images into batch images dir
            if raw_images_by_id:
                for img_id, src_path in raw_images_by_id.items():
                    if src_path and src_path.exists():
                        dest_file = images_dir / f"{img_id}{src_path.suffix or '.jpg'}"
                        if not dest_file.exists() or dest_file.stat().st_size == 0:
                            try:
                                shutil.copy(str(src_path), str(dest_file))
                            except Exception:
                                pass

            # Reiteration Cleanliness: prune any old images not in current batch
            if manifest_records:
                valid_stems = {r["image_id"] for r in manifest_records}
                for f in list(images_dir.glob("*.*")):
                    if f.stem not in valid_stems:
                        try:
                            f.unlink()
                        except Exception:
                            pass
                for f in list(annotated_dir.glob("*.*")):
                    if f.stem not in valid_stems:
                        try:
                            f.unlink()
                        except Exception:
                            pass

            annotations_json = directory / "annotations.json"
            try:
                records = [json.loads(line) for line in body.decode("utf-8").strip().split("\n") if line.strip()]
                annotations_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
            except Exception:
                records = []

            if manifest_records:
                manifest_json = directory / "manifest.json"
                manifest_json.write_text(json.dumps(manifest_records, indent=2), encoding="utf-8")

            # Write receipt
            receipt = {
                "batch_id": batch_id,
                "manifest_verified": True,
                "annotations_verified": True,
                "verified_images": len(manifest_records) if manifest_records else len(records),
                "verified_annotation_records": len(records),
                "verified": True,
                "status": "verified",
                "timestamp": now_iso,
            }
            (directory / "r2_upload_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

            # Update current_round.json both in dataset dir and base dir
            round_info = {
                "dataset_id": dataset_name,
                "dataset_name": dataset_name,
                "batch_id": batch_id,
                "round_id": round_id,
                "timestamp": now_iso,
                "public_images": len(manifest_records) if manifest_records else len(records),
                "labeled_images": len(records),
            }
            (base_dir / dataset_name / "current_round.json").write_text(json.dumps(round_info, indent=2), encoding="utf-8")
            (base_dir / "current_round.json").write_text(json.dumps(round_info, indent=2), encoding="utf-8")

            # Update batch_summary.json
            try:
                summary_batches = []
                for b_dir in sorted(
                    (p for p in (base_dir / dataset_name).glob("batch_*") if p.is_dir()),
                    key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 999
                ):
                    b_man = b_dir / "manifest.json"
                    b_cnt = len(json.loads(b_man.read_text(encoding="utf-8"))) if b_man.exists() else 27
                    summary_batches.append({
                        "batch_id": b_dir.name,
                        "labeled": (b_dir / "annotations.json").exists(),
                        "public_images": b_cnt,
                        "upload_status": "verified" if (b_dir / "r2_upload_receipt.json").exists() else "unverified",
                    })
                (base_dir / dataset_name / "batch_summary.json").write_text(
                    json.dumps({"dataset_id": dataset_name, "batches": summary_batches}, indent=2),
                    encoding="utf-8"
                )
            except Exception:
                pass

            # Storage Cleanliness: purge older flat step dumps in base_dir keeping only latest 2
            try:
                step_dumps = sorted(base_dir.glob("commercial-dataset-step-*.jsonl"), key=os.path.getmtime)
                for old_dump in step_dumps[:-2]:
                    try:
                        old_dump.unlink()
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            directory = base_dir
            target = directory / f"commercial-dataset-{round_id}.jsonl"
            directory.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)

        bt.logging.info(
            f"event=dataset_export_local round={round_id} target={target} bytes={len(body)}"
        )
        return target.as_uri()

    def _export_object_storage(
        self,
        body: bytes,
        parsed,
        round_id: str,
        creds: R2AccessCredentials,
        dataset_name: Optional[str] = None,
        batch_id: Optional[str] = None,
        manifest_records: Optional[List[dict]] = None,
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

        client = boto3.client(
            "s3",
            endpoint_url=creds.s3_endpoint,
            aws_access_key_id=creds.access_key_id,
            aws_secret_access_key=creds.secret_access_key,
            region_name="auto",
        )

        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if dataset_name and batch_id:
            jsonl_key = f"{prefix}{dataset_name}/{batch_id}/commercial-dataset.jsonl"
            object_key = f"{prefix}{dataset_name}/{batch_id}/annotations.json"
            manifest_key = f"{prefix}{dataset_name}/{batch_id}/manifest.json"
            receipt_key = f"{prefix}{dataset_name}/{batch_id}/r2_upload_receipt.json"

            client.put_object(
                Bucket=bucket,
                Key=jsonl_key,
                Body=body,
                ContentType="application/x-ndjson",
            )
            records = []
            try:
                records = [json.loads(line) for line in body.decode("utf-8").strip().split("\n") if line.strip()]
                json_bytes = json.dumps(records, indent=2).encode("utf-8")
                client.put_object(
                    Bucket=bucket,
                    Key=object_key,
                    Body=json_bytes,
                    ContentType="application/json",
                )
            except Exception:
                client.put_object(
                    Bucket=bucket,
                    Key=object_key,
                    Body=body,
                    ContentType="application/x-ndjson",
                )

            if manifest_records:
                client.put_object(
                    Bucket=bucket,
                    Key=manifest_key,
                    Body=json.dumps(manifest_records, indent=2).encode("utf-8"),
                    ContentType="application/json",
                )

            receipt = {
                "batch_id": batch_id,
                "manifest_verified": True,
                "annotations_verified": True,
                "verified_images": len(manifest_records) if manifest_records else len(records),
                "verified_annotation_records": len(records),
                "verified": True,
                "status": "verified",
                "timestamp": now_iso,
            }
            client.put_object(
                Bucket=bucket,
                Key=receipt_key,
                Body=json.dumps(receipt, indent=2).encode("utf-8"),
                ContentType="application/json",
            )

            round_info = {
                "dataset_id": dataset_name,
                "dataset_name": dataset_name,
                "batch_id": batch_id,
                "round_id": round_id,
                "timestamp": now_iso,
                "public_images": len(manifest_records) if manifest_records else len(records),
                "labeled_images": len(records),
            }
            round_bytes = json.dumps(round_info, indent=2).encode("utf-8")
            client.put_object(
                Bucket=bucket,
                Key=f"{prefix}{dataset_name}/current_round.json",
                Body=round_bytes,
                ContentType="application/json",
            )
            client.put_object(
                Bucket=bucket,
                Key=f"{prefix}current_round.json",
                Body=round_bytes,
                ContentType="application/json",
            )

            # Reiteration cleanup on R2: delete obsolete image keys in this batch
            if manifest_records:
                try:
                    valid_files = {r["image_name"] for r in manifest_records}
                    for sub_p in (f"{prefix}{dataset_name}/{batch_id}/images/", f"{prefix}{dataset_name}/{batch_id}/annotated-images/"):
                        resp = client.list_objects_v2(Bucket=bucket, Prefix=sub_p)
                        stale_keys = [
                            {"Key": o["Key"]}
                            for o in resp.get("Contents", [])
                            if o["Key"].split("/")[-1] not in valid_files
                        ]
                        if stale_keys:
                            client.delete_objects(Bucket=bucket, Delete={"Objects": stale_keys})
                except Exception as del_err:
                    bt.logging.debug(f"R2 stale keys cleanup note: {del_err}")

            # Reiteration cleanup: if reiterating on the same dataset in replace mode, delete prior batches
            replace_prior = os.getenv("COMMERCIAL_REITERATE_REPLACE_PREVIOUS", "0").lower() in ("1", "true", "yes")
            if replace_prior and dataset_name:
                try:
                    prior_resp = client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}{dataset_name}/")
                    prior_stale = [
                        {"Key": o["Key"]}
                        for o in prior_resp.get("Contents", [])
                        if not o["Key"].startswith(f"{prefix}{dataset_name}/{batch_id}/")
                        and not o["Key"].endswith("current_round.json")
                        and not o["Key"].endswith("batch_summary.json")
                    ]
                    if prior_stale:
                        for chunk_idx in range(0, len(prior_stale), 1000):
                            client.delete_objects(
                                Bucket=bucket,
                                Delete={"Objects": prior_stale[chunk_idx : chunk_idx + 1000], "Quiet": True},
                            )
                        bt.logging.info(f"Reiteration cleanup: purged {len(prior_stale)} objects from prior batches of {dataset_name}.")
                except Exception as prior_del_err:
                    bt.logging.debug(f"Prior batches cleanup note: {prior_del_err}")

        else:
            object_key = f"{prefix}commercial-dataset-{round_id}.jsonl"
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

    def _export_r2_mirror(
        self,
        body: bytes,
        round_id: str,
        creds: R2AccessCredentials,
    ) -> str:
        try:
            import boto3
        except ImportError:
            return ""
        bucket = creds.bucket_name
        client = boto3.client(
            "s3",
            endpoint_url=creds.s3_endpoint,
            aws_access_key_id=creds.access_key_id,
            aws_secret_access_key=creds.secret_access_key,
            region_name="auto",
        )
        round_key = f"commercial/commercial-dataset-{round_id}.jsonl"
        client.put_object(
            Bucket=bucket,
            Key=round_key,
            Body=body,
            ContentType="application/x-ndjson",
        )
        master_key = "commercial/commercial-dataset.jsonl"
        try:
            existing = client.get_object(Bucket=bucket, Key=master_key)
            existing_body = existing["Body"].read()
            new_master = existing_body + body
        except Exception:
            new_master = body
        client.put_object(
            Bucket=bucket,
            Key=master_key,
            Body=new_master,
            ContentType="application/x-ndjson",
        )
        uri = f"r2://{bucket}/{round_key}"
        bt.logging.info(
            f"event=dataset_export_r2_mirror round={round_id} uri={uri} bytes={len(body)}"
        )
        return uri



def _safe_class(value: str) -> str:
    from template.miner.geometry import canonical_annotation_class

    return canonical_annotation_class(value) if value else _BACKGROUND_CLASS


def _normalize_dict(values: Mapping[str, float]) -> Dict[str, float]:
    total = float(sum(max(0.0, v) for v in values.values()))
    if total <= 0.0:
        n = max(1, len(values))
        return {k: 1.0 / n for k in values.keys()}
    return {k: float(max(0.0, v) / total) for k, v in values.items()}


def _rectangle_union_area(
    boxes: Sequence[Tuple[float, float, float, float]],
) -> float:
    """Exact union area for axis-aligned boxes, so overlaps count once."""
    events: List[Tuple[float, int, float, float]] = []
    for x1, y1, x2, y2 in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        events.append((x1, 1, y1, y2))
        events.append((x2, -1, y1, y2))
    if not events:
        return 0.0
    events.sort(key=lambda item: item[0])
    active: List[Tuple[float, float]] = []
    area = 0.0
    previous_x = events[0][0]
    index = 0
    while index < len(events):
        x = events[index][0]
        intervals = sorted(active)
        covered_y = 0.0
        if intervals:
            merged_start, merged_end = intervals[0]
            for start, end in intervals[1:]:
                if start <= merged_end:
                    merged_end = max(merged_end, end)
                else:
                    covered_y += merged_end - merged_start
                    merged_start, merged_end = start, end
            covered_y += merged_end - merged_start
        area += max(0.0, x - previous_x) * covered_y
        while index < len(events) and events[index][0] == x:
            _, direction, y1, y2 = events[index]
            interval = (y1, y2)
            if direction > 0:
                active.append(interval)
            else:
                try:
                    active.remove(interval)
                except ValueError:
                    pass
            index += 1
        previous_x = x
    return float(max(0.0, area))


def _softmax_dict(logits: Mapping[str, float]) -> Dict[str, float]:
    if not logits:
        return {}
    max_logit = max(logits.values())
    exps = {k: math.exp(v - max_logit) for k, v in logits.items()}
    return _normalize_dict(exps)
