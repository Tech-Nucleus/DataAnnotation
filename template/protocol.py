from __future__ import annotations

from typing import List, Literal, Optional

import bittensor as bt
import math

from pydantic import BaseModel, Field, field_validator, model_validator

SeverityTier = Literal["none", "low", "medium", "high", "critical"]


def _orientation(a: List[float], b: List[float], c: List[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: List[float], b: List[float], p: List[float]) -> bool:
    return (
        min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def _segments_intersect(
    a: List[float], b: List[float], c: List[float], d: List[float]
) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if ((o1 > 0 > o2) or (o2 > 0 > o1)) and ((o3 > 0 > o4) or (o4 > 0 > o3)):
        return True
    return (
        (o1 == 0 and _on_segment(a, b, c))
        or (o2 == 0 and _on_segment(a, b, d))
        or (o3 == 0 and _on_segment(c, d, a))
        or (o4 == 0 and _on_segment(c, d, b))
    )


class UnlabeledAnnotationImage(BaseModel):
    image_url: str = Field(..., min_length=1)
    image_id: str = Field(..., min_length=1)


class PerImageAnnotationItem(BaseModel):
    hazard_class: str = Field(..., min_length=1)
    bounding_box: List[float] = Field(..., min_length=4, max_length=4)
    polygon: Optional[List[List[float]]] = Field(
        None,
        max_length=256,
        description="Optional polygon contour coordinates [[x,y],...]",
    )
    area: Optional[float] = Field(None, description="Annotated object area in pixels")
    weight: Optional[float] = Field(None, description="Ratio of object area to total image area")
    confidence: Optional[float] = Field(None, description="Detection confidence score in [0, 1]")

    @field_validator("bounding_box")
    @classmethod
    def validate_bounding_box(cls, value: List[float]) -> List[float]:
        if len(value) != 4 or not all(math.isfinite(v) for v in value):
            raise ValueError("bounding_box must contain four finite coordinates")
        x1, y1, x2, y2 = value
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bounding_box must have positive width and height")
        return value

    @field_validator("area", "weight", "confidence")
    @classmethod
    def validate_finite_metadata(cls, value: Optional[float], info) -> Optional[float]:
        if value is None:
            return value
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{info.field_name} must be finite and non-negative")
        if info.field_name == "confidence" and value > 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return value

    @field_validator("polygon")
    @classmethod
    def validate_polygon(
        cls, value: Optional[List[List[float]]]
    ) -> Optional[List[List[float]]]:
        if value is None:
            return value
        if len(value) < 3 or any(
            len(point) != 2 or not all(math.isfinite(coord) for coord in point)
            for point in value
        ):
            raise ValueError("polygon must contain at least three finite [x, y] points")
        return value

    @model_validator(mode="after")
    def validate_polygon_within_box(self) -> "PerImageAnnotationItem":
        if self.polygon is None:
            return self
        x1, y1, x2, y2 = self.bounding_box
        eps = 0.5
        if any(
            x < x1 - eps or y < y1 - eps or x > x2 + eps or y > y2 + eps
            for x, y in self.polygon
        ):
            raise ValueError("polygon vertices must lie within bounding_box")
        twice_area = abs(
            sum(
                self.polygon[index][0]
                * self.polygon[(index + 1) % len(self.polygon)][1]
                - self.polygon[(index + 1) % len(self.polygon)][0]
                * self.polygon[index][1]
                for index in range(len(self.polygon))
            )
        )
        if not math.isfinite(twice_area) or twice_area <= 0.0:
            raise ValueError("polygon must enclose positive finite area")
        count = len(self.polygon)
        for i in range(count):
            a, b = self.polygon[i], self.polygon[(i + 1) % count]
            for j in range(i + 1, count):
                # Adjacent edges share a vertex by definition and are valid.
                if j == i or j == (i + 1) % count or i == (j + 1) % count:
                    continue
                c, d = self.polygon[j], self.polygon[(j + 1) % count]
                if _segments_intersect(a, b, c, d):
                    raise ValueError("polygon must be a simple non-self-intersecting contour")
        return self


class LabeledTrainingImage(BaseModel):
    """A labeled image from the validator's public Training Pool.

    Miners may use these for fine-tuning.  The Training Pool is separate
    from the hidden Golden Set and is shared openly with every miner.
    """

    image_url: str = Field(..., min_length=1)
    image_id: str = Field(..., min_length=1)
    annotations: List[PerImageAnnotationItem] = Field(default_factory=list)


class ImageAnnotationDocument(BaseModel):
    image_id: str
    image_url: str = Field(..., min_length=1)
    miner_uid: str
    timestamp: str
    annotations: List[PerImageAnnotationItem] = Field(..., max_length=512)
    model_version: str = Field(..., min_length=8)
    image_name: Optional[str] = Field(None, description="Canonical image filename (e.g. climate_raw_042.jpg)")
    net_weight: Optional[float] = Field(None, description="Net tree coverage ratio in [0, 1]")
    tree_coverage_ratio: Optional[float] = Field(None, description="Synonym for net_weight")
    tree_coverage_percentage: Optional[float] = Field(None, description="Net tree coverage percentage (0-100%)")
    tree_count: Optional[int] = Field(None, description="Total number of trees/clusters detected in image")


class AnnotationsFilePayload(BaseModel):
    schema_version: str = Field("annotations.v1", min_length=1)
    task_id: str = Field("", min_length=0)
    batch_id: str = Field("", description="Active batch identifier matching task")
    round_num: int = Field(0, description="Active sequential round number matching task")
    records: List[ImageAnnotationDocument] = Field(..., max_length=8192)


class R2AccessCredentials(BaseModel):
    account_id: str = Field(..., min_length=4)
    bucket_name: str = Field(..., min_length=1)
    s3_endpoint: str = Field(..., min_length=8)
    access_key_id: str = Field(..., min_length=4)
    secret_access_key: str = Field(..., min_length=8)
    token: Optional[str] = Field(None)
    public_bucket_url: Optional[str] = Field(None)


class AnnotationTask(bt.Synapse):
    schema_version: str = Field("hazard.annotation.v1")
    task_id: str = Field("")
    batch_id: str = Field("", description="Active batch identifier (e.g. batch_1)")
    round_num: int = Field(0, description="Active sequential round number (1-indexed)")
    challenge_nonce: str = Field("")
    annotation_images: List[UnlabeledAnnotationImage] = Field(default_factory=list)
    training_pool: List[LabeledTrainingImage] = Field(default_factory=list)
    training_pool_hash: str = Field("")
    annotations_uri: str = Field("")
    miner_r2_credentials: Optional[R2AccessCredentials] = Field(None)
    miner_r2_credentials_json: Optional[str] = Field(None, description="Serialized miner R2 credentials for network transport")
    last_round_feedback_json: Optional[str] = Field(None, description="Serialized scoped agentic feedback receipt for miner hotkey")
    duration_ms: Optional[int] = Field(None, ge=0)
    error_message: Optional[str] = Field(None, max_length=1024)

    def deserialize(self) -> "AnnotationTask":
        return self
