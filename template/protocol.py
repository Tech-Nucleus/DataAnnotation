from __future__ import annotations

from typing import List, Literal, Optional

import bittensor as bt
from pydantic import BaseModel, Field

SeverityTier = Literal["none", "low", "medium", "high", "critical"]


class UnlabeledAnnotationImage(BaseModel):
    image_url: str = Field(..., min_length=1)
    image_id: str = Field(..., min_length=1)


class PerImageAnnotationItem(BaseModel):
    hazard_class: str = Field(..., min_length=1)
    bounding_box: List[float] = Field(..., min_length=4, max_length=4)


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
    annotations: List[PerImageAnnotationItem]
    model_version: str = Field(..., min_length=8)


class AnnotationsFilePayload(BaseModel):
    schema_version: str = Field("annotations.v1", min_length=1)
    task_id: str = Field("", min_length=0)
    records: List[ImageAnnotationDocument]


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
    challenge_nonce: str = Field("")
    annotation_images: List[UnlabeledAnnotationImage] = Field(default_factory=list)
    training_pool: List[LabeledTrainingImage] = Field(default_factory=list)
    training_pool_hash: str = Field("")
    annotations_uri: str = Field("")
    miner_r2_credentials: Optional[R2AccessCredentials] = Field(None)
    duration_ms: Optional[int] = Field(None, ge=0)
    error_message: Optional[str] = Field(None, max_length=1024)

    def deserialize(self) -> "AnnotationTask":
        return self
