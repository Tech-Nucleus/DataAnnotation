"""Pluggable model backends for the training-and-inference miner."""

from template.miner.backends.base import (
    BaseModelBackend,
    InferImage,
    TrainImage,
    TrainResult,
)
from template.miner.backends.factory import get_backend

__all__ = [
    "BaseModelBackend",
    "InferImage",
    "TrainImage",
    "TrainResult",
    "get_backend",
]
