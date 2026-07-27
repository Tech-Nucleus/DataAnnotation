"""Abstract base class and data types for model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from template.protocol import PerImageAnnotationItem


@dataclass
class TrainImage:
    """A single image for training, optionally with ground-truth labels."""

    image_id: str
    image_path: Path
    labels: Optional[List[PerImageAnnotationItem]] = None


@dataclass
class InferImage:
    """A single image for inference (no labels)."""

    image_id: str
    image_path: Path


@dataclass
class TrainResult:
    """Outcome of a training run."""

    model_version: str
    metrics: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: Optional[Path] = None


class BaseModelBackend(ABC):
    """Interface every model backend must implement.

    The auto-research loop and the ``ModelTrainingAnnotationEngine`` interact
    with backends *exclusively* through ``train()`` and ``infer()``.  Adding a
    new backend means implementing these two methods and registering the class
    in :pymod:`template.miner.backends.factory`.
    """

    @abstractmethod
    def train(
        self,
        train_images: List[TrainImage],
        config: Dict,
    ) -> TrainResult:
        """Fine-tune a model on *train_images*.

        Parameters
        ----------
        train_images:
            Images with optional ground-truth labels.  If labels are ``None``
            the backend may use pseudo-labeling or skip training.
        config:
            Hyperparameters (epochs, lr, batch, etc.) — backend-specific keys.

        Returns
        -------
        TrainResult
            Contains a ``model_version`` identifier usable by ``infer()`` and
            optional training metrics.
        """

    @abstractmethod
    def infer(
        self,
        inference_images: List[InferImage],
        model_version: str,
    ) -> Dict[str, List[PerImageAnnotationItem]]:
        """Run inference on *inference_images* using *model_version*.

        Returns
        -------
        dict
            Mapping ``image_id → list[PerImageAnnotationItem]``.
        """
