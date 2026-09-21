"""
Official IoU and Segmentation Metrics Evaluator.
Computes Micro and Macro IoU, Accuracy, F1, Precision, Recall, and 2-Class Mean IoU (mIoU).
Matches torchmetrics.JaccardIndex(task="multiclass", num_classes=2) and Accuracy(task="multiclass", num_classes=2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class ConfusionMatrix:
    """Accumulates true positives, false positives, false negatives, and true negatives."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, pred: np.ndarray, target: np.ndarray, target_class: int = 1) -> None:
        """Update confusion matrix for a specific target class."""
        pred_pos = (pred == target_class)
        pred_neg = (pred != target_class)
        targ_pos = (target == target_class)
        targ_neg = (target != target_class)

        self.tp += int(np.logical_and(pred_pos, targ_pos).sum())
        self.fp += int(np.logical_and(pred_pos, targ_neg).sum())
        self.fn += int(np.logical_and(pred_neg, targ_pos).sum())
        self.tn += int(np.logical_and(pred_neg, targ_neg).sum())

    @property
    def tree_iou(self) -> float:
        """IoU for foreground tree class (class 1)."""
        denom = self.tp + self.fp + self.fn
        return float(self.tp / denom) if denom > 0 else 1.0 if self.tp == 0 and self.fn == 0 else 0.0

    @property
    def bg_iou(self) -> float:
        """IoU for background class (class 0)."""
        denom = self.tn + self.fn + self.fp
        return float(self.tn / denom) if denom > 0 else 1.0 if self.tn == 0 and self.fp == 0 else 0.0

    @property
    def mean_iou(self) -> float:
        """2-class mean IoU (mIoU) matching torchmetrics.JaccardIndex(task='multiclass', num_classes=2)."""
        return float((self.tree_iou + self.bg_iou) / 2.0)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return float((self.tp + self.tn) / total) if total > 0 else 1.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


class SegmentationEvaluator:
    """Evaluates semantic segmentation predictions against ground truth masks."""

    def __init__(self, target_class: int = 1, ignore_index: Optional[int] = None):
        self.target_class = target_class
        self.ignore_index = ignore_index
        self.global_cm = ConfusionMatrix()
        self.per_image_tree_ious: List[float] = []
        self.per_image_mean_ious: List[float] = []
        self.per_image_accs: List[float] = []
        self.per_image_f1s: List[float] = []
        self.n_images: int = 0

    def add_batch(self, preds: np.ndarray, targets: np.ndarray) -> None:
        """Add a batch of predictions and targets (each shape [B, H, W] or [H, W])."""
        if preds.ndim == 2:
            preds = preds[np.newaxis, ...]
            targets = targets[np.newaxis, ...]

        for p, t in zip(preds, targets):
            if self.ignore_index is not None:
                valid = (t != self.ignore_index)
                p = p[valid]
                t = t[valid]

            img_cm = ConfusionMatrix()
            img_cm.update(p, t, target_class=self.target_class)

            self.global_cm.tp += img_cm.tp
            self.global_cm.fp += img_cm.fp
            self.global_cm.fn += img_cm.fn
            self.global_cm.tn += img_cm.tn

            self.per_image_tree_ious.append(img_cm.tree_iou)
            self.per_image_mean_ious.append(img_cm.mean_iou)
            self.per_image_accs.append(img_cm.accuracy)
            self.per_image_f1s.append(img_cm.f1)
            self.n_images += 1

    def compute(self) -> Dict[str, float]:
        """Return both micro (global) and macro (mean per-image) metrics."""
        return {
            "mean_iou": round(self.global_cm.mean_iou, 4),
            "tree_iou": round(self.global_cm.tree_iou, 4),
            "bg_iou": round(self.global_cm.bg_iou, 4),
            "macro_mean_iou": round(float(np.mean(self.per_image_mean_ious)), 4) if self.per_image_mean_ious else 0.0,
            "macro_tree_iou": round(float(np.mean(self.per_image_tree_ious)), 4) if self.per_image_tree_ious else 0.0,
            "accuracy": round(self.global_cm.accuracy, 4),
            "macro_accuracy": round(float(np.mean(self.per_image_accs)), 4) if self.per_image_accs else 0.0,
            "f1": round(self.global_cm.f1, 4),
            "macro_f1": round(float(np.mean(self.per_image_f1s)), 4) if self.per_image_f1s else 0.0,
            "precision": round(self.global_cm.precision, 4),
            "recall": round(self.global_cm.recall, 4),
            "tp": self.global_cm.tp,
            "fp": self.global_cm.fp,
            "fn": self.global_cm.fn,
            "tn": self.global_cm.tn,
            "n_images": self.n_images,
        }
