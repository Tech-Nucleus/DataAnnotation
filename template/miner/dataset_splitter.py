"""Deterministic hash-based dataset splitting.

Every image_id is assigned to a bucket via SHA-256.  The split is
reproducible given the same seed and split percentage.
"""

from __future__ import annotations

import hashlib
from typing import List, Tuple


def split_dataset(
    image_ids: List[str],
    *,
    train_split_pct: int = 70,
    random_seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split *image_ids* into (train, inference) lists.

    Parameters
    ----------
    image_ids:
        Flat list of image identifiers to partition.
    train_split_pct:
        Percentage of images allocated to the training split (0–100).
    random_seed:
        Seed mixed into the hash for reproducibility.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(train_ids, inference_ids)``
    """
    train_ids: List[str] = []
    inference_ids: List[str] = []
    for image_id in image_ids:
        h = hashlib.sha256((image_id + str(random_seed)).encode("utf-8")).hexdigest()
        bucket = int(h[:8], 16) % 100
        if bucket < train_split_pct:
            train_ids.append(image_id)
        else:
            inference_ids.append(image_id)
    return train_ids, inference_ids


def split_dataset_three_way(
    image_ids: List[str],
    *,
    train_pct: int = 60,
    val_pct: int = 20,
    random_seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Three-way split for auto-research: train / validation / inference.

    The remaining percentage after *train_pct + val_pct* is the inference
    (final) split.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        ``(train_ids, val_ids, inference_ids)``
    """
    if train_pct + val_pct > 100:
        raise ValueError(
            f"train_pct ({train_pct}) + val_pct ({val_pct}) exceeds 100"
        )
    train_ids: List[str] = []
    val_ids: List[str] = []
    inference_ids: List[str] = []
    for image_id in image_ids:
        h = hashlib.sha256((image_id + str(random_seed)).encode("utf-8")).hexdigest()
        bucket = int(h[:8], 16) % 100
        if bucket < train_pct:
            train_ids.append(image_id)
        elif bucket < train_pct + val_pct:
            val_ids.append(image_id)
        else:
            inference_ids.append(image_id)
    return train_ids, val_ids, inference_ids
