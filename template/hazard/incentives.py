from __future__ import annotations

import numpy as np


def broad_softmax_scores(
    scores: np.ndarray,
    *,
    temperature: float,
    floor: float,
    min_score: float,
) -> np.ndarray:
    """
    Convert raw miner value scores into a broad nonzero incentive surface.

    Miners below min_score receive zero. Every miner above it receives the
    configured floor plus a softmax-shaped share, so useful contributors are
    paid without flattening competition at the top.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if floor < 0:
        raise ValueError("floor must be non-negative")

    raw = np.asarray(scores, dtype=np.float64)
    eligible = np.isfinite(raw) & (raw >= min_score) & (raw > 0.0)
    shaped = np.zeros_like(raw, dtype=np.float64)
    if not eligible.any():
        return shaped.astype(np.float32)

    eligible_scores = raw[eligible]
    centered = eligible_scores - np.max(eligible_scores)
    exp_scores = np.exp(centered / temperature)
    exp_scores = exp_scores / np.sum(exp_scores)

    # Dynamically scale floor to prevent negative multipliers when many miners are eligible
    n_eligible = len(exp_scores)
    effective_floor = floor
    if floor * n_eligible > 0.9:
        effective_floor = 0.9 / n_eligible

    shaped_values = effective_floor + (1.0 - effective_floor * n_eligible) * exp_scores
    shaped_values = np.clip(shaped_values, 0.0, None)
    shaped[eligible] = shaped_values
    total = shaped.sum()
    if total > 0:
        shaped = shaped / total
    return shaped.astype(np.float32)

