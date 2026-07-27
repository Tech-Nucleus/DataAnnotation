"""
Final on-chain weight formula for the annotation-only subnet.

For every miner uid in the round:

  weight = alpha * annotation_score + (1 - alpha) * adoption_bonus

with a global hallucination multiplier that compresses the annotation
score by ``hallucination_penalty`` per hallucinated Golden annotation.

``adoption_bonus`` is the share of image_ids in the round whose winning
annotation came from this miner (normalized to [0, 1]).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from template.hazard.annotation_eval import PerMinerAnnotationScore
from template.hazard.dataset_assembler import AdoptionLedger, WinningAnnotation


@dataclass(frozen=True)
class DualFlywheelBreakdown:
    """Per-miner breakdown returned alongside the final weight."""

    uid: int
    annotation_score: float
    adoption_bonus: float
    hallucination_multiplier: float
    final_score: float
    fidelity_image_ids: int
    consensus_image_ids: int
    adopted_image_ids_round: int
    adopted_image_ids_total: int


@dataclass
class DualFlywheelRewardComposer:
    alpha: float = 0.7
    hallucination_penalty_per_event: float = 0.5
    golden_missing_penalty: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1] (got {self.alpha})")

    def compose(
        self,
        *,
        uids: Sequence[int],
        annotation_scores: Mapping[int, PerMinerAnnotationScore],
        ledger: AdoptionLedger,
        round_winners: Sequence[WinningAnnotation],
    ) -> tuple[np.ndarray, list[DualFlywheelBreakdown]]:
        round_share = ledger.round_contribution_share()
        winners_by_uid: Dict[int, int] = {}
        for w in round_winners:
            if w.escalation_required:
                continue
            winners_by_uid[w.chosen_uid] = winners_by_uid.get(w.chosen_uid, 0) + 1

        rewards: list[float] = []
        breakdowns: list[DualFlywheelBreakdown] = []
        for uid in uids:
            score = annotation_scores.get(uid)
            base_annotation = (
                score.average_score(golden_missing_penalty=self.golden_missing_penalty)
                if score is not None
                else 0.0
            )
            hallucination_mult = (
                score.hallucination_multiplier(self.hallucination_penalty_per_event)
                if score is not None
                else 1.0
            )
            annotation_score = float(max(0.0, min(1.0, base_annotation * hallucination_mult)))
            adoption_bonus = float(round_share.get(uid, 0.0))

            final = self.alpha * annotation_score + (1.0 - self.alpha) * adoption_bonus
            final = float(max(0.0, min(1.0, final)))

            rewards.append(final)
            breakdowns.append(
                DualFlywheelBreakdown(
                    uid=int(uid),
                    annotation_score=annotation_score,
                    adoption_bonus=adoption_bonus,
                    hallucination_multiplier=float(hallucination_mult),
                    final_score=final,
                    fidelity_image_ids=len(score.fidelity_scores_by_image_id) if score else 0,
                    consensus_image_ids=len(score.consensus_scores_by_image_id) if score else 0,
                    adopted_image_ids_round=int(winners_by_uid.get(uid, 0)),
                    adopted_image_ids_total=int(ledger.adoption_counts.get(uid, 0)),
                )
            )
        return np.asarray(rewards, dtype=np.float32), breakdowns
