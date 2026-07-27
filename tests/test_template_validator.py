from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from template.hazard.dataset_assembler import AdoptionLedger
from template.hazard.annotation_eval import PerMinerAnnotationScore
from template.hazard.dual_reward import DualFlywheelBreakdown
from template.hazard.incentives import broad_softmax_scores
from template.protocol import AnnotationTask, PerImageAnnotationItem
from template.validator.dual_forward import _validate_response_shape


def test_broad_softmax_pays_multiple_value_adding_miners():
    shaped = broad_softmax_scores(
        np.array([0.0, 0.2, 0.4, 0.8], dtype=float),
        temperature=0.35,
        floor=0.01,
        min_score=0.05,
    )
    assert shaped[0] == 0.0
    assert (shaped[1:] > 0.0).all()
    assert abs(float(shaped.sum()) - 1.0) < 1e-6


def test_response_shape_rejects_nonce_mismatch():
    response = AnnotationTask(task_id="t-1", challenge_nonce="bad-nonce")
    with pytest.raises(ValueError, match="Challenge nonce mismatch"):
        _validate_response_shape(
            response,
            expected_task_id="t-1",
            expected_nonce="good-nonce",
        )


def test_response_shape_requires_annotations_uri():
    response = AnnotationTask(task_id="t-2", challenge_nonce="nonce")
    with pytest.raises(ValueError, match="annotations_uri"):
        _validate_response_shape(
            response,
            expected_task_id="t-2",
            expected_nonce="nonce",
        )


def test_adoption_ledger_state_round_trip(tmp_path: Path):
    ledger = AdoptionLedger(
        adoption_counts={1: 4},
        last_round_counts={1: 2},
        adoption_contributions={1: 3.5},
        last_round_contributions={1: 1.5},
        rounds_observed=3,
    )
    path = tmp_path / "adoption_ledger.json"
    path.write_text(json.dumps(ledger.to_jsonable()), encoding="utf-8")
    restored = AdoptionLedger.from_jsonable(json.loads(path.read_text(encoding="utf-8")))
    assert restored.adoption_counts == {1: 4}
    assert restored.last_round_contributions == {1: 1.5}


def test_annotation_score_ema_update():
    state = SimpleNamespace(
        config=SimpleNamespace(neuron=SimpleNamespace(moving_average_alpha=0.25)),
        annotation_scores=np.array([0.0, 0.4], dtype=np.float32),
        adoption_bonus_scores=np.array([0.0, 0.2], dtype=np.float32),
    )

    def update_score_ledgers(breakdowns, uids):
        alpha = float(state.config.neuron.moving_average_alpha)
        for uid, item in zip(uids, breakdowns):
            state.annotation_scores[uid] = (
                alpha * item.annotation_score + (1.0 - alpha) * state.annotation_scores[uid]
            )
            state.adoption_bonus_scores[uid] = (
                alpha * item.adoption_bonus
                + (1.0 - alpha) * state.adoption_bonus_scores[uid]
            )

    update_score_ledgers(
        [
            DualFlywheelBreakdown(
                uid=1,
                annotation_score=1.0,
                adoption_bonus=0.8,
                hallucination_multiplier=1.0,
                final_score=0.94,
                fidelity_image_ids=2,
                consensus_image_ids=3,
                adopted_image_ids_round=2,
                adopted_image_ids_total=5,
            )
        ],
        [1],
    )
    assert state.annotation_scores[1] == pytest.approx(0.55)
    assert state.adoption_bonus_scores[1] == pytest.approx(0.35)


def test_per_miner_average_score_uses_golden_only():
    score = PerMinerAnnotationScore(
        uid=7,
        fidelity_scores_by_image_id={"g1": 0.9, "g2": 0.7},
        consensus_scores_by_image_id={"pool1": 0.1, "pool2": 0.2},
    )
    assert score.average_score() == pytest.approx(0.8)


def test_annotation_item_accepts_float_boxes():
    item = PerImageAnnotationItem(
        hazard_class="trip_hazard",
        bounding_box=[1.5, 2.5, 10.0, 12.0],
        severity="low",
    )
    assert item.bounding_box[0] == pytest.approx(1.5)


def test_broad_softmax_scaling_floor_edge_case():
    # N=20 eligible miners with floor=0.1, which would make floor * N = 2.0 > 1.0
    scores = np.ones(20, dtype=float)
    # Give the first miner a higher score
    scores[0] = 5.0
    
    shaped = broad_softmax_scores(
        scores,
        temperature=0.35,
        floor=0.1,
        min_score=0.05,
    )
    
    # Assert that the sum is exactly 1.0 (or extremely close)
    assert abs(float(shaped.sum()) - 1.0) < 1e-6
    # The first miner must have the highest incentive and not be zeroed out
    assert shaped[0] > shaped[1]
    assert shaped[1] == pytest.approx(shaped[-1])
    assert shaped[0] > 0.0
    assert (shaped >= 0.0).all()
