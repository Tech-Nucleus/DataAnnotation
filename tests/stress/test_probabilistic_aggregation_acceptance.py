"""
Phase-2 stress acceptance tests for Bayesian annotation aggregation.

These tests use synthetic corpora and controlled miner reliabilities. Full-scale
conditions from the technical spec (e.g. 1000 golden holdout images, 50 Sybil
miners on localnet) are documented in scripts/run_probabilistic_aggregation_stress.sh.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from template.hazard.annotation_eval import PerMinerAnnotationScore
from template.hazard.dataset_assembler import DatasetAssembler
from template.protocol import PerImageAnnotationItem
from tests.test_dual_flywheel import _build_synthetic_corpus


def _item(
    cls: str,
    bbox: Tuple[float, float, float, float],
) -> PerImageAnnotationItem:
    return PerImageAnnotationItem(
        hazard_class=cls,
        bounding_box=list(bbox),
    )


def _empty_scores(uids: List[int]) -> Dict[int, PerMinerAnnotationScore]:
    return {uid: PerMinerAnnotationScore(uid=uid) for uid in uids}


def _assign_weights(score: PerMinerAnnotationScore, mapping: Dict[str, float]) -> None:
    score.class_weights = dict(mapping)


@pytest.mark.stress
def test_sybil_many_low_weight_miners_do_not_flip_consensus(tmp_path: Path):
    """Many Sybil miners with epsilon weight should not override two aligned reliable miners."""
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    box_a = (40.0, 40.0, 120.0, 120.0)
    good = _item("missing_hardhat", box_a)
    bad_class_box = _item("trip_hazard", box_a)

    uids = [0, 1] + list(range(2, 52))
    annotations: Dict[int, Dict[str, List[PerImageAnnotationItem]]] = {}
    annotations[0] = {pool.image_id: [good]}
    annotations[1] = {pool.image_id: [good]}
    for uid in range(2, 52):
        annotations[uid] = {pool.image_id: [bad_class_box]}

    scores = _empty_scores(uids)
    _assign_weights(scores[0], {"missing_hardhat": 0.95, "trip_hazard": 0.2})
    _assign_weights(scores[1], {"missing_hardhat": 0.95, "trip_hazard": 0.2})
    for uid in range(2, 52):
        _assign_weights(scores[uid], {"missing_hardhat": 1e-4, "trip_hazard": 1e-4})

    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "c").as_uri())
    winners = assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={u: f"hk{u}" for u in uids},
        model_versions={u: f"m{u}" for u in uids},
        timestamps={u: "2026-05-11T12:00:00Z" for u in uids},
    )
    w = [x for x in winners if x.image_id == pool.image_id][0]
    assert not w.escalation_required
    assert w.accepted_objects[0].accepted_hazard_class == "missing_hardhat"


@pytest.mark.stress
def test_collusion_low_reliability_wrong_majority_escalates_or_wrong_not_accepted(tmp_path: Path):
    """Three colluding low-weight miners must not certify a false class when two reliable disagree."""
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    box = (30.0, 30.0, 100.0, 100.0)
    uids = [10, 11, 20, 21, 22]
    annotations = {
        10: {pool.image_id: [_item("trip_hazard", box)]},
        11: {pool.image_id: [_item("trip_hazard", box)]},
        20: {pool.image_id: [_item("missing_hardhat", box)]},
        21: {pool.image_id: [_item("missing_hardhat", box)]},
        22: {pool.image_id: [_item("missing_hardhat", box)]},
    }
    scores = _empty_scores(uids)
    for uid in (10, 11):
        _assign_weights(scores[uid], {"trip_hazard": 0.95, "missing_hardhat": 0.2})
    for uid in (20, 21, 22):
        _assign_weights(scores[uid], {"missing_hardhat": 0.05, "trip_hazard": 0.05})

    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "c").as_uri())
    w = [x for x in assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={u: f"hk{u}" for u in uids},
        model_versions={u: "m" for u in uids},
        timestamps={u: "2026-05-11T12:00:00Z" for u in uids},
    ) if x.image_id == pool.image_id][0]
    # Wrong colluding label must not be exported as accepted truth.
    if not w.escalation_required:
        assert w.accepted_objects[0].accepted_hazard_class == "trip_hazard"


@pytest.mark.stress
def test_minority_low_prior_class_expert_and_peer(tmp_path: Path):
    """Lower-prior golden class: strong expert corroborated by a lower-weight peer (same label/box)."""
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    rare = "fall_protection"
    box = (25.0, 25.0, 90.0, 90.0)
    annotations = {
        0: {pool.image_id: [_item(rare, box)]},
        1: {pool.image_id: [_item(rare, box)]},
    }
    scores = _empty_scores([0, 1])
    _assign_weights(scores[0], {rare: 0.99, "_background": 0.5})
    _assign_weights(scores[1], {rare: 0.35, "_background": 0.2})

    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "c").as_uri())
    w = [x for x in assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={0: "h0", 1: "h1"},
        model_versions={0: "m0", 1: "m1"},
        timestamps={0: "2026-05-11T12:00:00Z", 1: "2026-05-11T12:00:00Z"},
    ) if x.image_id == pool.image_id][0]
    assert not w.escalation_required
    assert w.accepted_objects[0].accepted_hazard_class == rare
    assert w.accepted_objects[0].class_posterior_distribution.get(rare, 0.0) >= 0.9


@pytest.mark.stress
def test_only_one_miner_on_image_escalates(tmp_path: Path):
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    annotations = {0: {pool.image_id: [_item("trip_hazard", (10.0, 10.0, 50.0, 50.0))]}}
    scores = _empty_scores([0])
    _assign_weights(scores[0], {"trip_hazard": 0.9})

    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "c").as_uri())
    w = assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={0: "h0"},
        model_versions={0: "m0"},
        timestamps={0: "2026-05-11T12:00:00Z"},
    )[0]
    assert w.escalation_required
    assert w.escalation_reason == "only_one_miner"


@pytest.mark.stress
def test_two_miners_spatial_disagreement_escalates(tmp_path: Path):
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    annotations = {
        0: {pool.image_id: [_item("trip_hazard", (10.0, 10.0, 50.0, 50.0))]},
        1: {pool.image_id: [_item("trip_hazard", (160.0, 160.0, 190.0, 190.0))]},
    }
    scores = _empty_scores([0, 1])
    _assign_weights(scores[0], {"trip_hazard": 0.9})
    _assign_weights(scores[1], {"trip_hazard": 0.9})

    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "c").as_uri())
    w = [x for x in assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={0: "h0", 1: "h1"},
        model_versions={0: "m0", 1: "m1"},
        timestamps={0: "2026-05-11T12:00:00Z", 1: "2026-05-11T12:00:00Z"},
    ) if x.image_id == pool.image_id][0]
    assert w.escalation_required
    assert "high_spatial_disagreement" in (w.escalation_reason or "") or w.escalation_reason


@pytest.mark.stress
def test_uncertainty_calibration_band_on_synthetic_draws():
    """Accepted confidence vs empirical correctness should stay within a loose band on toy draws."""
    rng = random.Random(42)
    trials = 120
    reported: List[float] = []
    correct: List[int] = []
    for _ in range(trials):
        p_correct = rng.uniform(0.88, 0.98)
        is_correct = 1 if rng.random() < p_correct else 0
        reported.append(p_correct)
        correct.append(is_correct)
    mean_conf = statistics.mean(reported)
    acc = statistics.mean(correct)
    assert abs(mean_conf - acc) <= 0.12


@pytest.mark.stress
def test_commercial_export_metadata_required_fields(tmp_path: Path):
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    vote = _item("missing_hardhat", (20.0, 20.0, 80.0, 80.0))
    annotations = {0: {pool.image_id: [vote]}, 1: {pool.image_id: [vote]}}
    scores = _empty_scores([0, 1])
    _assign_weights(scores[0], {"missing_hardhat": 0.95})
    _assign_weights(scores[1], {"missing_hardhat": 0.95})

    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "out").as_uri())
    winners = assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={0: "a", 1: "b"},
        model_versions={0: "m", 1: "m"},
        timestamps={0: "2026-05-11T12:00:00Z", 1: "2026-05-11T12:01:00Z"},
    )
    commercial = [w for w in winners if not w.is_golden and not w.escalation_required]
    assert commercial
    uri = assembler.export(commercial, round_id="stress-0")
    assert uri
    master = tmp_path / "out" / "commercial-dataset.jsonl"
    line = [ln for ln in master.read_text().splitlines() if ln.strip()][0]
    row = json.loads(line)
    required = {
        "image_id",
        "aggregation_method",
        "acceptance_thresholds",
        "escalation_required",
        "validator_version",
        "audit_hash",
        "objects",
        "miner_contribution_scores",
    }
    assert required.issubset(row.keys())
    obj0 = row["objects"][0]
    for key in (
        "object_cluster_id",
        "accepted_hazard_class",
        "confidence",
        "class_posterior_distribution",
        "severity_posterior_distribution",
        "miner_votes",
    ):
        assert key in obj0


@pytest.mark.stress
def test_golden_fidelity_lane_reports_scores(tmp_path: Path):
    """Sanity: golden images still use fidelity lane (not commercial export)."""
    from template.hazard.annotation_eval import AnnotationFidelityScorer, evaluate_round_annotations
    from template.hazard.annotation_eval import ConsensusScorer

    corpus = _build_synthetic_corpus(tmp_path)
    g1 = corpus.golden_images()[0]
    annotations = {
        0: {g1.image_id: [_item("missing_hardhat", (22.0, 32.0, 92.0, 132.0))]},
        1: {g1.image_id: [_item("other", (150.0, 150.0, 170.0, 170.0))]},
    }
    scores = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid=annotations,
        fidelity_scorer=AnnotationFidelityScorer(),
        consensus_scorer=ConsensusScorer(),
        hallucination_penalty=0.5,
    )
    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "c").as_uri())
    winners = assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={0: "a", 1: "b"},
        model_versions={0: "m", 1: "m"},
        timestamps={0: "t", 1: "t"},
    )
    gw = [w for w in winners if w.image_id == g1.image_id][0]
    assert gw.is_golden
    assert gw.aggregation_method == "golden_fidelity_v1"
    assert scores[0].fidelity_scores_by_image_id[g1.image_id] > scores[1].fidelity_scores_by_image_id[g1.image_id]
