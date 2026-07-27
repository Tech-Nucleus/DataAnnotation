"""End-to-end tests for the dual-flywheel validator pipeline.

The tests bypass HuggingFace dataset downloads by constructing an
``ImageCorpus`` with synthetic images plus hand-built Golden /
Annotation / Benchmark records. This exercises every scoring lane,
the injection planner, the dataset assembler, the commercial export,
and the final reward composer using only in-memory state.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from template.hazard.annotation_eval import (
    AnnotationFidelityScorer,
    ConsensusScorer,
    evaluate_round_annotations,
    iou_xyxy,
    _ReliabilityAccumulator,
)
from template.hazard.dataset_assembler import (
    AggregatedObject,
    AdoptionLedger,
    DatasetAssembler,
    MinerVote,
    WinningAnnotation,
)
from template.hazard.dual_reward import DualFlywheelRewardComposer
from template.hazard.image_corpus import (
    BenchmarkImage,
    GoldenAnnotation,
    GoldenImage,
    ImageCorpus,
    ImageCorpusConfig,
    UnlabeledImage,
    _severity_for_label,
)
from template.protocol import PerImageAnnotationItem
from template.validator.dual_forward import (
    _build_full_dataset_plan,
    _build_round_annotation_plan,
)


def test_image_corpus_normalized_annotation_entries_parses_at_split(tmp_path: Path):
    cfg = ImageCorpusConfig(
        cache_root=tmp_path,
        annotation_dataset_ids="foo/bar@test, baz/qux@validation, plain/id",
        annotation_split="train",
    )
    assert cfg.normalized_annotation_entries() == [
        ("foo/bar", "test"),
        ("baz/qux", "validation"),
        ("plain/id", "train"),
    ]


def _png_bytes(width: int, height: int, color=(120, 140, 160)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _build_synthetic_corpus(tmp_path: Path) -> ImageCorpus:
    """Build a fully-populated ImageCorpus without touching HuggingFace."""
    cache = tmp_path / "image_cache"
    cache.mkdir(parents=True, exist_ok=True)
    config = ImageCorpusConfig(cache_root=cache)
    corpus = ImageCorpus(config)
    # Deliberately bypass HF loading; populate the in-memory indexes directly.
    corpus._loaded = True

    def _materialize(image_id: str, payload: bytes, ext: str = "png") -> Path:
        path = cache / f"{image_id}.{ext}"
        path.write_bytes(payload)
        corpus._all_image_index[image_id] = path
        return path

    # Two Golden images with known boxes / classes / severities.
    g1_bytes = _png_bytes(200, 200, (100, 110, 120))
    g1_id = hashlib.sha256(g1_bytes).hexdigest()
    g1_path = _materialize(g1_id, g1_bytes)
    g1 = GoldenImage(
        image_id=g1_id,
        image_path=g1_path,
        image_url=g1_path.as_uri(),
        width=200,
        height=200,
        annotations=(
            GoldenAnnotation(
                hazard_class="missing_hardhat",
                bounding_box=(20, 30, 90, 130),
                severity="high",
            ),
        ),
    )
    corpus._golden.append(g1)
    corpus._golden_index[g1_id] = g1

    g2_bytes = _png_bytes(160, 160, (130, 90, 70))
    g2_id = hashlib.sha256(g2_bytes).hexdigest()
    g2_path = _materialize(g2_id, g2_bytes)
    g2 = GoldenImage(
        image_id=g2_id,
        image_path=g2_path,
        image_url=g2_path.as_uri(),
        width=160,
        height=160,
        annotations=(
            GoldenAnnotation(
                hazard_class="fall_protection",
                bounding_box=(10, 20, 60, 90),
                severity="high",
            ),
        ),
    )
    corpus._golden.append(g2)
    corpus._golden_index[g2_id] = g2

    # Three unlabeled annotation pool images.
    for idx, color in enumerate([(40, 60, 80), (90, 40, 50), (50, 90, 40)]):
        payload = _png_bytes(180, 180, color)
        img_id = hashlib.sha256(payload).hexdigest()
        path = _materialize(img_id, payload)
        corpus._annotation.append(
            UnlabeledImage(
                image_id=img_id,
                image_path=path,
                image_url=path.as_uri(),
                width=180,
                height=180,
                source_dataset=f"synthetic-{idx}",
            )
        )

    # One benchmark image.
    bench_bytes = _png_bytes(220, 220, (10, 10, 10))
    bench_id = hashlib.sha256(bench_bytes).hexdigest()
    bench_path = _materialize(bench_id, bench_bytes)
    corpus._benchmark.append(
        BenchmarkImage(
            image_id=bench_id,
            image_path=bench_path,
            width=220,
            height=220,
            annotations=(
                GoldenAnnotation(
                    hazard_class="fall_protection",
                    bounding_box=(30, 40, 100, 150),
                    severity="high",
                ),
            ),
            source_dataset="synthetic-benchmark",
        )
    )
    return corpus


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def test_severity_label_mapping_matches_spec():
    assert _severity_for_label("missing-hardhat") == "high"
    assert _severity_for_label("missing_vest") == "high"
    assert _severity_for_label("fall_protection") == "high"
    assert _severity_for_label("trip_hazard") == "low"
    assert _severity_for_label("hardhat") == "medium"


def test_iou_basic():
    assert iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert iou_xyxy([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175, rel=1e-4)


# ---------------------------------------------------------------------------
# Image corpus + full-dataset plan
# ---------------------------------------------------------------------------

def test_synthetic_corpus_builds(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    assert len(corpus.golden_images()) == 2
    assert len(corpus.annotation_images()) == 3
    assert len(corpus.benchmark_images()) == 1
    g1 = corpus.golden_images()[0]
    assert corpus.is_golden(g1.image_id)
    assert corpus.golden_lookup(g1.image_id) is g1


def test_full_dataset_plan_contains_golden_and_annotation_images(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    plan = _build_full_dataset_plan(corpus)
    assert len(plan.ordered_images) == 5
    assert len(plan.golden_image_ids) == 2
    assert len(plan.annotation_image_ids) == 3
    all_ids = {iid for iid, _ in plan.ordered_images}
    assert all_ids == set(plan.golden_image_ids) | set(plan.annotation_image_ids)


def test_round_annotation_plan_honors_request_size_and_golden_injection(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)

    class Ns:
        pass

    self_obj = Ns()
    self_obj.random = random.Random(7)
    self_obj.config = Ns()
    self_obj.config.neuron = Ns()
    self_obj.config.neuron.flywheel_annotation_request_size = 4
    self_obj.config.neuron.flywheel_golden_injection_per_request = 2

    plan = _build_round_annotation_plan(self_obj, corpus)
    assert len(plan.ordered_images) == 4
    assert len(plan.golden_image_ids) == 2
    assert len(plan.annotation_image_ids) == 2
    assert len({iid for iid, _ in plan.ordered_images}) == 4


# ---------------------------------------------------------------------------
# Annotation fidelity + consensus scoring
# ---------------------------------------------------------------------------

def _miner_item(
    *,
    cls: str = "missing_hardhat",
    bbox=(22, 32, 92, 132),
) -> PerImageAnnotationItem:
    return PerImageAnnotationItem(
        hazard_class=cls,
        bounding_box=list(bbox),
    )


def test_fidelity_scorer_high_quality_match(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    scorer = AnnotationFidelityScorer()
    g1 = corpus.golden_images()[0]
    components = scorer.score([_miner_item()], g1)
    assert components.iou > 0.7
    assert components.class_severity == pytest.approx(1.0)
    assert components.hallucination_penalty == 1.0
    assert components.fidelity > 0.6
    assert components.matched_count == 1


def test_fidelity_scorer_penalizes_hallucinations(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    scorer = AnnotationFidelityScorer(hallucination_penalty=0.5)
    g1 = corpus.golden_images()[0]
    halluc = _miner_item(
        cls="random_object",
        bbox=(140, 140, 180, 180),
    )
    components = scorer.score([_miner_item(), halluc], g1)
    assert components.hallucinated_count == 1
    assert components.hallucination_penalty == pytest.approx(0.5)
    assert components.fidelity < 0.5


def test_fidelity_scorer_zeros_for_total_miss(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    scorer = AnnotationFidelityScorer()
    g1 = corpus.golden_images()[0]
    items = [
        _miner_item(
            cls="random_class",
            bbox=(150, 150, 175, 175),
        )
    ]
    components = scorer.score(items, g1)
    assert components.iou == 0.0
    assert components.fidelity == 0.0
    assert components.matched_count == 0
    assert components.hallucinated_count == 1


def test_consensus_scorer_majority_class(tmp_path):
    scorer = ConsensusScorer()
    miner_items = [_miner_item(cls="missing_hardhat", bbox=(10, 10, 50, 50))]
    peers = {
        2: [_miner_item(cls="missing_hardhat", bbox=(12, 12, 52, 52))],
        3: [_miner_item(cls="missing_hardhat", bbox=(14, 14, 54, 54))],
        4: [_miner_item(cls="trip_hazard", bbox=(80, 80, 100, 100))],
    }
    comp = scorer.score(miner_items, peers)
    assert comp.peer_count == 3
    assert comp.majority_class_match == 1.0
    assert comp.consensus > 0.5


def test_consensus_scorer_zero_peers():
    scorer = ConsensusScorer()
    miner_items = [_miner_item()]
    comp = scorer.score(miner_items, {})
    assert comp.consensus == 0.0
    assert comp.peer_count == 0


def test_evaluate_round_annotations_penalizes_missing_golden(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    g1 = corpus.golden_images()[0]
    annotations_by_uid = {
        1: {g1.image_id: [_miner_item()]},
        2: {},
    }
    scores = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid=annotations_by_uid,
        fidelity_scorer=AnnotationFidelityScorer(),
        consensus_scorer=ConsensusScorer(),
        hallucination_penalty=0.5,
        golden_missing_penalty=0.5,
    )
    assert scores[2].golden_missing_count == len(corpus.golden_images())
    assert scores[2].average_score(golden_missing_penalty=0.5) < scores[1].average_score(
        golden_missing_penalty=0.5
    )


def test_evaluate_round_annotations_only_penalizes_missing_injected_golden(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    g1, g2 = corpus.golden_images()
    annotations_by_uid = {
        1: {g1.image_id: [_miner_item()]},
        2: {},
    }
    scores = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid=annotations_by_uid,
        fidelity_scorer=AnnotationFidelityScorer(),
        consensus_scorer=ConsensusScorer(),
        hallucination_penalty=0.5,
        golden_missing_penalty=0.5,
        expected_golden_ids_by_uid={
            1: [g1.image_id],
            2: [g1.image_id],
        },
    )
    assert scores[1].golden_missing_count == 0
    assert scores[2].golden_missing_count == 1
    assert g2.image_id not in scores[2].fidelity_scores_by_image_id


def test_evaluate_round_annotations_aggregates_correctly(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    g1 = corpus.golden_images()[0]
    g2 = corpus.golden_images()[1]
    pool = corpus.annotation_images()[0]
    annotations_by_uid = {
        1: {
            g1.image_id: [_miner_item()],
            pool.image_id: [_miner_item(cls="trip_hazard", bbox=(20, 20, 80, 80))],
        },
        2: {
            g1.image_id: [_miner_item(cls="other", bbox=(140, 140, 170, 170))],
            pool.image_id: [_miner_item(cls="trip_hazard", bbox=(22, 22, 82, 82))],
        },
    }
    fidelity = AnnotationFidelityScorer()
    consensus = ConsensusScorer()
    scores = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid=annotations_by_uid,
        fidelity_scorer=fidelity,
        consensus_scorer=consensus,
        hallucination_penalty=0.5,
    )
    assert scores[1].fidelity_scores_by_image_id[g1.image_id] > scores[2].fidelity_scores_by_image_id[g1.image_id]
    assert scores[1].consensus_scores_by_image_id[pool.image_id] > 0.0


# ---------------------------------------------------------------------------
# Dataset assembler
# ---------------------------------------------------------------------------

def test_dataset_assembler_picks_best_per_image_id(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    g1 = corpus.golden_images()[0]
    pool = corpus.annotation_images()[0]
    annotations = {
        1: {g1.image_id: [_miner_item()], pool.image_id: [_miner_item(cls="trip_hazard")]},
        2: {g1.image_id: [_miner_item(cls="other", bbox=(150, 150, 170, 170))],
            pool.image_id: [_miner_item(cls="trip_hazard")]},
    }
    fidelity = AnnotationFidelityScorer()
    consensus = ConsensusScorer()
    scores = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid=annotations,
        fidelity_scorer=fidelity,
        consensus_scorer=consensus,
        hallucination_penalty=0.5,
    )
    storage = (tmp_path / "commercial").as_uri()
    assembler = DatasetAssembler(corpus=corpus, storage_prefix=storage)
    winners = assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={1: "hk1", 2: "hk2"},
        model_versions={1: "v1", 2: "v2"},
        timestamps={1: "ts1", 2: "ts2"},
    )
    assert any(w.image_id == g1.image_id and w.chosen_uid == 1 for w in winners)
    assert assembler.ledger.adoption_counts.get(1, 0) >= 1


def test_dataset_assembler_export_local_jsonl(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    pool = corpus.annotation_images()[0]
    vote = MinerVote(
        miner_uid=7,
        miner_hotkey="hk7",
        class_voted="missing_hardhat",
        severity_voted="high",
        confidence=0.91,
        bounding_box=(22, 32, 92, 132),
        reliability_weight_at_aggregation=0.9,
    )
    obj = AggregatedObject(
        object_cluster_id=f"{pool.image_id}-cluster-0",
        accepted_hazard_class="missing_hardhat",
        accepted_severity="high",
        confidence=0.94,
        severity_confidence=0.91,
        class_posterior_distribution={"missing_hardhat": 0.94, "_background": 0.06},
        severity_posterior_distribution={"high": 0.91, "medium": 0.09},
        fused_bounding_box=(22, 32, 92, 132),
        spatial_mean_iou_to_median=0.95,
        miner_votes=[vote],
        escalation_reason=None,
    )
    winners = [
        WinningAnnotation(
            image_id=pool.image_id,
            score=0.85,
            chosen_uid=7,
            is_golden=False,
            aggregation_method="bayesian_dawid_skene_v1",
            image_url=pool.image_url,
            width=pool.width,
            height=pool.height,
            escalation_required=False,
            escalation_reason=None,
            accepted_objects=[obj],
            miner_contribution_scores={7: 1.0},
            reliability_window="2026-05-10T12:00:00Z/2026-05-11T12:00:00Z",
            acceptance_thresholds={"confidence": 0.9, "severity_confidence": 0.8, "min_voters": 2},
            validator_version="1.2.0",
            timestamp="2026-05-09T16:00:00Z",
        )
    ]
    target_dir = tmp_path / "commercial"
    assembler = DatasetAssembler(corpus=corpus, storage_prefix=target_dir.as_uri())
    uri = assembler.export(winners, round_id="step-1")
    assert uri.endswith("commercial-dataset-step-1.jsonl")
    master = target_dir / "commercial-dataset.jsonl"
    assert master.exists()
    parsed = [json.loads(line) for line in master.read_text().splitlines() if line.strip()]
    assert parsed[0]["image_id"] == pool.image_id
    assert parsed[0]["chosen_uid"] == 7
    assert parsed[0]["aggregation_method"] == "bayesian_dawid_skene_v1"
    assert parsed[0]["objects"][0]["accepted_hazard_class"] == "missing_hardhat"


def test_dataset_assembler_export_excludes_golden_rows(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    g1 = corpus.golden_images()[0]
    pool = corpus.annotation_images()[0]
    winners = [
        WinningAnnotation(
            image_id=g1.image_id,
            score=0.9,
            chosen_uid=1,
            is_golden=True,
            aggregation_method="golden_fidelity_v1",
            image_url=g1.image_url,
            width=g1.width,
            height=g1.height,
            escalation_required=False,
            escalation_reason=None,
            accepted_objects=[],
            miner_contribution_scores={1: 1.0},
            reliability_window="w",
            acceptance_thresholds={"confidence": 0.9, "severity_confidence": 0.8, "min_voters": 2},
            validator_version="1.2.0",
            timestamp="2026-05-09T16:00:00Z",
        ),
        WinningAnnotation(
            image_id=pool.image_id,
            score=0.7,
            chosen_uid=2,
            is_golden=False,
            aggregation_method="bayesian_dawid_skene_v1",
            image_url=pool.image_url,
            width=pool.width,
            height=pool.height,
            escalation_required=False,
            escalation_reason=None,
            accepted_objects=[],
            miner_contribution_scores={2: 1.0},
            reliability_window="w",
            acceptance_thresholds={"confidence": 0.9, "severity_confidence": 0.8, "min_voters": 2},
            validator_version="1.2.0",
            timestamp="2026-05-09T16:00:01Z",
        ),
    ]
    target_dir = tmp_path / "commercial"
    assembler = DatasetAssembler(corpus=corpus, storage_prefix=target_dir.as_uri())
    assembler.export(winners, round_id="step-g")
    master = target_dir / "commercial-dataset.jsonl"
    lines = [ln for ln in master.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["image_id"] == pool.image_id
    assert row["chosen_uid"] == 2


def test_dataset_assembler_export_skips_when_only_golden_winners(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    g1 = corpus.golden_images()[0]
    winners = [
        WinningAnnotation(
            image_id=g1.image_id,
            score=0.9,
            chosen_uid=1,
            is_golden=True,
            aggregation_method="golden_fidelity_v1",
            image_url=g1.image_url,
            width=g1.width,
            height=g1.height,
            escalation_required=False,
            escalation_reason=None,
            accepted_objects=[],
            miner_contribution_scores={1: 1.0},
            reliability_window="w",
            acceptance_thresholds={"confidence": 0.9, "severity_confidence": 0.8, "min_voters": 2},
            validator_version="1.2.0",
            timestamp="2026-05-09T16:00:00Z",
        ),
    ]
    assembler = DatasetAssembler(corpus=corpus, storage_prefix=(tmp_path / "c").as_uri())
    assert assembler.export(winners, round_id="step-x") == ""


# ---------------------------------------------------------------------------
# Reward composer
# ---------------------------------------------------------------------------

def test_dual_reward_composer_combines_three_signals(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    g1 = corpus.golden_images()[0]
    pool = corpus.annotation_images()[0]
    annotations = {
        1: {g1.image_id: [_miner_item()], pool.image_id: [_miner_item(cls="trip_hazard")]},
        2: {g1.image_id: [_miner_item(cls="other", bbox=(150, 150, 170, 170))],
            pool.image_id: [_miner_item(cls="trip_hazard")]},
    }
    scores = evaluate_round_annotations(
        corpus=corpus,
        annotations_by_uid=annotations,
        fidelity_scorer=AnnotationFidelityScorer(),
        consensus_scorer=ConsensusScorer(),
        hallucination_penalty=0.5,
    )
    storage = (tmp_path / "commercial").as_uri()
    assembler = DatasetAssembler(corpus=corpus, storage_prefix=storage)
    winners = assembler.assemble(
        per_miner_scores=scores,
        annotations_by_uid=annotations,
        miner_hotkeys={1: "hk1", 2: "hk2"},
        model_versions={1: "v1", 2: "v2"},
        timestamps={1: "ts1", 2: "ts2"},
    )

    composer = DualFlywheelRewardComposer(alpha=0.7)
    rewards, breakdowns = composer.compose(
        uids=[1, 2],
        annotation_scores=scores,
        ledger=assembler.ledger,
        round_winners=winners,
    )
    assert rewards[0] > rewards[1]  # uid 1 should clearly win
    assert breakdowns[0].annotation_score > 0.0
    assert breakdowns[0].adoption_bonus >= 0.0
    assert breakdowns[0].final_score == pytest.approx(rewards[0], abs=1e-6)


def test_dual_reward_composer_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        DualFlywheelRewardComposer(alpha=1.5)


# ---------------------------------------------------------------------------
# Adoption ledger persistence
# ---------------------------------------------------------------------------

def test_adoption_ledger_round_trips():
    ledger = AdoptionLedger()
    ledger.adoption_counts = {1: 5, 2: 3}
    ledger.last_round_counts = {1: 2}
    ledger.adoption_contributions = {1: 2.5, 2: 1.0}
    ledger.last_round_contributions = {1: 0.8}
    ledger.rounds_observed = 7
    serialized = ledger.to_jsonable()
    restored = AdoptionLedger.from_jsonable(serialized)
    assert restored.adoption_counts == {1: 5, 2: 3}
    assert restored.last_round_counts == {1: 2}
    assert restored.adoption_contributions == {1: 2.5, 2: 1.0}
    assert restored.last_round_contributions == {1: 0.8}
    assert restored.rounds_observed == 7


# ---------------------------------------------------------------------------
# Reliability accumulator serialization/decay
# ---------------------------------------------------------------------------

def test_reliability_accumulator_serialization_and_decay(tmp_path):
    corpus = _build_synthetic_corpus(tmp_path)
    g1 = corpus.golden_images()[0]
    acc = _ReliabilityAccumulator(decay=0.9)
    acc.update(1, [_miner_item()], g1)
    acc.update(2, [_miner_item(cls="other", bbox=(150, 150, 170, 170))], g1)

    # Verify serialization
    data = acc.to_jsonable()
    restored = _ReliabilityAccumulator.from_jsonable(data)
    assert restored.tp[1]["missing_hardhat"] == 1.0
    assert restored.fp[2]["other"] == 1.0

    # Verify decay
    restored.decay = 0.5
    restored.decay_all()
    assert restored.tp[1]["missing_hardhat"] == 0.5
    assert restored.fp[2]["other"] == 0.5
