"""
Tests for the three critical fixes:
  A. Zero-detection handling (fidelity scoring for empty golden images)
  B. Self-contained commercial export (image_url validation)
  C. Single-miner fallback policy
"""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from template.hazard.annotation_eval import (
    AnnotationFidelityScorer,
    ConsensusScorer,
    FidelityComponents,
    evaluate_round_annotations,
)
from template.hazard.image_corpus import GoldenAnnotation, GoldenImage


# ===========================================================================
# Task A: Zero-detection handling
# ===========================================================================

class TestZeroDetectionFidelity:
    """Verify that zero-detection golden images are scored correctly."""

    def _make_golden(self, annotations=()):
        """Build a minimal GoldenImage for testing."""
        return GoldenImage(
            image_id="test_image_001",
            image_path=Path("/dev/null"),
            image_url="http://example.com/test.jpg",
            width=640,
            height=480,
            annotations=tuple(annotations),
        )

    def test_empty_golden_empty_miner_is_perfect(self):
        """Miner correctly reports nothing on a clean image → fidelity=1.0."""
        scorer = AnnotationFidelityScorer()
        golden = self._make_golden(annotations=())
        result = scorer.score(miner_items=[], golden=golden)
        assert result.fidelity == 1.0
        assert result.iou == 1.0
        assert result.class_severity == 1.0
        assert result.hallucination_penalty == 1.0
        assert result.hallucinated_count == 0

    def test_empty_golden_miner_hallucinated(self):
        """Miner fabricates detections on a clean image → fidelity=0.0."""
        from template.protocol import PerImageAnnotationItem

        scorer = AnnotationFidelityScorer()
        golden = self._make_golden(annotations=())
        fake_item = PerImageAnnotationItem(
            hazard_class="hardhat",
            bounding_box=[10, 10, 50, 50],
            confidence=0.9,
            severity="medium",
        )
        result = scorer.score(miner_items=[fake_item], golden=golden)
        assert result.fidelity == 0.0
        assert result.hallucinated_count == 1
        assert result.hallucination_penalty < 1.0

    def test_nonempty_golden_miner_empty_is_zero(self):
        """Miner reports nothing on an image with actual hazards → fidelity=0.0."""
        scorer = AnnotationFidelityScorer()
        golden = self._make_golden(annotations=(
            GoldenAnnotation(
                hazard_class="hardhat",
                bounding_box=(10, 10, 50, 50),
                severity="high",
            ),
        ))
        result = scorer.score(miner_items=[], golden=golden)
        assert result.fidelity == 0.0
        assert result.matched_count == 0


# ===========================================================================
# Task C: Single-miner fallback
# ===========================================================================

class TestSingleMinerFallback:
    """Verify that the single-miner fallback policy works correctly."""

    def test_single_miner_fallback_enabled(self):
        """With fallback enabled, a reliable single miner is adopted."""
        from template.hazard.annotation_eval import PerMinerAnnotationScore
        from template.hazard.dataset_assembler import DatasetAssembler
        from template.protocol import PerImageAnnotationItem

        corpus = MagicMock()
        corpus.is_golden.return_value = False
        corpus.golden_images.return_value = []
        corpus.annotation_images.return_value = []
        corpus.golden_lookup.return_value = None

        assembler = DatasetAssembler(
            corpus=corpus,
            storage_prefix="file:///tmp/test",
        )

        item = PerImageAnnotationItem(
            hazard_class="hardhat",
            bounding_box=[10, 10, 50, 50],
            confidence=0.9,
            severity="medium",
        )

        score = PerMinerAnnotationScore(uid=0)
        score.fidelity_scores_by_image_id = {"golden1": 0.8}

        with patch.dict(os.environ, {
            "FALLBACK_SINGLE_MINER_ENABLED": "1",
            "FALLBACK_SINGLE_MINER_MIN_RELIABILITY": "0.3",
        }):
            # We need to reimport to pick up the env var
            import importlib
            import template.hazard.dataset_assembler as da_mod
            orig_enabled = da_mod._FALLBACK_SINGLE_MINER_ENABLED
            orig_reliability = da_mod._FALLBACK_SINGLE_MINER_MIN_RELIABILITY
            da_mod._FALLBACK_SINGLE_MINER_ENABLED = True
            da_mod._FALLBACK_SINGLE_MINER_MIN_RELIABILITY = 0.3

            try:
                result = assembler._aggregate_image(
                    image_id="pool_image_1",
                    image_votes={0: [item]},
                    per_miner_scores={0: score},
                    miner_hotkeys={0: "hotkey_0"},
                    priors={"_background": 0.3, "hardhat": 0.7},
                )
            finally:
                da_mod._FALLBACK_SINGLE_MINER_ENABLED = orig_enabled
                da_mod._FALLBACK_SINGLE_MINER_MIN_RELIABILITY = orig_reliability

        assert result["escalation_required"] is False
        assert result["chosen_uid"] == 0
        assert len(result["objects"]) == 1
        assert result["objects"][0].aggregation_method == "single_miner_fallback_v1"

    def test_single_miner_fallback_disabled_escalates(self):
        """With fallback disabled, a single miner always triggers escalation."""
        from template.hazard.annotation_eval import PerMinerAnnotationScore
        from template.hazard.dataset_assembler import DatasetAssembler
        from template.protocol import PerImageAnnotationItem

        corpus = MagicMock()
        corpus.is_golden.return_value = False
        corpus.golden_images.return_value = []

        assembler = DatasetAssembler(
            corpus=corpus,
            storage_prefix="file:///tmp/test",
        )

        item = PerImageAnnotationItem(
            hazard_class="hardhat",
            bounding_box=[10, 10, 50, 50],
            confidence=0.9,
            severity="medium",
        )

        score = PerMinerAnnotationScore(uid=0)
        score.fidelity_scores_by_image_id = {"golden1": 0.8}

        import template.hazard.dataset_assembler as da_mod
        orig_enabled = da_mod._FALLBACK_SINGLE_MINER_ENABLED
        da_mod._FALLBACK_SINGLE_MINER_ENABLED = False

        try:
            result = assembler._aggregate_image(
                image_id="pool_image_1",
                image_votes={0: [item]},
                per_miner_scores={0: score},
                miner_hotkeys={0: "hotkey_0"},
                priors={"_background": 0.3, "hardhat": 0.7},
            )
        finally:
            da_mod._FALLBACK_SINGLE_MINER_ENABLED = orig_enabled

        assert result["escalation_required"] is True
        assert result["escalation_reason"] == "only_one_miner"


# ===========================================================================
# Task B: Commercial export image_url validation
# ===========================================================================

class TestCommercialImageUrlValidation:
    """Verify schema validation enforces HTTP(S) image_url."""

    def test_file_url_fails_validation(self):
        """file:// URLs must fail the schema check."""
        from scripts.production_readiness_eval import validate_commercial_row

        row = {
            "image_id": "abc",
            "image_url": "file:///home/user/image.jpg",
            "annotated_image_url": "https://pub-xxx.r2.dev/commercial/annotated-images/abc.jpg",
            "width": 640,
            "height": 480,
            "is_golden": False,
            "aggregation_method": "bayesian_dawid_skene_v1",
            "acceptance_thresholds": {},
            "reliability_window": "",
            "escalation_required": False,
            "escalation_reason": None,
            "validator_version": "1.2.0",
            "audit_hash": "abc",
            "miner_contribution_scores": {},
            "objects": [{}],
            "score": 0.9,
            "chosen_uid": 0,
            "timestamp": "2025-01-01T00:00:00Z",
        }
        errors = validate_commercial_row(row, line_no=1)
        url_errors = [e for e in errors if "image_url" in e and "HTTP(S)" in e]
        assert len(url_errors) > 0, f"Expected image_url HTTP(S) error, got: {errors}"

    def test_https_url_passes_validation(self):
        """https:// URLs must pass the schema check for image_url."""
        from scripts.production_readiness_eval import validate_commercial_row

        row = {
            "image_id": "abc",
            "image_url": "https://pub-xxx.r2.dev/commercial-images/abc.jpg",
            "annotated_image_url": "https://pub-xxx.r2.dev/commercial/annotated-images/abc.jpg",
            "width": 640,
            "height": 480,
            "is_golden": False,
            "aggregation_method": "bayesian_dawid_skene_v1",
            "acceptance_thresholds": {},
            "reliability_window": "",
            "escalation_required": False,
            "escalation_reason": None,
            "validator_version": "1.2.0",
            "audit_hash": "placeholder",
            "miner_contribution_scores": {},
            "objects": [{}],
            "score": 0.9,
            "chosen_uid": 0,
            "timestamp": "2025-01-01T00:00:00Z",
        }
        errors = validate_commercial_row(row, line_no=1)
        url_errors = [e for e in errors if "image_url" in e]
        assert len(url_errors) == 0, f"Unexpected image_url errors: {url_errors}"

    def test_empty_url_fails_validation(self):
        """Empty image_url must fail."""
        from scripts.production_readiness_eval import validate_commercial_row

        row = {
            "image_id": "abc",
            "image_url": "",
            "annotated_image_url": "https://pub-xxx.r2.dev/commercial/annotated-images/abc.jpg",
            "width": 640,
            "height": 480,
            "is_golden": False,
            "aggregation_method": "test",
            "acceptance_thresholds": {},
            "reliability_window": "",
            "escalation_required": True,
            "escalation_reason": "test",
            "validator_version": "1.0",
            "audit_hash": "",
            "miner_contribution_scores": {},
            "objects": [],
            "score": 0.0,
            "chosen_uid": -1,
            "timestamp": "",
        }
        errors = validate_commercial_row(row, line_no=1)
        url_errors = [e for e in errors if "image_url" in e and "empty" in e]
        assert len(url_errors) > 0
