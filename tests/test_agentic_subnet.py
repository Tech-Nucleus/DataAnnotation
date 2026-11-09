"""Unit and integration tests for CanopyMRV Subnet 498 Agentic Upgrade.

Covers:
1. Side-by-side composite comparison grid generation strictly in memory.
2. Zero-leakage feedback sanitization.
3. Scoped per-miner leaderboard receipt construction.
4. Canary verification & automatic rollback guard.
5. Registration guardrail (human-in-the-loop registration boundary).
6. Watchdog sleep tuning and token/call circuit breaker enforcement.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from template.hazard.image_corpus import GoldenAnnotation, GoldenImage
from template.miner.agentic_controller import (
    ALLOWLISTED_PATHS,
    DEFAULT_PARAMS,
    MinerAgenticController,
    is_path_allowlisted,
    load_miner_params,
    run_canary_verification,
    save_miner_params,
)
from template.protocol import AnnotationTask, PerImageAnnotationItem
from template.validator.agentic_feedback import (
    ValidatorAgenticEngine,
    build_comparison_grid,
    compute_chip_geometry_metrics,
    sanitize_feedback,
)
from template.utils.agentic_watchdog import AgenticWatchdog


# -------------------------------------------------------------------------
# Test 1: Composite Image Generation in Memory
# -------------------------------------------------------------------------
def test_composite_image_generation_in_memory(tmp_path):
    """Verify single stitched side-by-side composite JPEG is built strictly in RAM."""
    # Create 2 synthetic golden images
    dummy_img = np.full((512, 512, 3), 100, dtype=np.uint8)
    img_path1 = tmp_path / "golden1.jpg"
    img_path2 = tmp_path / "golden2.jpg"
    cv2.imwrite(str(img_path1), dummy_img)
    cv2.imwrite(str(img_path2), dummy_img)

    g1 = GoldenImage(
        image_id="golden_sample_01",
        image_url="file://golden1.jpg",
        image_path=img_path1,
        width=512,
        height=512,
        annotations=[
            GoldenAnnotation(
                hazard_class="dense_tree",
                bounding_box=[50.0, 50.0, 150.0, 150.0],
                severity="medium",
                polygon=((50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0)),
            )
        ],
    )
    g2 = GoldenImage(
        image_id="golden_sample_02",
        image_url="file://golden2.jpg",
        image_path=img_path2,
        width=512,
        height=512,
        annotations=[
            GoldenAnnotation(
                hazard_class="ordinary_tree",
                bounding_box=[100.0, 100.0, 200.0, 200.0],
                severity="low",
                polygon=((100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)),
            )
        ],
    )

    miner_items1 = [
        PerImageAnnotationItem(
            hazard_class="dense_tree",
            bounding_box=[55.0, 55.0, 145.0, 145.0],
            polygon=[[55, 55], [145, 55], [145, 145], [55, 145]],
            confidence=0.88,
        )
    ]
    miner_items2 = [
        PerImageAnnotationItem(
            hazard_class="ordinary_tree",
            bounding_box=[95.0, 95.0, 205.0, 205.0],
            polygon=[[95, 95], [205, 95], [205, 205], [95, 205]],
            confidence=0.91,
        )
    ]

    samples = [(g1, miner_items1), (g2, miner_items2)]
    jpeg_bytes = build_comparison_grid(samples, thumbnail_size=384, jpeg_quality=80)

    assert isinstance(jpeg_bytes, bytes)
    assert len(jpeg_bytes) > 1000
    # Standard JPEG SOI marker \xff\xd8
    assert jpeg_bytes[:2] == b"\xff\xd8"

    # Decode and check dimensions: width = 768, height = 384 * 2 = 768
    dec = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert dec is not None
    assert dec.shape[1] == 768
    assert dec.shape[0] == 384 * 2


# -------------------------------------------------------------------------
# Test 2: Zero-Leakage Feedback Sanitizer
# -------------------------------------------------------------------------
def test_zero_leakage_sanitizer():
    """Ensure golden image IDs, coordinates, chip references, and filenames are purged."""
    raw = (
        "In chip 1 (golden_id_948194) golden_img_02.jpg at [124.5, 342.1], the miner missed "
        "small crowns. Polygons are overly coarse 4-point boxes. Please lower conf threshold."
    )
    sanitized = sanitize_feedback(raw, golden_image_ids=["golden_id_948194"])

    assert "golden_id_948194" not in sanitized
    assert "golden_img_02.jpg" not in sanitized
    assert "[124.5, 342.1]" not in sanitized
    assert "chip 1" not in sanitized.lower()

    # Verify sentence count <= 2 and word count <= 45
    sentences = [s.strip() for s in sanitized.split(".") if s.strip()]
    assert len(sentences) <= 2
    assert len(sanitized.split()) <= 45


# -------------------------------------------------------------------------
# Test 3: Scoped Per-Miner Leaderboard Receipts
# -------------------------------------------------------------------------
def test_scoped_per_miner_leaderboard_receipt():
    """Verify each miner receives ONLY their scoped telemetry with no cross-miner leakage."""
    watchdog = AgenticWatchdog(role="validator", max_daily_calls=10, state_dir="./.test_state")
    engine = ValidatorAgenticEngine(watchdog=watchdog, enabled=True)

    # 2 miners
    uids = [0, 1]
    hotkeys = ["5MinerOneHotkey1111111111111111111111111111111", "5MinerTwoHotkey2222222222222222222222222222222"]

    score1 = MagicMock()
    score1.fidelity_scores_by_image_id = {"c1": 0.85, "c2": 0.75}
    score2 = MagicMock()
    score2.fidelity_scores_by_image_id = {"c1": 0.45, "c2": 0.55}
    scores_map = {0: score1, 1: score2}

    corpus = MagicMock()
    corpus.golden_lookup = {}
    annotations_by_uid = {0: {}, 1: {}}

    receipts = engine.process_round(
        round_num=10,
        batch_id="batch_10",
        uids=uids,
        hotkeys=hotkeys,
        per_miner_scores=scores_map,
        corpus=corpus,
        annotations_by_uid=annotations_by_uid,
    )

    r1 = receipts[hotkeys[0]]
    r2 = receipts[hotkeys[1]]

    assert r1["schema_version"] == "agentic.feedback.v1"
    assert r1["miner_uid"] == 0
    assert r1["hotkey_ss58"] == hotkeys[0]
    assert r1["leaderboard"]["self_rank"] == 1
    assert r1["leaderboard"]["total_active_miners"] == 2
    assert r1["leaderboard"]["self_fidelity_score"] == 0.80

    assert r2["miner_uid"] == 1
    assert r2["hotkey_ss58"] == hotkeys[1]
    assert r2["leaderboard"]["self_rank"] == 2
    assert r2["leaderboard"]["self_fidelity_score"] == 0.50

    # Ensure secondary transport helper returns valid JSON string
    json_str = engine.get_latest_feedback_json(hotkeys[0])
    assert json_str is not None
    parsed = json.loads(json_str)
    assert parsed["miner_uid"] == 0

    # Cleanup
    shutil.rmtree("./.test_state", ignore_errors=True)


# -------------------------------------------------------------------------
# Test 4: Canary Verification & Snapshot Rollback Guard
# -------------------------------------------------------------------------
def test_canary_verification_and_snapshot_rollback(tmp_path):
    """Verify canary runs on test_clustered_trees.jpg and bad edits trigger clean rollback."""
    # Run valid canary pass
    ok, msg, stats = run_canary_verification()
    assert ok is True
    assert "passed" in msg.lower()
    assert stats["annotations_count"] >= 0

    # Test snapshot & rollback
    controller = MinerAgenticController()
    test_params = {"conf_thres": 0.22, "iou_thres": 0.44}
    save_miner_params(test_params)

    snap_tag = "test_snap_01"
    controller.create_snapshot(snap_tag)

    # Corrupt parameters
    save_miner_params({"conf_thres": 0.999, "iou_thres": 0.999})
    assert load_miner_params()["conf_thres"] == 0.999

    # Rollback
    rb_ok = controller.rollback(snap_tag)
    assert rb_ok is True
    restored = load_miner_params()
    assert restored["conf_thres"] == 0.22
    assert restored["iou_thres"] == 0.44

    # Cleanup
    shutil.rmtree(controller.snapshots_dir / snap_tag, ignore_errors=True)


# -------------------------------------------------------------------------
# Test 5: Registration Guardrail
# -------------------------------------------------------------------------
def test_registration_guardrail():
    """Verify miner agent enters standby if hotkey is not registered on netuid."""
    mock_wallet = MagicMock()
    mock_wallet.hotkey.ss58_address = "5UnregisteredHotkey"
    mock_metagraph = MagicMock()
    mock_metagraph.hotkeys = ["5RegisteredHotkeyA", "5RegisteredHotkeyB"]

    controller = MinerAgenticController(wallet=mock_wallet, metagraph=mock_metagraph, enabled=True)
    res = controller.step()
    # Must return None and refuse to invoke btcli
    assert res is None


# -------------------------------------------------------------------------
# Test 6: Watchdog Sleep & Token Circuit Breaker
# -------------------------------------------------------------------------
def test_watchdog_sleep_and_budget_enforcement(tmp_path):
    """Verify dynamic sleep interval overrides and daily API call limits."""
    state_dir = tmp_path / ".agentic_state"
    wd = AgenticWatchdog(role="miner", sleep_seconds=100.0, max_daily_calls=2, state_dir=state_dir)

    assert wd.get_sleep_seconds() == 100.0
    wd.set_sleep_seconds(45.0)
    assert wd.get_sleep_seconds() == 45.0

    # Call budget checks
    assert wd.check_budget() is True
    wd.record_call(prompt_tokens=150, completion_tokens=30)
    assert wd.check_budget() is True
    wd.record_call(prompt_tokens=150, completion_tokens=30)

    # Limit reached
    assert wd.check_budget() is False
    ledger = wd.get_ledger_summary()
    assert ledger["circuit_breaker_tripped"] is True
    assert ledger["total_calls"] == 2
    assert ledger["total_prompt_tokens"] == 300
    assert ledger["total_completion_tokens"] == 60


# -------------------------------------------------------------------------
# Test 7: Allowlist and Blocklist Enforcement
# -------------------------------------------------------------------------
def test_path_allowlist_enforcement():
    """Verify agent cannot edit validator, protocol, or wallet files."""
    assert is_path_allowlisted("template/miner/detector_annotate.py") is True
    assert is_path_allowlisted("template/miner/geometry.py") is True
    assert is_path_allowlisted("neurons/validator.py") is False
    assert is_path_allowlisted("template/validator/agentic_feedback.py") is False
    assert is_path_allowlisted("template/protocol.py") is False
    assert is_path_allowlisted("~/.bittensor/wallets/miner/coldkey") is False
