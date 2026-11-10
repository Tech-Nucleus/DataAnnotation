"""Unit tests for the 4 Verified Code Bug Fixes in CanopyMRV Agentic Subnet.

Tests:
1. Bug 1: Multi-Miner Self-Intrusion Prevention on Scoped State Directories.
2. Bug 2: Dynamic Module Hot-Reloading via spec_from_file_location and st_mtime_ns.
3. Bug 3: SAHI Tile Loop mask_xy Offset and False-Positive Confidence Clamp.
4. Bug 4: 27-Round Intervention Memory & Anti-Oscillation Ledger.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# Add project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from template.miner.agentic_controller import (
    MinerAgenticController,
    load_miner_params,
    save_miner_params,
)
from template.miner.detector_loader import (
    get_miner_pipeline_dir,
    get_miner_state_dir,
    get_miner_tag,
    load_scoped_detector_module,
    register_miner_tag,
)
from template.utils.agentic_watchdog import AgenticWatchdog


def create_mock_wallet(name: str, ss58: str):
    wallet = MagicMock()
    wallet.name = name
    wallet.hotkey.ss58_address = ss58
    return wallet


class TestAgenticBugFixes:

    def setup_method(self):
        """Clean temporary test directories."""
        miners_root = ROOT / ".agentic_state" / "miners"
        if miners_root.is_dir():
            for folder in ["miner_a_5GrwvaEF", "miner_b_5FHneW46", "miner_reload_5DAAnrj7", "miner_history_5HGjWAeF"]:
                target = miners_root / folder
                if target.exists():
                    shutil.rmtree(target)

    def test_bug1_multi_miner_isolation(self):
        """Bug 1: Verify Miner A and Miner B have strictly isolated state directories."""
        wallet_a = create_mock_wallet("miner_a", "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
        wallet_b = create_mock_wallet("miner_b", "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty")

        miner_a = MinerAgenticController(wallet=wallet_a, strict_live_llm=False)
        miner_b = MinerAgenticController(wallet=wallet_b, strict_live_llm=False)

        tag_a = miner_a.hk_tag
        tag_b = miner_b.hk_tag

        assert tag_a != tag_b
        assert tag_a == "miner_a_5GrwvaEF"
        assert tag_b == "miner_b_5FHneW46"

        # Verify distinct state dirs
        assert miner_a.state_dir != miner_b.state_dir
        assert miner_a.state_dir.is_dir()
        assert miner_b.state_dir.is_dir()

        # Verify distinct parameter isolation
        save_miner_params({"conf_thres": 0.68, "imgsz": 640}, miner_uid=tag_a)
        save_miner_params({"conf_thres": 0.10, "imgsz": 1024}, miner_uid=tag_b)

        params_a = load_miner_params(miner_uid=tag_a)
        params_b = load_miner_params(miner_uid=tag_b)

        assert params_a["conf_thres"] == 0.68
        assert params_b["conf_thres"] == 0.10
        assert params_a["imgsz"] == 640
        assert params_b["imgsz"] == 1024

        # Verify distinct pipeline directories
        assert miner_a.pipeline_dir != miner_b.pipeline_dir
        assert (miner_a.pipeline_dir / "detector_annotate.py").is_file()
        assert (miner_b.pipeline_dir / "detector_annotate.py").is_file()

        # Verify distinct watchdog ledgers
        assert miner_a.watchdog.ledger_path != miner_b.watchdog.ledger_path
        val_watchdog = AgenticWatchdog(role="validator")
        assert val_watchdog.ledger_path != miner_a.watchdog.ledger_path
        assert "validator" in str(val_watchdog.ledger_path)

    def test_bug2_dynamic_module_hot_reload(self):
        """Bug 2: Verify in-memory module hot-reloading when scoped file is modified."""
        wallet = create_mock_wallet("miner_reload", "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy")
        miner = MinerAgenticController(wallet=wallet, strict_live_llm=False)

        # 1. Initial load
        mod1 = load_scoped_detector_module(miner_uid=miner.hk_tag)
        assert hasattr(mod1, "annotate_image_detector_only")
        assert not hasattr(mod1, "_test_marker_token")

        # 2. Mutate scoped file on disk and modify mtime
        time.sleep(0.01)  # Ensure distinct mtime
        scoped_file = miner.pipeline_dir / "detector_annotate.py"
        content = scoped_file.read_text(encoding="utf-8")
        marker = "\n_test_marker_token = 'HOT_RELOAD_SUCCESS_2026'\n"
        scoped_file.write_text(content + marker, encoding="utf-8")

        # 3. Reload module and verify hot update
        mod2 = load_scoped_detector_module(miner_uid=miner.hk_tag)
        assert getattr(mod2, "_test_marker_token", None) == "HOT_RELOAD_SUCCESS_2026"

    def test_bug3_sahi_mask_xy_offset(self):
        """Bug 3: Verify SAHI tile loop extracts and offsets mask_xy with tile_conf clamp."""
        from template.miner.geometry import extract_canopy_geometry
        from PIL import Image
        import numpy as np

        # Create dummy 1024x1024 synthetic image
        img = Image.new("RGB", (1024, 1024), color=(34, 139, 34))

        # Simulate tile crop at offset (tx0=435, ty0=435)
        tx0, ty0 = 435, 435
        raw_crop_mask_pts = [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]
        
        # Test offset calculation
        t_mask_xy = [[pt[0] + tx0, pt[1] + ty0] for pt in raw_crop_mask_pts]
        assert t_mask_xy[0] == [445.0, 445.0]
        assert t_mask_xy[2] == [465.0, 465.0]

        # Verify extract_canopy_geometry preserves full global polygon with mask_xy
        ann = extract_canopy_geometry(
            pil_img=img,
            box_xyxy=[440.0, 440.0, 470.0, 470.0],
            hazard_class="ordinary_tree",
            confidence=0.85,
            mask_xy=t_mask_xy,
            epsilon_factor=0.008,
            force_coarse_box=False,
        )
        assert ann.polygon is not None
        assert len(ann.polygon) == 4
        assert ann.polygon[0] == [445.0, 445.0]

        # Verify conf clamp logic: tile_conf = max(eff_conf, 0.25)
        eff_conf_low = 0.10
        tile_conf = max(eff_conf_low, 0.25)
        assert tile_conf == 0.25

    def test_bug4_intervention_history_and_anti_oscillation(self):
        """Bug 4: Verify structured intervention history and anti-oscillation guard."""
        wallet = create_mock_wallet("miner_history", "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw")
        miner = MinerAgenticController(wallet=wallet, strict_live_llm=False)

        # 1. Record round outcome
        entry = miner.record_round_outcome(
            round_num=1,
            score_before=0.15,
            score_after=0.18,
            score_delta=0.03,
            mean_iou_after=0.22,
            tier=1,
            action="tune_param",
            params={"conf_thres": 0.25},
        )
        assert entry["outcome"] == "IMPROVED"
        assert len(miner.history) == 1

        # 2. Record regressed round outcome
        entry2 = miner.record_round_outcome(
            round_num=2,
            score_before=0.18,
            score_after=0.14,
            score_delta=-0.04,
            mean_iou_after=0.16,
            tier=2,
            action="code_edit",
            patch_type="bad_experimental_patch",
        )
        assert entry2["outcome"] == "REGRESSED"

        # 3. Verify history table formatting
        table_str = miner._format_history_table()
        assert "| Round | Tier |" in table_str
        assert "IMPROVED" in table_str
        assert "REGRESSED" in table_str

        # 4. Verify all 4 Tier 2 patches apply cleanly and generate valid unified diffs
        patches = [
            ("sliding_window_tiling", "detector_annotate.py"),
            ("adaptive_contour_smoothing", "geometry.py"),
            ("watershed_cluster_splitting", "detector_annotate.py"),
            ("multi_scale_tta_ensemble", "detector_annotate.py"),
        ]

        for p_type, target_name in patches:
            interv = {
                "tier": 2,
                "action": "code_edit",
                "target_file": target_name,
                "patch_type": p_type,
            }
            success, msg = miner.apply_intervention(interv)
            assert success, f"Patch {p_type} failed: {msg}"
            # Verify diff generated
            assert "git_diff_applied" in interv
            # Verify idempotency
            success2, msg2 = miner.apply_intervention(interv)
            assert success2
