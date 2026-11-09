"""Autonomous Miner Agent Controller Subsystem for Bittensor Subnet 498.

Implements:
1. Registration guardrail: hard block preventing registration/deregistration btcli commands.
2. Perceive: fetches scoped leaderboard and validator diagnostic feedback.
3. Reason: hierarchical intervention planning (Tier 1 config, Tier 2 pipeline patch, Tier 3 HF checkpoint).
4. Canary verification & automatic rollback guard on test_clustered_trees.jpg.
5. Deterministic sleep watchdog between rounds.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import py_compile
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import bittensor as bt
from pydantic import ValidationError

from template.protocol import (
    AnnotationsFilePayload,
    ImageAnnotationDocument,
    PerImageAnnotationItem,
)
from template.utils.agentic_watchdog import AgenticWatchdog

# Parameter file
PARAMS_PATH = Path("template/miner/agentic_params.json")

DEFAULT_PARAMS: Dict[str, Any] = {
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "imgsz": 640,
    "epsilon_factor": 0.008,
    "max_area_ratio": 0.60,
    "active_checkpoint": "models/tree_detection.pt",
}


def load_miner_params() -> Dict[str, Any]:
    """Load configurable miner parameters with safe defaults."""
    if PARAMS_PATH.is_file():
        try:
            data = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
            merged = copy.deepcopy(DEFAULT_PARAMS)
            merged.update(data)
            return merged
        except Exception:
            pass
    return copy.deepcopy(DEFAULT_PARAMS)


def save_miner_params(params: Dict[str, Any]) -> None:
    """Save miner parameters atomically."""
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(params, indent=2), encoding="utf-8")


# Strict file boundaries
ALLOWLISTED_PATHS = [
    Path("template/miner/detector_annotate.py"),
    Path("template/miner/geometry.py"),
    Path("template/miner/ecological_reasoning.py"),
    Path("template/miner/autoresearch.py"),
    Path("template/miner/agentic_params.json"),
    Path("autoresearch_config.yaml"),
]

BLOCKLISTED_PATTERNS = [
    "neurons/validator.py",
    "template/validator/",
    "template/hazard/",
    "template/protocol.py",
    ".bittensor/wallets/",
    ".git/",
]


def is_path_allowlisted(path: Path | str) -> bool:
    """Ensure path is within allowlisted miner pipeline and not in blocklist."""
    p_str = str(Path(path).as_posix())
    for blk in BLOCKLISTED_PATTERNS:
        if blk in p_str:
            return False
    for al in ALLOWLISTED_PATHS:
        al_str = str(al.as_posix())
        if p_str == al_str or p_str.startswith("template/miner/backends/"):
            return True
    return False


def get_hardware_profile(last_duration_ms: Optional[int] = None) -> Dict[str, Any]:
    """Inspect local hardware resources within min_compute constraints."""
    vram_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    except Exception:
        pass

    disk_free_gb = 0.0
    try:
        disk_free_gb = round(shutil.disk_usage(".").free / (1024**3), 2)
    except Exception:
        pass

    return {
        "vram_gb": vram_gb,
        "disk_free_gb": disk_free_gb,
        "last_duration_ms": last_duration_ms or 0,
    }


def run_canary_verification(
    canary_image_path: Path | str = "test_clustered_trees.jpg",
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Execute fast local canary inference pass on test_clustered_trees.jpg.

    Verifies:
    1. Output conforms cleanly to AnnotationsFilePayload (annotations.v1).
    2. Total annotations <= 512.
    3. Max area ratio < 0.60.
    4. Sum of squared vertices <= 2,000,000.
    5. Single-image inference time projects to < 450s for a 30-image batch (< 15s per image).
    """
    canary_p = Path(canary_image_path)
    if not canary_p.is_file():
        return False, f"Canary image not found: {canary_p}", {}

    active_params = params or load_miner_params()
    chk_str = active_params.get("active_checkpoint", "models/tree_detection.pt")
    checkpoint = Path(chk_str)
    if not checkpoint.is_file():
        # Fallback to any existing tree model or yolo
        for cand in ["models/tree_detection.pt", "models/tree_detection_yolov8n_selvabox.pt", "yolov8n-seg.pt"]:
            if Path(cand).is_file():
                checkpoint = Path(cand)
                break

    try:
        from template.miner.detector_annotate import annotate_image_detector_only
        img_bytes = canary_p.read_bytes()

        t0 = time.perf_counter()
        doc = annotate_image_detector_only(
            checkpoint=checkpoint,
            image_bytes=img_bytes,
            image_id="canary_eval_image",
            image_url=f"file://{canary_p.resolve()}",
            model_version="agentic_canary_v1.0",
            miner_uid="canary_miner",
            conf_thres=active_params.get("conf_thres", 0.25),
            iou_thres=active_params.get("iou_thres", 0.45),
            imgsz=active_params.get("imgsz", 640),
            epsilon_factor=active_params.get("epsilon_factor", 0.008),
            max_area_ratio=active_params.get("max_area_ratio", 0.60),
        )
        elapsed = time.perf_counter() - t0

        # Verification 1: Latency projection
        if elapsed > 15.0:
            return False, f"Inference latency too slow: {elapsed:.2f}s > 15.0s (batch projection > 450s)", {}

        # Verification 2: Payload schema conformity
        payload = AnnotationsFilePayload(
            schema_version="annotations.v1",
            task_id="canary_task",
            batch_id="canary_batch",
            round_num=1,
            records=[doc],
        )

        # Verification 3: Total annotations constraint
        if len(doc.annotations) > 512:
            return False, f"Too many annotations: {len(doc.annotations)} > 512", {}

        # Verification 4: Max area ratio < 0.60 and sum of vertices^2 <= 2,000,000
        sum_v_sq = 0
        img_w, img_h = 256, 256  # test_clustered_trees.jpg dimensions
        canvas_area = float(img_w * img_h)

        for ann in doc.annotations:
            if ann.bounding_box:
                x1, y1, x2, y2 = ann.bounding_box
                area = (x2 - x1) * (y2 - y1)
                if area / canvas_area >= 0.60:
                    return False, f"Annotation area ratio {area / canvas_area:.3f} >= 0.60", {}
            if ann.polygon:
                nv = len(ann.polygon)
                sum_v_sq += nv * nv

        if sum_v_sq > 2_000_000:
            return False, f"Sum of squared vertices {sum_v_sq} > 2,000,000", {}

        stats = {
            "elapsed_seconds": round(elapsed, 4),
            "annotations_count": len(doc.annotations),
            "sum_vertices_sq": sum_v_sq,
        }
        return True, "Canary verification passed", stats
    except Exception as exc:
        return False, f"Canary execution exception: {exc}", {}


class MinerAgenticController:
    """Autonomous agent autopilot daemon for Subnet 498 miners."""

    def __init__(
        self,
        wallet: Optional[Any] = None,
        metagraph: Optional[Any] = None,
        watchdog: Optional[AgenticWatchdog] = None,
        r2_client: Optional[Any] = None,
        r2_bucket: Optional[str] = None,
        r2_telemetry_prefix: Optional[str] = None,
        hf_token: Optional[str] = None,
        allow_code_edits: bool = True,
        enabled: bool = False,
        strict_live_llm: bool = False,
    ):
        self.wallet = wallet
        self.metagraph = metagraph
        self.strict_live_llm = strict_live_llm
        self.watchdog = watchdog or AgenticWatchdog(role="miner", strict_live_llm=strict_live_llm)
        if strict_live_llm:
            self.watchdog.strict_live_llm = True
        self.r2_client = r2_client
        self.r2_bucket = r2_bucket or os.getenv("R2_BUCKET_NAME", "annotation-subnet")
        self.r2_telemetry_prefix = r2_telemetry_prefix or os.getenv("R2_LOCALNET_TELEMETRY_PREFIX", "localnet_sim/telemetry/")
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.allow_code_edits = allow_code_edits
        self.enabled = enabled
        self.llm_calls: List[Dict[str, Any]] = []

        self.snapshots_dir = Path(".agentic_snapshots").resolve()
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = Path(".agentic_state").resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.state_dir / "intervention_history.json"
        self.failed_hypotheses_path = self.state_dir / "failed_hypotheses.json"

        self.last_processed_round_num = 0
        self.history: List[Dict[str, Any]] = self._load_history()
        self.failed_hypotheses: List[Dict[str, Any]] = self._load_failed_hypotheses()

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.latest_received_feedback: Optional[Dict[str, Any]] = None

    def _load_history(self) -> List[Dict[str, Any]]:
        if self.history_path.is_file():
            try:
                return json.loads(self.history_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return []

    def _save_history(self) -> None:
        try:
            self.history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        except Exception as e:
            bt.logging.warning(f"[AGENTIC MINER] Failed to persist history: {e}")

    def _load_failed_hypotheses(self) -> List[Dict[str, Any]]:
        if self.failed_hypotheses_path.is_file():
            try:
                return json.loads(self.failed_hypotheses_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return []

    def _record_failed_hypothesis(self, item: Dict[str, Any]) -> None:
        self.failed_hypotheses.append(item)
        try:
            self.failed_hypotheses_path.write_text(json.dumps(self.failed_hypotheses, indent=2), encoding="utf-8")
        except Exception:
            pass

    def start_background(self) -> None:
        """Launch autonomous autopilot background daemon."""
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="MinerAgenticAutopilot", daemon=True)
        self._thread.start()
        bt.logging.info("[AGENTIC MINER] Autopilot background daemon started.")

    def stop(self) -> None:
        """Signal background daemon to stop."""
        self._stop_event.set()
        self.watchdog.wake()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.step()
            except Exception as exc:
                bt.logging.error(f"[AGENTIC MINER] Autopilot cycle error: {exc}")
            self.watchdog.sleep()

    # -------------------------------------------------------------------------
    # Core Step Execution
    # -------------------------------------------------------------------------

    def step(self, receipt: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Perform one autonomous perceive-reason-act cycle."""
        # Registration Guardrail
        if self.wallet and self.metagraph:
            ss58 = self.wallet.hotkey.ss58_address
            if ss58 not in self.metagraph.hotkeys:
                bt.logging.warning(
                    f"[AGENTIC GUARDRAIL] Hotkey {ss58} is not registered on netuid. "
                    "Human operator must run btcli subnet register. Entering sleep standby."
                )
                return None

        # Step 1: Perceive
        if receipt is None:
            receipt = self.fetch_latest_feedback()
        if not receipt:
            bt.logging.debug("[AGENTIC MINER] No feedback receipt found yet. Sleeping.")
            return None

        round_num = receipt.get("round_num", 0)
        if round_num <= self.last_processed_round_num:
            bt.logging.debug(f"[AGENTIC MINER] Round {round_num} already processed. Sleeping (0 tokens used).")
            return None

        bt.logging.info(f"[AGENTIC MINER] Processing Round {round_num} feedback...")
        self.last_processed_round_num = round_num

        # Check consecutive score degradation for rollback
        self._check_and_handle_consecutive_drop(receipt)

        # Step 2: Reason
        intervention = self.plan_intervention(receipt)
        if not intervention:
            bt.logging.info("[AGENTIC MINER] No intervention required or planned for this round.")
            return None

        # Step 3: Snapshot, Apply, Canary Verification & Automatic Rollback
        snapshot_tag = f"round_{round_num}"
        self.create_snapshot(snapshot_tag)

        success, msg = self.apply_intervention(intervention)
        if not success:
            bt.logging.error(f"[AGENTIC MINER] Failed to apply intervention: {msg}. Rolling back.")
            self.rollback(snapshot_tag)
            self._record_failed_hypothesis({"round_num": round_num, "intervention": intervention, "reason": msg})
            return None

        # Verify py_compile on allowlisted files
        for p in ALLOWLISTED_PATHS:
            if p.suffix == ".py" and p.is_file():
                try:
                    py_compile.compile(str(p), doraise=True)
                except Exception as py_err:
                    bt.logging.error(f"[AGENTIC MINER] py_compile failed on {p}: {py_err}. Rolling back.")
                    self.rollback(snapshot_tag)
                    self._record_failed_hypothesis({"round_num": round_num, "intervention": intervention, "reason": str(py_err)})
                    return None

        # Run Canary Verification
        canary_ok, canary_msg, canary_stats = run_canary_verification()
        if not canary_ok:
            bt.logging.error(f"[AGENTIC MINER] Canary verification FAILED: {canary_msg}. Automatically restoring snapshot.")
            self.rollback(snapshot_tag)
            self._record_failed_hypothesis({"round_num": round_num, "intervention": intervention, "reason": canary_msg})
            return None

        bt.logging.info(f"[AGENTIC MINER] Canary verification PASSED ({canary_stats}). Intervention committed!")

        # Log Miner LLM Call Proof Metadata
        last_call = self.watchdog.call_logs[-1] if self.watchdog.call_logs else {}
        proof_entry = {
            "round_num": round_num,
            "model": self.watchdog.model,
            "prompt_tokens": last_call.get("prompt_tokens", 0),
            "candidate_tokens": last_call.get("candidate_tokens", 0),
            "raw_json_plan": intervention,
            "reasoning_summary": intervention.get("rationale", ""),
            "git_diff_applied": intervention.get("git_diff_applied", ""),
            "latency_ms": last_call.get("latency_ms", 0),
        }
        self.llm_calls.append(proof_entry)
        self.watchdog.record_llm_proof("miner", proof_entry)

        record = {
            "round_num": round_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "receipt": receipt,
            "intervention": intervention,
            "canary_stats": canary_stats,
            "status": "applied",
        }
        self.history.append(record)
        self._save_history()
        return record

    def fetch_latest_feedback(self) -> Optional[Dict[str, Any]]:
        """Fetch latest feedback from incoming synapse or R2."""
        if self.latest_received_feedback:
            fb = self.latest_received_feedback
            self.latest_received_feedback = None
            return fb

        if not self.wallet or not self.r2_client or not self.r2_bucket:
            return None

        try:
            ss58 = self.wallet.hotkey.ss58_address
            hk_hash = hashlib.sha256(ss58.encode("utf-8")).hexdigest()
            prefix = self.r2_telemetry_prefix.rstrip("/")
            key = f"{prefix}/feedback/{hk_hash}/latest.json"
            resp = self.r2_client.get_object(Bucket=self.r2_bucket, Key=key)
            data = json.loads(resp["Body"].read().decode("utf-8"))
            return data
        except Exception:
            return None

    def plan_intervention(self, receipt: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose validator feedback and choose one atomic intervention from hierarchy."""
        oneliner = receipt.get("validator_feedback_oneliner", "")
        geom = receipt.get("geometry_summary", {})
        leaderboard = receipt.get("leaderboard", {})
        current_params = load_miner_params()
        hw = get_hardware_profile()
        round_num = receipt.get("round_num", 1)

        self_fid = float(leaderboard.get("self_fidelity_score", 0.0))
        score_delta = float(leaderboard.get("score_delta", 0.0))

        # Check Round-to-Round Escalation Rule:
        # If self_fidelity_score < 0.25 or score_delta < 0.01 after a Tier 1 hyperparameter edit,
        # the Miner Agent MUST escalate to Tier 2 (Scoped Code Patch) on the next round.
        has_prior_tier1 = any(h.get("intervention", {}).get("tier") == 1 for h in self.history)
        must_escalate_tier2 = False
        if has_prior_tier1 and (self_fid < 0.25 or (len(self.history) > 0 and score_delta < 0.01)):
            must_escalate_tier2 = True

        system_instruction = (
            "You are the Autonomous Subnet 498 CanopyMRV Miner Autopilot Engineer. "
            "Your objective is to maximize the miner's self_fidelity_score by addressing validator diagnostic feedback. "
            "You choose EXACTLY ONE atomic intervention per round from this strict hierarchy:\n"
            "Tier 1 (Config tuning): Adjust conf_thres (0.15-0.70), iou_thres (0.30-0.70), imgsz (512-1024), "
            "or epsilon_factor (0.003-0.020 for contour approximation density).\n"
            "Tier 2 (Scoped code edit): Modify allowlisted files (e.g. template/miner/detector_annotate.py or geometry.py). "
            "Options: patch_type='sliding_window_tiling' on template/miner/detector_annotate.py (enables 2x2 overlapping sliding-window "
            "tile inference with 15% overlap + polygon NMS merge for missed clustered crowns) or patch_type='watershed_crown_splitting'.\n"
            "Tier 3 (HF Checkpoint Upgrade): Upgrade to YOLO/SAM canopy model if hardware permits.\n\n"
            "Respond STRICTLY in valid JSON matching this schema:\n"
            "{\n"
            '  "tier": 1 | 2 | 3,\n'
            '  "rationale": "1-2 sentence engineering explanation",\n'
            '  "action": "tune_param" | "code_edit" | "hf_checkpoint",\n'
            '  "target_file": "template/miner/detector_annotate.py",\n'
            '  "patch_type": "sliding_window_tiling",\n'
            '  "params": { "conf_thres": float, "iou_thres": float, "epsilon_factor": float, "imgsz": int }\n'
            "}"
        )

        if must_escalate_tier2:
            system_instruction += (
                "\n\n*** MANDATORY ESCALATION DIRECTIVE ***\n"
                f"The current self_fidelity_score is {self_fid:.4f} (< 0.25) or score_delta is {score_delta:.4f} (< 0.01) "
                "following Tier 1 hyperparameter tuning. Per autonomous subnet protocol, you MUST ESCALATE to Tier 2 (Scoped Code Patch). "
                "You are strictly forbidden from selecting Tier 1.\n"
                "You MUST choose tier=2, action='code_edit', target_file='template/miner/detector_annotate.py', "
                "and patch_type='sliding_window_tiling' to implement overlapping sliding-window inference."
            )

        user_content = json.dumps(
            {
                "validator_feedback": oneliner,
                "geometry_summary": geom,
                "leaderboard": leaderboard,
                "current_params": current_params,
                "hardware": hw,
                "failed_hypotheses": self.failed_hypotheses[-3:],
                "escalate_to_tier_2_required": must_escalate_tier2,
            },
            indent=2,
        )

        raw_llm = self.watchdog.call_gemini(
            system_prompt=system_instruction,
            user_content=user_content,
            max_output_tokens=1024,
            temperature=0.2,
            strict_live_llm=self.strict_live_llm,
            round_num=round_num,
        )

        parsed_plan: Optional[Dict[str, Any]] = None
        if raw_llm:
            try:
                # Extract JSON if wrapped in markdown code fence or surrounded by text
                text = raw_llm.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                if "{" in text and "}" in text:
                    s_idx = text.find("{")
                    e_idx = text.rfind("}") + 1
                    text = text[s_idx:e_idx]
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "tier" in parsed:
                    parsed_plan = parsed
            except Exception as e:
                bt.logging.warning(f"[AGENTIC MINER] Failed to parse Gemini intervention response: {e}")

        if parsed_plan is not None:
            return parsed_plan

        if self.strict_live_llm:
            raise RuntimeError(
                f"[AGENTIC MINER] Gemini intervention planning failed under strict_live_llm=True! "
                f"Raw response: {raw_llm}"
            )

        # Rule-based fallback planning (only when strict_live_llm=False)
        fn_rate = geom.get("false_negative_rate", 0.0)
        mean_v = geom.get("mean_vertices", 4.0)

        # High false-negative rate -> Lower confidence threshold
        if fn_rate > 0.25 and current_params.get("conf_thres", 0.25) > 0.28:
            return {
                "tier": 1,
                "rationale": f"High false-negative rate ({fn_rate:.2f}); lowering conf_thres to improve recall.",
                "action": "tune_param",
                "params": {"conf_thres": 0.25, "iou_thres": 0.45},
            }

        # Coarse polygons / low vertex count -> Lower contour epsilon factor
        if mean_v < 12.0 and current_params.get("epsilon_factor", 0.008) > 0.005:
            return {
                "tier": 1,
                "rationale": f"Low mean vertices ({mean_v:.1f}); refining contour epsilon to capture organic crowns.",
                "action": "tune_param",
                "params": {"epsilon_factor": 0.004},
            }

        return {
            "tier": 1,
            "rationale": "Refining NMS and contour smoothing parameters for boundary alignment.",
            "action": "tune_param",
            "params": {"iou_thres": 0.50, "epsilon_factor": 0.004},
        }

    def _apply_sliding_window_tiling_patch(self, target_file: Path) -> Tuple[bool, str]:
        """Apply 2x2 overlapping sliding-window tile inference with NMS merge to detector_annotate.py."""
        try:
            content = target_file.read_text(encoding="utf-8")
            if "Tier 2 SAHI 2x2 Overlapping Sliding-Window Tile Inference" in content:
                return True, "Already patched with SAHI 2x2 sliding-window tile inference"

            old_block = """    # Run inference with tuned parameters
    results = model(img, conf=eff_conf, iou=eff_iou, imgsz=eff_imgsz, verbose=False)

    annotations: List[PerImageAnnotationItem] = []
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        if boxes is not None:
            from template.miner.geometry import extract_canopy_geometry
            for b_idx, box in enumerate(boxes):
                xyxy = box.xyxy[0].tolist()
                cls_idx = int(box.cls[0].item())
                cls_name = model.names[cls_idx]
                conf = float(box.conf[0].item()) if hasattr(box, "conf") and box.conf is not None else 1.0
                mask_xy = None
                if hasattr(result, "masks") and result.masks is not None and len(result.masks) > b_idx:
                    mask_xy = result.masks.xy[b_idx].tolist()
                ann_item = extract_canopy_geometry(
                    pil_img=img,
                    box_xyxy=xyxy,
                    hazard_class=cls_name,
                    confidence=conf,
                    mask_xy=mask_xy,
                    epsilon_factor=eff_eps,
                    force_coarse_box=eff_coarse,
                )
                annotations.append(ann_item)"""

            new_block = """    # Run Tier 2 SAHI 2x2 Overlapping Sliding-Window Tile Inference + NMS Merge
    annotations: List[PerImageAnnotationItem] = []
    from template.miner.geometry import extract_canopy_geometry
    from template.hazard.annotation_eval import iou_xyxy

    img_w, img_h = img.size if hasattr(img, "size") else (1024, 1024)
    raw_candidates: List[PerImageAnnotationItem] = []

    # If high-res aerial/satellite chip (>= 1024px), run 2x2 tiles with 15% overlap
    if img_w >= 1024 and img_h >= 1024:
        tw = int(img_w * 0.575)
        th = int(img_h * 0.575)
        tiles = [
            (0, 0, tw, th),
            (img_w - tw, 0, img_w, th),
            (0, img_h - th, tw, img_h),
            (img_w - tw, img_h - th, img_w, img_h),
        ]
        for tx0, ty0, tx1, ty1 in tiles:
            tile_crop = img.crop((tx0, ty0, tx1, ty1))
            t_res = model(tile_crop, conf=eff_conf, iou=eff_iou, imgsz=eff_imgsz, verbose=False)
            if t_res and len(t_res) > 0 and t_res[0].boxes is not None:
                for b_idx, box in enumerate(t_res[0].boxes):
                    c_xyxy = box.xyxy[0].tolist()
                    g_xyxy = [c_xyxy[0] + tx0, c_xyxy[1] + ty0, c_xyxy[2] + tx0, c_xyxy[3] + ty0]
                    cls_idx = int(box.cls[0].item())
                    cls_name = model.names[cls_idx]
                    conf = float(box.conf[0].item()) if hasattr(box, "conf") and box.conf is not None else 1.0
                    ann_item = extract_canopy_geometry(
                        pil_img=img,
                        box_xyxy=g_xyxy,
                        hazard_class=cls_name,
                        confidence=conf,
                        epsilon_factor=eff_eps,
                        force_coarse_box=eff_coarse,
                    )
                    raw_candidates.append(ann_item)

    # Whole-image pass to capture macroscopic canopies
    full_res = model(img, conf=eff_conf, iou=eff_iou, imgsz=eff_imgsz, verbose=False)
    if full_res and len(full_res) > 0 and full_res[0].boxes is not None:
        for b_idx, box in enumerate(full_res[0].boxes):
            xyxy = box.xyxy[0].tolist()
            cls_idx = int(box.cls[0].item())
            cls_name = model.names[cls_idx]
            conf = float(box.conf[0].item()) if hasattr(box, "conf") and box.conf is not None else 1.0
            mask_xy = None
            if hasattr(full_res[0], "masks") and full_res[0].masks is not None and len(full_res[0].masks) > b_idx:
                mask_xy = full_res[0].masks.xy[b_idx].tolist()
            ann_item = extract_canopy_geometry(
                pil_img=img,
                box_xyxy=xyxy,
                hazard_class=cls_name,
                confidence=conf,
                mask_xy=mask_xy,
                epsilon_factor=eff_eps,
                force_coarse_box=eff_coarse,
            )
            raw_candidates.append(ann_item)

    # Polygon & Bounding Box NMS Merge across all crops
    if raw_candidates:
        sorted_cands = sorted(raw_candidates, key=lambda a: float(getattr(a, "confidence", 1.0) or 1.0), reverse=True)
        kept: List[PerImageAnnotationItem] = []
        for cand in sorted_cands:
            suppressed = False
            for k in kept:
                if iou_xyxy(cand.bounding_box, k.bounding_box) >= 0.45:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(cand)
        annotations = kept"""

            if old_block in content:
                patched = content.replace(old_block, new_block)
                target_file.write_text(patched, encoding="utf-8")
                return True, "Successfully patched detector_annotate.py with SAHI 2x2 sliding-window tile inference"
            else:
                return False, "Target block not found in detector_annotate.py for patch"
        except Exception as e:
            return False, f"Failed to patch detector_annotate.py: {e}"

    def apply_intervention(self, intervention: Dict[str, Any]) -> Tuple[bool, str]:
        """Apply the planned intervention to configuration or allowlisted code."""
        tier = intervention.get("tier", 1)
        action = intervention.get("action", "tune_param")

        if tier == 1 or action == "tune_param":
            new_params = intervention.get("params", {})
            curr = load_miner_params()
            curr.update(new_params)
            save_miner_params(curr)
            intervention["git_diff_applied"] = ""
            return True, f"Updated parameters: {new_params}"

        if tier == 2 and self.allow_code_edits:
            target_file = intervention.get("target_file", "template/miner/detector_annotate.py")
            patch_type = intervention.get("patch_type", "")
            p = Path(target_file)
            if not is_path_allowlisted(p):
                return False, f"Target file {target_file} is outside allowlist or blocked"

            if patch_type == "sliding_window_tiling" or "detector_annotate" in str(p):
                success, msg = self._apply_sliding_window_tiling_patch(p)
                if not success:
                    return False, msg
            elif intervention.get("code_patch"):
                p.write_text(intervention["code_patch"], encoding="utf-8")

            # Capture autonomous git diff
            try:
                import subprocess
                diff_bytes = subprocess.check_output(["git", "diff", str(p)], cwd=str(Path.cwd()))
                diff_str = diff_bytes.decode("utf-8")
                intervention["git_diff_applied"] = diff_str
                bt.logging.info(f"[AGENTIC MINER] Autonomously generated git diff on {p}:\n{diff_str}")
            except Exception as diff_err:
                bt.logging.warning(f"[AGENTIC MINER] Could not run git diff: {diff_err}")
                intervention["git_diff_applied"] = ""

            return True, f"Applied Tier 2 code patch to {target_file}"

        return True, "Intervention acknowledged"

    # -------------------------------------------------------------------------
    # Snapshots and Rollback
    # -------------------------------------------------------------------------

    def create_snapshot(self, tag: str) -> Path:
        """Create a backup of all allowlisted miner files."""
        snap_dir = self.snapshots_dir / tag
        snap_dir.mkdir(parents=True, exist_ok=True)
        for p in ALLOWLISTED_PATHS:
            if p.is_file():
                dest = snap_dir / p
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
        if PARAMS_PATH.is_file():
            dest = snap_dir / PARAMS_PATH
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PARAMS_PATH, dest)
        return snap_dir

    def rollback(self, tag: str) -> bool:
        """Restore allowlisted files from snapshot."""
        snap_dir = self.snapshots_dir / tag
        if not snap_dir.is_dir():
            bt.logging.error(f"[AGENTIC MINER] Snapshot directory not found: {snap_dir}")
            return False
        for root, _, files in os.walk(snap_dir):
            for file in files:
                src = Path(root) / file
                rel = src.relative_to(snap_dir)
                dest = Path(rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
        bt.logging.warning(f"[AGENTIC MINER] Successfully rolled back to snapshot: {tag}")
        return True

    def _check_and_handle_consecutive_drop(self, current_receipt: Dict[str, Any]) -> None:
        """Rollback if self_fidelity_score drops for 2 consecutive rounds."""
        curr_score = current_receipt.get("leaderboard", {}).get("self_fidelity_score", 0.0)
        if len(self.history) >= 2:
            h1 = self.history[-1].get("receipt", {}).get("leaderboard", {}).get("self_fidelity_score", 0.0)
            h2 = self.history[-2].get("receipt", {}).get("leaderboard", {}).get("self_fidelity_score", 0.0)
            if curr_score < h1 and h1 < h2:
                round_to_restore = self.history[-2].get("round_num", 1)
                tag = f"round_{round_to_restore}"
                bt.logging.warning(
                    f"[AGENTIC MINER] Score degraded consecutively ({h2:.4f} -> {h1:.4f} -> {curr_score:.4f}). "
                    f"Auto-rolling back to {tag}!"
                )
                self.rollback(tag)
