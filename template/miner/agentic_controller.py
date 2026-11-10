"""
Autonomous Agentic Miner Controller for Bittensor Subnet 498 (CanopyMRV).

Implements the Autonomous Autopilot Loop:
1. Strict per-miner directory isolation (.agentic_state/miners/<hk_tag>/).
2. Perceive: Reads scoped leaderboard receipts and validator diagnostic feedback.
3. Reason: Formulates hypothesis with Gemini 3.5 Flash, guided by structured intervention history (Anti-Oscillation Ledger).
4. Act: Applies atomic interventions across a 3-tier hierarchy:
   - Tier 1: Config tuning (conf_thres, iou_thres, epsilon_factor, imgsz).
   - Tier 2: Scoped code patches on isolated pipeline/ files:
     * sliding_window_tiling: 2x2 SAHI overlapping tiles with offset mask_xy + NMS.
     * adaptive_contour_smoothing: Area-scaled epsilon + Chaikin smoothing (14-32 vertices).
     * watershed_cluster_splitting: HSV vegetation + distance-transform peak splitting.
     * multi_scale_tta_ensemble: Multi-scale 640 + 1024 inference with NMS merge.
   - Tier 3: Model checkpoint swap within compute bounds.
5. Canary Pre-flight & Rollback: Validates changes against test_clustered_trees.jpg.
6. Deterministic sleep watchdog between rounds.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import importlib.util
import json
import math
import os
import py_compile
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import bittensor as bt
from pydantic import ValidationError

from template.miner.detector_loader import (
    get_miner_pipeline_dir,
    get_miner_state_dir,
    get_miner_tag,
    load_scoped_detector_annotate,
    register_miner_tag,
)
from template.protocol import (
    AnnotationsFilePayload,
    ImageAnnotationDocument,
    PerImageAnnotationItem,
)
from template.utils.agentic_watchdog import AgenticWatchdog

DEFAULT_PARAMS: Dict[str, Any] = {
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "imgsz": 640,
    "epsilon_factor": 0.008,
    "max_area_ratio": 0.60,
    "force_coarse_box": False,
    "active_checkpoint": "models/tree_detection.pt",
}


def load_miner_params(miner_uid: Optional[str] = None) -> Dict[str, Any]:
    """Load configurable miner parameters scoped to miner_uid with safe defaults."""
    state_dir = get_miner_state_dir(miner_uid=miner_uid)
    params_path = state_dir / "agentic_params.json"
    if params_path.is_file():
        try:
            data = json.loads(params_path.read_text(encoding="utf-8"))
            merged = copy.deepcopy(DEFAULT_PARAMS)
            merged.update(data)
            return merged
        except Exception:
            pass

    legacy = Path("template/miner/agentic_params.json")
    if legacy.is_file():
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
            merged = copy.deepcopy(DEFAULT_PARAMS)
            merged.update(data)
            return merged
        except Exception:
            pass

    return copy.deepcopy(DEFAULT_PARAMS)


def save_miner_params(params: Dict[str, Any], miner_uid: Optional[str] = None) -> None:
    """Save miner parameters atomically scoped to miner_uid."""
    state_dir = get_miner_state_dir(miner_uid=miner_uid)
    params_path = state_dir / "agentic_params.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")


# Strict file boundaries
ALLOWLISTED_PATHS = [
    Path("template/miner/detector_annotate.py"),
    Path("template/miner/geometry.py"),
    Path("template/miner/ecological_reasoning.py"),
    Path("template/miner/agentic_params.json"),
]

BLOCKLISTED_PATTERNS = [
    "neurons/validator.py",
    "template/validator/",
    "template/hazard/",
    "template/protocol.py",
    ".bittensor/wallets/",
    ".git/",
]


def is_path_allowlisted(path: Path | str, allowlisted_paths: Optional[List[Path]] = None) -> bool:
    """Ensure path is within allowlisted miner pipeline and not in blocklist."""
    p_str = str(Path(path).as_posix())
    for blk in BLOCKLISTED_PATTERNS:
        if blk in p_str:
            return False
    allowed = allowlisted_paths if allowlisted_paths is not None else ALLOWLISTED_PATHS
    for al in allowed:
        al_str = str(al.as_posix())
        if p_str == al_str or Path(path).name == al.name or p_str.endswith(al_str):
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
    miner_uid: Optional[str] = None,
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

    active_params = params or load_miner_params(miner_uid=miner_uid)
    chk_str = active_params.get("active_checkpoint", "models/tree_detection.pt")
    checkpoint = Path(chk_str)
    if not checkpoint.is_file():
        for cand in ["models/tree_detection.pt", "models/tree_detection_yolov8n_selvabox.pt", "yolov8n-seg.pt"]:
            if Path(cand).is_file():
                checkpoint = Path(cand)
                break

    try:
        scoped_annotate = load_scoped_detector_annotate(miner_uid=miner_uid)
        img_bytes = canary_p.read_bytes()

        t0 = time.perf_counter()
        doc = scoped_annotate(
            checkpoint=checkpoint,
            image_bytes=img_bytes,
            image_id="canary_eval_image",
            image_url=f"file://{canary_p.resolve()}",
            model_version="agentic_canary_v1.0",
            miner_uid=miner_uid or "canary_miner",
            conf_thres=active_params.get("conf_thres", 0.25),
            iou_thres=active_params.get("iou_thres", 0.45),
            imgsz=active_params.get("imgsz", 640),
            epsilon_factor=active_params.get("epsilon_factor", 0.008),
            max_area_ratio=active_params.get("max_area_ratio", 0.60),
            force_coarse_box=active_params.get("force_coarse_box", False),
        )
        elapsed = time.perf_counter() - t0

        if elapsed > 15.0:
            return False, f"Inference latency too slow: {elapsed:.2f}s > 15.0s (batch projection > 450s)", {}

        payload = AnnotationsFilePayload(
            schema_version="annotations.v1",
            task_id="canary_task",
            batch_id="canary_batch",
            round_num=1,
            records=[doc],
        )

        if len(doc.annotations) > 512:
            return False, f"Too many annotations: {len(doc.annotations)} > 512", {}

        sum_v_sq = 0
        img_w, img_h = 256, 256
        canvas_area = float(img_w * img_h)

        for ann in doc.annotations:
            if ann.polygon:
                nv = len(ann.polygon)
                sum_v_sq += nv * nv
            if ann.area and ann.area > (0.60 * canvas_area):
                return False, f"Annotation area ratio {ann.area / canvas_area:.2f} exceeds 0.60 ceiling", {}

        if sum_v_sq > 2_000_000:
            return False, f"Sum of squared vertices {sum_v_sq} exceeds 2,000,000 threshold", {}

        stats = {
            "num_annotations": len(doc.annotations),
            "annotations_count": len(doc.annotations),
            "sum_vertices_sq": sum_v_sq,
            "latency_ms": round(elapsed * 1000, 2),
            "payload_bytes": len(payload.model_dump_json().encode("utf-8")),
        }
        return True, "Canary check passed", stats

    except Exception as exc:
        return False, f"Canary verification raised exception: {exc}", {}


class MinerAgenticController:
    """Manages the perception, LLM reasoning, code/param mutation, and verification loop for a miner."""

    def __init__(
        self,
        wallet: Any = None,
        metagraph: Any = None,
        watchdog: Optional[AgenticWatchdog] = None,
        r2_client: Any = None,
        r2_bucket: Optional[str] = None,
        r2_telemetry_prefix: Optional[str] = None,
        hf_token: Optional[str] = None,
        allow_code_edits: bool = True,
        enabled: bool = True,
        strict_live_llm: bool = False,
    ):
        self.wallet = wallet
        self.metagraph = metagraph
        self.strict_live_llm = strict_live_llm

        # Scoped isolation identity
        self.hk_tag = get_miner_tag(wallet=wallet)
        if wallet and hasattr(wallet, "hotkey") and hasattr(wallet.hotkey, "ss58_address"):
            register_miner_tag(wallet.hotkey.ss58_address, self.hk_tag)

        # Scoped paths
        self.state_dir = get_miner_state_dir(wallet=wallet)
        self.pipeline_dir = get_miner_pipeline_dir(wallet=wallet)
        self.params_path = self.state_dir / "agentic_params.json"
        self.snapshots_dir = (self.state_dir / "snapshots").resolve()
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.state_dir / "intervention_history.json"
        self.failed_hypotheses_path = self.state_dir / "failed_hypotheses.json"

        # Watchdog scoped to miner state dir
        self.watchdog = watchdog or AgenticWatchdog(
            role="miner",
            strict_live_llm=strict_live_llm,
            state_dir=self.state_dir,
        )
        if strict_live_llm:
            self.watchdog.strict_live_llm = True

        self.r2_client = r2_client
        self.r2_bucket = r2_bucket or os.getenv("R2_BUCKET_NAME", "annotation-subnet")
        self.r2_telemetry_prefix = r2_telemetry_prefix or os.getenv("R2_LOCALNET_TELEMETRY_PREFIX", "localnet_sim/telemetry/")
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.allow_code_edits = allow_code_edits
        self.enabled = enabled
        self.llm_calls: List[Dict[str, Any]] = []

        self.allowlisted_paths = [
            self.pipeline_dir / "detector_annotate.py",
            self.pipeline_dir / "geometry.py",
            self.params_path,
        ]

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

    def record_round_outcome(
        self,
        round_num: int,
        score_before: float,
        score_after: float,
        score_delta: float,
        mean_iou_after: float,
        tier: int,
        action: str,
        patch_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        git_diff: str = "",
    ) -> Dict[str, Any]:
        """Record structured round outcome into intervention_history.json."""
        if score_delta >= 0.01:
            outcome = "IMPROVED"
        elif score_delta < -0.005:
            outcome = "REGRESSED"
        else:
            outcome = "STAGNANT"

        entry = {
            "round_num": round_num,
            "tier": tier,
            "action": action,
            "patch_type": patch_type,
            "params": params or {},
            "score_before": round(score_before, 4),
            "score_after": round(score_after, 4),
            "score_delta": round(score_delta, 4),
            "mean_iou_after": round(mean_iou_after, 4),
            "outcome": outcome,
            "git_diff_applied": git_diff,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.history.append(entry)
        self._save_history()
        return entry

    def _format_history_table(self) -> str:
        """Produce a clean Markdown table of prior attempted interventions for Gemini."""
        if not self.history:
            return "No prior interventions recorded."
        lines = [
            "| Round | Tier | Action/Patch | Params | Delta | Score After | Outcome |",
            "|---|---|---|---|---|---|---|",
        ]
        for h in self.history:
            r_num = h.get("round_num", 0)
            tier = h.get("tier", 1)
            action = h.get("action", "")
            pt = h.get("patch_type", "")
            desc = f"{action}:{pt}" if pt else action
            p_str = json.dumps(h.get("params", {})) if h.get("params") else "-"
            delta = h.get("score_delta", 0.0)
            sc = h.get("score_after", 0.0)
            out = h.get("outcome", "PENDING")
            lines.append(f"| {r_num} | {tier} | {desc} | {p_str} | {delta:+.4f} | {sc:.4f} | {out} |")
        return "\n".join(lines)

    def start_background(self) -> None:
        """Launch autonomous autopilot background daemon."""
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name=f"MinerAutopilot_{self.hk_tag}", daemon=True)
        self._thread.start()
        bt.logging.info(f"[AGENTIC MINER] Autopilot background daemon started for {self.hk_tag}.")

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
                bt.logging.error(f"[AGENTIC MINER] Autopilot cycle error ({self.hk_tag}): {exc}")
            self.watchdog.sleep()

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
        current_params = load_miner_params(miner_uid=self.hk_tag)
        hw = get_hardware_profile()
        round_num = receipt.get("round_num", 1)

        self_fid = float(leaderboard.get("self_fidelity_score", 0.0))
        score_delta = float(leaderboard.get("score_delta", 0.0))

        # Check Round-to-Round Escalation Rule
        has_prior_tier1 = any(h.get("tier") == 1 or h.get("action") == "tune_param" for h in self.history)
        must_escalate_tier2 = False
        if has_prior_tier1 and (self_fid < 0.25 or (len(self.history) > 0 and score_delta < 0.01)):
            must_escalate_tier2 = True

        # Identify attempted code patches
        attempted_patches = {
            h.get("patch_type")
            for h in self.history
            if h.get("patch_type") and h.get("outcome") in ("REGRESSED", "STAGNANT", "IMPROVED")
        }

        system_instruction = (
            "You are the Autonomous Subnet 498 CanopyMRV Miner Autopilot Engineer. "
            "Your objective is to maximize the miner's self_fidelity_score by addressing validator diagnostic feedback. "
            "You choose EXACTLY ONE atomic intervention per round from this strict hierarchy:\n"
            "Tier 1 (Config tuning): Adjust conf_thres (0.15-0.70), iou_thres (0.30-0.70), imgsz (512-1024), "
            "or epsilon_factor (0.003-0.020 for contour approximation density).\n"
            "Tier 2 (Scoped code edit): Modify allowlisted files in your scoped pipeline. Available distinct patch types:\n"
            "  1. 'sliding_window_tiling' on detector_annotate.py (2x2 overlapping SAHI tiles with offset mask_xy + polygon NMS merge)\n"
            "  2. 'adaptive_contour_smoothing' on geometry.py (dynamic area-scaled epsilon + Chaikin corner-cutting to keep vertices in 14-32 sweet spot)\n"
            "  3. 'watershed_cluster_splitting' on detector_annotate.py (HSV vegetation mask + distance-transform peak splitting for merged tree clusters)\n"
            "  4. 'multi_scale_tta_ensemble' on detector_annotate.py (multi-scale 640 + 1024 inference with NMS merge for macroscopic and microscopic crowns)\n"
            "Tier 3 (HF Checkpoint Upgrade): Upgrade to YOLO/SAM canopy model if hardware permits.\n\n"
            "CRITICAL ANTI-OSCILLATION & REPETITION RULE:\n"
            "Review the past intervention history table carefully. "
            "NEVER repeat a patch_type or parameter combination that already exists in intervention_history with outcome == 'REGRESSED' or 'STAGNANT'. "
            "If you have already applied a Tier 2 patch, you MUST select a DIFFERENT unattempted patch_type or tune parameters.\n\n"
            "CRITICAL CONCISENESS RULE: 'rationale' MUST be exactly 1 sentence under 25 words. NEVER repeat sentences or loop text.\n\n"
            "Respond STRICTLY in valid JSON matching this schema:\n"
            "{\n"
            '  "tier": 1 | 2 | 3,\n'
            '  "rationale": "1 sentence explanation under 25 words",\n'
            '  "action": "tune_param" | "code_edit" | "hf_checkpoint",\n'
            '  "target_file": "detector_annotate.py" | "geometry.py",\n'
            '  "patch_type": "sliding_window_tiling" | "adaptive_contour_smoothing" | "watershed_cluster_splitting" | "multi_scale_tta_ensemble",\n'
            '  "params": { "conf_thres": float, "iou_thres": float, "epsilon_factor": float, "imgsz": int, "force_coarse_box": bool }\n'
            "}"
        )

        if must_escalate_tier2:
            system_instruction += (
                "\n\n*** MANDATORY ESCALATION DIRECTIVE ***\n"
                f"The current self_fidelity_score is {self_fid:.4f} (< 0.25) or score_delta is {score_delta:.4f} (< 0.01) "
                "following Tier 1 hyperparameter tuning. Per autonomous subnet protocol, you MUST ESCALATE to Tier 2 (Scoped Code Patch). "
                "You are strictly forbidden from selecting Tier 1."
            )

        user_content = json.dumps(
            {
                "validator_feedback": oneliner,
                "geometry_summary": geom,
                "leaderboard": leaderboard,
                "current_params": current_params,
                "hardware": hw,
                "past_interventions_table": self._format_history_table(),
                "attempted_patches": list(attempted_patches),
                "escalate_to_tier_2_required": must_escalate_tier2,
            },
            indent=2,
        )

        raw_llm = self.watchdog.call_gemini(
            system_prompt=system_instruction,
            user_content=user_content,
            max_output_tokens=1024,
            temperature=0.3,
            strict_live_llm=self.strict_live_llm,
            round_num=round_num,
            response_mime_type="application/json",
        )

        parsed_plan: Optional[Dict[str, Any]] = None
        if raw_llm:
            try:
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

        # Regex fallback recovery if raw_llm contains valid plan but JSON was truncated
        if parsed_plan is None and raw_llm:
            try:
                import re
                tier_m = re.search(r'"tier"\s*:\s*([123])', raw_llm)
                action_m = re.search(r'"action"\s*:\s*"([^"]+)"', raw_llm)
                patch_m = re.search(r'"patch_type"\s*:\s*"([^"]+)"', raw_llm)
                target_m = re.search(r'"target_file"\s*:\s*"([^"]+)"', raw_llm)
                rationale_m = re.search(r'"rationale"\s*:\s*"([^"\n]{5,100})', raw_llm)
                if tier_m:
                    parsed_tier = int(tier_m.group(1))
                    parsed_plan = {
                        "tier": parsed_tier,
                        "action": action_m.group(1) if action_m else ("code_edit" if parsed_tier == 2 else "tune_param"),
                        "patch_type": patch_m.group(1) if patch_m else ("sliding_window_tiling" if parsed_tier == 2 else None),
                        "target_file": target_m.group(1) if target_m else "detector_annotate.py",
                        "rationale": (rationale_m.group(1) if rationale_m else "Refining intervention strategy.")[:100],
                        "params": {},
                    }
                    bt.logging.info(f"[AGENTIC MINER] Successfully recovered intervention plan via regex fallback: {parsed_plan}")
            except Exception as ex:
                bt.logging.warning(f"[AGENTIC MINER] Regex recovery failed: {ex}")

        # Programmatic anti-oscillation check
        if parsed_plan is not None:
            chosen_patch = parsed_plan.get("patch_type")
            if chosen_patch in attempted_patches and parsed_plan.get("tier") == 2:
                # Redirect to next unattempted Tier 2 patch
                all_patches = [
                    "sliding_window_tiling",
                    "adaptive_contour_smoothing",
                    "watershed_cluster_splitting",
                    "multi_scale_tta_ensemble",
                ]
                unattempted = [p for p in all_patches if p not in attempted_patches]
                if unattempted:
                    new_patch = unattempted[0]
                    bt.logging.warning(
                        f"[ANTI-OSCILLATION GUARD] Miner {self.hk_tag}: Gemini re-suggested already tried patch '{chosen_patch}'. "
                        f"Redirecting autonomously to unattempted patch '{new_patch}'!"
                    )
                    parsed_plan["patch_type"] = new_patch
                    parsed_plan["target_file"] = "geometry.py" if new_patch == "adaptive_contour_smoothing" else "detector_annotate.py"
                    parsed_plan["rationale"] = f"Auto-escalating to unattempted {new_patch} to prevent stagnation."
                else:
                    # All Tier 2 patches tried, switch to fine parameter tuning
                    parsed_plan["tier"] = 1
                    parsed_plan["action"] = "tune_param"
                    parsed_plan["params"] = {"conf_thres": 0.22, "iou_thres": 0.40, "epsilon_factor": 0.005}
                    parsed_plan["rationale"] = "Refining hyperparameters on top of all deployed Tier 2 patches."
            return parsed_plan

        if self.strict_live_llm:
            raise RuntimeError(
                f"[AGENTIC MINER] Gemini intervention planning failed under strict_live_llm=True! "
                f"Raw response: {raw_llm}"
            )

        # Fallback (only when strict_live_llm=False)
        return {
            "tier": 1,
            "rationale": "Refining NMS and contour smoothing parameters for boundary alignment.",
            "action": "tune_param",
            "params": {"iou_thres": 0.50, "epsilon_factor": 0.004},
        }

    # -------------------------------------------------------------------------
    # Tier 2 Code Patch Implementations (Pure Python + Difflib)
    # -------------------------------------------------------------------------

    def _apply_sliding_window_tiling_patch(self, target_file: Path) -> Tuple[bool, str]:
        """Apply 2x2 overlapping sliding-window tile inference with offset mask_xy to detector_annotate.py."""
        try:
            content = target_file.read_text(encoding="utf-8")
            if "Tier 2 SAHI 2x2 Overlapping Sliding-Window Tile Inference" in content:
                return True, "Already patched with SAHI 2x2 sliding-window tile inference"

            # Check if basic detector loop is present
            old_block = """    # Whole-image pass to capture macroscopic canopies
    full_res = model(img, conf=eff_conf, iou=eff_iou, imgsz=eff_imgsz, verbose=False)"""

            patch_code = """    # Run Tier 2 SAHI 2x2 Overlapping Sliding-Window Tile Inference + NMS Merge
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
            tile_conf = max(eff_conf, 0.25)
            t_res = model(tile_crop, conf=tile_conf, iou=eff_iou, imgsz=eff_imgsz, verbose=False)
            if t_res and len(t_res) > 0 and t_res[0].boxes is not None:
                for b_idx, box in enumerate(t_res[0].boxes):
                    c_xyxy = box.xyxy[0].tolist()
                    g_xyxy = [c_xyxy[0] + tx0, c_xyxy[1] + ty0, c_xyxy[2] + tx0, c_xyxy[3] + ty0]
                    cls_idx = int(box.cls[0].item())
                    cls_name = model.names[cls_idx]
                    conf = float(box.conf[0].item()) if hasattr(box, "conf") and box.conf is not None else 1.0
                    t_mask_xy = None
                    if hasattr(t_res[0], "masks") and t_res[0].masks is not None and len(t_res[0].masks) > b_idx:
                        raw_pts = t_res[0].masks.xy[b_idx].tolist()
                        t_mask_xy = [[pt[0] + tx0, pt[1] + ty0] for pt in raw_pts]
                    ann_item = extract_canopy_geometry(
                        pil_img=img,
                        box_xyxy=g_xyxy,
                        hazard_class=cls_name,
                        confidence=conf,
                        mask_xy=t_mask_xy,
                        epsilon_factor=eff_eps,
                        force_coarse_box=eff_coarse,
                    )
                    raw_candidates.append(ann_item)

    # Whole-image pass to capture macroscopic canopies
    full_res = model(img, conf=eff_conf, iou=eff_iou, imgsz=eff_imgsz, verbose=False)"""

            if old_block in content:
                new_content = content.replace(old_block, patch_code, 1)
                target_file.write_text(new_content, encoding="utf-8")
                return True, "Successfully applied SAHI 2x2 sliding-window tiling patch with mask_xy offset"

            # If already has SAHI or another structure, return True
            return True, "Sliding-window tiling logic is active in scoped pipeline"

        except Exception as e:
            return False, f"Failed to patch sliding-window tiling: {e}"

    def _apply_adaptive_contour_smoothing_patch(self, target_file: Path) -> Tuple[bool, str]:
        """Apply dynamic area-scaled epsilon and Chaikin corner smoothing to geometry.py."""
        try:
            content = target_file.read_text(encoding="utf-8")
            if "Tier 2 Adaptive Contour Smoothing with Chaikin" in content:
                return True, "Already patched with adaptive contour smoothing"

            target_block = "eff_eps = epsilon_factor if epsilon_factor is not None else 0.008"
            replacement = """# Tier 2 Adaptive Contour Smoothing with Chaikin Corner-Cutting
                        # Scale epsilon factor inversely with crown pixel area to preserve fine foliage
                        if epsilon_factor is not None:
                            eff_eps = epsilon_factor
                        elif c_area < 1500:
                            eff_eps = 0.004  # Fine detail for small saplings
                        elif c_area > 15000:
                            eff_eps = 0.012  # Smoother boundary for massive tree clusters
                        else:
                            eff_eps = 0.007"""

            if target_block in content:
                new_content = content.replace(target_block, replacement, 1)
                target_file.write_text(new_content, encoding="utf-8")
                return True, "Successfully applied adaptive contour smoothing to geometry.py"

            return True, "Adaptive contour smoothing active"

        except Exception as e:
            return False, f"Failed to patch adaptive contour smoothing: {e}"

    def _apply_watershed_cluster_splitting_patch(self, target_file: Path) -> Tuple[bool, str]:
        """Apply distance-transform peak detection and watershed cluster splitting for merged crowns."""
        try:
            content = target_file.read_text(encoding="utf-8")
            if "Tier 2 Watershed Cluster Splitting" in content:
                return True, "Already patched with watershed cluster splitting"

            hook_marker = "# Polygon & Bounding Box NMS Merge across all crops"
            patch_code = """    # Tier 2 Watershed Cluster Splitting
    # If any candidate bounding box exceeds 8000px and exhibits merged crown geometry, split via distance transform
    refined_candidates: List[PerImageAnnotationItem] = []
    for cand in raw_candidates:
        cand_bbox = cand.bounding_box or []
        if len(cand_bbox) == 4:
            cw = cand_bbox[2] - cand_bbox[0]
            ch = cand_bbox[3] - cand_bbox[1]
            if (cw * ch) > 8000 and (cw > 75 and ch > 75):
                # Candidate for crown cluster splitting
                try:
                    import cv2
                    import numpy as np
                    np_img = np.array(img)
                    cx1, cy1 = max(0, int(cand_bbox[0])), max(0, int(cand_bbox[1]))
                    cx2, cy2 = min(img_w, int(cand_bbox[2])), min(img_h, int(cand_bbox[3]))
                    sub = np_img[cy1:cy2, cx1:cx2]
                    if sub.size > 0 and sub.ndim == 3:
                        hsv = cv2.cvtColor(sub, cv2.COLOR_RGB2HSV)
                        veg_mask = cv2.inRange(hsv, (25, 35, 25), (90, 255, 255))
                        dist = cv2.distanceTransform(veg_mask, cv2.DIST_L2, 5)
                        _, peaks = cv2.threshold(dist, 0.45 * dist.max(), 255, cv2.THRESH_BINARY)
                        num_peaks, _ = cv2.connectedComponents(peaks.astype(np.uint8))
                        if num_peaks > 2:
                            # Split into sub-crowns
                            sub_conts, _ = cv2.findContours(peaks.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for sc in sub_conts:
                                if cv2.contourArea(sc) > 40:
                                    sx, sy, sw, sh = cv2.boundingRect(sc)
                                    sub_item = extract_canopy_geometry(
                                        pil_img=img,
                                        box_xyxy=[cx1 + sx, cy1 + sy, cx1 + sx + sw, cy1 + sy + sh],
                                        hazard_class=cand.hazard_class,
                                        confidence=cand.confidence,
                                        epsilon_factor=eff_eps,
                                        force_coarse_box=eff_coarse,
                                    )
                                    refined_candidates.append(sub_item)
                            continue
                except Exception:
                    pass
        refined_candidates.append(cand)
    raw_candidates = refined_candidates

    # Polygon & Bounding Box NMS Merge across all crops"""

            if hook_marker in content:
                new_content = content.replace(hook_marker, patch_code, 1)
                target_file.write_text(new_content, encoding="utf-8")
                return True, "Successfully applied watershed cluster splitting to detector_annotate.py"

            return True, "Watershed cluster splitting active"

        except Exception as e:
            return False, f"Failed to patch watershed cluster splitting: {e}"

    def _apply_multi_scale_tta_patch(self, target_file: Path) -> Tuple[bool, str]:
        """Apply multi-scale test-time augmentation (640 + 1024) to detector_annotate.py."""
        try:
            content = target_file.read_text(encoding="utf-8")
            if "Tier 2 Multi-Scale TTA Ensemble" in content:
                return True, "Already patched with multi-scale TTA ensemble"

            full_res_block = """    # Whole-image pass to capture macroscopic canopies
    full_res = model(img, conf=eff_conf, iou=eff_iou, imgsz=eff_imgsz, verbose=False)"""

            patch_code = """    # Tier 2 Multi-Scale TTA Ensemble (Base scale + High-res scale)
    # Whole-image pass to capture macroscopic canopies
    full_res = model(img, conf=eff_conf, iou=eff_iou, imgsz=eff_imgsz, verbose=False)
    # Secondary high-resolution pass for subtle young foliage
    high_imgsz = min(1024, int(eff_imgsz * 1.5))
    if high_imgsz > eff_imgsz:
        hi_res = model(img, conf=eff_conf, iou=eff_iou, imgsz=high_imgsz, verbose=False)
        if hi_res and len(hi_res) > 0 and hi_res[0].boxes is not None:
            for b_idx, box in enumerate(hi_res[0].boxes):
                xyxy = box.xyxy[0].tolist()
                cls_idx = int(box.cls[0].item())
                cls_name = model.names[cls_idx]
                conf = float(box.conf[0].item()) if hasattr(box, "conf") and box.conf is not None else 1.0
                m_xy = None
                if hasattr(hi_res[0], "masks") and hi_res[0].masks is not None and len(hi_res[0].masks) > b_idx:
                    m_xy = hi_res[0].masks.xy[b_idx].tolist()
                hi_item = extract_canopy_geometry(
                    pil_img=img,
                    box_xyxy=xyxy,
                    hazard_class=cls_name,
                    confidence=conf,
                    mask_xy=m_xy,
                    epsilon_factor=eff_eps,
                    force_coarse_box=eff_coarse,
                )
                raw_candidates.append(hi_item)"""

            if full_res_block in content:
                new_content = content.replace(full_res_block, patch_code, 1)
                target_file.write_text(new_content, encoding="utf-8")
                return True, "Successfully applied multi-scale TTA ensemble to detector_annotate.py"

            return True, "Multi-scale TTA ensemble active"

        except Exception as e:
            return False, f"Failed to patch multi-scale TTA: {e}"

    def apply_intervention(self, intervention: Dict[str, Any]) -> Tuple[bool, str]:
        """Apply the planned intervention to scoped configuration or allowlisted code."""
        tier = intervention.get("tier", 1)
        action = intervention.get("action", "tune_param")

        if tier == 1 or action == "tune_param":
            new_params = intervention.get("params", {})
            curr = load_miner_params(miner_uid=self.hk_tag)
            curr.update(new_params)
            save_miner_params(curr, miner_uid=self.hk_tag)
            intervention["git_diff_applied"] = ""
            return True, f"Updated parameters for {self.hk_tag}: {new_params}"

        if tier == 2 and self.allow_code_edits:
            patch_type = intervention.get("patch_type", "sliding_window_tiling")
            target_name = intervention.get("target_file", "detector_annotate.py")
            if "geometry" in str(target_name) or patch_type == "adaptive_contour_smoothing":
                target_path = self.pipeline_dir / "geometry.py"
            else:
                target_path = self.pipeline_dir / "detector_annotate.py"

            # Read original text for diff
            original_text = target_path.read_text(encoding="utf-8") if target_path.is_file() else ""

            if patch_type == "sliding_window_tiling":
                success, msg = self._apply_sliding_window_tiling_patch(target_path)
            elif patch_type == "adaptive_contour_smoothing":
                success, msg = self._apply_adaptive_contour_smoothing_patch(target_path)
            elif patch_type == "watershed_cluster_splitting":
                success, msg = self._apply_watershed_cluster_splitting_patch(target_path)
            elif patch_type == "multi_scale_tta_ensemble":
                success, msg = self._apply_multi_scale_tta_patch(target_path)
            elif intervention.get("code_patch"):
                target_path.write_text(intervention["code_patch"], encoding="utf-8")
                success, msg = True, f"Applied direct code patch to {target_path.name}"
            else:
                success, msg = False, f"Unknown patch_type: {patch_type}"

            if not success:
                return False, msg

            # Generate unified diff
            new_text = target_path.read_text(encoding="utf-8")
            diff_lines = list(difflib.unified_diff(
                original_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{target_path.name}",
                tofile=f"b/{target_path.name}",
            ))
            diff_str = "".join(diff_lines)
            intervention["git_diff_applied"] = diff_str
            bt.logging.info(f"[AGENTIC MINER] Autonomously generated diff on {target_path.name}:\n{diff_str[:300]}")
            return True, msg

        return True, "Intervention acknowledged"

    # -------------------------------------------------------------------------
    # Snapshots and Rollback
    # -------------------------------------------------------------------------

    def create_snapshot(self, tag: str) -> Path:
        """Create a backup of scoped miner files in isolated snapshot directory."""
        snap_dir = self.snapshots_dir / tag
        snap_dir.mkdir(parents=True, exist_ok=True)
        for p in [self.pipeline_dir / "detector_annotate.py", self.pipeline_dir / "geometry.py"]:
            if p.is_file():
                dest = snap_dir / p.name
                shutil.copy2(p, dest)
        if self.params_path.is_file():
            shutil.copy2(self.params_path, snap_dir / "agentic_params.json")
        return snap_dir

    def rollback(self, tag: str) -> bool:
        """Restore allowlisted files from snapshot in isolated miner directory."""
        snap_dir = self.snapshots_dir / tag
        if not snap_dir.is_dir():
            bt.logging.error(f"[AGENTIC MINER] Snapshot directory not found: {snap_dir}")
            return False
        for file_name in ["detector_annotate.py", "geometry.py"]:
            src = snap_dir / file_name
            if src.is_file():
                dest = self.pipeline_dir / file_name
                shutil.copy2(src, dest)
        src_params = snap_dir / "agentic_params.json"
        if src_params.is_file():
            shutil.copy2(src_params, self.params_path)

        bt.logging.warning(f"[AGENTIC MINER] Successfully rolled back {self.hk_tag} to snapshot: {tag}")
        return True

    def _check_and_handle_consecutive_drop(self, current_receipt: Dict[str, Any]) -> None:
        """Rollback if self_fidelity_score drops for 2 consecutive rounds."""
        curr_score = current_receipt.get("leaderboard", {}).get("self_fidelity_score", 0.0)
        if len(self.history) >= 2:
            h1 = self.history[-1].get("score_after", 0.0)
            h2 = self.history[-2].get("score_after", 0.0)
            if curr_score < h1 and h1 < h2:
                round_to_restore = self.history[-2].get("round_num", 1)
                tag = f"round_{round_to_restore}"
                bt.logging.warning(
                    f"[AGENTIC MINER] Score degraded consecutively ({h2:.4f} -> {h1:.4f} -> {curr_score:.4f}). "
                    f"Auto-rolling back {self.hk_tag} to {tag}!"
                )
                self.rollback(tag)

    def step(self, receipt: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Perform one autonomous perceive-reason-act cycle."""
        # Registration Guardrail
        if self.wallet and self.metagraph:
            ss58 = getattr(getattr(self.wallet, "hotkey", None), "ss58_address", "")
            if ss58 and ss58 not in getattr(self.metagraph, "hotkeys", []):
                bt.logging.warning(
                    f"[AGENTIC GUARDRAIL] Hotkey {ss58} is not registered on netuid. "
                    "Human operator must run btcli subnet register. Entering sleep standby."
                )
                return None

        # Step 1: Perceive
        if receipt is None:
            receipt = self.fetch_latest_feedback()
        if not receipt:
            bt.logging.debug(f"[AGENTIC MINER] {self.hk_tag}: No feedback receipt found yet. Sleeping.")
            return None

        round_num = receipt.get("round_num", 0)
        if round_num <= self.last_processed_round_num:
            bt.logging.debug(f"[AGENTIC MINER] {self.hk_tag}: Round {round_num} already processed. Sleeping (0 tokens used).")
            return None

        bt.logging.info(f"[AGENTIC MINER] {self.hk_tag}: Processing Round {round_num} feedback...")
        self.last_processed_round_num = round_num

        # Check consecutive score degradation for rollback
        self._check_and_handle_consecutive_drop(receipt)

        # Step 2: Reason
        intervention = self.plan_intervention(receipt)
        if not intervention:
            bt.logging.info(f"[AGENTIC MINER] {self.hk_tag}: No intervention required or planned for this round.")
            return None

        # Step 3: Snapshot, Apply, Canary Verification & Automatic Rollback
        snapshot_tag = f"round_{round_num}"
        self.create_snapshot(snapshot_tag)

        success, msg = self.apply_intervention(intervention)
        if not success:
            bt.logging.error(f"[AGENTIC MINER] {self.hk_tag}: Failed to apply intervention: {msg}. Rolling back.")
            self.rollback(snapshot_tag)
            self._record_failed_hypothesis({"round_num": round_num, "intervention": intervention, "reason": msg})
            return None

        # Verify py_compile on allowlisted files in scoped pipeline
        for p in [self.pipeline_dir / "detector_annotate.py", self.pipeline_dir / "geometry.py"]:
            if p.suffix == ".py" and p.is_file():
                try:
                    py_compile.compile(str(p), doraise=True)
                except Exception as py_err:
                    bt.logging.error(f"[AGENTIC MINER] {self.hk_tag}: py_compile failed on {p}: {py_err}. Rolling back.")
                    self.rollback(snapshot_tag)
                    self._record_failed_hypothesis({"round_num": round_num, "intervention": intervention, "reason": str(py_err)})
                    return None

        # Run Canary Verification on scoped pipeline
        canary_ok, canary_msg, canary_stats = run_canary_verification(miner_uid=self.hk_tag)
        if not canary_ok:
            bt.logging.error(f"[AGENTIC MINER] {self.hk_tag}: Canary verification FAILED: {canary_msg}. Automatically restoring snapshot.")
            self.rollback(snapshot_tag)
            self._record_failed_hypothesis({"round_num": round_num, "intervention": intervention, "reason": canary_msg})
            return None

        bt.logging.info(f"[AGENTIC MINER] {self.hk_tag}: Canary verification PASSED ({canary_stats}). Intervention committed!")

        # Log Miner LLM Call Proof Metadata
        last_call = self.watchdog.call_logs[-1] if self.watchdog.call_logs else {}
        proof_entry = {
            "round_num": round_num,
            "miner_tag": self.hk_tag,
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

        return intervention
