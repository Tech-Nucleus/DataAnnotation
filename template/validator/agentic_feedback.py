"""Validator Multimodal Diagnostic Engine & Scoped Leaderboard Receipt Subsystem.

Implements:
1. Representative golden chip selection (2 lowest-scoring + 1 median-scoring chips).
2. RAM-only side-by-side composite comparison grid (zero disk/R2 leakage of golden chips).
3. Quantitative geometry error metric extraction.
4. Multimodal diagnostic querying via Gemini 2.5 Flash with strict zero-leakage sanitization.
5. Scoped per-miner leaderboard receipt publishing via Cloudflare R2 and Synapse piggybacking.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import bittensor as bt
import cv2
import numpy as np

from template.hazard.annotation_eval import (
    AnnotationFidelityScorer,
    FidelityComponents,
    PerMinerAnnotationScore,
    iou_xyxy,
)
from template.hazard.geometry_quality import is_crude_box, raster_polygon_iou, shoelace_area
from template.hazard.image_corpus import GoldenImage, ImageCorpus
from template.protocol import PerImageAnnotationItem
from template.utils.agentic_watchdog import AgenticWatchdog


def sanitize_feedback(feedback: str, golden_image_ids: Sequence[str] = ()) -> str:
    """Sanitize LLM diagnostic output to strictly prevent golden data leakage.

    Enforces:
    - Zero mention of image IDs, chip hashes, filenames, or coordinates.
    - Maximum 2 sentences.
    - Maximum 45 words.
    """
    if not feedback:
        return "Canopy annotations submitted; optimize confidence threshold and contour vertex resolution."

    sanitized = feedback.strip()

    # Remove golden image IDs and hashes
    for gid in golden_image_ids:
        if gid:
            sanitized = sanitized.replace(gid, "")

    # Remove filenames
    sanitized = re.sub(r"[\w\-_]+\.(?:jpg|jpeg|png|tif|tiff)", "", sanitized, flags=re.IGNORECASE)

    # Remove explicit chip or sample references (e.g. "chip 1", "sample #2")
    sanitized = re.sub(r"\b(?:chip|sample|image|patch)\s*(?:#?\d+|[a-f0-9]{6,})", "sector", sanitized, flags=re.IGNORECASE)

    # Remove coordinate tuples or arrays (e.g. [120, 240] or (45.2, 89.1))
    sanitized = re.sub(r"\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]", "", sanitized)
    sanitized = re.sub(r"\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)", "", sanitized)

    # Normalize whitespace
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    sanitized = re.sub(r"\s+([.,!?;:])", r"\1", sanitized)

    # Enforce maximum 2 sentences (avoiding splitting on numbers like 0.25 or list indices like 1.)
    sentences = [s.strip() for s in re.split(r"(?<=[a-zA-Z]{2}[.!?])\s+", sanitized) if s.strip()]
    if len(sentences) > 2:
        sanitized = " ".join(sentences[:2])
    elif sentences:
        sanitized = " ".join(sentences)

    # Enforce maximum 45 words
    words = sanitized.split()
    if len(words) > 45:
        sanitized = " ".join(words[:45]).rstrip(",;:-") + "."

    if not sanitized.endswith((".", "!", "?")):
        sanitized += "."

    return sanitized


def compute_chip_geometry_metrics(
    miner_items: Sequence[PerImageAnnotationItem],
    golden_annotations: Sequence[Any],
    image_width: int = 1024,
    image_height: int = 1024,
) -> Dict[str, Any]:
    """Extract quantitative geometric and error metrics across golden chip annotations."""
    total_miner = len(miner_items)
    total_gt = len(golden_annotations)

    matched_ious: List[float] = []
    used_miner = set()

    # Match gt to miner
    for gt in golden_annotations:
        gt_box = gt.bounding_box
        gt_poly = getattr(gt, "polygon", None)
        best_iou = 0.0
        best_idx = -1
        for idx, item in enumerate(miner_items):
            if idx in used_miner:
                continue
            item_poly = getattr(item, "polygon", None)
            if gt_poly and item_poly:
                iou = raster_polygon_iou(item_poly, gt_poly, item.bounding_box, gt_box)
            else:
                iou = iou_xyxy(item.bounding_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= 0.30:
            matched_ious.append(best_iou)
            used_miner.add(best_idx)
        else:
            matched_ious.append(0.0)

    mean_iou = float(sum(matched_ious) / max(1, total_gt))
    false_negative_rate = float((total_gt - len(used_miner)) / max(1, total_gt))
    hallucinated = max(0, total_miner - len(used_miner))
    false_positive_rate = float(hallucinated / max(1, total_miner))

    # Vertex analysis
    miner_vertices = [len(m.polygon) for m in miner_items if m.polygon and len(m.polygon) >= 3]
    mean_vertices = float(sum(miner_vertices) / max(1, len(miner_vertices))) if miner_vertices else 4.0

    box_like = sum(1 for m in miner_items if is_crude_box(m.polygon, m.bounding_box))
    box_like_ratio = float(box_like / max(1, total_miner))

    total_canvas = float(image_width * image_height)
    oversized = 0
    for m in miner_items:
        if m.bounding_box and len(m.bounding_box) == 4:
            x1, y1, x2, y2 = m.bounding_box
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area / max(1.0, total_canvas) >= 0.50:
                oversized += 1
    oversized_ratio = float(oversized / max(1, total_miner))

    # Class distribution skew
    class_skew: Dict[str, int] = {}
    for m in miner_items:
        c = (m.hazard_class or "ordinary_tree").strip().lower()
        class_skew[c] = class_skew.get(c, 0) + 1

    return {
        "mean_iou": round(mean_iou, 4),
        "boundary_f1_or_distance_error": round(1.0 - mean_iou, 4),
        "hallucination_penalty": round(1.0 - false_positive_rate, 4),
        "false_negative_rate": round(false_negative_rate, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "mean_vertices_per_polygon": round(mean_vertices, 1),
        "box_like_polygon_ratio": round(box_like_ratio, 4),
        "oversized_canvas_ratio": round(oversized_ratio, 4),
        "class_distribution_skew": class_skew,
    }


def build_comparison_grid(
    selected_samples: Sequence[Tuple[GoldenImage, List[PerImageAnnotationItem]]],
    thumbnail_size: int = 384,
    jpeg_quality: int = 80,
) -> bytes:
    """Build a single stitched side-by-side composite JPEG image in memory.

    Left Panel: Part A: Miner Prediction (satellite chip + miner predicted polygons in cyan/red + dots + text)
    Right Panel: Part B: Golden Ground Truth (satellite chip + golden contours in green + missed in yellow + false positives in red)

    Kept strictly in memory (RAM). Never persists golden chips to disk or R2.
    """
    rows = []

    for golden_image, miner_items in selected_samples:
        img_path = str(golden_image.image_path) if golden_image.image_path else ""
        base_img = None
        if img_path and Path(img_path).is_file():
            base_img = cv2.imread(img_path)

        if base_img is None:
            # Fallback synthetic aerial background (dark green canopy background)
            base_img = np.full((1024, 1024, 3), (30, 60, 35), dtype=np.uint8)

        orig_h, orig_w = base_img.shape[:2]
        scale_x = thumbnail_size / float(orig_w)
        scale_y = thumbnail_size / float(orig_h)

        thumb = cv2.resize(base_img, (thumbnail_size, thumbnail_size), interpolation=cv2.INTER_AREA)

        # Left Panel: Miner Prediction
        left = thumb.copy()
        cv2.rectangle(left, (0, 0), (thumbnail_size, 26), (0, 0, 0), -1)
        cv2.putText(
            left,
            "Part A: Miner Prediction",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for item in miner_items:
            poly = item.polygon
            if poly and len(poly) >= 3:
                pts = np.array(
                    [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in poly],
                    dtype=np.int32,
                )
                cv2.polylines(left, [pts], isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                for pt in pts:
                    cv2.circle(left, tuple(pt), radius=2, color=(0, 255, 255), thickness=-1)
            elif item.bounding_box and len(item.bounding_box) == 4:
                bx1, by1, bx2, by2 = item.bounding_box
                p1 = (int(bx1 * scale_x), int(by1 * scale_y))
                p2 = (int(bx2 * scale_x), int(by2 * scale_y))
                cv2.rectangle(left, p1, p2, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            if item.bounding_box and len(item.bounding_box) == 4:
                bx1, by1, _, _ = item.bounding_box
                lbl = f"{item.hazard_class[:8]}:{item.confidence or 0.0:.2f}"
                cv2.putText(
                    left,
                    lbl,
                    (max(2, int(bx1 * scale_x)), max(12, int(by1 * scale_y) - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        # Right Panel: Golden Ground Truth
        right = thumb.copy()
        cv2.rectangle(right, (0, 0), (thumbnail_size, 26), (0, 0, 0), -1)
        cv2.putText(
            right,
            "Part B: Golden Ground Truth",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        gt_annotations = golden_image.annotations or []
        for gt in gt_annotations:
            gt_poly = getattr(gt, "polygon", None)
            if gt_poly and len(gt_poly) >= 3:
                pts = np.array(
                    [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in gt_poly],
                    dtype=np.int32,
                )
                cv2.polylines(right, [pts], isClosed=True, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            elif gt.bounding_box and len(gt.bounding_box) == 4:
                bx1, by1, bx2, by2 = gt.bounding_box
                p1 = (int(bx1 * scale_x), int(by1 * scale_y))
                p2 = (int(bx2 * scale_x), int(by2 * scale_y))
                cv2.rectangle(right, p1, p2, (0, 255, 0), 1, lineType=cv2.LINE_AA)

        # Highlight false-negatives (missed) in yellow and hallucinations in red
        used_m = set()
        for gt in gt_annotations:
            matched = False
            for m_idx, m_item in enumerate(miner_items):
                if m_idx in used_m:
                    continue
                if iou_xyxy(m_item.bounding_box, gt.bounding_box) >= 0.30:
                    matched = True
                    used_m.add(m_idx)
                    break
            if not matched and gt.bounding_box:
                bx1, by1, bx2, by2 = gt.bounding_box
                p1 = (int(bx1 * scale_x), int(by1 * scale_y))
                p2 = (int(bx2 * scale_x), int(by2 * scale_y))
                cv2.rectangle(right, p1, p2, (0, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(right, "MISS", (p1[0], max(12, p1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        # Spurious miner hallucinations in red on right
        for m_idx, m_item in enumerate(miner_items):
            if m_idx not in used_m and m_item.bounding_box:
                bx1, by1, bx2, by2 = m_item.bounding_box
                p1 = (int(bx1 * scale_x), int(by1 * scale_y))
                p2 = (int(bx2 * scale_x), int(by2 * scale_y))
                cv2.rectangle(right, p1, p2, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        pair_row = np.hstack([left, right])
        rows.append(pair_row)

    if not rows:
        blank = np.zeros((thumbnail_size, thumbnail_size * 2, 3), dtype=np.uint8)
        rows.append(blank)

    composite = np.vstack(rows)
    success, encoded = cv2.imencode(".jpg", composite, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not success:
        raise RuntimeError("Failed to encode composite image to JPEG")
    return encoded.tobytes()


def generate_validator_feedback(
    watchdog: AgenticWatchdog,
    composite_jpeg_bytes: bytes,
    geometry_summary: Dict[str, Any],
    golden_image_ids: Sequence[str] = (),
    strict_live_llm: bool = False,
    round_num: int = 1,
) -> str:
    """Pass single stitched composite JPEG + metrics block to Gemini 2.5 Flash."""
    system_instruction = (
        "You are the Subnet 498 CanopyMRV Validator Diagnostic Judge. Compare Left (Miner Prediction) "
        "vs. Right (Golden Ground Truth) across these satellite canopy samples and accompanying polygon geometry statistics. "
        "Identify the #1 and #2 systematic failure patterns (e.g., missing small clustered crowns, overly coarse "
        "4-point bounding-box polygons instead of organic crown contours, over-merging adjacent canopies, "
        "confidence threshold too high/low, or class mislabeling). Output STRICTLY 1 to 2 concise sentences "
        "(maximum 45 words) of actionable engineering feedback for the miner. NEVER mention image IDs, chip indices, "
        "or pixel coordinates."
    )

    user_payload = (
        f"Polygon Geometry & Quality Statistics:\n{json.dumps(geometry_summary, indent=2)}\n\n"
        "Provide your 1-2 sentence engineering diagnosis:"
    )

    raw_response = watchdog.call_gemini(
        system_prompt=system_instruction,
        user_content=user_payload,
        image_bytes=composite_jpeg_bytes,
        mime_type="image/jpeg",
        max_output_tokens=120,
        strict_live_llm=strict_live_llm,
        round_num=round_num,
    )

    if raw_response:
        sanitized = sanitize_feedback(raw_response, golden_image_ids)
        if sanitized:
            return sanitized

    if strict_live_llm:
        raise RuntimeError(
            "Validator Gemini diagnostic call failed or returned empty response in strict_live_llm mode! "
            "Rule-based fallback is disabled."
        )

    # Rule-based fallback if LLM unavailable or circuit breaker tripped
    fn_rate = geometry_summary.get("false_negative_rate", 0.0)
    box_ratio = geometry_summary.get("box_like_polygon_ratio", 0.0)
    fp_rate = geometry_summary.get("false_positive_rate", 0.0)

    if fn_rate > 0.30 and box_ratio > 0.40:
        return (
            f"Missing {int(fn_rate * 100)}% of canopy crowns and producing coarse box-like polygons; "
            "lower detection confidence threshold and increase contour approximation density."
        )
    elif fn_rate > 0.25:
        return (
            f"Missing {int(fn_rate * 100)}% of small clustered tree crowns; "
            "lower detection confidence threshold (~0.25) to improve recall in dense sectors."
        )
    elif box_ratio > 0.30:
        return "Producing coarse rectangular polygons; refine contour smoothing and increase vertex count to match natural crowns."
    elif fp_rate > 0.25:
        return "Elevated false-positive detections on background terrain; increase confidence threshold and filter low-vegetation regions."
    else:
        return "Polygon contours align well with canopy boundaries; tune NMS IoU threshold and edge smoothing for marginal gains."


class ValidatorAgenticEngine:
    """Manages round feedback generation, scoped receipt delivery, and dual transport."""

    def __init__(
        self,
        watchdog: Optional[AgenticWatchdog] = None,
        r2_client: Optional[Any] = None,
        r2_bucket: Optional[str] = None,
        r2_telemetry_prefix: Optional[str] = None,
        enabled: bool = True,
        strict_live_llm: bool = False,
    ):
        self.watchdog = watchdog or AgenticWatchdog(role="validator", strict_live_llm=strict_live_llm)
        self.strict_live_llm = strict_live_llm
        if strict_live_llm:
            self.watchdog.strict_live_llm = True
        self.r2_client = r2_client
        self.r2_bucket = r2_bucket or os.getenv("R2_BUCKET_NAME", "annotation-subnet")
        self.r2_telemetry_prefix = r2_telemetry_prefix or os.getenv("R2_LOCALNET_TELEMETRY_PREFIX", "localnet_sim/telemetry/")
        self.enabled = enabled
        self.llm_calls: List[Dict[str, Any]] = []

        self.latest_feedback_by_hotkey: Dict[str, Dict[str, Any]] = {}
        self.prev_scores_by_hotkey: Dict[str, float] = {}

    def get_latest_feedback_json(self, hotkey_ss58: str) -> Optional[str]:
        """Return serialized feedback receipt for piggybacking on next round Synapse dispatch."""
        data = self.latest_feedback_by_hotkey.get(hotkey_ss58)
        if data:
            return json.dumps(data)
        return None

    def process_round(
        self,
        round_num: int,
        batch_id: str,
        uids: Sequence[int],
        hotkeys: Sequence[str],
        per_miner_scores: Dict[int, PerMinerAnnotationScore],
        corpus: ImageCorpus,
        annotations_by_uid: Dict[int, Dict[str, List[PerImageAnnotationItem]]],
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate round miners, compute scoped receipts, and publish via R2 + RAM cache."""
        if not self.enabled:
            return {}

        timestamp_iso = datetime.now(timezone.utc).isoformat()
        active_entries = []

        for uid in uids:
            hk = hotkeys[uid] if uid < len(hotkeys) else f"uid_{uid}"
            score_obj = per_miner_scores.get(uid)
            fidelity_scores = score_obj.fidelity_scores_by_image_id if score_obj else {}
            avg_fid = float(sum(fidelity_scores.values()) / max(1, len(fidelity_scores))) if fidelity_scores else 0.0
            active_entries.append({"uid": uid, "hotkey": hk, "fidelity": avg_fid, "score_obj": score_obj})

        # Rank miners descending by fidelity score
        active_entries.sort(key=lambda x: x["fidelity"], reverse=True)
        total_active = len(active_entries)
        leader_score = active_entries[0]["fidelity"] if active_entries else 0.0

        receipts_by_hotkey: Dict[str, Dict[str, Any]] = {}

        for rank_idx, entry in enumerate(active_entries, start=1):
            uid = entry["uid"]
            hk = entry["hotkey"]
            fid = entry["fidelity"]
            score_obj = entry["score_obj"]

            prev_score = self.prev_scores_by_hotkey.get(hk, fid)
            score_delta = round(fid - prev_score, 4)
            self.prev_scores_by_hotkey[hk] = fid

            percentile = round((total_active - rank_idx + 1) / max(1, total_active) * 100.0, 1)
            leader_gap = round(leader_score - fid, 4)

            # Golden samples selection for this miner
            miner_submissions = annotations_by_uid.get(uid, {})
            golden_scores = score_obj.fidelity_scores_by_image_id if score_obj else {}

            selected_samples: List[Tuple[GoldenImage, List[PerImageAnnotationItem]]] = []
            golden_chip_ids = []

            # Sort golden images evaluated for this miner by fidelity ascending
            sorted_chips = sorted(golden_scores.items(), key=lambda kv: kv[1])
            if sorted_chips:
                k_chips = []
                if len(sorted_chips) <= 3:
                    k_chips = sorted_chips
                else:
                    k_chips.append(sorted_chips[0])
                    k_chips.append(sorted_chips[1])
                    median_idx = len(sorted_chips) // 2
                    k_chips.append(sorted_chips[median_idx])

                for chip_id, _ in k_chips:
                    g_img = None
                    gl = getattr(corpus, "golden_lookup", None)
                    if callable(gl):
                        g_img = gl(chip_id)
                    elif isinstance(gl, dict):
                        g_img = gl.get(chip_id)
                    if g_img:
                        m_items = miner_submissions.get(chip_id, [])
                        selected_samples.append((g_img, m_items))
                        golden_chip_ids.append(chip_id)

            # Compute quantitative geometry metrics across golden samples
            all_miner_items = []
            all_gt_items = []
            for g_img, m_items in selected_samples:
                all_miner_items.extend(m_items)
                all_gt_items.extend(g_img.annotations or [])

            geom_metrics = compute_chip_geometry_metrics(all_miner_items, all_gt_items)

            # Build stitched side-by-side composite image strictly in RAM
            try:
                composite_bytes = build_comparison_grid(selected_samples, thumbnail_size=384, jpeg_quality=80)
            except Exception as comp_err:
                bt.logging.warning(f"[AGENTIC FEEDBACK] Failed to build composite grid for UID {uid}: {comp_err}")
                composite_bytes = b""

            # Query Gemini / Heuristic diagnostic judge
            feedback_oneliner = generate_validator_feedback(
                watchdog=self.watchdog,
                composite_jpeg_bytes=composite_bytes,
                geometry_summary=geom_metrics,
                golden_image_ids=golden_chip_ids,
                strict_live_llm=self.strict_live_llm,
                round_num=round_num,
            )

            # Record call metadata proof
            if self.watchdog.call_logs:
                last_call = self.watchdog.call_logs[-1]
                proof_entry = {
                    "round_num": round_num,
                    "model": last_call.get("model", self.watchdog.model),
                    "prompt_tokens": last_call.get("prompt_tokens", 0),
                    "image_tokens": last_call.get("image_tokens", 516),
                    "candidate_tokens": last_call.get("candidate_tokens", 0),
                    "raw_response_text": last_call.get("raw_response_text", ""),
                    "latency_ms": last_call.get("latency_ms", 0),
                }
                self.llm_calls.append(proof_entry)
                self.watchdog.record_llm_proof("validator", proof_entry)

            # Build scoped receipt (schema: agentic.feedback.v1)
            receipt = {
                "schema_version": "agentic.feedback.v1",
                "round_num": round_num,
                "batch_id": batch_id,
                "timestamp": timestamp_iso,
                "miner_uid": uid,
                "hotkey_ss58": hk,
                "leaderboard": {
                    "self_rank": rank_idx,
                    "total_active_miners": total_active,
                    "percentile": percentile,
                    "self_fidelity_score": round(fid, 4),
                    "prev_round_score": round(prev_score, 4),
                    "score_delta": score_delta,
                    "leader_score_gap": leader_gap,
                },
                "geometry_summary": {
                    "mean_iou": geom_metrics["mean_iou"],
                    "false_negative_rate": geom_metrics["false_negative_rate"],
                    "false_positive_rate": geom_metrics["false_positive_rate"],
                    "mean_vertices": geom_metrics["mean_vertices_per_polygon"],
                },
                "validator_feedback_oneliner": feedback_oneliner,
            }

            self.latest_feedback_by_hotkey[hk] = receipt
            receipts_by_hotkey[hk] = receipt

            # Primary Delivery: Cloudflare R2 Upload
            self._upload_receipt_to_r2(hk, receipt, round_num)

        return receipts_by_hotkey

    def _upload_receipt_to_r2(self, hotkey_ss58: str, receipt: Dict[str, Any], round_num: int) -> None:
        """Upload scoped receipt to R2 under isolated telemetry prefix."""
        if not self.r2_client or not self.r2_bucket:
            return
        try:
            hk_hash = hashlib.sha256(hotkey_ss58.encode("utf-8")).hexdigest()
            prefix = self.r2_telemetry_prefix.rstrip("/")
            body = json.dumps(receipt, indent=2).encode("utf-8")

            # Upload latest.json
            key_latest = f"{prefix}/feedback/{hk_hash}/latest.json"
            self.r2_client.put_object(
                Bucket=self.r2_bucket,
                Key=key_latest,
                Body=body,
                ContentType="application/json",
            )

            # Upload round_<round_num>.json
            key_round = f"{prefix}/feedback/{hk_hash}/round_{round_num}.json"
            self.r2_client.put_object(
                Bucket=self.r2_bucket,
                Key=key_round,
                Body=body,
                ContentType="application/json",
            )
            bt.logging.debug(f"[AGENTIC FEEDBACK] Uploaded scoped receipt to R2: {key_latest}")
        except Exception as exc:
            bt.logging.warning(f"[AGENTIC FEEDBACK] Failed to upload receipt to R2 for {hotkey_ss58}: {exc}")
