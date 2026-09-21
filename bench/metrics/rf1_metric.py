"""
Raster-level F1 (RF1_75) Metric and Algorithm 1 Threshold Optimization.
Implements the SelvaBox / CanopyRS evaluation protocol (ICLR 2026, arXiv:2507.00170).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of bounding boxes [x1, y1, x2, y2].
    boxes_a: shape (N, 4)
    boxes_b: shape (M, 4)
    Returns: shape (N, M)
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou.astype(np.float32)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    Standard Non-Maximum Suppression (NMS).
    Returns indices of kept boxes.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


def compute_rf1(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, np.ndarray]],
    iou_threshold: float = 0.75,
    score_threshold: float = 0.3,
    nms_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute raster-level F1 (RF1_75) across multiple tiles.
    predictions: list of dicts with 'boxes' (N, 4) and 'scores' (N,)
    ground_truths: list of dicts with 'boxes' (M, 4)
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, gt in zip(predictions, ground_truths):
        p_boxes = pred.get("boxes", np.zeros((0, 4), dtype=np.float32))
        p_scores = pred.get("scores", np.zeros((0,), dtype=np.float32))
        gt_boxes = gt.get("boxes", np.zeros((0, 4), dtype=np.float32))

        # 1. Filter by score threshold
        valid_idx = np.where(p_scores >= score_threshold)[0]
        p_boxes = p_boxes[valid_idx]
        p_scores = p_scores[valid_idx]

        # 2. Apply NMS
        if len(p_boxes) > 0 and nms_threshold < 1.0:
            keep = nms_boxes(p_boxes, p_scores, nms_threshold)
            p_boxes = p_boxes[keep]
            p_scores = p_scores[keep]

        # 3. Match predictions to ground truth at iou_threshold
        if len(gt_boxes) == 0:
            total_fp += len(p_boxes)
            continue
        if len(p_boxes) == 0:
            total_fn += len(gt_boxes)
            continue

        # Sort predictions by descending score
        order = np.argsort(-p_scores)
        p_boxes = p_boxes[order]

        ious = box_iou(p_boxes, gt_boxes)
        matched_gt = set()
        matched_pred = set()

        for p_idx in range(len(p_boxes)):
            gt_matches = np.where(ious[p_idx] >= iou_threshold)[0]
            if len(gt_matches) > 0:
                # Find best unmatched ground truth
                best_gt = None
                best_iou = -1.0
                for g_idx in gt_matches:
                    if g_idx not in matched_gt and ious[p_idx, g_idx] > best_iou:
                        best_iou = ious[p_idx, g_idx]
                        best_gt = g_idx
                if best_gt is not None:
                    matched_gt.add(best_gt)
                    matched_pred.add(p_idx)

        tp = len(matched_pred)
        fp = len(p_boxes) - tp
        fn = len(gt_boxes) - len(matched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    denom_p = total_tp + total_fp
    denom_r = total_tp + total_fn
    precision = float(total_tp / denom_p) if denom_p > 0 else 0.0
    recall = float(total_tp / denom_r) if denom_r > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "rf1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def optimize_thresholds_algorithm_1(
    val_predictions: List[Dict[str, np.ndarray]],
    val_ground_truths: List[Dict[str, np.ndarray]],
    iou_threshold: float = 0.75,
    s_min_range: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    tau_nms_range: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
) -> Tuple[float, float, float]:
    """
    Implements Algorithm 1 from SelvaBox (ICLR 2026):
    Grid search over (s_min, tau_nms) on the validation set to maximize RF1_75.
    Returns: (best_s_min, best_tau_nms, best_val_rf1)
    """
    best_rf1 = -1.0
    best_s_min = 0.3
    best_tau_nms = 0.5

    for s_min in s_min_range:
        for tau_nms in tau_nms_range:
            res = compute_rf1(
                val_predictions,
                val_ground_truths,
                iou_threshold=iou_threshold,
                score_threshold=s_min,
                nms_threshold=tau_nms,
            )
            if res["rf1"] > best_rf1:
                best_rf1 = res["rf1"]
                best_s_min = s_min
                best_tau_nms = tau_nms

    return best_s_min, best_tau_nms, best_rf1
