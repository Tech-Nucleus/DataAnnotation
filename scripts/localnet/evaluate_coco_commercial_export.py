#!/usr/bin/env python3
"""Compare commercial export vs COCO holdout GT on the 180-image pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from template.hazard.annotation_eval import iou_xyxy


def _load_export(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _evaluate(rows: Sequence[dict], holdout: Dict[str, list]) -> dict:
    tp = fp = fn = 0
    iou_sum = 0.0
    matched = 0
    golden_leaks = sum(1 for r in rows if r.get("is_golden"))
    for row in rows:
        if row.get("is_golden"):
            continue
        if row.get("escalation_required"):
            continue
        image_id = str(row["image_id"])
        gt_list = holdout.get(image_id, [])
        used: set[int] = set()
        objects = row.get("objects") or []
        for obj in objects:
            if obj.get("escalation_reason"):
                continue
            pred_cls = str(obj.get("accepted_hazard_class") or "").lower().strip()
            pred_box = obj.get("fused_bounding_box") or []
            best_idx = -1
            best_iou = 0.0
            for idx, gt in enumerate(gt_list):
                if idx in used:
                    continue
                if pred_cls != str(gt["hazard_class"]).lower().strip():
                    continue
                iou = iou_xyxy(pred_box, gt["bounding_box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= 0.5:
                tp += 1
                iou_sum += best_iou
                matched += 1
                used.add(best_idx)
            else:
                fp += 1
        fn += max(0, len(gt_list) - len(used))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "export_rows": len(rows),
        "golden_leaks": golden_leaks,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": (iou_sum / matched) if matched else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--min-f1", type=float, default=0.0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    holdout = manifest.get("holdout_ground_truth") or {}
    rows = _load_export(args.export)
    metrics = _evaluate(rows, holdout)
    print(json.dumps(metrics, indent=2))
    if metrics["golden_leaks"] > 0:
        raise SystemExit("FAIL: golden images leaked into commercial export")
    if metrics["f1"] < args.min_f1:
        raise SystemExit(f"FAIL: f1 {metrics['f1']:.4f} < min {args.min_f1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
