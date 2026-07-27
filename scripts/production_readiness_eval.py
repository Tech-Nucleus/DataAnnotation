#!/usr/bin/env python3
"""
Production-readiness checks for probabilistic aggregation + commercial JSONL.

Usage:
  PYTHONPATH=. python scripts/production_readiness_eval.py schema --glob 'artifacts/**/*.jsonl'
  PYTHONPATH=. python scripts/production_readiness_eval.py golden-holdout \\
      --golden path/to/golden_manifest.json --commercial path/to/commercial.jsonl \\
      --holdout-count 1000 --holdout-seed 42

Exit code 0 only when all invoked checks pass thresholds (see --help per subcommand).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# --- Schema (must match template/hazard/dataset_assembler WinningAnnotation export) ---

ROOT_COMMERCIAL_KEYS: frozenset[str] = frozenset(
    {
        "image_id",
        "aggregation_method",
        "acceptance_thresholds",
        "reliability_window",
        "escalation_required",
        "escalation_reason",
        "validator_version",
        "audit_hash",
        "miner_contribution_scores",
        "objects",
        "score",
        "chosen_uid",
        "image_url",
        "annotated_image_url",
        "width",
        "height",
        "is_golden",
        "timestamp",
    }
)

OBJECT_KEYS: frozenset[str] = frozenset(
    {
        "aggregation_method",
        "object_cluster_id",
        "accepted_hazard_class",
        "accepted_severity",
        "confidence",
        "severity_confidence",
        "class_posterior_distribution",
        "severity_posterior_distribution",
        "fused_bounding_box",
        "spatial_mean_iou_to_median",
        "miner_votes",
        "escalation_reason",
    }
)

VOTE_KEYS: frozenset[str] = frozenset(
    {
        "miner_uid",
        "miner_hotkey",
        "class_voted",
        "severity_voted",
        "confidence",
        "bounding_box",
        "reliability_weight_at_aggregation",
    }
)


def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(x) for x in a)
    bx1, by1, bx2, by2 = (float(x) for x in b)
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, inter / denom)))


def _norm_class(c: str) -> str:
    return (c or "").strip().lower().replace(" ", "_")


def validate_commercial_row(row: Mapping[str, Any], *, line_no: int) -> List[str]:
    errors: List[str] = []
    missing_root = ROOT_COMMERCIAL_KEYS - frozenset(row.keys())
    if missing_root:
        errors.append(f"line {line_no}: missing root keys {sorted(missing_root)}")
    # Validate image_url is a reachable HTTP(S) link (not file://, r2://, or empty)
    image_url = str(row.get("image_url") or "").strip()
    if not image_url:
        errors.append(f"line {line_no}: image_url is empty")
    elif not image_url.startswith(("http://", "https://")):
        errors.append(
            f"line {line_no}: image_url must be HTTP(S), got {image_url[:80]!r}"
        )
    # Validate annotated_image_url is a reachable HTTP(S) link
    annotated_image_url = str(row.get("annotated_image_url") or "").strip()
    if not annotated_image_url:
        errors.append(f"line {line_no}: annotated_image_url is empty")
    elif not annotated_image_url.startswith(("http://", "https://")):
        errors.append(
            f"line {line_no}: annotated_image_url must be HTTP(S), got {annotated_image_url[:80]!r}"
        )
    if row.get("escalation_required") is False:
        if not row.get("objects"):
            errors.append(f"line {line_no}: accepted row must have non-empty objects")
    for i, obj in enumerate(row.get("objects") or []):
        if not isinstance(obj, dict):
            errors.append(f"line {line_no}: objects[{i}] not an object")
            continue
        mo = OBJECT_KEYS - frozenset(obj.keys())
        if mo:
            errors.append(f"line {line_no} objects[{i}]: missing keys {sorted(mo)}")
        for j, vote in enumerate(obj.get("miner_votes") or []):
            if not isinstance(vote, dict):
                errors.append(f"line {line_no} objects[{i}].miner_votes[{j}] not an object")
                continue
            mv = VOTE_KEYS - frozenset(vote.keys())
            if mv:
                errors.append(
                    f"line {line_no} objects[{i}].miner_votes[{j}]: missing keys {sorted(mv)}"
                )
    ah = hashlib.sha256()
    probe = {k: v for k, v in sorted(row.items()) if k != "audit_hash"}
    ah.update(json.dumps(probe, sort_keys=True).encode("utf-8"))
    expected = ah.hexdigest()
    got = row.get("audit_hash")
    if got != expected:
        errors.append(
            f"line {line_no}: audit_hash mismatch (recomputed {expected[:16]}… vs stored {got!r})"
        )
    return errors


def cmd_schema(args: argparse.Namespace) -> int:
    patterns = args.glob
    paths: List[Path] = []
    for pat in patterns:
        paths.extend(Path(p) for p in glob.glob(pat, recursive=True))
    paths = sorted({p for p in paths if p.is_file()})
    if not paths:
        print("schema: no files matched", file=sys.stderr)
        return 1
    all_errors: List[str] = []
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"{path}:{lineno}: JSON error {e}")
                continue
            all_errors.extend(validate_commercial_row(row, line_no=lineno))
    if all_errors:
        for e in all_errors[:200]:
            print(e, file=sys.stderr)
        if len(all_errors) > 200:
            print(f"... and {len(all_errors) - 200} more errors", file=sys.stderr)
        return 1
    print(f"schema: OK ({len(paths)} file(s))")
    return 0


def _load_golden_manifest(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    images = data.get("images") or []
    out: Dict[str, List[Dict[str, Any]]] = {}
    for im in images:
        iid = im.get("image_id")
        if not iid:
            continue
        out[str(iid)] = list(im.get("annotations") or [])
    return out


def _load_labels_map(path: Optional[Path]) -> Dict[str, List[Dict[str, Any]]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "labels_by_image_id" in data:
        raw = data["labels_by_image_id"]
        return {str(k): list(v or []) for k, v in raw.items()}
    if "images" in data:
        return _load_golden_manifest(path)
    raise ValueError(f"Unsupported labels file (need labels_by_image_id or images): {path}")


def _merge_gt(
    primary: Dict[str, List[Dict[str, Any]]],
    extra: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    out = {k: list(v) for k, v in primary.items()}
    for k, v in extra.items():
        out.setdefault(k, [])
        out[k].extend(v)
    return out


def _holdout_image_ids(
    candidates: Sequence[str],
    *,
    holdout_count: int,
    seed: int,
) -> Set[str]:
    ids = sorted(set(candidates))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = min(holdout_count, len(ids))
    return set(ids[:n])


def cmd_golden_holdout(args: argparse.Namespace) -> int:
    golden = _load_golden_manifest(Path(args.golden))
    extra = _load_labels_map(Path(args.extra_labels)) if args.extra_labels else {}
    golden = _merge_gt(golden, extra)
    commercial_path = Path(args.commercial)
    rows: List[Dict[str, Any]] = []
    for line in commercial_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    pool_ids = [r["image_id"] for r in rows if not r.get("is_golden", False)]
    overlap = [iid for iid in pool_ids if iid in golden]
    holdout = _holdout_image_ids(overlap, holdout_count=args.holdout_count, seed=args.holdout_seed)

    iou_match = args.iou_match_threshold
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)
    total_match = 0
    total = 0

    for row in rows:
        if row.get("is_golden"):
            continue
        if row.get("escalation_required"):
            continue
        iid = row["image_id"]
        if iid not in holdout:
            continue
        gt_list = golden.get(iid) or []
        gt_boxes = [
            ( _norm_class(a.get("hazard_class", "")), [float(x) for x in (a.get("bounding_box") or [])])
            for a in gt_list
            if len(a.get("bounding_box") or []) == 4
        ]
        used_gt: Set[int] = set()
        for obj in row.get("objects") or []:
            pred_cls = obj.get("accepted_hazard_class")
            if pred_cls is None:
                continue
            pred_cls_n = _norm_class(str(pred_cls))
            fused = obj.get("fused_bounding_box")
            if not fused or len(fused) != 4:
                continue
            best_j = -1
            best_iou = 0.0
            for j, (gcls, gbox) in enumerate(gt_boxes):
                if j in used_gt:
                    continue
                iou = _iou_xyxy(fused, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j < 0 or best_iou < iou_match:
                fp[pred_cls_n] += 1
                total += 1
                continue
            used_gt.add(best_j)
            gcls_n = gt_boxes[best_j][0]
            total += 1
            if pred_cls_n == gcls_n:
                total_match += 1
                tp[pred_cls_n] += 1
            else:
                fp[pred_cls_n] += 1
                fn[gcls_n] += 1
        for j, (gcls_n, _) in enumerate(gt_boxes):
            if j not in used_gt:
                fn[gcls_n] += 1

    classes = sorted(set(tp.keys()) | set(fp.keys()) | set(fn.keys()))
    f1s: Dict[str, float] = {}
    for c in classes:
        p_tp, p_fp, p_fn = tp[c], fp[c], fn[c]
        prec = p_tp / (p_tp + p_fp) if (p_tp + p_fp) else 0.0
        rec = p_tp / (p_tp + p_fn) if (p_tp + p_fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s[c] = f1

    overall_acc = (total_match / total) if total else 0.0
    min_f1 = min(f1s.values()) if f1s else 0.0

    report = {
        "holdout_images": len(holdout),
        "evaluated_pairs": total,
        "overall_exact_class_accuracy": overall_acc,
        "per_class_f1": f1s,
        "min_per_class_f1": min_f1,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    ok = overall_acc >= args.min_overall_acc and min_f1 >= args.min_class_f1
    if not ok:
        print(
            f"golden-holdout: FAIL need overall>={args.min_overall_acc} "
            f"and min_class_f1>={args.min_class_f1}",
            file=sys.stderr,
        )
        return 1
    print("golden-holdout: PASS")
    return 0


def cmd_calibration(args: argparse.Namespace) -> int:
    golden = _load_golden_manifest(Path(args.golden))
    extra = _load_labels_map(Path(args.extra_labels)) if args.extra_labels else {}
    golden = _merge_gt(golden, extra)
    commercial_path = Path(args.commercial)
    pairs: List[Tuple[float, int]] = []
    for line in commercial_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("is_golden") or row.get("escalation_required"):
            continue
        iid = row["image_id"]
        gt_list = golden.get(iid) or []
        gt_boxes = [
            (_norm_class(a.get("hazard_class", "")), [float(x) for x in (a.get("bounding_box") or [])])
            for a in gt_list
            if len(a.get("bounding_box") or []) == 4
        ]
        for obj in row.get("objects") or []:
            pred_cls = obj.get("accepted_hazard_class")
            fused = obj.get("fused_bounding_box")
            conf = float(obj.get("confidence") or 0.0)
            if pred_cls is None or not fused or len(fused) != 4:
                continue
            pred_n = _norm_class(str(pred_cls))
            best_iou = 0.0
            best_g = ""
            for gcls, gbox in gt_boxes:
                iou = _iou_xyxy(fused, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_g = gcls
            if best_iou < args.iou_match_threshold:
                continue
            correct = 1 if pred_n == best_g else 0
            pairs.append((conf, correct))

    if len(pairs) < args.min_samples:
        print(
            f"calibration: FAIL need at least {args.min_samples} matched samples, got {len(pairs)}",
            file=sys.stderr,
        )
        return 1
    mean_conf = sum(p[0] for p in pairs) / len(pairs)
    acc = sum(p[1] for p in pairs) / len(pairs)
    delta = abs(mean_conf - acc)
    print(
        json.dumps(
            {
                "n": len(pairs),
                "mean_reported_confidence": mean_conf,
                "empirical_accuracy": acc,
                "abs_delta": delta,
                "max_allowed_delta": args.max_delta,
            },
            indent=2,
        )
    )
    if delta > args.max_delta:
        print("calibration: FAIL", file=sys.stderr)
        return 1
    print("calibration: PASS")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Run synthetic scenario suite (same logic as pytest stress; no chain)."""
    root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(root)}
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(root / "tests/stress/test_probabilistic_aggregation_acceptance.py"),
            "-q",
            "--tb=line",
        ],
        cwd=str(root),
        env=env,
    )
    if r.returncode != 0:
        print("simulate: FAIL (pytest stress suite)", file=sys.stderr)
        return r.returncode
    print("simulate: PASS (synthetic Sybil / collusion / minority / low-miner / calibration / schema / golden lane)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("schema", help="Validate commercial JSONL lines + audit_hash")
    ps.add_argument("--glob", nargs="+", required=True, help="Glob(s) for jsonl files")
    ps.set_defaults(func=cmd_schema)

    pg = sub.add_parser("golden-holdout", help="Holdout accuracy vs golden manifest")
    pg.add_argument("--golden", required=True, type=Path, help="golden_manifest.v1 JSON")
    pg.add_argument(
        "--extra-labels",
        type=Path,
        default=None,
        help="Optional JSON: {labels_by_image_id: {image_id: [{hazard_class, bounding_box}]}} for pool rows",
    )
    pg.add_argument("--commercial", required=True, type=Path)
    pg.add_argument("--holdout-count", type=int, default=1000)
    pg.add_argument("--holdout-seed", type=int, default=42)
    pg.add_argument("--iou-match-threshold", type=float, default=0.5)
    pg.add_argument("--min-overall-acc", type=float, default=0.90)
    pg.add_argument("--min-class-f1", type=float, default=0.85)
    pg.set_defaults(func=cmd_golden_holdout)

    pc = sub.add_parser("calibration", help="Mean confidence vs empirical accuracy (matched objects)")
    pc.add_argument("--golden", required=True, type=Path)
    pc.add_argument("--extra-labels", type=Path, default=None)
    pc.add_argument("--commercial", required=True, type=Path)
    pc.add_argument("--iou-match-threshold", type=float, default=0.5)
    pc.add_argument("--min-samples", type=int, default=100)
    pc.add_argument("--max-delta", type=float, default=0.05)
    pc.set_defaults(func=cmd_calibration)

    px = sub.add_parser("simulate", help="Run offline pytest stress scenarios")
    px.set_defaults(func=cmd_simulate)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
