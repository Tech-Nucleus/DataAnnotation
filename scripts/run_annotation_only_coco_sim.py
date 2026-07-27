#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from template.hazard.annotation_eval import (
    AnnotationFidelityScorer,
    ConsensusScorer,
    evaluate_round_annotations,
    iou_xyxy,
)
from template.hazard.dataset_assembler import DatasetAssembler
from template.hazard.dual_reward import DualFlywheelRewardComposer
from template.hazard.image_corpus import (
    GoldenAnnotation,
    GoldenImage,
    ImageCorpus,
    ImageCorpusConfig,
    UnlabeledImage,
)
from template.protocol import PerImageAnnotationItem


@dataclass(frozen=True)
class CocoImage:
    image_id: int
    file_name: str
    width: int
    height: int


def _load_coco_subset(
    *,
    annotations_file: Path,
    images_dir: Path,
    dataset_size: int,
    golden_ratio: float,
    seed: int,
) -> tuple[ImageCorpus, Dict[str, list[GoldenAnnotation]]]:
    payload = json.loads(annotations_file.read_text(encoding="utf-8"))
    categories = {int(c["id"]): str(c["name"]).strip().lower().replace(" ", "_") for c in payload["categories"]}
    images = [CocoImage(int(im["id"]), im["file_name"], int(im["width"]), int(im["height"])) for im in payload["images"]]
    anns_by_image: Dict[int, list[dict]] = {}
    for ann in payload["annotations"]:
        if ann.get("iscrowd"):
            continue
        bbox = ann.get("bbox") or []
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    eligible = [im for im in images if im.image_id in anns_by_image]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen = eligible[:dataset_size]
    if len(chosen) < dataset_size:
        raise RuntimeError(f"Requested {dataset_size} images but only found {len(chosen)} eligible COCO images.")

    golden_count = max(1, int(round(dataset_size * golden_ratio)))
    golden_ids = {im.image_id for im in chosen[:golden_count]}

    cache_root = Path(tempfile.mkdtemp(prefix="coco-annotation-only-"))
    corpus = ImageCorpus(ImageCorpusConfig(cache_root=cache_root))
    corpus._loaded = True
    gt_by_hash: Dict[str, list[GoldenAnnotation]] = {}

    for im in chosen:
        image_path = images_dir / im.file_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing COCO image: {image_path}")
        image_bytes = image_path.read_bytes()
        image_hash = __import__("hashlib").sha256(image_bytes).hexdigest()
        corpus._all_image_index[image_hash] = image_path
        anns: list[GoldenAnnotation] = []
        for ann in anns_by_image[im.image_id]:
            x, y, w, h = [float(v) for v in ann["bbox"]]
            from template.hazard.image_corpus import _severity_for_label
            cls = categories[int(ann["category_id"])]
            anns.append(
                GoldenAnnotation(
                    hazard_class=cls,
                    bounding_box=(int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))),
                    severity=_severity_for_label(cls),
                )
            )
        gt_by_hash[image_hash] = anns
        if im.image_id in golden_ids:
            golden = GoldenImage(
                image_id=image_hash,
                image_path=image_path,
                image_url=image_path.as_uri(),
                width=im.width,
                height=im.height,
                annotations=tuple(anns),
            )
            corpus._golden.append(golden)
            corpus._golden_index[image_hash] = golden
        else:
            corpus._annotation.append(
                UnlabeledImage(
                    image_id=image_hash,
                    image_path=image_path,
                    image_url=image_path.as_uri(),
                    width=im.width,
                    height=im.height,
                    source_dataset="coco2017",
                )
            )
    return corpus, gt_by_hash


def _anno(cls: str, bbox: Sequence[float]) -> PerImageAnnotationItem:
    return PerImageAnnotationItem(
        hazard_class=cls,
        bounding_box=[float(v) for v in bbox],
    )


def _simulate_miner_annotations(
    *,
    corpus: ImageCorpus,
    gt_by_hash: Dict[str, list[GoldenAnnotation]],
    quality: str,
    seed: int,
) -> Dict[str, list[PerImageAnnotationItem]]:
    rng = random.Random(seed)
    out: Dict[str, list[PerImageAnnotationItem]] = {}
    image_ids = [g.image_id for g in corpus.golden_images()] + [u.image_id for u in corpus.annotation_images()]
    for image_id in image_ids:
        gt = gt_by_hash[image_id]
        dims = corpus.golden_lookup(image_id)
        if dims is not None:
            width, height = dims.width, dims.height
        else:
            unlabeled = next(image for image in corpus.annotation_images() if image.image_id == image_id)
            width, height = unlabeled.width, unlabeled.height
        if quality == "good":
            items = [_anno(a.hazard_class, a.bounding_box) for a in gt]
        elif quality == "medium":
            items = []
            for a in gt:
                x1, y1, x2, y2 = a.bounding_box
                dx = rng.randint(-6, 6)
                dy = rng.randint(-6, 6)
                noisy = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
                cls = a.hazard_class if rng.random() < 0.8 else gt[0].hazard_class
                items.append(_anno(cls, noisy))
            if rng.random() < 0.15:
                items.append(_anno("random_object", [5.0, 5.0, 25.0, 25.0]))
        elif quality == "random":
            items = []
            for _ in range(max(1, len(gt))):
                x1 = rng.randint(0, max(1, width // 2))
                y1 = rng.randint(0, max(1, height // 2))
                x2 = x1 + rng.randint(10, max(11, width // 3))
                y2 = y1 + rng.randint(10, max(11, height // 3))
                items.append(_anno("random_object", [float(x1), float(y1), float(x2), float(y2)]))
        else:
            raise ValueError(f"Unknown quality {quality!r}")
        out[image_id] = items
    return out


def _evaluate_export(
    *,
    exported_rows: Sequence[dict],
    gt_by_hash: Dict[str, list[GoldenAnnotation]],
) -> dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    total_iou = 0.0
    matched = 0
    for row in exported_rows:
        gt = list(gt_by_hash.get(row["image_id"], []))
        used_gt: set[int] = set()
        for obj in row.get("objects", []):
            pred_cls = obj.get("accepted_hazard_class")
            pred_box = obj.get("fused_bounding_box") or []
            best_idx = -1
            best_iou = 0.0
            for idx, ann in enumerate(gt):
                if idx in used_gt or pred_cls != ann.hazard_class:
                    continue
                iou = iou_xyxy(pred_box, ann.bounding_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= 0.5:
                tp += 1
                total_iou += best_iou
                matched += 1
                used_gt.add(best_idx)
            else:
                fp += 1
        fn += max(0, len(gt) - len(used_gt))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": (total_iou / matched) if matched else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run annotation-only COCO local simulation.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations-file", type=Path, required=True)
    parser.add_argument("--dataset-size", type=int, default=200)
    parser.add_argument("--golden-ratio", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    corpus, gt_by_hash = _load_coco_subset(
        annotations_file=args.annotations_file,
        images_dir=args.images_dir,
        dataset_size=args.dataset_size,
        golden_ratio=args.golden_ratio,
        seed=args.seed,
    )
    export_root = Path(tempfile.mkdtemp(prefix="coco-export-"))
    assembler = DatasetAssembler(corpus=corpus, storage_prefix=export_root.as_uri())
    fidelity = AnnotationFidelityScorer()
    consensus = ConsensusScorer()
    reward = DualFlywheelRewardComposer(alpha=0.8)

    miner_labels = {1: "good", 2: "medium", 3: "random"}
    latest_rows: list[dict] = []
    round_summaries: list[dict] = []
    for round_idx in range(args.rounds):
        annotations_by_uid = {
            uid: _simulate_miner_annotations(
                corpus=corpus,
                gt_by_hash=gt_by_hash,
                quality=quality,
                seed=args.seed + round_idx * 101 + uid,
            )
            for uid, quality in miner_labels.items()
        }
        per_miner_scores = evaluate_round_annotations(
            corpus=corpus,
            annotations_by_uid=annotations_by_uid,
            fidelity_scorer=fidelity,
            consensus_scorer=consensus,
            hallucination_penalty=0.5,
        )
        winners = assembler.assemble(
            per_miner_scores=per_miner_scores,
            annotations_by_uid=annotations_by_uid,
            miner_hotkeys={uid: f"miner-{uid}" for uid in miner_labels},
            model_versions={uid: f"sim-{quality}" for uid, quality in miner_labels.items()},
            timestamps={uid: f"round-{round_idx}" for uid in miner_labels},
        )
        rewards, breakdowns = reward.compose(
            uids=list(miner_labels.keys()),
            annotation_scores=per_miner_scores,
            ledger=assembler.ledger,
            round_winners=winners,
        )
        export_uri = assembler.export(winners, round_id=f"round-{round_idx}")
        latest_rows = []
        if export_uri:
            export_path = Path(export_uri.removeprefix("file://"))
            latest_rows = [json.loads(line) for line in export_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        metrics = _evaluate_export(exported_rows=latest_rows, gt_by_hash=gt_by_hash)
        round_summaries.append(
            {
                "round": round_idx,
                "rewards": {uid: float(reward) for uid, reward in zip(miner_labels.keys(), rewards)},
                "breakdowns": {
                    item.uid: {
                        "annotation_score": item.annotation_score,
                        "adoption_bonus": item.adoption_bonus,
                        "final_score": item.final_score,
                    }
                    for item in breakdowns
                },
                "metrics": metrics,
                "export_rows": len(latest_rows),
                "contains_golden": any(row.get("is_golden") for row in latest_rows),
            }
        )

    print(
        json.dumps(
            {
                "dataset_size": args.dataset_size,
                "golden_images": len(corpus.golden_images()),
                "pool_images": len(corpus.annotation_images()),
                "rounds": round_summaries,
                "export_root": str(export_root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
