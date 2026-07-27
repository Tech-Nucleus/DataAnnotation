#!/usr/bin/env python3
"""Download COCO val2017 (if needed) and build a 200-image localnet manifest (20 golden / 180 pool)."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        print(f"skip download (exists): {dest}")
        return
    print(f"downloading {url} -> {dest}")
    urlretrieve(url, dest)  # noqa: S310 — localnet prep only


def _ensure_coco(root: Path) -> tuple[Path, Path]:
    images_dir = root / "val2017"
    ann_file = root / "annotations" / "instances_val2017.json"
    if images_dir.is_dir() and ann_file.is_file():
        return images_dir, ann_file

    cache = root / "_downloads"
    cache.mkdir(parents=True, exist_ok=True)
    val_zip = cache / "val2017.zip"
    ann_zip = cache / "annotations_trainval2017.zip"
    _download(COCO_VAL_IMAGES_URL, val_zip)
    _download(COCO_ANNOTATIONS_URL, ann_zip)
    if not images_dir.is_dir():
        with zipfile.ZipFile(val_zip) as zf:
            zf.extractall(root)
    if not ann_file.is_file():
        with zipfile.ZipFile(ann_zip) as zf:
            zf.extractall(root)
    if not images_dir.is_dir() or not ann_file.is_file():
        raise RuntimeError("COCO extract failed; check zip integrity.")
    return images_dir, ann_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/localnet/coco200"),
        help="Isolated output root (not committed).",
    )
    parser.add_argument("--dataset-size", type=int, default=200)
    parser.add_argument("--golden-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--coco-root", type=Path, default=Path("artifacts/localnet/coco_upstream"))
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    images_dir, ann_file = _ensure_coco(args.coco_root.resolve())
    payload = json.loads(ann_file.read_text(encoding="utf-8"))
    categories = {
        int(c["id"]): str(c["name"]).strip().lower().replace(" ", "_")
        for c in payload["categories"]
    }
    images = [
        (int(im["id"]), str(im["file_name"]), int(im["width"]), int(im["height"]))
        for im in payload["images"]
    ]
    anns_by_image: dict[int, list[dict]] = {}
    for ann in payload["annotations"]:
        if ann.get("iscrowd"):
            continue
        bbox = ann.get("bbox") or []
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    eligible = [im for im in images if im[0] in anns_by_image]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    chosen = eligible[: args.dataset_size]
    if len(chosen) < args.dataset_size:
        raise SystemExit(
            f"Need {args.dataset_size} annotated images, found {len(chosen)} eligible."
        )

    golden_n = max(1, int(round(args.dataset_size * args.golden_ratio)))
    golden_ids = {im[0] for im in chosen[:golden_n]}

    manifest_images: list[dict] = []
    holdout_gt: dict[str, list[dict]] = {}

    for im_id, file_name, width, height in chosen:
        src = images_dir / file_name
        if not src.is_file():
            raise FileNotFoundError(src)
        image_hash = _sha256_file(src)
        dest_name = f"{image_hash[:16]}_{file_name}"
        dest = out_images / dest_name
        if not dest.is_file():
            shutil.copy2(src, dest)
        anns = []
        for ann in anns_by_image[im_id]:
            x, y, w, h = [float(v) for v in ann["bbox"]]
            cls = categories[int(ann["category_id"])]
            box = [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))]
            anns.append(
                {
                    "hazard_class": cls,
                    "bounding_box": box,
                }
            )
        is_golden = im_id in golden_ids
        row = {
            "image_id": image_hash,
            "relative_path": f"images/{dest_name}",
            "coco_image_id": im_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "is_golden": is_golden,
            "annotations": anns,
        }
        manifest_images.append(row)
        if not is_golden:
            holdout_gt[image_hash] = anns

    manifest = {
        "version": 1,
        "dataset_size": len(manifest_images),
        "golden_count": sum(1 for r in manifest_images if r["is_golden"]),
        "pool_count": sum(1 for r in manifest_images if not r["is_golden"]),
        "seed": args.seed,
        "golden_ratio": args.golden_ratio,
        "images": manifest_images,
        "holdout_ground_truth": holdout_gt,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "golden": manifest["golden_count"],
                "pool": manifest["pool_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
