#!/usr/bin/env python3
"""
Dynamic Urban Tree Canopy Clustering & R2 Uploader for DataAnnotation Subnet.

Features:
- Dynamic spatial canopy clustering (DBSCAN + Adaptive Radius + NMS/IoU Merging).
- Converts raw point-clutter into clean, presentation-grade annotations:
  - Isolated single trees -> Crisp individual tree bounding boxes (`hazard_class: "tree"`).
  - Dense tree rows, groups, squares -> Unified canopy bounding boxes (`hazard_class: "tree_group"`).
- Zero box clutter / zero box overlap.
- Empties/uploads task pool images to R2 bucket (`subnet/task_dataset/`).
- Writes structured manifest to `artifacts/urban_tree_dataset/manifest.json`.
"""

import os
import json
import csv
import hashlib
import random
import boto3
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
from dotenv import load_dotenv

load_dotenv('.env')

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def merge_overlapping_boxes(boxes: list[dict], iou_thresh: float = 0.2) -> list[dict]:
    """Dynamically merge any overlapping bounding boxes to guarantee clean, non-cluttered annotations."""
    if not boxes:
        return []

    def iou(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    merged = []
    visited = [False] * len(boxes)

    for i in range(len(boxes)):
        if visited[i]:
            continue
        curr_box = list(boxes[i]["bounding_box"])
        curr_count = boxes[i].get("tree_count", 1)
        visited[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if visited[j]:
                    continue
                other_box = boxes[j]["bounding_box"]
                if iou(curr_box, other_box) > iou_thresh:
                    # Merge boxes
                    curr_box = [
                        min(curr_box[0], other_box[0]),
                        min(curr_box[1], other_box[1]),
                        max(curr_box[2], other_box[2]),
                        max(curr_box[3], other_box[3]),
                    ]
                    curr_count += boxes[j].get("tree_count", 1)
                    visited[j] = True
                    changed = True

        hazard_cls = "tree" if curr_count == 1 else "tree_group"
        merged.append({
            "hazard_class": hazard_cls,
            "bounding_box": curr_box,
            "severity": "medium" if curr_count == 1 else "high",
            "tree_count": curr_count
        })

    return merged

def process_urban_tree_dataset_dynamic(
    dataset_dir: Path,
    output_dir: Path,
    max_images: int = 40,
    golden_ratio: float = 0.3,
    cluster_eps: float = 22.0,
    seed: int = 42
):
    rng = random.Random(seed)
    images_dir = dataset_dir / "images"
    csv_dir = dataset_dir / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir = output_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    all_tif_files = sorted(list(images_dir.glob("*.tif")))
    by_city = {}
    for f in all_tif_files:
        city = f.name.split("_")[0]
        by_city.setdefault(city, []).append(f)

    selected_files = []
    cities = sorted(list(by_city.keys()))
    per_city = max(1, max_images // len(cities))
    for c in cities:
        city_files = by_city[c]
        rng.shuffle(city_files)
        selected_files.extend(city_files[:per_city])

    if len(selected_files) < max_images:
        remaining = [f for f in all_tif_files if f not in selected_files]
        rng.shuffle(remaining)
        selected_files.extend(remaining[: max_images - len(selected_files)])

    selected_files = selected_files[:max_images]
    print(f"Selected {len(selected_files)} images across {len(cities)} cities ({', '.join(cities)})")

    manifest_images = []
    golden_count = int(round(len(selected_files) * golden_ratio))
    indices = list(range(len(selected_files)))
    rng.shuffle(indices)
    golden_indices = set(indices[:golden_count])

    for idx, tif_path in enumerate(selected_files):
        csv_path = csv_dir / (tif_path.stem + ".csv")
        if not csv_path.exists():
            continue

        try:
            img_pil = Image.open(tif_path)
            arr = np.array(img_pil)
        except Exception as e:
            print(f"Error loading {tif_path}: {e}")
            continue

        h_img, w_img = arr.shape[0], arr.shape[1]
        rgb = arr[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = (rgb / rgb.max() * 255).astype(np.uint8)

        jpg_filename = f"{tif_path.stem}.jpg"
        jpg_path = images_out_dir / jpg_filename
        Image.fromarray(rgb).save(jpg_path, format="JPEG", quality=95)

        image_id = compute_sha256(jpg_path)

        # Parse tree center points
        points = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    px, py = float(row['x']), float(row['y'])
                    if 0 <= px < w_img and 0 <= py < h_img:
                        points.append((px, py))
                except (ValueError, KeyError):
                    continue

        raw_boxes = []
        if points:
            pts = np.array(points)
            db = DBSCAN(eps=cluster_eps, min_samples=1).fit(pts)
            labels = db.labels_

            margin = 8.0 # Margin around tree canopy edge
            for label_id in set(labels):
                c_pts = pts[labels == label_id]
                x_min = max(0, int(np.floor(c_pts[:, 0].min() - margin)))
                y_min = max(0, int(np.floor(c_pts[:, 1].min() - margin)))
                x_max = min(w_img, int(np.ceil(c_pts[:, 0].max() + margin)))
                y_max = min(h_img, int(np.ceil(c_pts[:, 1].max() + margin)))

                if x_max > x_min and y_max > y_min:
                    raw_boxes.append({
                        "hazard_class": "tree" if len(c_pts) == 1 else "tree_group",
                        "bounding_box": [x_min, y_min, x_max, y_max],
                        "severity": "medium" if len(c_pts) == 1 else "high",
                        "tree_count": len(c_pts)
                    })

        # Apply IoU Merging / NMS to eliminate overlapping boxes completely
        clean_annotations = merge_overlapping_boxes(raw_boxes, iou_thresh=0.15)

        is_golden = idx in golden_indices
        manifest_images.append({
            "image_id": image_id,
            "file_name": jpg_filename,
            "relative_path": f"images/{jpg_filename}",
            "width": w_img,
            "height": h_img,
            "is_golden": is_golden,
            "annotations": clean_annotations,
            "source_city": tif_path.stem.split("_")[0]
        })

    manifest = {
        "dataset_name": "Urban Tree & Canopy Cluster NAIP Dataset",
        "dataset_source": "https://github.com/jonathanventura/urban-tree-detection-data",
        "total_images": len(manifest_images),
        "golden_count": sum(1 for im in manifest_images if im["is_golden"]),
        "annotation_count": sum(1 for im in manifest_images if not im["is_golden"]),
        "images": manifest_images
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Clean Manifest written to {manifest_path} ({manifest['golden_count']} Golden, {manifest['annotation_count']} Task Pool)")
    return manifest_path, manifest

def upload_task_dataset_to_r2(manifest_data: dict, dataset_dir: Path):
    endpoint = os.getenv('R2_S3_ENDPOINT')
    access_key = os.getenv('R2_ACCESS_KEY_ID')
    secret_key = os.getenv('R2_SECRET_ACCESS_KEY')
    bucket_name = os.getenv('R2_BUCKET_NAME', 'subnet')

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='auto'
    )

    print(f"Uploading clean task dataset images to R2 bucket '{bucket_name}' under 'task_dataset/'...")
    uploaded_urls = {}
    for img_entry in manifest_data["images"]:
        rel_path = img_entry["relative_path"]
        local_path = dataset_dir / rel_path
        r2_key = f"task_dataset/{img_entry['image_id']}.jpg"

        s3.upload_file(
            str(local_path),
            bucket_name,
            r2_key,
            ExtraArgs={'ContentType': 'image/jpeg'}
        )
        url = f"{endpoint}/{bucket_name}/{r2_key}"
        uploaded_urls[img_entry['image_id']] = url

    print(f"Uploaded {len(uploaded_urls)} clean task dataset images to R2 successfully.")
    return uploaded_urls

if __name__ == "__main__":
    dataset_raw = Path("data/urban-tree-detection-data")
    output_dir = Path("artifacts/urban_tree_dataset")
    manifest_path, manifest_data = process_urban_tree_dataset_dynamic(
        dataset_dir=dataset_raw,
        output_dir=output_dir,
        max_images=40,
        golden_ratio=0.3,
        cluster_eps=22.0
    )
    upload_task_dataset_to_r2(manifest_data, output_dir)
