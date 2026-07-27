from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from template.protocol import ImageAnnotationDocument, PerImageAnnotationItem


def annotate_image_detector_only(
    *,
    checkpoint: Path,
    image_bytes: bytes,
    image_id: str,
    image_url: str,
    model_version: str,
    miner_uid: str,
) -> ImageAnnotationDocument:
    """Run YOLO-only detector on the provided image and return annotations."""
    try:
        from PIL import Image
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "pillow and ultralytics are required to run YOLO annotations."
        ) from exc

    # Load YOLO model
    model = YOLO(str(checkpoint))

    # Read image
    img = Image.open(io.BytesIO(image_bytes))

    # Run inference
    results = model(img, verbose=False)

    annotations: List[PerImageAnnotationItem] = []
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get xyxy coordinates
                xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
                # Get class label/index
                cls_idx = int(box.cls[0].item())
                cls_name = model.names[cls_idx]

                annotations.append(
                    PerImageAnnotationItem(
                        hazard_class=cls_name,
                        bounding_box=xyxy,
                    )
                )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ImageAnnotationDocument(
        image_id=image_id,
        image_url=image_url,
        miner_uid=miner_uid,
        timestamp=ts,
        annotations=annotations,
        model_version=model_version,
    )
