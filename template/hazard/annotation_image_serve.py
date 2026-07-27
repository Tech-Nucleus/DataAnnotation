"""
Validator-side preparation of annotation-task images for miners.

Strips embedded metadata, re-encodes pixels to a normalized JPEG, and emits
opaque file URLs so miners cannot fingerprint Golden Set rows from filenames,
EXIF camera tags, or raw file structure. Optional per-image timing jitter
homogenizes wall-clock fetch latency across the batch.
"""

from __future__ import annotations

import asyncio
import io
import secrets
from pathlib import Path
from typing import List, Sequence

import bittensor as bt

from template.hazard.golden_injection import InjectionPlan
from template.hazard.image_corpus import ImageCorpus
from template.protocol import UnlabeledAnnotationImage


def public_url_for_local_path(local_path: Path, serving_base_url: str) -> str:
    """Return a miner-fetchable URL for a local file.

    When ``serving_base_url`` is empty, use ``file://`` (localnet). When set,
    the URL uses only the basename; the HTTP docroot must expose that file.
    """

    base = (serving_base_url or "").strip()
    if not base:
        return local_path.resolve().as_uri()
    if not base.endswith("/"):
        base = base + "/"
    return base + local_path.name


def reencode_strip_metadata(image_bytes: bytes, rng) -> bytes:
    """Decode image bytes, drop metadata, re-encode as baseline JPEG."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pillow is required for annotation image camouflage.") from exc

    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB")
    buf = io.BytesIO()
    quality = int(rng.randint(88, 93))
    rgb.save(
        buf,
        format="JPEG",
        quality=quality,
        optimize=True,
        subsampling=2,
    )
    return buf.getvalue()


async def build_camouflaged_annotation_images(
    *,
    corpus: ImageCorpus,
    plan: InjectionPlan,
    cache_root: Path,
    step: int,
    uid: int,
    rng,
    serving_base_url: str,
    jitter_ms_max: int,
    ephemeral_paths: List[Path],
) -> List[UnlabeledAnnotationImage]:
    """Materialize per-request annotation images with camouflaged bytes."""

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    out: List[UnlabeledAnnotationImage] = []
    jitter_ms_max = max(0, int(jitter_ms_max))

    creds = None
    try:
        from template.hazard.r2_storage import load_r2_credentials_from_env
        creds = load_r2_credentials_from_env()
    except Exception:
        pass

    for idx, (image_id, _legacy_url) in enumerate(plan.ordered_images):
        path = corpus.known_image_path(image_id)
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"Missing corpus file for image_id={image_id}; cannot serve annotation image."
            )
        raw = path.read_bytes()
        payload = reencode_strip_metadata(raw, rng)
        token = secrets.token_hex(8)
        dest = cache_root / f"ann_step{step}_uid{uid}_{idx}_{token}.jpg"
        dest.write_bytes(payload)
        ephemeral_paths.append(dest)

        url = None
        if creds is not None:
            try:
                from template.hazard.r2_storage import upload_image_to_r2
                object_key = f"camouflaged/ann_step{step}_uid{uid}_{idx}_{token}.jpg"
                url = upload_image_to_r2(dest, object_key=object_key, creds=creds)
                bt.logging.debug(f"Uploaded camouflaged task image to R2: {url}")
            except Exception as e:
                bt.logging.error(f"Failed to upload camouflaged image to R2: {e}")

        if not url:
            url = public_url_for_local_path(dest, serving_base_url)

        out.append(UnlabeledAnnotationImage(image_url=url, image_id=image_id))
        if jitter_ms_max > 0:
            await asyncio.sleep(rng.uniform(0.0, jitter_ms_max / 1000.0))

    bt.logging.debug(
        f"event=annotation_images_camouflaged step={step} uid={uid} count={len(out)}"
    )
    return out



def cleanup_ephemeral_annotation_files(paths: Sequence[Path]) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover
            bt.logging.warning(f"event=annotation_ephemeral_cleanup_failed path={p} err={exc}")
