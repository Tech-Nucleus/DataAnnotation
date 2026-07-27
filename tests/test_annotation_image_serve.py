from __future__ import annotations

import asyncio
import random
from pathlib import Path
from urllib.parse import urlparse

from template.hazard.annotation_image_serve import (
    build_camouflaged_annotation_images,
    public_url_for_local_path,
    reencode_strip_metadata,
)
from template.hazard.golden_injection import InjectionPlan


def test_reencode_produces_jpeg_without_crashing():
    from template.miner.annotation import build_synthetic_labeled_png

    raw = build_synthetic_labeled_png(32, 32)
    out = reencode_strip_metadata(raw, random.Random(0))
    assert out[:2] == b"\xff\xd8"


def test_public_url_file_vs_http():
    p = Path("/tmp/not_real_test.jpg")
    assert public_url_for_local_path(p, "").startswith("file://")
    assert public_url_for_local_path(p, "https://cdn.example/x/") == (
        "https://cdn.example/x/not_real_test.jpg"
    )


def test_build_camouflaged_annotation_images_opaque_names(tmp_path):
    from template.hazard.image_corpus import (
        GoldenAnnotation,
        GoldenImage,
        ImageCorpus,
        ImageCorpusConfig,
    )

    cache = tmp_path / "cache"
    corpus = ImageCorpus(ImageCorpusConfig(cache_root=cache))
    corpus._loaded = True
    g_bytes = __import__("io").BytesIO()
    from PIL import Image

    Image.new("RGB", (8, 8), (1, 2, 3)).save(g_bytes, format="PNG")
    gb = g_bytes.getvalue()
    import hashlib

    gid = hashlib.sha256(gb).hexdigest()
    gp = cache / f"{gid}.png"
    gp.write_bytes(gb)
    corpus._all_image_index[gid] = gp
    corpus._golden.append(
        GoldenImage(
            image_id=gid,
            image_path=gp,
            image_url=gp.as_uri(),
            width=8,
            height=8,
            annotations=(
                GoldenAnnotation(
                    hazard_class="x",
                    bounding_box=(0, 0, 1, 1),
                    severity="low",
                ),
            ),
        )
    )
    corpus._golden_index[gid] = corpus._golden[0]

    ub = __import__("io").BytesIO()
    Image.new("RGB", (8, 8), (4, 5, 6)).save(ub, format="PNG")
    ubytes = ub.getvalue()
    uid_img = hashlib.sha256(ubytes).hexdigest()
    up = cache / f"{uid_img}.png"
    up.write_bytes(ubytes)
    corpus._all_image_index[uid_img] = up

    plan = InjectionPlan(
        ordered_images=((gid, gp.as_uri()), (uid_img, up.as_uri())),
        golden_image_ids=(gid,),
        annotation_image_ids=(uid_img,),
    )
    ephemeral: list[Path] = []
    imgs = asyncio.run(
        build_camouflaged_annotation_images(
            corpus=corpus,
            plan=plan,
            cache_root=cache,
            step=1,
            uid=7,
            rng=random.Random(99),
            serving_base_url="",
            jitter_ms_max=0,
            ephemeral_paths=ephemeral,
        )
    )
    assert len(imgs) == 2
    assert imgs[0].image_id == gid
    for im in imgs:
        url = im.image_url
        assert gid not in url and uid_img not in url
        p = Path(urlparse(url).path)
        assert p.suffix == ".jpg"
        assert p.read_bytes()[:2] == b"\xff\xd8"
