from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from template.hazard.r2_storage import upload_bytes_to_r2
from template.miner.annotation import (
    AnnotationEngine,
    build_synthetic_labeled_png,
    fetch_url_bytes,
)
from template.protocol import (
    AnnotationTask,
    AnnotationsFilePayload,
    ImageAnnotationDocument,
    PerImageAnnotationItem,
    R2AccessCredentials,
    UnlabeledAnnotationImage,
)


def _minimal_miner_config(tmp_path: Path) -> MagicMock:
    detector = tmp_path / "detector.pt"
    detector.write_bytes(b"fake-yolo-weights")
    m = MagicMock()
    m.annotation_workspace = str(tmp_path / "annotation_ws")
    m.annotation_backend = "yolo"
    m.dual_flywheel_r2_prefix = "miners/annotations"
    m.detector_checkpoint = str(detector)
    return m


def test_annotations_file_payload_roundtrip():
    payload = AnnotationsFilePayload(
        task_id="t1",
        records=[
            ImageAnnotationDocument(
                image_id="i1",
                image_url="http://example.com/i1.png",
                miner_uid="hk",
                timestamp="2026-05-01T12:00:00Z",
                annotations=[
                    PerImageAnnotationItem(
                        hazard_class="missing_fall_protection",
                        bounding_box=[245.0, 130.0, 412.0, 389.0],
                    )
                ],
                model_version="mv1" * 4,
            )
        ],
    )
    data = payload.model_dump()
    restored = AnnotationsFilePayload.model_validate(data)
    assert restored.records[0].annotations[0].hazard_class == "missing_fall_protection"


def test_annotation_engine_rejects_deterministic_config(tmp_path: Path):
    class _Cfg:
        miner = MagicMock()
        miner.annotation_workspace = str(tmp_path)
        miner.annotation_backend = "deterministic"

    with pytest.raises(ValueError, match="deterministic.*removed"):
        AnnotationEngine(config=_Cfg())


def test_annotation_engine_rejects_removed_simulation_backends(tmp_path: Path):
    class _Cfg:
        miner = _minimal_miner_config(tmp_path)

    _Cfg.miner.annotation_backend = "yolo_medium"
    with pytest.raises(ValueError, match="removed from production"):
        AnnotationEngine(config=_Cfg())


def test_annotation_engine_uploads_annotations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    creds = R2AccessCredentials(
        account_id="abcd1234",
        bucket_name="testbucket",
        s3_endpoint="https://00000000000000000000000000000000.r2.cloudflarestorage.com",
        access_key_id="accesskey",
        secret_access_key="secretaccesssecret",
    )
    monkeypatch.setattr(
        "template.miner.annotation.load_r2_credentials_from_env",
        lambda: creds,
    )
    monkeypatch.setattr(
        "template.miner.annotation.upload_bytes_to_r2",
        lambda *a, **k: "r2://testbucket/miners/annotations/task-x/annotations.json",
    )

    png = build_synthetic_labeled_png(64, 64)

    def fake_annotate_detector_only(**kwargs):
        from datetime import datetime, timezone

        return ImageAnnotationDocument(
            image_id=kwargs["image_id"],
            image_url=kwargs.get("image_url", "http://example.invalid/unl.png"),
            miner_uid=kwargs["miner_uid"],
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            annotations=[
                PerImageAnnotationItem(
                    hazard_class="missing_hardhat",
                    bounding_box=[1.0, 1.0, 20.0, 20.0],
                )
            ],
            model_version=kwargs["model_version"],
        )

    monkeypatch.setattr(
        "template.miner.detector_annotate.annotate_image_detector_only",
        fake_annotate_detector_only,
    )

    class _Cfg:
        miner = _minimal_miner_config(tmp_path)

    engine = AnnotationEngine(config=_Cfg())
    engine._fetch_image = lambda _url: png

    synapse = AnnotationTask(
        task_id="task-x",
        challenge_nonce="n1",
        annotation_images=[
            UnlabeledAnnotationImage(
                image_url="http://example.invalid/unl.png",
                image_id="cs_000142",
            )
        ],
    )
    out = engine.run(synapse, miner_hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty")
    assert out.error_message is None
    assert out.annotations_uri.startswith("r2://")
    assert out.miner_r2_credentials is not None
    assert out.duration_ms is not None


def test_annotation_engine_requires_annotation_images(tmp_path: Path):
    class _Cfg:
        miner = _minimal_miner_config(tmp_path)

    engine = AnnotationEngine(config=_Cfg())
    synapse = AnnotationTask(task_id="t", annotation_images=[])
    out = engine.run(synapse, miner_hotkey="hk")
    assert out.error_message is not None
    assert "annotation_images" in out.error_message


def test_fetch_file_url(tmp_path: Path):
    p = tmp_path / "x.png"
    p.write_bytes(build_synthetic_labeled_png())
    data = fetch_url_bytes(p.as_uri())
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_r2_upload_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    put_objects: list[tuple[str, bytes]] = []

    class FakeClient:
        def put_object(self, **kwargs):
            put_objects.append((kwargs["Key"], kwargs["Body"]))

    monkeypatch.setattr("boto3.client", lambda *a, **k: FakeClient())
    creds = R2AccessCredentials(
        account_id="abcd1234",
        bucket_name="bk",
        s3_endpoint="https://ex.r2.cloudflarestorage.com",
        access_key_id="keyid",
        secret_access_key="longsecretvalue",
    )
    uri = upload_bytes_to_r2(
        b'{"a":1}',
        object_key="pref/a.json",
        creds=creds,
        content_type="application/json",
    )
    assert uri == "r2://bk/pref/a.json"
    assert put_objects[0][0] == "pref/a.json"
