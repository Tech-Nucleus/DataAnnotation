from types import SimpleNamespace

import pytest

from template.mock import MockDendrite, MockWallet
from template.protocol import AnnotationTask, UnlabeledAnnotationImage


@pytest.mark.asyncio
async def test_mock_dendrite_returns_annotation_task_response(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "template.miner.annotation.load_r2_credentials_from_env",
        lambda: SimpleNamespace(
            account_id="acc",
            bucket_name="bucket",
            s3_endpoint="https://example.invalid",
            access_key_id="key1",
            secret_access_key="secret1",
            token=None,
            public_bucket_url=None,
        ),
    )
    monkeypatch.setattr(
        "template.miner.annotation.upload_bytes_to_r2",
        lambda *a, **k: "r2://bucket/miners/annotations/task-1/annotations.json",
    )

    from template.protocol import ImageAnnotationDocument, PerImageAnnotationItem

    monkeypatch.setattr(
        "template.miner.detector_annotate.annotate_image_detector_only",
        lambda **kwargs: ImageAnnotationDocument(
            image_id=kwargs["image_id"],
            image_url=kwargs.get("image_url", "http://example.com/mock.png"),
            miner_uid=kwargs["miner_uid"],
            timestamp="2026-05-19T00:00:00Z",
            annotations=[
                PerImageAnnotationItem(
                    hazard_class="missing_hardhat",
                    bounding_box=[1.0, 2.0, 3.0, 4.0],
                )
            ],
            model_version=kwargs["model_version"],
        ),
    )

    wallet = MockWallet()
    dendrite = MockDendrite(wallet)
    synapse = AnnotationTask(
        task_id="task-1",
        challenge_nonce="nonce",
        annotation_images=[
            UnlabeledAnnotationImage(
                image_url=(tmp_path / "img.png").as_uri(),
                image_id="img-1",
            )
        ],
    )
    (tmp_path / "img.png").write_bytes(b"fake")
    axon = SimpleNamespace(uid=1, hotkey="miner-hotkey-1")
    response = (await dendrite([axon], synapse, timeout=5))[0]
    assert response.task_id == "task-1"
    assert response.annotations_uri.startswith("file://")
    assert response.error_message is None
