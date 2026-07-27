import hashlib
from typing import Any, Dict, List, Union

import bittensor as bt
from bittensor.subnets import SubnetsAPI

from template.protocol import AnnotationTask, UnlabeledAnnotationImage


class HazardAPI(SubnetsAPI):
    def __init__(self, wallet: "bt.wallet"):
        super().__init__(wallet)
        self.netuid = 33
        self.name = "hazard"

    def prepare_synapse(
        self,
        *,
        task_id: str,
        challenge_nonce: str,
        image_urls: List[str],
    ) -> AnnotationTask:
        return AnnotationTask(
            task_id=task_id,
            challenge_nonce=challenge_nonce,
            annotation_images=[
                UnlabeledAnnotationImage(
                    image_url=url,
                    image_id=hashlib.sha256(url.encode("utf-8")).hexdigest(),
                )
                for url in image_urls
            ],
        )

    def process_responses(
        self, responses: List[Union["bt.Synapse", Any]]
    ) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for response in responses:
            if response.dendrite.status_code != 200:
                continue
            outputs.append(
                {
                    "task_id": response.task_id,
                    "annotations_uri": response.annotations_uri,
                    "duration_ms": response.duration_ms,
                    "error_message": response.error_message,
                }
            )
        return outputs
