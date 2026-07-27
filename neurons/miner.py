# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer):TECHNOLOGY NUCLEUS
# Copyright © 2023 TECHNOLOGY NUCLEUS

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import typing

import template.compat.bittensor_commit_hotkey  # noqa: F401 — before bittensor: drand hotkey + subtensor rebind

import bittensor as bt

import template
from template.miner import AnnotationEngine

from template.base.miner import BaseMinerNeuron


class Miner(BaseMinerNeuron):
    """Annotation miner: supports legacy single-pass and multi-backend training modes."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # Select engine based on --miner.model_backend
        backend = str(
            getattr(getattr(self.config, "miner", object()), "model_backend", "") or ""
        ).strip()

        if backend in ("yolo_local", "self_hosted", "openai_vision"):
            from template.miner.training_engine import ModelTrainingAnnotationEngine

            self.annotation_engine = ModelTrainingAnnotationEngine(config=self.config)
            bt.logging.info(
                f"Miner using ModelTrainingAnnotationEngine with backend={backend}"
            )
        else:
            # Legacy path — original AnnotationEngine
            self.annotation_engine = AnnotationEngine(config=self.config)
            bt.logging.info("Miner using legacy AnnotationEngine")

        self.axon.attach(
            forward_fn=self.forward_annotation,
            blacklist_fn=self.blacklist_annotation,
            priority_fn=self.priority_annotation,
        )

    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        """Stub: annotation-only miner handles tasks via forward_annotation."""
        bt.logging.warning("Generic forward called — annotation miner expects AnnotationTask.")
        return synapse

    async def blacklist(self, synapse: bt.Synapse) -> typing.Tuple[bool, str]:
        """Stub for generic Synapse blacklist — only AnnotationTask accepted."""
        return True, "Use AnnotationTask protocol"

    async def priority(self, synapse: bt.Synapse) -> float:
        """Stub for generic Synapse priority."""
        return 0.0

    async def forward_annotation(
        self, synapse: template.protocol.AnnotationTask
    ) -> template.protocol.AnnotationTask:
        """Annotate unlabeled images and upload annotations.json."""
        try:
            synapse = self.annotation_engine.run(
                synapse, miner_hotkey=self.wallet.hotkey.ss58_address
            )
        except Exception as exc:
            bt.logging.error(f"Annotation task {synapse.task_id} failed: {exc}")
            synapse.error_message = str(exc)
        return synapse

    async def blacklist_annotation(
        self, synapse: template.protocol.AnnotationTask
    ) -> typing.Tuple[bool, str]:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        if self.config.blacklist.force_validator_permit:
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority_annotation(
        self, synapse: template.protocol.AnnotationTask
    ) -> float:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[caller_uid])
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# The main function parses the configuration and runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...")
            import time
            time.sleep(5)
