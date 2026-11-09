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

import os
import sys
import typing

# Restrict worker threads per process to prevent OpenMP/PyTorch thread contention and SIGFPE
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

from dotenv import load_dotenv

load_dotenv()

import template.compat.bittensor_commit_hotkey  # noqa: F401
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

        if backend in ("segformer", "yolo_local", "self_hosted", "openai_vision", "qwen_vl"):
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

        # Check if agentic autopilot mode is enabled
        self.agentic_controller = None
        is_agentic = getattr(self.config, "agentic", False)
        if is_agentic:
            from template.miner.agentic_controller import MinerAgenticController
            from template.utils.agentic_watchdog import AgenticWatchdog
            from template.hazard.r2_storage import load_r2_credentials_from_env

            ag_cfg = getattr(self.config, "agentic", object())
            api_key = getattr(ag_cfg, "api_key", None) or os.getenv("GEMINI_API_KEY", "")
            sleep_sec = float(getattr(ag_cfg, "sleep_seconds", 300))
            hf_token = getattr(ag_cfg, "hf_token", None) or os.getenv("HF_TOKEN")
            allow_edits = bool(getattr(ag_cfg, "allow_code_edits", True))

            watchdog = AgenticWatchdog(
                role="miner",
                sleep_seconds=sleep_sec,
                api_key=api_key,
            )

            r2_client = None
            r2_bucket = os.getenv("R2_BUCKET_NAME", "annotation-subnet")
            r2_prefix = os.getenv("R2_LOCALNET_TELEMETRY_PREFIX", "localnet_sim/telemetry/")
            try:
                creds = load_r2_credentials_from_env()
                if creds:
                    import boto3
                    r2_client = boto3.client(
                        "s3",
                        endpoint_url=creds.s3_endpoint,
                        aws_access_key_id=creds.access_key_id,
                        aws_secret_access_key=creds.secret_access_key,
                        region_name="auto",
                    )
                    r2_bucket = creds.bucket_name
            except Exception as r2_err:
                bt.logging.warning(f"Could not initialize R2 client for miner agent: {r2_err}")

            self.agentic_controller = MinerAgenticController(
                wallet=self.wallet,
                metagraph=self.metagraph,
                watchdog=watchdog,
                r2_client=r2_client,
                r2_bucket=r2_bucket,
                r2_telemetry_prefix=r2_prefix,
                hf_token=hf_token,
                allow_code_edits=allow_edits,
                enabled=True,
            )
            self.agentic_controller.start_background()

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, "agentic_controller") and self.agentic_controller:
            self.agentic_controller.stop()
        super().__exit__(exc_type, exc_value, traceback)

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
        # Ingest validator feedback if piggybacked on synapse
        if hasattr(self, "agentic_controller") and self.agentic_controller is not None:
            fb_json = getattr(synapse, "last_round_feedback_json", None)
            if fb_json:
                try:
                    import json
                    fb_data = json.loads(fb_json)
                    self.agentic_controller.latest_received_feedback = fb_data
                    self.agentic_controller.watchdog.wake()
                    bt.logging.info("[AGENTIC MINER] Received piggybacked round feedback from validator!")
                except Exception as fb_err:
                    bt.logging.warning(f"[AGENTIC MINER] Failed to parse received feedback: {fb_err}")

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
        heartbeat_counter = 0
        while True:
            if hasattr(miner, "thread") and miner.thread and not miner.thread.is_alive():
                bt.logging.error("CRITICAL: Miner background thread died! Exiting process for automatic recovery.")
                sys.exit(1)
            heartbeat_counter += 1
            if heartbeat_counter % 12 == 0:
                bt.logging.info(f"Miner running... (step={getattr(miner, 'step', 0)})")
            import time
            time.sleep(5)
