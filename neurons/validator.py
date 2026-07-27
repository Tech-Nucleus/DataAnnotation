import json
import os
import time
import random
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import template.compat.bittensor_commit_hotkey  # noqa: F401

import bittensor as bt
_orig_wallet = bt.wallet
class PasswordWallet(_orig_wallet):
    @property
    def coldkey(self):
        return self.get_coldkey(password="5121472")
    def unlock_coldkey(self):
        return self.get_coldkey(password="5121472")
bt.wallet = PasswordWallet
bt.subtensor.commit_reveal_enabled = lambda self, netuid, block=None: False
import numpy as np

from template.base.validator import BaseValidatorNeuron
from template.hazard.annotation_eval import AnnotationFidelityScorer, ConsensusScorer, _ReliabilityAccumulator
from template.hazard.dataset_assembler import AdoptionLedger, DatasetAssembler
from template.hazard.dual_reward import DualFlywheelBreakdown, DualFlywheelRewardComposer
from template.hazard.image_corpus import ImageCorpus, ImageCorpusConfig
from template.hazard.incentives import broad_softmax_scores
from template.hazard.r2_storage import load_r2_credentials_from_env
from template.protocol import R2AccessCredentials
from template.validator.dual_forward import dual_flywheel_forward


class Validator(BaseValidatorNeuron):
    """Annotation-only validator neuron."""

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        self.random = random.Random(self.config.neuron.scheduler_seed)

        self.image_corpus = ImageCorpus(self._build_corpus_config())
        self.fidelity_scorer = AnnotationFidelityScorer(
            hallucination_penalty=float(self.config.neuron.flywheel_hallucination_penalty),
        )
        self.consensus_scorer = ConsensusScorer()
        self.reliability = _ReliabilityAccumulator()
        # Retrieve draw_boxes and annotated_prefix config values robustly
        draw_boxes = self.config.get("commercial_draw_boxes")
        if draw_boxes is None:
            draw_boxes = getattr(self.config.neuron, "flywheel_commercial_draw_boxes", None)
        if draw_boxes is None:
            draw_boxes = True

        annotated_prefix = self.config.get("commercial_annotated_image_prefix")
        if annotated_prefix is None:
            annotated_prefix = getattr(self.config.neuron, "flywheel_commercial_annotated_image_prefix", None)
        if annotated_prefix is None:
            annotated_prefix = "commercial/annotated-images/"

        self.dataset_assembler = DatasetAssembler(
            corpus=self.image_corpus,
            storage_prefix=str(self.config.neuron.flywheel_commercial_dataset_prefix),
            draw_boxes=draw_boxes,
            annotated_prefix=str(annotated_prefix),
        )
        self.reward_composer = DualFlywheelRewardComposer(
            alpha=float(self.config.neuron.flywheel_alpha_annotation),
            hallucination_penalty_per_event=float(
                self.config.neuron.flywheel_hallucination_penalty
            ),
            golden_missing_penalty=float(
                getattr(self.config.neuron, "flywheel_golden_missing_penalty", 0.5)
            ),
        )

        self.annotation_scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.adoption_bonus_scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.last_commercial_dataset_uri: Optional[str] = None

        bt.logging.info("event=validator_init mode=annotation_only")
        self.load_state()

    async def forward(self):
        return await dual_flywheel_forward(self)

    def update_score_ledgers(self, breakdowns: List[DualFlywheelBreakdown], uids: List[int]) -> None:
        alpha = float(self.config.neuron.moving_average_alpha)
        for uid, item in zip(uids, breakdowns):
            self.annotation_scores[uid] = (
                alpha * item.annotation_score + (1.0 - alpha) * self.annotation_scores[uid]
            )
            self.adoption_bonus_scores[uid] = (
                alpha * item.adoption_bonus
                + (1.0 - alpha) * self.adoption_bonus_scores[uid]
            )

    def set_weights(self):
        raw_scores = self.scores.copy()
        self.scores = broad_softmax_scores(
            raw_scores,
            temperature=self.config.neuron.incentive_temperature,
            floor=self.config.neuron.incentive_floor,
            min_score=self.config.neuron.incentive_min_score,
        )
        try:
            endpoint = str(getattr(self.config.subtensor, "chain_endpoint", ""))
            force_local = os.environ.get("FORCE_LOCAL_SET_WEIGHTS", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if (
                not force_local
                and (
                    endpoint.startswith("ws://127.0.0.1")
                    or endpoint.startswith("ws://localhost")
                )
            ):
                bt.logging.warning(
                    f"Skipping on-chain set_weights in local dev endpoint {endpoint} due commit API mismatch. "
                    "Set FORCE_LOCAL_SET_WEIGHTS=1 to attempt commits anyway."
                )
                return
            super().set_weights()
        finally:
            self.scores = raw_scores

    def resync_metagraph(self):
        previous_size = len(self.scores)
        super().resync_metagraph()
        if len(self.scores) == previous_size:
            return
        new_size = len(self.scores)

        def _resize(array: np.ndarray) -> np.ndarray:
            new_arr = np.zeros(new_size, dtype=np.float32)
            copy_len = min(len(array), new_size)
            new_arr[:copy_len] = array[:copy_len]
            return new_arr

        self.annotation_scores = _resize(self.annotation_scores)
        self.adoption_bonus_scores = _resize(self.adoption_bonus_scores)

    def save_state(self):
        bt.logging.info("Saving validator state.")
        current_n = int(getattr(self.metagraph, "n", 0))
        annotation_scores = getattr(self, "annotation_scores", None)
        if annotation_scores is None:
            annotation_scores = np.zeros(current_n, dtype=np.float32)
        adoption_bonus_scores = getattr(self, "adoption_bonus_scores", None)
        if adoption_bonus_scores is None:
            adoption_bonus_scores = np.zeros(current_n, dtype=np.float32)
        last_commercial_dataset_uri = getattr(self, "last_commercial_dataset_uri", None)
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
            annotation_scores=annotation_scores,
            adoption_bonus_scores=adoption_bonus_scores,
            last_commercial_dataset_uri=last_commercial_dataset_uri or "",
        )
        if hasattr(self, "dataset_assembler"):
            ledger_path = Path(self.config.neuron.full_path) / "adoption_ledger.json"
            ledger_path.write_text(
                json.dumps(self.dataset_assembler.ledger.to_jsonable(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if hasattr(self, "reliability"):
            reliability_path = Path(self.config.neuron.full_path) / "reliability_state.json"
            reliability_path.write_text(
                json.dumps(self.reliability.to_jsonable(), indent=2, sort_keys=True),
                encoding="utf-8",
            )

    def load_state(self):
        bt.logging.info("Loading validator state.")
        state_path = Path(self.config.neuron.full_path) / "state.npz"
        if not state_path.exists():
            bt.logging.warning("No prior validator state found; starting fresh.")
            return
        state = np.load(state_path, allow_pickle=False)
        self.step = int(state["step"])
        current_n = int(self.metagraph.n)

        def _fit(arr: np.ndarray) -> np.ndarray:
            out = np.zeros(current_n, dtype=np.float32)
            copy_len = min(len(arr), current_n)
            out[:copy_len] = arr[:copy_len]
            return out

        def _fit_hotkeys(arr: np.ndarray) -> np.ndarray:
            out = np.array(list(self.metagraph.hotkeys))
            if len(arr) and len(out) == len(arr):
                return arr
            return out

        self.scores = _fit(state["scores"])
        self.hotkeys = _fit_hotkeys(state["hotkeys"])
        if "annotation_scores" in state:
            self.annotation_scores = _fit(state["annotation_scores"])
        if "adoption_bonus_scores" in state:
            self.adoption_bonus_scores = _fit(state["adoption_bonus_scores"])
        if "last_commercial_dataset_uri" in state:
            value = state["last_commercial_dataset_uri"].item()
            self.last_commercial_dataset_uri = value if value else None
        ledger_path = Path(self.config.neuron.full_path) / "adoption_ledger.json"
        if ledger_path.exists():
            payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.dataset_assembler.ledger = AdoptionLedger.from_jsonable(payload)
        reliability_path = Path(self.config.neuron.full_path) / "reliability_state.json"
        if reliability_path.exists():
            try:
                payload = json.loads(reliability_path.read_text(encoding="utf-8"))
                self.reliability = _ReliabilityAccumulator.from_jsonable(payload)
            except Exception as exc:
                bt.logging.warning(f"Failed to load reliability state: {exc}; starting fresh reliability.")
                self.reliability = _ReliabilityAccumulator()
        else:
            self.reliability = _ReliabilityAccumulator()

    def _build_corpus_config(self) -> ImageCorpusConfig:
        golden_ratio = getattr(self.config.neuron, "flywheel_golden_ratio", None)
        if golden_ratio is None:
            golden_ratio = getattr(self.config.validator, "golden_split_ratio", 0.1)

        # Climate MRV fields (only used when golden_dataset_id == 'climate_mrv')
        gee_project = str(
            getattr(self.config.neuron, "climate_mrv_gee_project", "") or ""
        )
        n_raw_chips = int(getattr(self.config.neuron, "climate_mrv_n_raw_chips", 200) or 200)
        n_golden_chips = int(getattr(self.config.neuron, "climate_mrv_n_golden_chips", 60) or 60)
        fallback_dir = str(
            getattr(self.config.neuron, "climate_mrv_fallback_dir", "") or ""
        )
        fallback_golden_manifest = str(
            getattr(self.config.neuron, "climate_mrv_fallback_golden_manifest", "") or ""
        )

        return ImageCorpusConfig(
            cache_root=Path(self.config.neuron.flywheel_image_cache_root),
            serving_base_url=str(self.config.neuron.flywheel_image_serving_base_url),
            golden_dataset_id=str(self.config.neuron.flywheel_golden_dataset_id),
            golden_split=str(self.config.neuron.flywheel_golden_split),
            golden_ratio=float(golden_ratio),
            golden_split_seed=int(self.config.neuron.flywheel_golden_split_seed),
            annotation_dataset_ids=str(self.config.neuron.flywheel_annotation_dataset_ids),
            annotation_split=str(self.config.neuron.flywheel_annotation_split),
            annotation_max_per_dataset=int(
                self.config.neuron.flywheel_annotation_max_per_dataset
            ),
            benchmark_max_samples=0,
            hf_revision=str(self.config.neuron.flywheel_hf_revision),
            coco_manifest_path=str(getattr(self.config.neuron, "flywheel_coco_manifest", "") or ""),
            # Climate MRV
            climate_mrv_gee_project=gee_project,
            climate_mrv_n_raw_chips=n_raw_chips,
            climate_mrv_n_golden_chips=n_golden_chips,
            climate_mrv_fallback_dir=fallback_dir,
            climate_mrv_fallback_golden_manifest=fallback_golden_manifest,
        )

    def _should_export_commercial(self, step: int) -> bool:
        every = max(1, int(self.config.neuron.flywheel_commercial_export_every))
        return step % every == 0

    def _load_commercial_credentials(self) -> Optional[R2AccessCredentials]:
        try:
            return load_r2_credentials_from_env()
        except Exception:
            return None


if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
