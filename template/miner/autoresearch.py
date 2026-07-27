"""Karpathy-style auto-research hyperparameter search loop.

When enabled (``--miner.enable_autoresearch``), the miner runs multiple
training passes with different hyperparameter configurations and selects
the best one based on a hold-out validation metric.

The loop is backend-agnostic — it only calls ``backend.train()`` and
``backend.infer()`` — so it works identically for ``yolo_local``,
``self_hosted``, and ``openai_vision``.
"""

from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bittensor as bt

from template.miner.backends.base import (
    BaseModelBackend,
    InferImage,
    TrainImage,
    TrainResult,
)
from template.miner.dataset_splitter import split_dataset_three_way
from template.protocol import PerImageAnnotationItem


def load_search_space(config_path: str) -> Dict[str, list]:
    """Load hyperparameter search space from YAML or JSON file.

    Expected format::

        search_space:
          epochs: [10, 20, 50]
          lr0: [0.001, 0.01]
          batch: [8, 16]
    """
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Auto-research config not found: {path}")

    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    space = data.get("search_space", data)
    if not isinstance(space, dict):
        raise ValueError("search_space must be a dict of param_name → list_of_values")
    return space


def _generate_configs(
    search_space: Dict[str, list],
    max_trials: int = 0,
    seed: int = 42,
) -> List[Dict]:
    """Generate hyperparameter configurations from the search space.

    If *max_trials* is 0, returns the full Cartesian product.
    Otherwise, returns a random sample of *max_trials* configurations.
    """
    keys = sorted(search_space.keys())
    values = [search_space[k] for k in keys]
    all_configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    if max_trials > 0 and max_trials < len(all_configs):
        rng = random.Random(seed)
        all_configs = rng.sample(all_configs, max_trials)

    return all_configs


def _score_validation_annotations(
    val_annotations: Dict[str, List[PerImageAnnotationItem]],
) -> float:
    """Score validation annotations using self-consistency.

    Since there is no ground truth for the validation split, we use a
    proxy metric: the total number of detections normalized by image
    count, penalizing images with zero detections.  This is a simple
    heuristic — in production, miners with access to private labels
    would implement a better scorer.

    A more sophisticated version would compare across trials using
    inter-model agreement, but for the single-trial case we use this
    simpler metric.
    """
    if not val_annotations:
        return 0.0

    total_detections = 0
    images_with_detections = 0
    for image_id, anns in val_annotations.items():
        if anns:
            total_detections += len(anns)
            images_with_detections += 1

    coverage = images_with_detections / max(1, len(val_annotations))
    avg_detections = total_detections / max(1, len(val_annotations))

    # Favor configs with good coverage and reasonable detection count
    # Penalize extremes (too few or too many detections)
    detection_score = min(1.0, avg_detections / 5.0)  # Normalize to ~5 detections/image
    return 0.6 * coverage + 0.4 * detection_score


def run_autoresearch(
    backend: BaseModelBackend,
    all_images: List[TrainImage],
    inference_images_out: List[InferImage],
    *,
    search_space: Dict[str, list],
    max_trials: int = 0,
    random_seed: int = 42,
    workspace: Path,
    train_pct: int = 60,
    val_pct: int = 20,
) -> Tuple[TrainResult, Dict]:
    """Run the auto-research hyperparameter search loop.

    Parameters
    ----------
    backend:
        The model backend to use for training and inference.
    all_images:
        All available images (train + val + inference) as TrainImage objects.
    inference_images_out:
        Output list — populated with the final inference-split images.
    search_space:
        Hyperparameter search space dict.
    max_trials:
        Maximum number of trial configurations to evaluate (0 = full grid).
    random_seed:
        Seed for the dataset split and trial sampling.
    workspace:
        Directory for logging and saving results.
    train_pct:
        Percentage for training candidates in the 3-way split.
    val_pct:
        Percentage for validation selection in the 3-way split.

    Returns
    -------
    tuple[TrainResult, dict]
        The best training result and the best hyperparameter config.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    log_path = workspace / "autoresearch.log"

    # 1. Split images three-way
    all_ids = [img.image_id for img in all_images]
    train_ids, val_ids, infer_ids = split_dataset_three_way(
        all_ids,
        train_pct=train_pct,
        val_pct=val_pct,
        random_seed=random_seed,
    )

    id_to_image = {img.image_id: img for img in all_images}
    train_candidates = [id_to_image[iid] for iid in train_ids if iid in id_to_image]
    val_select = [
        InferImage(image_id=iid, image_path=id_to_image[iid].image_path)
        for iid in val_ids
        if iid in id_to_image
    ]
    final_infer = [
        InferImage(image_id=iid, image_path=id_to_image[iid].image_path)
        for iid in infer_ids
        if iid in id_to_image
    ]

    bt.logging.info(
        f"AutoResearch: split — train={len(train_candidates)}, "
        f"val={len(val_select)}, infer={len(final_infer)}"
    )

    # 2. Generate configs
    configs = _generate_configs(search_space, max_trials=max_trials, seed=random_seed)
    bt.logging.info(f"AutoResearch: {len(configs)} trial configurations to evaluate")

    # 3. Run trials
    best_score = -1.0
    best_config: Dict = {}
    best_result: Optional[TrainResult] = None
    trial_log: List[Dict] = []

    for trial_idx, config in enumerate(configs):
        bt.logging.info(
            f"AutoResearch: trial {trial_idx + 1}/{len(configs)} — config={config}"
        )
        started = time.time()

        try:
            # Train on train_candidates
            result = backend.train(train_candidates, config)

            # Infer on val_select
            val_annotations = backend.infer(val_select, result.model_version)

            # Score
            score = _score_validation_annotations(val_annotations)

            elapsed = time.time() - started
            entry = {
                "trial": trial_idx + 1,
                "config": config,
                "score": score,
                "model_version": result.model_version,
                "metrics": result.metrics,
                "elapsed_seconds": round(elapsed, 1),
            }
            trial_log.append(entry)

            bt.logging.info(
                f"AutoResearch: trial {trial_idx + 1} — score={score:.4f}, "
                f"elapsed={elapsed:.1f}s"
            )

            if score > best_score:
                best_score = score
                best_config = config
                best_result = result
                bt.logging.info(
                    f"AutoResearch: new best! score={score:.4f}, config={config}"
                )

        except Exception as exc:
            bt.logging.error(
                f"AutoResearch: trial {trial_idx + 1} failed: {exc}"
            )
            trial_log.append({
                "trial": trial_idx + 1,
                "config": config,
                "error": str(exc),
            })

    # 4. Write trial log
    with open(log_path, "w") as f:
        for entry in trial_log:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
    bt.logging.info(f"AutoResearch: trial log written to {log_path}")

    # 5. Save best config
    best_config_path = workspace / "best_config.json"
    best_config_path.write_text(json.dumps(best_config, indent=2, sort_keys=True))
    bt.logging.info(
        f"AutoResearch: best config saved to {best_config_path} — score={best_score:.4f}"
    )

    # 6. Final training on combined train + val with best config
    bt.logging.info(
        f"AutoResearch: final training on {len(train_candidates) + len(val_select)} images "
        f"with best config={best_config}"
    )
    combined_train = list(train_candidates)
    for vi in val_select:
        if vi.image_id in id_to_image:
            combined_train.append(id_to_image[vi.image_id])

    final_result = backend.train(combined_train, best_config)

    # 7. Populate inference images output
    inference_images_out.clear()
    inference_images_out.extend(final_infer)

    return final_result, best_config
