# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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
import subprocess
import argparse
from pathlib import Path
import bittensor as bt
from .logging import setup_events_logger

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


_ENV_ARG_MAP = {
    "NETUID": "--netuid",
    "WALLET_NAME": "--wallet.name",
    "WALLET_HOTKEY": "--wallet.hotkey",
    "SUBTENSOR_NETWORK": "--subtensor.network",
    "SUBTENSOR_CHAIN_ENDPOINT": "--subtensor.chain_endpoint",
    "MINER_MODEL_BACKEND": "--miner.model_backend",
    "MINER_ANNOTATION_WORKSPACE": "--miner.annotation_workspace",
    "MINER_R2_PREFIX": "--miner.dual_flywheel_r2_prefix",
    "SELF_HOSTED_TRAIN_URL": "--miner.self_hosted_train_url",
    "SELF_HOSTED_INFER_URL": "--miner.self_hosted_infer_url",
    "SELF_HOSTED_API_KEY": "--miner.self_hosted_api_key",
    "SELF_HOSTED_POLL_INTERVAL_SECONDS": "--miner.self_hosted_poll_interval_seconds",
    "YOLO_MODEL_PATH": "--miner.yolo_pretrained_weights",
    "YOLO_EPOCHS": "--miner.yolo_epochs",
    "YOLO_IMGSZ": "--miner.yolo_imgsz",
    "YOLO_BATCH": "--miner.yolo_batch",
    "OPENAI_API_KEY": "--miner.openai_api_key",
    "OPENAI_BASE_URL": "--miner.openai_base_url",
    "OPENAI_BASE_MODEL": "--miner.openai_base_model",
    "OPENAI_N_EPOCHS": "--miner.openai_n_epochs",
    "OPENAI_BATCH_SIZE": "--miner.openai_batch_size",
    "OPENAI_LEARNING_RATE_MULTIPLIER": "--miner.openai_learning_rate_multiplier",
    # Validator dataset configuration
    "VALIDATOR_GOLDEN_DATASET": "--neuron.flywheel_golden_dataset_id",
    "VALIDATOR_GOLDEN_SPLIT": "--neuron.flywheel_golden_split",
    "VALIDATOR_GOLDEN_RATIO": "--neuron.flywheel_golden_ratio",
    "VALIDATOR_GOLDEN_SPLIT_SEED": "--neuron.flywheel_golden_split_seed",
    "VALIDATOR_ANNOTATION_DATASET": "--neuron.flywheel_annotation_dataset_ids",
    "VALIDATOR_ANNOTATION_SPLIT": "--neuron.flywheel_annotation_split",
    "VALIDATOR_ANNOTATION_MAX_PER_DATASET": "--neuron.flywheel_annotation_max_per_dataset",
    "VALIDATOR_COCO_MANIFEST": "--neuron.flywheel_coco_manifest",
    # Climate MRV dataset
    "GEE_PROJECT": "--neuron.climate_mrv_gee_project",
    "CLIMATE_MRV_N_RAW_CHIPS": "--neuron.climate_mrv_n_raw_chips",
    "CLIMATE_MRV_N_GOLDEN_CHIPS": "--neuron.climate_mrv_n_golden_chips",
    "CLIMATE_MRV_FALLBACK_DIR": "--neuron.climate_mrv_fallback_dir",
    "CLIMATE_MRV_FALLBACK_GOLDEN_MANIFEST": "--neuron.climate_mrv_fallback_golden_manifest",
    # Validator infra
    "VALIDATOR_IMAGE_CACHE_ROOT": "--neuron.flywheel_image_cache_root",
    "VALIDATOR_IMAGE_SERVING_BASE_URL": "--neuron.flywheel_image_serving_base_url",
    "VALIDATOR_REQUEST_SIZE": "--neuron.flywheel_annotation_request_size",
    "VALIDATOR_GOLDEN_INJECTION_PER_REQUEST": "--neuron.flywheel_golden_injection_per_request",
    "VALIDATOR_COMMERCIAL_DATASET_PREFIX": "--neuron.flywheel_commercial_dataset_prefix",
    "VALIDATOR_COMMERCIAL_EXPORT_EVERY": "--neuron.flywheel_commercial_export_every",
    "VALIDATOR_FORWARD_STEP_SLEEP_SECONDS": "--neuron.forward_step_sleep_seconds",
    "VALIDATOR_SAMPLE_SIZE": "--neuron.sample_size",
    "VALIDATOR_TIMEOUT": "--neuron.timeout",
    "VALIDATOR_ANNOTATION_TIMEOUT": "--neuron.annotation_timeout",
    "VALIDATOR_NUM_CONCURRENT_FORWARDS": "--neuron.num_concurrent_forwards",
    "COMMERCIAL_DRAW_BOXES": "--neuron.flywheel_commercial_draw_boxes",
    "COMMERCIAL_ANNOTATED_IMAGE_PREFIX": "--neuron.flywheel_commercial_annotated_image_prefix",
}

_ENV_FLAG_MAP = {
    "TEST_MODE": "--test-mode",
    "MINER_SKIP_TRAINING": "--miner.skip_training",
    "MINER_FORCE_RETRAIN": "--miner.force_retrain",
    "VALIDATOR_DISABLE_SET_WEIGHTS": "--neuron.disable_set_weights",
    "VALIDATOR_AXON_OFF": "--neuron.axon_off",
}


def _truthy_env(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _load_env_file() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _argv_has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def _argv_with_env_defaults(argv: list[str]) -> list[str]:
    out = list(argv)
    for env_name, option in _ENV_ARG_MAP.items():
        value = os.getenv(env_name, "").strip()
        if value and not _argv_has_option(out, option):
            out.extend([option, value])
    for env_name, option in _ENV_FLAG_MAP.items():
        value = os.getenv(env_name, "").strip()
        if value and _truthy_env(value) and not _argv_has_option(out, option):
            out.append(option)
    return out


def is_cuda_available():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.STDOUT
        )
        if "NVIDIA" in output.decode("utf-8"):
            return "cuda"
    except Exception:
        pass
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        if "release" in output:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    print("full path:", full_path)
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        events_logger = setup_events_logger(
            config.neuron.full_path, config.neuron.events_retention_size
        )
        bt.logging.register_primary_logger(events_logger.name)


def add_args(cls, parser):
    """
    Adds relevant arguments to the parser for operation.
    """

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default=is_cuda_available(),
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=100,
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use in-memory mock wallet/subtensor/metagraph/dendrite/axon for tests only.",
        default=False,
    )

    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default=2 * 1024 * 1024 * 1024,  # 2 GB
    )

    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )

    parser.add_argument(
        "--wandb.off",
        action="store_true",
        help="Turn off wandb.",
        default=False,
    )

    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        help="Runs wandb in offline mode.",
        default=False,
    )

    parser.add_argument(
        "--wandb.notes",
        type=str,
        help="Notes to add to the wandb run.",
        default="",
    )


def add_miner_args(cls, parser):
    """Add miner specific arguments to the parser."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="miner",
    )

    parser.add_argument(
        "--blacklist.force_validator_permit",
        action="store_true",
        help="If set, we will force incoming requests to have a permit.",
        default=False,
    )

    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_true",
        help="If set, miners will accept queries from non registered entities. (Dangerous!)",
        default=False,
    )

    parser.add_argument(
        "--miner.annotation_workspace",
        type=str,
        help="Writable directory for miner annotation scratch files.",
        default=str(Path.cwd() / "artifacts" / "miner_annotation"),
    )

    # --- Legacy annotation backend (preserved for backward compatibility) ---
    parser.add_argument(
        "--miner.annotation_backend",
        type=str,
        choices=["yolo"],
        help=(
            "Legacy annotation backend: yolo (YOLO-only detection). Synthetic backends were removed "
            "from production paths."
        ),
        default="yolo",
    )
    parser.add_argument(
        "--miner.detector_checkpoint",
        type=str,
        help="Local YOLO weights path used for Stage-1 detection in the annotation pipeline.",
        default="yolov8s.pt",
    )

    parser.add_argument(
        "--miner.dual_flywheel_r2_prefix",
        type=str,
        help="R2 key prefix for annotation artifacts (per-task subfolders are appended).",
        default="miners/annotations",
    )

    # ===================================================================
    # Multi-backend training & inference arguments
    # ===================================================================

    parser.add_argument(
        "--miner.model_backend",
        type=str,
        choices=["yolo_local", "self_hosted", "openai_vision"],
        help=(
            "Model backend for training and inference: yolo_local (Ultralytics YOLO on GPU), "
            "self_hosted (external REST API), openai_vision (OpenAI fine-tuning)."
        ),
        default="",
    )

    # --- Dataset splitting ---
    parser.add_argument(
        "--miner.split_seed",
        type=int,
        help="Seed for deterministic hash-based dataset splitting.",
        default=42,
    )
    parser.add_argument(
        "--miner.train_split_pct",
        type=int,
        help="Percentage of images allocated to the training split (0-100).",
        default=70,
    )

    # --- Class taxonomy ---
    parser.add_argument(
        "--miner.class_taxonomy_path",
        type=str,
        help="Path to JSON list of valid hazard class strings.",
        default="",
    )

    # --- Training control ---
    parser.add_argument(
        "--miner.skip_training",
        action="store_true",
        help="Skip fine-tuning entirely; run inference with base/pretrained model.",
        default=False,
    )
    parser.add_argument(
        "--miner.force_retrain",
        action="store_true",
        help="Force retraining even if a cached model exists.",
        default=False,
    )
    parser.add_argument(
        "--miner.model_cache_dir",
        type=str,
        help="Directory for cached fine-tuned model checkpoints.",
        default="",
    )

    # --- YOLO local backend ---
    parser.add_argument(
        "--miner.yolo_pretrained_weights",
        type=str,
        help="Path to pretrained YOLO weights for yolo_local backend.",
        default="yolov8s.pt",
    )
    parser.add_argument("--miner.yolo_epochs", type=int, help="YOLO training epochs.", default=50)
    parser.add_argument("--miner.yolo_imgsz", type=int, help="YOLO image size.", default=640)
    parser.add_argument("--miner.yolo_batch", type=int, help="YOLO batch size.", default=16)
    parser.add_argument("--miner.yolo_lr0", type=float, help="YOLO initial learning rate.", default=0.01)
    parser.add_argument("--miner.yolo_lrf", type=float, help="YOLO final LR factor.", default=0.01)
    parser.add_argument("--miner.yolo_momentum", type=float, help="YOLO SGD momentum.", default=0.937)
    parser.add_argument("--miner.yolo_weight_decay", type=float, help="YOLO weight decay.", default=0.0005)
    parser.add_argument("--miner.yolo_warmup_epochs", type=float, help="YOLO warmup epochs.", default=3.0)
    parser.add_argument("--miner.yolo_optimizer", type=str, help="YOLO optimizer (auto, SGD, Adam, AdamW).", default="auto")
    parser.add_argument("--miner.yolo_augment", action="store_true", help="Enable YOLO augmentation.", default=True)
    parser.add_argument(
        "--miner.yolo_pseudo_label_conf",
        type=float,
        help="Confidence threshold for YOLO pseudo-labeling on unlabeled images.",
        default=0.5,
    )
    parser.add_argument(
        "--miner.seed_labels_path",
        type=str,
        help="Path to a directory or JSON file with seed labels for YOLO training.",
        default="",
    )

    # --- Self-hosted backend ---
    parser.add_argument(
        "--miner.self_hosted_train_url",
        type=str,
        help="URL for the self-hosted /train endpoint.",
        default="",
    )
    parser.add_argument(
        "--miner.self_hosted_infer_url",
        type=str,
        help="URL for the self-hosted /infer endpoint.",
        default="",
    )
    parser.add_argument(
        "--miner.self_hosted_api_key",
        type=str,
        help="Bearer token for self-hosted API authentication.",
        default="",
    )
    parser.add_argument(
        "--miner.self_hosted_poll_interval_seconds",
        type=int,
        help="Seconds between polls when waiting for self-hosted training to complete.",
        default=30,
    )

    # --- OpenAI Vision backend ---
    parser.add_argument(
        "--miner.openai_api_key",
        type=str,
        help="OpenAI API key for the openai_vision backend.",
        default="",
    )
    parser.add_argument(
        "--miner.openai_base_url",
        type=str,
        help="Optional OpenAI-compatible API base URL for integration tests or gateways.",
        default="",
    )
    parser.add_argument(
        "--miner.openai_base_model",
        type=str,
        help="OpenAI base model ID for fine-tuning.",
        default="gpt-4o-2024-08-06",
    )
    parser.add_argument("--miner.openai_n_epochs", type=int, help="OpenAI fine-tuning epochs.", default=3)
    parser.add_argument("--miner.openai_batch_size", type=int, help="OpenAI fine-tuning batch size.", default=1)
    parser.add_argument(
        "--miner.openai_learning_rate_multiplier",
        type=float,
        help="OpenAI fine-tuning learning rate multiplier.",
        default=1.8,
    )

    # --- Auto-research ---
    parser.add_argument(
        "--miner.enable_autoresearch",
        action="store_true",
        help="Enable Karpathy-style auto-research hyperparameter search loop.",
        default=False,
    )
    parser.add_argument(
        "--miner.autoresearch_config_path",
        type=str,
        help="Path to YAML/JSON file defining the hyperparameter search space.",
        default="",
    )
    parser.add_argument(
        "--miner.autoresearch_max_trials",
        type=int,
        help="Max number of trial configurations (0 = full Cartesian product).",
        default=0,
    )

    # --- W&B ---
    parser.add_argument(
        "--wandb.project_name",
        type=str,
        default="template-miners",
        help="Wandb project to log to.",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        default="opentensor-dev",
        help="Wandb entity to log to.",
    )


def add_validator_args(cls, parser):
    """Add validator specific arguments to the parser."""

    parser.add_argument(
        "--neuron.forward_step_sleep_seconds",
        type=float,
        help=(
            "Wall-clock seconds to sleep after each successful validation step (operator pacing; "
            "e.g. 300 for five-minute rounds)."
        ),
        default=0.0,
    )

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="validator",
    )

    parser.add_argument(
        "--neuron.timeout",
        type=float,
        help="The timeout for each forward call in seconds.",
        default=10,
    )

    parser.add_argument(
        "--neuron.annotation_timeout",
        type=float,
        help="Dendrite timeout for miner annotation tasks (0 = use neuron.timeout).",
        default=0.0,
    )

    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        help="The number of concurrent forwards running at any time.",
        default=1,
    )

    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        help="The number of miners to query in a single step.",
        default=50,
    )

    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disables setting weights.",
        default=False,
    )

    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Moving average alpha parameter, how much to add of the new observation.",
        default=0.1,
    )

    parser.add_argument(
        "--neuron.axon_off",
        "--axon_off",
        action="store_true",
        # Note: the validator needs to serve an Axon with their IP or they may
        #   be blacklisted by the firewall of serving peers on the network.
        help="Set this flag to not attempt to serve an Axon.",
        default=False,
    )

    parser.add_argument(
        "--neuron.vpermit_tao_limit",
        type=int,
        help="The maximum number of TAO allowed to query a validator with a vpermit.",
        default=4096,
    )

    parser.add_argument(
        "--neuron.scheduler_seed",
        type=int,
        help="Deterministic seed for annotation request sampling.",
        default=13,
    )

    parser.add_argument(
        "--neuron.incentive_temperature",
        type=float,
        help="Temperature for broad softmax incentive shaping.",
        default=0.25,
    )

    parser.add_argument(
        "--neuron.incentive_floor",
        type=float,
        help="Minimum nonzero share for eligible value-adding miners.",
        default=0.002,
    )

    parser.add_argument(
        "--neuron.incentive_min_score",
        type=float,
        help="Minimum EMA score required before a miner receives broad-softmax share.",
        default=0.05,
    )

    parser.add_argument(
        "--neuron.flywheel_image_cache_root",
        type=str,
        help=(
            "Local filesystem root where the validator materializes images for "
            "miner consumption. Each image is keyed by its content-hash image_id."
        ),
        default=str(Path(__file__).resolve().parents[2] / "data" / "flywheel" / "image_cache"),
    )
    parser.add_argument(
        "--neuron.flywheel_image_serving_base_url",
        type=str,
        help=(
            "Optional public base URL prefix for serving images to miners. When "
            "empty the validator emits file:// URIs (single-host / localnet)."
        ),
        default="",
    )
    parser.add_argument(
        "--neuron.flywheel_golden_dataset_id",
        type=str,
        help=(
            "Dataset source for the golden labeled set and annotation pool. "
            "Use 'climate_mrv' (default) for the Climate MRV Sentinel-2 + Hansen satellite "
            "imagery pipeline, or any HuggingFace dataset ID for the legacy path."
        ),
        default="climate_mrv",
    )
    parser.add_argument(
        "--neuron.flywheel_golden_split",
        type=str,
        help="Split within the golden dataset to load before applying the 30/70 partition.",
        default="train",
    )
    parser.add_argument(
        "--validator.golden_split_ratio",
        "--neuron.flywheel_golden_ratio",
        type=float,
        help="Fraction of the shared dataset reserved as the validator-only Golden Set.",
        default=0.1,
    )
    parser.add_argument(
        "--neuron.flywheel_golden_split_seed",
        type=int,
        help="Seed used to deterministically split the shared dataset into Golden vs annotation pools.",
        default=20260601,
    )
    parser.add_argument(
        "--neuron.climate_mrv_gee_project",
        type=str,
        help="Google Earth Engine cloud project ID (optional; leave empty for personal auth).",
        default="",
    )
    parser.add_argument(
        "--neuron.climate_mrv_n_raw_chips",
        type=int,
        help="Number of Sentinel-2 annotation-pool chips to sample from GEE per run.",
        default=200,
    )
    parser.add_argument(
        "--neuron.climate_mrv_n_golden_chips",
        type=int,
        help="Number of Hansen/ESA golden chips to export per run (validator-only).",
        default=60,
    )
    parser.add_argument(
        "--neuron.climate_mrv_fallback_dir",
        type=str,
        help=(
            "Directory containing pre-exported Climate MRV chip files for offline mode. "
            "Defaults to data/climate_mrv/samples/ in the repo root when empty."
        ),
        default="",
    )
    parser.add_argument(
        "--neuron.climate_mrv_fallback_golden_manifest",
        type=str,
        help="Path to golden_labels.json for fallback chips. Auto-detected when empty.",
        default="",
    )
    parser.add_argument(
        "--neuron.flywheel_annotation_dataset_ids",
        type=str,
        help=(
            "Optional comma-separated extra Hugging Face dataset ids for the annotation pool. "
            "By default the pool comes from the non-Golden portion of flywheel_golden_dataset_id."
        ),
        default="",
    )
    parser.add_argument(
        "--neuron.flywheel_annotation_split",
        type=str,
        help="Split name to load from each annotation dataset (use 'train' if unsure).",
        default="train",
    )
    parser.add_argument(
        "--neuron.flywheel_annotation_max_per_dataset",
        type=int,
        help="Per-dataset cap for the unlabeled annotation pool. Use 0 for no cap.",
        default=512,
    )
    parser.add_argument(
        "--neuron.flywheel_hf_revision",
        type=str,
        help=(
            "Hugging Face git revision for Hub datasets (use refs/convert/parquet when the "
            "repo still ships a legacy dataset script). Empty = Hub default branch loader."
        ),
        default="refs/convert/parquet",
    )
    parser.add_argument(
        "--neuron.flywheel_annotation_image_jitter_ms",
        type=int,
        help=(
            "Upper bound (uniform 0..N ms) for async sleep between camouflaged annotation "
            "images when building full-dataset miner requests; homogenizes per-image latency."
        ),
        default=40,
    )
    parser.add_argument(
        "--neuron.flywheel_annotation_request_size",
        type=int,
        help=(
            "Number of images to include in each miner annotation request. "
            "Use 0 to send the full corpus for the round."
        ),
        default=0,
    )
    parser.add_argument(
        "--neuron.flywheel_golden_injection_per_request",
        type=int,
        help=(
            "How many of the request images come from the hidden Golden Set. "
            "Ignored when flywheel_annotation_request_size is 0."
        ),
        default=0,
    )
    parser.add_argument(
        "--neuron.flywheel_alpha_annotation",
        type=float,
        help=(
            "Weight on annotation fidelity/consensus in the final on-chain score; "
            "adoption bonus receives (1 - alpha)."
        ),
        default=0.7,
    )
    parser.add_argument(
        "--neuron.flywheel_hallucination_penalty",
        type=float,
        help="Multiplicative penalty applied per hallucinated annotation on a Golden image.",
        default=0.5,
    )
    parser.add_argument(
        "--neuron.flywheel_golden_missing_penalty",
        type=float,
        help=(
            "Multiplicative penalty per Golden image_id the miner failed to annotate "
            "when that image was in the round task."
        ),
        default=0.5,
    )
    parser.add_argument(
        "--neuron.flywheel_coco_manifest",
        type=str,
        help=(
            "Path to COCO localnet manifest.json (from scripts/localnet/prepare_coco_val2017_subset.py). "
            "When set, HuggingFace corpus loading is skipped."
        ),
        default="",
    )
    parser.add_argument(
        "--neuron.flywheel_commercial_dataset_prefix",
        type=str,
        help=(
            "Object-storage prefix (file://, r2://, or s3://) where the assembled per-image_id "
            "winning annotations are exported as the subnet's commercial dataset."
        ),
        default=str(
            (Path(__file__).resolve().parents[2] / "artifacts" / "commercial_dataset").as_uri()
        ),
    )
    parser.add_argument(
        "--neuron.flywheel_commercial_export_every",
        type=int,
        help="Number of validator forward steps between commercial dataset exports.",
        default=10,
    )
    parser.add_argument(
        "--commercial-draw-boxes",
        "--neuron.flywheel_commercial_draw_boxes",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        help="Draw bounding boxes and class labels onto exported commercial images.",
        default=True,
    )
    parser.add_argument(
        "--commercial-annotated-image-prefix",
        "--neuron.flywheel_commercial_annotated_image_prefix",
        type=str,
        help="Prefix under which annotated images are uploaded in the commercial export.",
        default="commercial/annotated-images/",
    )
    parser.add_argument(
        "--wandb.project_name",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="template-validators",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="opentensor-dev",
    )


def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    _load_env_file()
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser)
    original_argv = sys.argv
    sys.argv = _argv_with_env_defaults(sys.argv)
    try:
        cfg = bt.config(parser)
    finally:
        sys.argv = original_argv
    cfg.mock = bool(getattr(cfg, "test_mode", False))
    return cfg
