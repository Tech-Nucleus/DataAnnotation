"""Hash-based model cache manager.

The miner persists its fine-tuned model checkpoint alongside a hash of the
training pool.  On subsequent rounds, if the hash has not changed and no
force-retrain flag is set, the cached model is reused — saving enormous
compute for miners.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

import bittensor as bt


class ModelCache:
    """Manages cached model checkpoints keyed by training-pool hash."""

    def __init__(self, cache_root: Path):
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _version_dir(self, training_pool_hash: str) -> Path:
        return self.cache_root / training_pool_hash

    def has_cached_model(self, training_pool_hash: str) -> bool:
        """Return True if a valid cached model exists for this hash."""
        vdir = self._version_dir(training_pool_hash)
        meta_path = vdir / "cache_meta.json"
        return meta_path.is_file()

    def get_cached_model_version(self, training_pool_hash: str) -> Optional[str]:
        """Return the cached ``model_version`` string, or None."""
        meta_path = self._version_dir(training_pool_hash) / "cache_meta.json"
        if not meta_path.is_file():
            return None
        try:
            data = json.loads(meta_path.read_text())
            return data.get("model_version")
        except (json.JSONDecodeError, OSError):
            return None

    def get_cached_checkpoint_path(self, training_pool_hash: str) -> Optional[Path]:
        """Return the path to the cached checkpoint, or None."""
        meta_path = self._version_dir(training_pool_hash) / "cache_meta.json"
        if not meta_path.is_file():
            return None
        try:
            data = json.loads(meta_path.read_text())
            cp = data.get("checkpoint_path")
            if cp and Path(cp).is_file():
                return Path(cp)
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def get_cached_best_config(self, training_pool_hash: str) -> Optional[dict]:
        """Return the cached best hyper-parameter config (auto-research), or None."""
        best_path = self._version_dir(training_pool_hash) / "best_config.json"
        if not best_path.is_file():
            return None
        try:
            return json.loads(best_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def save(
        self,
        training_pool_hash: str,
        model_version: str,
        checkpoint_path: Optional[Path] = None,
        best_config: Optional[dict] = None,
    ) -> None:
        """Persist a trained model to the cache."""
        vdir = self._version_dir(training_pool_hash)
        vdir.mkdir(parents=True, exist_ok=True)

        meta = {
            "model_version": model_version,
            "training_pool_hash": training_pool_hash,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        }
        (vdir / "cache_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True)
        )

        if best_config:
            (vdir / "best_config.json").write_text(
                json.dumps(best_config, indent=2, sort_keys=True)
            )

        bt.logging.info(
            f"Model cache saved: hash={training_pool_hash[:16]}… "
            f"version={model_version[:16]}…"
        )

    def invalidate(self, training_pool_hash: str) -> None:
        """Remove a cached model version."""
        vdir = self._version_dir(training_pool_hash)
        if vdir.exists():
            shutil.rmtree(vdir, ignore_errors=True)
            bt.logging.info(
                f"Model cache invalidated: hash={training_pool_hash[:16]}…"
            )

    @staticmethod
    def compute_training_pool_hash(training_pool_data: list) -> str:
        """Compute a deterministic hash of training pool content.

        *training_pool_data* is a list of dicts (or Pydantic model dumps)
        representing ``LabeledTrainingImage`` entries.
        """
        canonical = json.dumps(
            sorted(training_pool_data, key=lambda x: x.get("image_id", "")),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
