"""Backend registry and factory function."""

from __future__ import annotations

import importlib
from typing import Dict, Type

from template.miner.backends.base import BaseModelBackend

# Lazy-loaded registry: backend name → fully-qualified class path.
# Each value is imported only when requested, so heavy deps (openai, etc.)
# are never loaded unless the miner actually selects that backend.
_BACKEND_REGISTRY: Dict[str, str] = {
    "yolo_local": "template.miner.backends.yolo_local.YoloLocalBackend",
    "self_hosted": "template.miner.backends.self_hosted.SelfHostedBackend",
    "openai_vision": "template.miner.backends.openai_vision.OpenAIVisionBackend",
}


def register_backend(name: str, class_path: str) -> None:
    """Register a custom backend so third-party plugins can extend the miner."""
    _BACKEND_REGISTRY[name] = class_path


def get_backend(name: str, config: object) -> BaseModelBackend:
    """Instantiate the named backend, passing the full miner config.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    class_path = _BACKEND_REGISTRY.get(name)
    if class_path is None:
        available = ", ".join(sorted(_BACKEND_REGISTRY))
        raise ValueError(
            f"Unknown model backend {name!r}. Available: {available}"
        )

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls: Type[BaseModelBackend] = getattr(module, class_name)
    return cls(config=config)
