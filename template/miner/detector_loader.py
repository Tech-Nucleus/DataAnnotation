"""
Dynamic and Isolated Module Loader for Autonomous Miners.

Provides:
1. Strict per-miner directory isolation (.agentic_state/miners/<hk_tag>/).
2. Dynamic hot-reloading of scoped pipeline modules via importlib.util.spec_from_file_location
   using file st_mtime_ns to ensure in-memory code updates on Tier 2 patches or rollbacks.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_BASE_DETECTOR = _ROOT_DIR / "template" / "miner" / "detector_annotate.py"
_BASE_GEOMETRY = _ROOT_DIR / "template" / "miner" / "geometry.py"

# In-memory mapping of hotkey/SS58/UID -> hk_tag
_MINER_TAG_REGISTRY: Dict[str, str] = {}
_MODULE_CACHE: Dict[str, Tuple[int, Any]] = {}  # tag -> (mtime_ns, module)


def get_miner_tag(wallet: Any = None, miner_uid: Optional[str] = None) -> str:
    """Resolve a clean, deterministic tag for a miner instance.
    
    Format: <wallet_name>_<ss58_prefix[:8]> or miner_<uid[:8]>.
    """
    if wallet is not None:
        w_name = getattr(wallet, "name", "miner") or "miner"
        hk = getattr(getattr(wallet, "hotkey", None), "ss58_address", "") or ""
        if hk:
            tag = f"{w_name}_{hk[:8]}"
            _MINER_TAG_REGISTRY[hk] = tag
            _MINER_TAG_REGISTRY[tag] = tag
            return tag
        return f"{w_name}_default"

    if miner_uid:
        # Check in-memory registry
        if miner_uid in _MINER_TAG_REGISTRY:
            return _MINER_TAG_REGISTRY[miner_uid]

        # Check persisted registry file
        reg_file = _ROOT_DIR / ".agentic_state" / "miners" / "active_miners.json"
        if reg_file.is_file():
            try:
                reg_data = json.loads(reg_file.read_text(encoding="utf-8"))
                if miner_uid in reg_data:
                    tag = reg_data[miner_uid]
                    _MINER_TAG_REGISTRY[miner_uid] = tag
                    return tag
            except Exception:
                pass

        # Check existing directories in .agentic_state/miners/
        miners_root = _ROOT_DIR / ".agentic_state" / "miners"
        if miners_root.is_dir():
            prefix = miner_uid[:8]
            for child in miners_root.iterdir():
                if child.is_dir():
                    if child.name == miner_uid or child.name.endswith(f"_{prefix}") or prefix in child.name:
                        _MINER_TAG_REGISTRY[miner_uid] = child.name
                        return child.name

        # If already formatted as a tag
        if "_" in miner_uid:
            return miner_uid
        return f"miner_{miner_uid[:8]}"

    return "miner_default"


def register_miner_tag(miner_identifier: str, tag: str) -> None:
    """Register mapping between an identifier (ss58 or name) and a tag."""
    _MINER_TAG_REGISTRY[miner_identifier] = tag
    reg_file = _ROOT_DIR / ".agentic_state" / "miners" / "active_miners.json"
    try:
        reg_file.parent.mkdir(parents=True, exist_ok=True)
        reg_data = {}
        if reg_file.is_file():
            reg_data = json.loads(reg_file.read_text(encoding="utf-8"))
        reg_data[miner_identifier] = tag
        reg_file.write_text(json.dumps(reg_data, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_miner_state_dir(wallet: Any = None, miner_uid: Optional[str] = None) -> Path:
    """Get isolated state directory for a miner: .agentic_state/miners/<hk_tag>/."""
    tag = get_miner_tag(wallet=wallet, miner_uid=miner_uid)
    state_dir = _ROOT_DIR / ".agentic_state" / "miners" / tag
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def get_miner_pipeline_dir(wallet: Any = None, miner_uid: Optional[str] = None) -> Path:
    """Get isolated pipeline directory containing copies of detector_annotate.py and geometry.py."""
    state_dir = get_miner_state_dir(wallet=wallet, miner_uid=miner_uid)
    pipeline_dir = state_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    # Initialize isolated copies if not present
    target_detector = pipeline_dir / "detector_annotate.py"
    if not target_detector.is_file() and _BASE_DETECTOR.is_file():
        shutil.copy2(_BASE_DETECTOR, target_detector)

    target_geometry = pipeline_dir / "geometry.py"
    if not target_geometry.is_file() and _BASE_GEOMETRY.is_file():
        shutil.copy2(_BASE_GEOMETRY, target_geometry)

    return pipeline_dir


def load_scoped_detector_module(miner_uid: Optional[str] = None) -> Any:
    """Dynamically load or reload the miner's scoped detector_annotate.py module.
    
    Checks st_mtime_ns on the scoped file so that any Tier 2 code patch or rollback
    is immediately reflected in memory without process restart.
    """
    pipeline_dir = get_miner_pipeline_dir(miner_uid=miner_uid)
    target_detector = pipeline_dir / "detector_annotate.py"
    if not target_detector.is_file():
        target_detector = _BASE_DETECTOR

    tag = get_miner_tag(miner_uid=miner_uid)
    current_mtime = target_detector.stat().st_mtime_ns

    cached = _MODULE_CACHE.get(tag)
    if cached is not None and cached[0] == current_mtime:
        return cached[1]

    # Dynamically load scoped geometry first if present
    target_geometry = pipeline_dir / "geometry.py"
    geom_mod = None
    if target_geometry.is_file():
        geom_mod_name = f"template.miner.scoped_{tag}.geometry"
        geom_spec = importlib.util.spec_from_file_location(geom_mod_name, str(target_geometry))
        if geom_spec and geom_spec.loader:
            geom_mod = importlib.util.module_from_spec(geom_spec)
            sys.modules[geom_mod_name] = geom_mod
            geom_spec.loader.exec_module(geom_mod)

    # Dynamically load scoped detector
    det_mod_name = f"template.miner.scoped_{tag}.detector_annotate"
    spec = importlib.util.spec_from_file_location(det_mod_name, str(target_detector))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {target_detector}")

    mod = importlib.util.module_from_spec(spec)
    if geom_mod is not None:
        mod.__dict__["geometry"] = geom_mod
        if hasattr(geom_mod, "extract_canopy_geometry"):
            mod.__dict__["extract_canopy_geometry"] = geom_mod.extract_canopy_geometry

    sys.modules[det_mod_name] = mod
    spec.loader.exec_module(mod)

    _MODULE_CACHE[tag] = (current_mtime, mod)
    return mod


def load_scoped_detector_annotate(miner_uid: Optional[str] = None) -> Callable[..., Any]:
    """Retrieve the hot-reloaded annotate_image_detector_only function for this miner."""
    mod = load_scoped_detector_module(miner_uid=miner_uid)
    fn = getattr(mod, "annotate_image_detector_only", None)
    if fn is None:
        raise AttributeError(f"Module {mod} does not define annotate_image_detector_only")
    return fn
