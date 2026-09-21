#!/usr/bin/env python3
"""
Dataset and Weight Fetcher for CanopyMRV SOTA Benchmark Harness.
Downloads official benchmark splits and weights, computing SHA-256 checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench.fetcher")


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_track_a(dest_dir: Path) -> dict:
    """Fetch Track A (OAM-TCD) test split and incumbent weights."""
    tcd_dir = dest_dir / "oam_tcd"
    tcd_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching OAM-TCD test split from huggingface: restor/tcd ...")
    test_parquet_path = hf_hub_download(
        repo_id="restor/tcd",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
        local_dir=str(tcd_dir),
    )
    test_p = Path(test_parquet_path)
    test_sha = compute_sha256(test_p)
    logger.info("Downloaded test split to %s (SHA-256: %s)", test_p, test_sha)

    # Download incumbent weights: restor/tcd-segformer-mit-b5
    weights_dir = dest_dir / "weights" / "tcd-segformer-mit-b5"
    logger.info("Fetching incumbent weights: restor/tcd-segformer-mit-b5 ...")
    snapshot_download(
        repo_id="restor/tcd-segformer-mit-b5",
        local_dir=str(weights_dir),
    )

    manifest = {
        "track": "A",
        "dataset": "restor/tcd",
        "split": "test",
        "file": str(test_p.relative_to(dest_dir.parent)),
        "size_bytes": test_p.stat().st_size,
        "sha256": test_sha,
        "incumbent_model": "restor/tcd-segformer-mit-b5",
        "weights_dir": str(weights_dir.relative_to(dest_dir.parent)),
    }

    manifest_path = tcd_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest saved to %s", manifest_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch benchmark datasets and weights")
    parser.add_argument("--track", choices=["A", "B", "C", "D", "all"], default="A")
    parser.add_argument("--dest", default="bench/data", help="Destination root directory")
    args = parser.parse_args()

    dest = Path(args.dest).resolve()
    if args.track in ("A", "all"):
        fetch_track_a(dest)
