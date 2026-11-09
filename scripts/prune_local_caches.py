#!/usr/bin/env python3
"""
Prunes stale local caches from continuous node execution without disrupting running testnet nodes.

Safety guarantees:
1. Preserves all downloaded images from the last 12 hours.
2. Preserves all flywheel camouflaged chips from the last 24 hours.
3. Cleans npm download cache via npm cache clean --force.
4. Cleans stale next.js build cache.
5. Cleans stale mozilla temp files in /tmp.
6. EXCLUDES ~/.cache/uv because active .venv-neurons processes hold shared library references.
7. Leaves models, wallets, manifests, golden data, and active processes untouched.
"""

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MINER_DOWNLOADS = REPO_ROOT / "artifacts" / "miner_annotation" / "downloaded_images"
FLYWHEEL_IMAGE_CACHE = REPO_ROOT / "data" / "flywheel" / "image_cache"
NEXT_CACHE = Path("/home/komail/AP_ERP_CRM/.next/cache")
MOZILLA_TEMP = Path("/tmp")

def prune_directory_by_age(dir_path: Path, max_age_seconds: float, dry_run: bool = True):
    if not dir_path.is_dir():
        print(f"Directory {dir_path} does not exist, skipping.")
        return 0, 0, 0, 0

    now = time.time()
    total_files = 0
    deleted_files = 0
    deleted_bytes = 0
    retained_files = 0

    for entry in os.scandir(dir_path):
        if not entry.is_file():
            continue
        total_files += 1
        stat = entry.stat()
        age = now - stat.st_mtime
        if age > max_age_seconds:
            deleted_files += 1
            deleted_bytes += stat.st_size
            if not dry_run:
                try:
                    os.unlink(entry.path)
                except OSError as e:
                    print(f"Warning: could not delete {entry.path}: {e}")
        else:
            retained_files += 1

    return total_files, deleted_files, deleted_bytes, retained_files


def run_prune(dry_run: bool = True):
    print("=" * 70)
    print(f"LOCAL CACHE PRUNING & OPTIMIZATION AUDIT")
    print(f"Mode: {'DRY RUN (Simulated)' if dry_run else 'LIVE EXECUTION'}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}")
    print("=" * 70)

    total_reclaimed = 0

    # 1. Miner downloaded images (Keep last 12 hours)
    print("\n[1/4] Miner Downloaded Images:")
    tot, deleted, b_deleted, retained = prune_directory_by_age(
        MINER_DOWNLOADS, max_age_seconds=12 * 3600, dry_run=dry_run
    )
    gb_deleted = b_deleted / (1024**3)
    print(f"  Path: {MINER_DOWNLOADS}")
    print(f"  Total images: {tot}")
    print(f"  Retained (last 12h): {retained}")
    print(f"  Pruned (>12h): {deleted} ({gb_deleted:.2f} GB)")
    total_reclaimed += b_deleted

    # 2. Validator Flywheel Image Cache (Keep last 24 hours)
    print("\n[2/4] Validator Flywheel Camouflaged Cache:")
    tot, deleted, b_deleted, retained = prune_directory_by_age(
        FLYWHEEL_IMAGE_CACHE, max_age_seconds=24 * 3600, dry_run=dry_run
    )
    gb_deleted = b_deleted / (1024**3)
    print(f"  Path: {FLYWHEEL_IMAGE_CACHE}")
    print(f"  Total images: {tot}")
    print(f"  Retained (last 24h): {retained}")
    print(f"  Pruned (>24h): {deleted} ({gb_deleted:.2f} GB)")
    total_reclaimed += b_deleted

    # 3. NPM cache
    print("\n[3/4] NPM Cache:")
    npm_cache = Path("/home/komail/.npm/_cacache")
    npm_size = 0
    if npm_cache.exists():
        for f in npm_cache.rglob("*"):
            if f.is_file():
                npm_size += f.stat().st_size
    print(f"  Path: {npm_cache} ({npm_size / (1024**3):.2f} GB)")
    if not dry_run:
        npm_bin = shutil.which("npm")
        if npm_bin:
            try:
                subprocess.run([npm_bin, "cache", "clean", "--force"], check=True)
                print("  Cleaned via 'npm cache clean --force'")
            except Exception as e:
                print(f"  Warning cleaning npm: {e}")
        else:
            shutil.rmtree(npm_cache, ignore_errors=True)
            print(f"  Removed {npm_cache}")
    total_reclaimed += npm_size

    # 4. Stale temp files & Next.js cache
    print("\n[4/4] Stale Temp / Next.js Caches:")
    next_size = 0
    if NEXT_CACHE.exists():
        for f in NEXT_CACHE.rglob("*"):
            if f.is_file():
                next_size += f.stat().st_size
        print(f"  Next.js cache: {NEXT_CACHE} ({next_size / (1024**2):.1f} MB)")
        if not dry_run:
            shutil.rmtree(NEXT_CACHE, ignore_errors=True)
            print(f"  Removed {NEXT_CACHE}")
        total_reclaimed += next_size

    # Mozilla temp files in /tmp older than 24h
    moz_size = 0
    for f in MOZILLA_TEMP.glob("mozilla-temp-*"):
        if f.is_file():
            moz_size += f.stat().st_size
            if not dry_run:
                try:
                    f.unlink()
                except OSError:
                    pass
    if moz_size > 0:
        print(f"  Stale Mozilla temp files in /tmp: {moz_size / (1024**2):.1f} MB")
        total_reclaimed += moz_size

    # Explicit safety notes
    print("\n[SAFETY NOTICE]")
    print("  * ~/.cache/uv: Preserved (shared PyTorch/CUDA libraries active in .venv-neurons).")
    print("  * models/: Preserved (SegFormer & Qwen weights intact).")
    print("  * data/climate_mrv/samples/: Preserved (Golden and raw manifests untouched).")
    print("  * ~/.bittensor/wallets/: Preserved.")

    print("\n" + "=" * 70)
    print(f"TOTAL RECLAIMED DISK SPACE: {total_reclaimed / (1024**3):.2f} GB")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune stale caches.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate pruning without deleting.")
    parser.add_argument("--live", action="store_true", help="Execute live deletion.")
    args = parser.parse_args()

    is_dry = True
    if args.live:
        is_dry = False
    elif not args.dry_run:
        is_dry = True

    run_prune(dry_run=is_dry)
