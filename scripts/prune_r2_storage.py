#!/usr/bin/env python3
"""Cloudflare R2 Bucket Storage Pruning & Optimization Utility.

Safely audits and prunes transient deadweight storage in the subnet R2 bucket:
1. Stale miner task submission annotations in `miners/annotations/` older than --retention-hours.
2. Orphan benchmark run artifacts in `benchmark-oam-tcd-*/`.

STRICTLY PRESERVES:
- All raw image datasets (`camouflaged/`, etc.)
- Active commercial datasets (`climate_mrv/`, etc.)
- Website article imagery (`articles/`)
- Active/recent miner submissions within the retention window.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from collections import defaultdict
from typing import List, Tuple

import boto3
from dotenv import load_dotenv

load_dotenv("/home/komail/DataAnnotation/.env")


def _get_s3_client():
    endpoint = os.getenv("R2_ENDPOINT_URL") or os.getenv("R2_S3_ENDPOINT")
    key = os.getenv("R2_ACCESS_KEY_ID")
    secret = os.getenv("R2_SECRET_ACCESS_KEY")
    if not (endpoint and key and secret):
        raise RuntimeError("Missing R2 credentials in environment (.env).")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )


def fmt_size(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:6.2f} {unit}"
        n /= 1024.0
    return f"{n:6.2f} PB"


def audit_and_prune(
    bucket_name: str,
    retention_hours: float = 2.0,
    dry_run: bool = True,
) -> Tuple[int, int]:
    client = _get_s3_client()
    paginator = client.get_paginator("list_objects_v2")

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=retention_hours)

    print("=" * 65)
    print(f"R2 BUCKET PRUNING & AUDIT: {bucket_name}")
    print(f"Mode: {'DRY RUN (no deletions)' if dry_run else 'LIVE EXECUTION'}")
    print(f"Miner annotation retention window: {retention_hours:.1f} hours (cutoff: {cutoff.isoformat()})")
    print("=" * 65)

    keys_to_delete: List[str] = []
    bytes_to_delete: int = 0
    preserved_bytes: int = 0
    preserved_count: int = 0

    category_delete_counts = defaultdict(int)
    category_delete_bytes = defaultdict(int)

    # 1. Scan miners/annotations/
    print("\n[1/3] Scanning 'miners/annotations/' for transient submissions...")
    for page in paginator.paginate(Bucket=bucket_name, Prefix="miners/annotations/"):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            sz = obj["Size"]
            lm = obj["LastModified"]
            if lm < cutoff:
                keys_to_delete.append(k)
                bytes_to_delete += sz
                category_delete_counts["miners/annotations (stale)"] += 1
                category_delete_bytes["miners/annotations (stale)"] += sz
            else:
                preserved_count += 1
                preserved_bytes += sz

    # 2. Scan orphan benchmarks
    print("[2/3] Scanning orphan benchmark runs ('benchmark-oam-tcd-')...")
    for page in paginator.paginate(Bucket=bucket_name, Prefix="benchmark-oam-tcd-"):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            sz = obj["Size"]
            keys_to_delete.append(k)
            bytes_to_delete += sz
            category_delete_counts["benchmark artifacts (orphan)"] += 1
            category_delete_bytes["benchmark artifacts (orphan)"] += sz

    print("\n--- PRUNING SUMMARY BY CATEGORY ---")
    for cat, cnt in category_delete_counts.items():
        print(f"  * {cat:<35}: {cnt:>6} objects | {fmt_size(category_delete_bytes[cat])}")

    print(f"\nTotal objects identified for pruning: {len(keys_to_delete):,}")
    print(f"Total space to be reclaimed:          {fmt_size(bytes_to_delete)} ({bytes_to_delete:,} bytes)")
    print(f"Active miner objects preserved (<{retention_hours}h): {preserved_count:,} ({fmt_size(preserved_bytes)})")

    if dry_run:
        print("\n[DRY RUN COMPLETE] No objects were deleted. Use --execute to proceed.")
        return 0, bytes_to_delete

    # Execute deletion in chunks of 1000
    print(f"\n[3/3] Executing deletion of {len(keys_to_delete)} objects in batches of 1,000...")
    deleted_total = 0
    for i in range(0, len(keys_to_delete), 1000):
        chunk = keys_to_delete[i : i + 1000]
        del_spec = [{"Key": k} for k in chunk]
        resp = client.delete_objects(
            Bucket=bucket_name,
            Delete={"Objects": del_spec, "Quiet": True},
        )
        errors = resp.get("Errors", [])
        if errors:
            print(f"  Warning: {len(errors)} errors in batch {i//1000 + 1}")
        deleted_batch = len(chunk) - len(errors)
        deleted_total += deleted_batch
        print(f"  Deleted batch {i//1000 + 1}: {deleted_batch} objects (running total: {deleted_total:,})")

    print("\n" + "=" * 65)
    print(f"PRUNING COMPLETED: Successfully deleted {deleted_total:,} objects.")
    print(f"Reclaimed Storage: {fmt_size(bytes_to_delete)}")
    print("=" * 65)
    return deleted_total, bytes_to_delete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune stale R2 bucket objects.")
    parser.add_argument(
        "--bucket",
        default=os.getenv("R2_BUCKET_NAME", "subnet"),
        help="Target R2 bucket name (default: from .env)",
    )
    parser.add_argument(
        "--retention-hours",
        type=float,
        default=2.0,
        help="Retention window in hours for miner annotations (default: 2.0)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute deletions (default: dry run)",
    )
    args = parser.parse_args()

    audit_and_prune(
        bucket_name=args.bucket,
        retention_hours=args.retention_hours,
        dry_run=not args.execute,
    )
