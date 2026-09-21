#!/usr/bin/env python3
"""
Fast parallel range downloader for large files (e.g. Hugging Face LFS / S3).
Splits file into concurrent byte chunks using HTTP Range requests.
"""

import sys
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def download_chunk(url, start_byte, end_byte, dest_path, chunk_idx, total_chunks):
    headers = {"Range": f"bytes={start_byte}-{end_byte}"}
    for attempt in range(5):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=30)
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code}")
            with open(dest_path, "r+b") as f:
                f.seek(start_byte)
                for block in r.iter_content(chunk_size=1024*1024):
                    if block:
                        f.write(block)
            return chunk_idx, end_byte - start_byte + 1
        except Exception as e:
            if attempt == 4:
                raise RuntimeError(f"Chunk {chunk_idx} failed after 5 attempts: {e}")
            time.sleep(1 + attempt)

def parallel_download(url: str, dest_path: str, num_workers: int = 16):
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Resolving {url} ...")
    session = requests.Session()
    r = session.head(url, allow_redirects=True, timeout=30)
    real_url = r.url
    total_size = int(r.headers.get("Content-Length", 0))
    if total_size == 0:
        raise RuntimeError("Could not determine content length")
    
    print(f"Total size: {total_size / (1024*1024):.2f} MB. Workers: {num_workers}")
    
    # Pre-allocate file
    with open(dest, "wb") as f:
        f.truncate(total_size)
        
    chunk_size = (total_size + num_workers - 1) // num_workers
    ranges = []
    for i in range(num_workers):
        start = i * chunk_size
        end = min(start + chunk_size - 1, total_size - 1)
        if start <= end:
            ranges.append((i, start, end))
            
    t0 = time.time()
    completed_bytes = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_chunk, real_url, start, end, str(dest), i, len(ranges)): (i, end - start + 1)
            for i, start, end in ranges
        }
        for future in as_completed(futures):
            idx, n_bytes = future.result()
            completed_bytes += n_bytes
            pct = 100.0 * completed_bytes / total_size
            elapsed = time.time() - t0
            speed_mb = (completed_bytes / (1024 * 1024)) / (elapsed + 1e-6)
            sys.stdout.write(f"\rProgress: {pct:5.1f}% ({completed_bytes/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB) | Speed: {speed_mb:.2f} MB/s | Elapsed: {elapsed:.1f}s")
            sys.stdout.flush()
            
    print(f"\nSuccessfully downloaded {dest} in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: parallel_download.py <url> <dest_path> [num_workers]")
        sys.exit(1)
    url = sys.argv[1]
    dest = sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    parallel_download(url, dest, workers)
