"""HTTP(S) GET for validator-side downloads (R2/S3 presigned URLs, etc.)."""

from __future__ import annotations


def fetch_url_bytes(url: str, *, timeout: float = 120.0) -> bytes:
    """GET ``url`` and return body bytes. Uses ``requests`` for better S3/R2 presigned compatibility."""
    import requests

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content
