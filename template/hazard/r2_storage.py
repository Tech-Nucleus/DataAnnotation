from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

from template.protocol import R2AccessCredentials


def _s3_client(creds: R2AccessCredentials):
    """Build an S3-compatible client for R2 (optional temporary session token)."""
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError as exc:  # pragma: no cover
        raise ImportError("boto3 is required for Cloudflare R2 access.") from exc
    kwargs = dict(
        endpoint_url=creds.s3_endpoint,
        aws_access_key_id=creds.access_key_id,
        aws_secret_access_key=creds.secret_access_key,
        region_name="auto",
        config=BotoConfig(signature_version="s3v4"),
    )
    if creds.token:
        kwargs["aws_session_token"] = creds.token
    # Static R2 API tokens must not pick up AWS_SESSION_TOKEN from the environment;
    # that often triggers R2 InvalidArgument on X-Amz-Security-Token.
    env_backup: dict[str, str] = {}
    if not creds.token:
        for key in ("AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN"):
            if key in os.environ:
                env_backup[key] = os.environ.pop(key)
    try:
        return boto3.client("s3", **kwargs)
    finally:
        os.environ.update(env_backup)


def r2_uri_bucket_key(uri: str) -> Tuple[str, str]:
    """Return ``(bucket, object_key)`` from an ``r2://bucket/key`` URI."""
    parsed = urlparse(uri)
    if parsed.scheme != "r2":
        raise ValueError(f"Expected r2:// URI, got {uri!r}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid r2 URI (missing bucket or key): {uri!r}")
    return bucket, key


def presigned_get_url_expires_seconds() -> int:
    raw = os.getenv("R2_PRESIGNED_EXPIRES_SECONDS", "7200").strip()
    return max(300, min(604800, int(raw)))


def generate_presigned_get_url(
    *,
    creds: R2AccessCredentials,
    bucket: str,
    object_key: str,
    expires_in: int | None = None,
) -> str:
    """
    Issue a short-lived HTTPS GET URL for Cloudflare R2 (S3-compatible).

    Prefer ``r2://`` URIs plus validator-side ``load_r2_credentials_from_env`` for
    subnet downloads; presigned URLs remain available for ad-hoc sharing.
    """
    ttl = int(expires_in if expires_in is not None else presigned_get_url_expires_seconds())
    if ttl < 300 or ttl > 604800:
        raise ValueError("expires_in must be between 300 and 604800 seconds.")
    client = _s3_client(creds)
    params = {"Bucket": bucket, "Key": object_key}
    url: str = client.generate_presigned_url(
        "get_object",
        Params=params,
        ExpiresIn=ttl,
    )
    return url


def presign_r2_object_uri(*, creds: R2AccessCredentials, r2_uri: str) -> str:
    """Build a presigned HTTPS GET URL for a single object behind ``r2_uri``."""
    bucket, key = r2_uri_bucket_key(r2_uri)
    return generate_presigned_get_url(creds=creds, bucket=bucket, object_key=key)


def upload_bytes_to_r2(
    data: bytes,
    *,
    object_key: str,
    creds: R2AccessCredentials,
    content_type: str = "application/octet-stream",
) -> str:
    client = _s3_client(creds)
    client.put_object(
        Bucket=creds.bucket_name,
        Key=object_key,
        Body=data,
        ContentType=content_type,
    )
    return f"r2://{creds.bucket_name}/{object_key}"


def upload_directory_to_r2(
    local_dir: Path,
    *,
    key_prefix: str,
    creds: R2AccessCredentials,
) -> str:
    """
    Upload every file under ``local_dir`` preserving relative paths beneath ``key_prefix``.

    ``key_prefix`` should use forward slashes and typically end with ``/``.
    """
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Expected directory: {local_dir}")
    prefix = key_prefix.rstrip("/") + "/"
    client = _s3_client(creds)
    for path in sorted(local_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(local_dir).as_posix()
        object_key = f"{prefix}{rel}"
        client.upload_file(str(path), creds.bucket_name, object_key)
    return f"r2://{creds.bucket_name}/{prefix}"


def load_r2_credentials_from_env() -> R2AccessCredentials:
    endpoint = (
        os.getenv("R2_ENDPOINT_URL", "").strip()
        or os.getenv("R2_S3_ENDPOINT", "").strip()
    )
    required = {
        "R2_ACCOUNT_ID": os.getenv("R2_ACCOUNT_ID", "").strip() or "r2",
        "R2_BUCKET_NAME": os.getenv("R2_BUCKET_NAME", "").strip(),
        "R2_ENDPOINT_URL": endpoint,
        "R2_ACCESS_KEY_ID": os.getenv("R2_ACCESS_KEY_ID", "").strip(),
        "R2_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing required R2 env vars for annotation storage: {', '.join(missing)}"
        )
    # Cloudflare R2 S3 API tokens are normally (access_key_id, secret_access_key) only.
    # A stray R2_TOKEN / STS-style value breaks ListObjects with InvalidArgument on
    # X-Amz-Security-Token. Opt in explicitly when you have real temporary credentials.
    raw_tok = os.getenv("R2_TOKEN")
    session_tok = (raw_tok or "").strip() or None
    use_sess = os.getenv("R2_USE_SESSION_TOKEN", "").strip().lower() in ("1", "true", "yes")
    if not use_sess:
        session_tok = None
    return R2AccessCredentials(
        account_id=required["R2_ACCOUNT_ID"],
        bucket_name=required["R2_BUCKET_NAME"],
        s3_endpoint=required["R2_ENDPOINT_URL"],
        access_key_id=required["R2_ACCESS_KEY_ID"],
        secret_access_key=required["R2_SECRET_ACCESS_KEY"],
        token=session_tok,
        public_bucket_url=os.getenv("R2_PUBLIC_BUCKET_URL"),
    )


def download_bytes_from_r2(uri: str, *, creds: R2AccessCredentials) -> bytes:
    """Download a single object from R2 via S3 API (validator / tests; no presigned GET)."""
    bucket, key = r2_uri_bucket_key(uri)
    client = _s3_client(creds)
    obj = client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"]
    try:
        return body.read()
    finally:
        body.close()


def upload_image_to_r2(
    image_path: Path,
    *,
    object_key: str,
    creds: R2AccessCredentials,
) -> str:
    """Upload a local image to R2 and return a reachable HTTP(S) URL.

    If ``creds.public_bucket_url`` is set (e.g.
    ``https://pub-xxx.r2.dev``), the URL is built directly from the
    public domain. Otherwise a presigned GET URL is issued.
    """
    content_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    ext = image_path.suffix.lower()
    ct = content_types.get(ext, "application/octet-stream")

    data = image_path.read_bytes()
    upload_bytes_to_r2(data, object_key=object_key, creds=creds, content_type=ct)

    # Build a reachable HTTP(S) URL
    public_base = (creds.public_bucket_url or "").strip().rstrip("/")
    if public_base:
        return f"{public_base}/{object_key}"

    # Fallback: issue a presigned GET URL
    return generate_presigned_get_url(
        creds=creds,
        bucket=creds.bucket_name,
        object_key=object_key,
        expires_in=presigned_get_url_expires_seconds(),
    )
