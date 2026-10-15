from __future__ import annotations

import re
from typing import Iterable


_client = None
_bucket = ""
_prefix = ""


def configure_object_storage(secrets=None) -> None:
    global _client, _bucket, _prefix
    secrets = secrets or {}
    bucket = str(secrets.get("S3_BUCKET", "")).strip()
    access_key = str(secrets.get("S3_ACCESS_KEY_ID", "")).strip()
    secret_key = str(secrets.get("S3_SECRET_ACCESS_KEY", "")).strip()
    endpoint = str(secrets.get("S3_ENDPOINT_URL", "")).strip() or None
    region = str(secrets.get("S3_REGION", "")).strip() or "us-east-1"
    addressing_style = str(secrets.get("S3_ADDRESSING_STYLE", "auto")).strip().lower()
    prefix = str(secrets.get("S3_PREFIX", "license-plate-system")).strip().strip("/")
    configured = (bucket, access_key, secret_key)
    if not any(configured):
        _client, _bucket, _prefix = None, "", ""
        return
    if not all(configured):
        raise ValueError(
            "Cần cấu hình đầy đủ S3_BUCKET, S3_ACCESS_KEY_ID và S3_SECRET_ACCESS_KEY."
        )
    if addressing_style not in {"auto", "path", "virtual"}:
        raise ValueError("S3_ADDRESSING_STYLE phải là auto, path hoặc virtual.")
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "Đã cấu hình object storage nhưng thiếu boto3. Cài dependencies từ requirements.txt."
        ) from exc
    _client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(s3={"addressing_style": addressing_style}),
    )
    _bucket, _prefix = bucket, prefix


def object_storage_enabled() -> bool:
    return _client is not None


def is_object_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def put_bytes(relative_key: str, content: bytes, content_type: str = "application/octet-stream") -> str:
    if _client is None:
        raise RuntimeError("Object storage chưa được cấu hình.")
    key = "/".join(part.strip("/") for part in (_prefix, relative_key) if part.strip("/"))
    _client.put_object(Bucket=_bucket, Key=key, Body=content, ContentType=content_type)
    return f"s3://{_bucket}/{key}"


def delete_uri(uri: str) -> None:
    _require_client()
    bucket, key = _parse_uri(uri)
    _client.delete_object(Bucket=bucket, Key=key)


def read_bytes(path: str) -> bytes:
    if is_object_uri(path):
        _require_client()
        bucket, key = _parse_uri(path)
        response = _client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()
    with open(path, "rb") as source:
        return source.read()


def exists(path: str) -> bool:
    if is_object_uri(path):
        _require_client()
        from botocore.exceptions import ClientError
        bucket, key = _parse_uri(path)
        try:
            _client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
    from pathlib import Path
    return Path(path).is_file()


def current_prefix_uri(category: str = "") -> str:
    key = "/".join(part for part in (_prefix, category.strip("/")) if part)
    return f"s3://{_bucket}/{key}" if _bucket else ""


def objects(category: str) -> Iterable[tuple[str, bytes]]:
    if _client is None:
        return ()
    prefix = "/".join(part for part in (_prefix, category.strip("/")) if part).rstrip("/") + "/"
    paginator = _client.get_paginator("list_objects_v2")

    def iterate():
        for page in paginator.paginate(Bucket=_bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                response = _client.get_object(Bucket=_bucket, Key=key)
                yield key[len(prefix):], response["Body"].read()
    return iterate()


def _parse_uri(uri: str) -> tuple[str, str]:
    match = re.fullmatch(r"s3://([^/]+)/(.+)", uri)
    if not match:
        raise ValueError("Đường dẫn object storage không hợp lệ.")
    return match.group(1), match.group(2)


def _require_client() -> None:
    if _client is None:
        raise RuntimeError(
            "Dữ liệu ảnh tham chiếu tới object storage nhưng S3 chưa được cấu hình."
        )
