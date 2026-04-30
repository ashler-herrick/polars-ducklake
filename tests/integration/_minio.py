"""MinIO helper for integration tests.

Exposes a small client wrapper that:

* refuses to operate on any bucket other than the configured test bucket,
* scopes objects under a unique per-session prefix so parallel runs and
  leftover state from prior runs cannot cross-contaminate,
* cleans up its own prefix on teardown.

The point is to make it *impossible* for the integration suite to mutate
any bucket the developer cares about. Every destructive call goes
through :meth:`MinIOTestClient._guard_bucket`.

Configuration loads from ``tests/.env`` (project-local docker-compose
defaults), overridden by ``tests/.env.local`` and real env vars — see
:mod:`tests.integration._env`.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import boto3
import duckdb
from botocore.client import Config
from botocore.exceptions import ClientError, EndpointConnectionError

from tests.integration._env import env

# boto3 ships no type stubs, so the s3 client is just Any here. Adding
# boto3-stubs as a dev dep would let us tighten this, but for a small
# guarded helper the value isn't worth the install cost.
S3Client = Any


@dataclass(frozen=True)
class MinIOConfig:
    endpoint: str
    access_key: str
    secret_key: str
    region: str
    bucket: str

    @property
    def s3_uri_prefix(self) -> str:
        """``s3://<bucket>/`` — useful for scan_parquet test paths."""
        return f"s3://{self.bucket}/"


def load_config() -> MinIOConfig | None:
    """Return MinIO config, or None if any required key is missing.

    A None return signals "skip integration tests"; callers should not
    fall back to defaults.
    """
    values = env()
    required = ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_TEST_BUCKET")
    if not all(values.get(k) for k in required):
        return None
    return MinIOConfig(
        endpoint=values["MINIO_ENDPOINT"],
        access_key=values["MINIO_ACCESS_KEY"],
        secret_key=values["MINIO_SECRET_KEY"],
        region=values.get("MINIO_REGION", "us-east-1"),
        bucket=values["MINIO_TEST_BUCKET"],
    )


class MinIOTestClient:
    """A guarded boto3 wrapper that can only mutate the configured bucket.

    Any attempt to write/delete outside ``config.bucket`` raises
    ``RuntimeError`` before touching the wire. List/read calls on other
    buckets are allowed (used to confirm we're not mucking with them).
    """

    def __init__(self, config: MinIOConfig, *, prefix: str | None = None) -> None:
        self.config = config
        self.prefix = prefix or f"sessions/{uuid.uuid4().hex[:12]}/"
        if not self.prefix.endswith("/"):
            self.prefix = f"{self.prefix}/"
        self._s3: S3Client = boto3.client(
            "s3",
            endpoint_url=config.endpoint,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            config=Config(signature_version="s3v4"),
            region_name=config.region,
        )

    @property
    def s3(self) -> S3Client:
        return self._s3

    @property
    def s3_uri(self) -> str:
        """``s3://<bucket>/<prefix>`` — pass to scan_parquet / DuckLake DATA_PATH."""
        return f"s3://{self.config.bucket}/{self.prefix}"

    def reachable(self) -> bool:
        """Return True iff we can list the test bucket."""
        try:
            self._s3.head_bucket(Bucket=self.config.bucket)
            return True
        except (ClientError, EndpointConnectionError):
            return False

    def _guard_bucket(self, bucket: str) -> None:
        if bucket != self.config.bucket:
            raise RuntimeError(
                f"Refusing destructive op on bucket {bucket!r}: integration "
                f"tests are restricted to {self.config.bucket!r}."
            )

    def _guard_prefix(self, key: str) -> None:
        if not key.startswith(self.prefix):
            raise RuntimeError(
                f"Refusing destructive op on key {key!r}: outside the "
                f"per-session prefix {self.prefix!r}."
            )

    def put_object(self, key: str, body: bytes) -> None:
        self._guard_prefix(key)
        self._s3.put_object(Bucket=self.config.bucket, Key=key, Body=body)

    def cleanup(self) -> int:
        """Delete every object under this client's prefix. Returns the count.

        Safe by construction: only acts on ``self.config.bucket`` and only
        on keys under ``self.prefix``.
        """
        self._guard_bucket(self.config.bucket)
        deleted = 0
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=self.prefix):
            objects = page.get("Contents") or []
            if not objects:
                continue
            # Sanity-check every key before batching the delete request.
            for obj in objects:
                self._guard_prefix(obj["Key"])
            self._s3.delete_objects(
                Bucket=self.config.bucket,
                Delete={
                    "Objects": [{"Key": obj["Key"]} for obj in objects],
                    "Quiet": True,
                },
            )
            deleted += len(objects)
        return deleted


_TEST_SECRET_NAME = "polars_ducklake_test_minio"


def configure_duckdb_for_minio(con: Any, config: MinIOConfig) -> None:
    """Configure a DuckDB connection to read/write the test MinIO bucket.

    Uses a bucket-scoped DuckDB secret rather than session ``SET`` for
    two reasons:

    1. **Precedence over existing secrets.** DuckDB's secret subsystem
       picks the most-specific scope match, so any pre-existing
       ``s3://``-scoped secret on the developer's machine (e.g., one
       pointing at a different MinIO) silently wins over plain ``SET``
       statements. A bucket-scoped secret is more specific than the
       broader ``s3://`` and so it wins.
    2. **No mutation of persisted secrets.** This secret is dropped at
       the end of the test (see :func:`teardown_duckdb_minio_secret`)
       so the developer's stored secrets are untouched.

    Path-style addressing is required because MinIO doesn't support
    virtual-hosted-style URLs by default.
    """
    endpoint = config.endpoint.replace("https://", "").replace("http://", "")
    use_ssl = "true" if config.endpoint.startswith("https://") else "false"
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute(
        f"""
        CREATE OR REPLACE SECRET {_TEST_SECRET_NAME} (
            TYPE s3,
            KEY_ID '{config.access_key}',
            SECRET '{config.secret_key}',
            ENDPOINT '{endpoint}',
            USE_SSL {use_ssl},
            URL_STYLE 'path',
            REGION '{config.region}',
            SCOPE 's3://{config.bucket}/'
        )
        """
    )


def teardown_duckdb_minio_secret(con: Any) -> None:
    """Drop the test secret created by :func:`configure_duckdb_for_minio`.

    Idempotent: if the secret was never created (or already dropped),
    silently swallows the error.
    """
    with contextlib.suppress(Exception):
        con.execute(f"DROP SECRET {_TEST_SECRET_NAME}")


@contextmanager
def duckdb_writer(config: MinIOConfig) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a configured DuckDB connection scoped to the test MinIO.

    Tests use this instead of managing connect / configure / cleanup by
    hand. The bucket-scoped secret is created on enter and dropped on
    exit so the developer's persisted secrets are never modified.
    """
    con = duckdb.connect(":memory:")
    try:
        configure_duckdb_for_minio(con, config)
        yield con
    finally:
        teardown_duckdb_minio_secret(con)
        con.close()


def polars_storage_options(config: MinIOConfig) -> dict[str, str]:
    """Return ``storage_options`` that make ``pl.scan_parquet`` talk to MinIO.

    Polars uses object_store under the hood; option names follow that crate.
    """
    return {
        "aws_endpoint_url": config.endpoint,
        "aws_access_key_id": config.access_key,
        "aws_secret_access_key": config.secret_key,
        "aws_region": config.region,
        "aws_allow_http": "true",
        # MinIO needs path-style, not virtual-hosted-style.
        "aws_virtual_hosted_style_request": "false",
    }


def iter_session_bytes(client: MinIOTestClient) -> Iterator[tuple[str, int]]:
    """Yield ``(key, size)`` for every object under the session prefix.

    Used in tests to assert what was actually written.
    """
    paginator = client.s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=client.config.bucket, Prefix=client.prefix):
        for obj in page.get("Contents") or []:
            yield obj["Key"], obj["Size"]
