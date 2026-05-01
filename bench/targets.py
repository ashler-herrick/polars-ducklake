"""Per-backend (catalog, data path) targets for the bench harness.

A :class:`CatalogTarget` is everything :func:`bench.seed.materialize_lake`
needs to write into a lake, and everything the runners need later to
read from it. The same struct is what gets serialized into
``bench/state.json``.

CI tier: sqlite catalog file + local FS data path. No infra needed.

Full tier: one of three catalog backends (sqlite, postgres, duckdb)
with data on MinIO under a backend-specific bucket so catalog-migration
quirks between backends cannot bleed across runs (the user's
motivation, see issue #9 discussion).

We deliberately *do not* support MySQL here — the DuckDB ducklake-mysql
extension is upstream-unstable per the project README.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlalchemy
from sqlalchemy import text


_BENCH_BUCKET_PREFIX = "polars-ducklake-bench-"
SUPPORTED_BACKENDS_FULL = ("sqlite", "postgres", "duckdb")


@dataclass
class CatalogTarget:
    """Everything bench.seed and the runners need to talk to one lake.

    ``catalog_local_files``: paths the seed step created on local disk
    (e.g. a sqlite or duckdb catalog file). Tracked so a future
    ``--rebuild`` can clean them up.
    ``storage_options``: forwarded to :func:`scan_ducklake` for cloud
    reads; ``None`` for purely local data paths.
    ``duckdb_setup_sql``: SQL the runner should run on a fresh DuckDB
    connection before ATTACH (e.g. ``CREATE SECRET`` for MinIO).
    """

    backend: str
    sqlalchemy_url: str
    ducklake_native: str
    data_path: str
    storage_options: dict[str, str] | None = None
    duckdb_setup_sql: list[str] = field(default_factory=list)
    catalog_local_files: list[str] = field(default_factory=list)


# ----------------------------- CI tier ---------------------------------


def ci_sqlite_local(root: Path) -> CatalogTarget:
    """SQLite catalog + local-FS data, all under ``root``. No infra."""
    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    catalog_path = root / "metadata.db"
    return CatalogTarget(
        backend="sqlite",
        sqlalchemy_url=f"sqlite:///{catalog_path}",
        ducklake_native=f"ducklake:sqlite:{catalog_path}",
        data_path=f"{data_dir}/",
        catalog_local_files=[str(catalog_path)],
    )


# ---------------------------- Full tier --------------------------------
#
# Building these targets has side effects on shared infrastructure:
# they reset the bench bucket on MinIO and (for postgres) drop the
# ducklake metadata tables in the test DB. The user explicitly invoked
# ``python -m bench.seed`` so this is the expected destructive op.


def _minio_storage_options(cfg: Any) -> dict[str, str]:
    return {
        "aws_endpoint_url": cfg.endpoint,
        "aws_access_key_id": cfg.access_key,
        "aws_secret_access_key": cfg.secret_key,
        "aws_region": cfg.region,
        "aws_allow_http": "true",
        "aws_virtual_hosted_style_request": "false",
    }


def _minio_duckdb_setup_sql(cfg: Any, secret_name: str) -> list[str]:
    endpoint = cfg.endpoint.replace("https://", "").replace("http://", "")
    use_ssl = "true" if cfg.endpoint.startswith("https://") else "false"
    return [
        "INSTALL httpfs;",
        "LOAD httpfs;",
        f"""
        CREATE OR REPLACE SECRET {secret_name} (
            TYPE s3,
            KEY_ID '{cfg.access_key}',
            SECRET '{cfg.secret_key}',
            ENDPOINT '{endpoint}',
            USE_SSL {use_ssl},
            URL_STYLE 'path',
            REGION '{cfg.region}',
            SCOPE 's3://{cfg.bucket}/'
        )
        """.strip(),
    ]


def _ensure_bench_bucket(client: Any, bucket: str) -> None:
    s3 = client._s3  # the guarded boto3 client
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)


def _purge_bench_bucket(client: Any, bucket: str) -> int:
    """Delete all objects in ``bucket``. Refuses to touch anything else.

    The bucket name guard mirrors :class:`MinIOTestClient`'s safety
    check: every bench bucket starts with ``polars-ducklake-bench-`` and
    we refuse to operate on anything else.
    """
    if not bucket.startswith(_BENCH_BUCKET_PREFIX):
        raise RuntimeError(f"refusing to purge non-bench bucket {bucket!r}")
    s3 = client._s3
    deleted = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        objects = page.get("Contents") or []
        if not objects:
            continue
        s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": o["Key"]} for o in objects], "Quiet": True},
        )
        deleted += len(objects)
    return deleted


def _bench_bucket(backend: str) -> str:
    return f"{_BENCH_BUCKET_PREFIX}{backend}"


def full_sqlite_minio(client: Any, root: Path) -> CatalogTarget:
    """SQLite catalog file (local) + data on the sqlite bench MinIO bucket."""
    root.mkdir(parents=True, exist_ok=True)
    catalog_path = root / "metadata.db"
    if catalog_path.exists():
        catalog_path.unlink()
    bucket = _bench_bucket("sqlite")
    _ensure_bench_bucket(client, bucket)
    _purge_bench_bucket(client, bucket)

    cfg = type(client.config)(
        endpoint=client.config.endpoint,
        access_key=client.config.access_key,
        secret_key=client.config.secret_key,
        region=client.config.region,
        bucket=bucket,
    )
    return CatalogTarget(
        backend="sqlite",
        sqlalchemy_url=f"sqlite:///{catalog_path}",
        ducklake_native=f"ducklake:sqlite:{catalog_path}",
        data_path=f"s3://{bucket}/",
        storage_options=_minio_storage_options(cfg),
        duckdb_setup_sql=_minio_duckdb_setup_sql(cfg, "polars_ducklake_bench_sqlite"),
        catalog_local_files=[str(catalog_path)],
    )


def full_duckdb_minio(client: Any, root: Path) -> CatalogTarget:
    """DuckDB catalog file (local) + data on the duckdb bench MinIO bucket."""
    root.mkdir(parents=True, exist_ok=True)
    catalog_path = root / "metadata.duckdb"
    if catalog_path.exists():
        catalog_path.unlink()
    bucket = _bench_bucket("duckdb")
    _ensure_bench_bucket(client, bucket)
    _purge_bench_bucket(client, bucket)

    cfg = type(client.config)(
        endpoint=client.config.endpoint,
        access_key=client.config.access_key,
        secret_key=client.config.secret_key,
        region=client.config.region,
        bucket=bucket,
    )
    return CatalogTarget(
        backend="duckdb",
        sqlalchemy_url=f"duckdb:///{catalog_path}",
        ducklake_native=f"ducklake:duckdb:{catalog_path}",
        data_path=f"s3://{bucket}/",
        storage_options=_minio_storage_options(cfg),
        duckdb_setup_sql=_minio_duckdb_setup_sql(cfg, "polars_ducklake_bench_duckdb"),
        catalog_local_files=[str(catalog_path)],
    )


def full_postgres_minio(minio_client: Any, pg_config: Any) -> CatalogTarget:
    """Postgres catalog (test DB) + data on the postgres bench MinIO bucket.

    Drops every ducklake metadata table in the test DB before returning
    so the seed step starts from a clean slate. Only operates on the
    test database (``PostgresTestClient`` enforces the guard).
    """
    bucket = _bench_bucket("postgres")
    _ensure_bench_bucket(minio_client, bucket)
    _purge_bench_bucket(minio_client, bucket)

    engine = sqlalchemy.create_engine(pg_config.sqlalchemy_url)
    try:
        with engine.begin() as conn:
            tables = conn.execute(
                text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname='public' AND tablename LIKE 'ducklake_%'"
                )
            ).fetchall()
            for (name,) in tables:
                conn.execute(text(f'DROP TABLE IF EXISTS "{name}" CASCADE'))
    finally:
        engine.dispose()

    cfg = type(minio_client.config)(
        endpoint=minio_client.config.endpoint,
        access_key=minio_client.config.access_key,
        secret_key=minio_client.config.secret_key,
        region=minio_client.config.region,
        bucket=bucket,
    )
    return CatalogTarget(
        backend="postgres",
        sqlalchemy_url=pg_config.sqlalchemy_url,
        ducklake_native=pg_config.ducklake_native_string,
        data_path=f"s3://{bucket}/",
        storage_options=_minio_storage_options(cfg),
        duckdb_setup_sql=_minio_duckdb_setup_sql(cfg, "polars_ducklake_bench_postgres"),
        catalog_local_files=[],
    )
