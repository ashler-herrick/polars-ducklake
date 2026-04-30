"""Postgres-as-catalog integration tests.

Drives the DuckDB ducklake-extension writer with a Postgres-backed
catalog (data on the project-local MinIO) and reads via our
scan_ducklake. Covers the SQLAlchemy-URL and ducklake-native string
addressing forms.

The test database (configured via ``POSTGRES_TEST_DATABASE`` in
``tests/.env``) is dedicated to the integration suite — every
ducklake_* table created during a test is dropped on teardown via
:class:`PostgresTestClient.drop_all_ducklake_tables`. We never touch
any other database on the same Postgres instance.
"""

from __future__ import annotations

import duckdb
import pytest

import polars_ducklake as pdl
from tests.integration._minio import (
    MinIOConfig,
    MinIOTestClient,
    duckdb_writer,
    polars_storage_options,
)
from tests.integration._postgres import PostgresConfig, PostgresTestClient

pytestmark = pytest.mark.integration


def _attach_postgres_catalog(
    con: duckdb.DuckDBPyConnection,
    *,
    pg: PostgresConfig,
    data_path: str,
    data_inlining_row_limit: int | None = None,
) -> None:
    """ATTACH a DuckLake whose catalog is the dedicated Postgres test DB.

    The DuckDB ducklake extension reads libpq-style key=value parameters
    after ``ducklake:postgres:``.
    """
    if "'" in data_path:
        raise ValueError("Refusing to ATTACH with single-quote in data_path")
    con.execute("INSTALL ducklake;")
    con.execute("LOAD ducklake;")
    con.execute("INSTALL postgres;")
    con.execute("LOAD postgres;")
    extras = ""
    if data_inlining_row_limit is not None:
        extras = f", DATA_INLINING_ROW_LIMIT {int(data_inlining_row_limit)}"
    attach = pg.duckdb_attach_string
    con.execute(f"ATTACH '{attach}' AS lake (DATA_PATH '{data_path}'{extras})")
    con.execute("USE lake;")


def test_postgres_catalog_round_trip_via_sqlalchemy_url(
    minio_client: MinIOTestClient,
    minio_config: MinIOConfig,
    postgres_client: PostgresTestClient,
) -> None:
    """Real DuckDB writes to a Postgres catalog; we read via SQLAlchemy URL.

    This is the canonical "I have a Postgres-backed lake" path. Tests
    every layer end-to-end: psycopg connection, MVCC visibility on the
    Postgres-side catalog rows, multi-file Parquet read on MinIO.
    """
    pg = postgres_client.config

    with duckdb_writer(minio_config) as con:
        _attach_postgres_catalog(
            con,
            pg=pg,
            data_path=minio_client.s3_uri,
            data_inlining_row_limit=0,
        )
        con.execute("CREATE TABLE sales (id INTEGER, region VARCHAR);")
        con.execute(
            "INSERT INTO sales VALUES (1,'us'),(2,'eu'),(3,'us'),(4,'apac'),(5,'eu'),(6,'us');"
        )
        con.execute("CHECKPOINT;")

    df = (
        pdl.scan_ducklake(
            pg.sqlalchemy_url,
            table="sales",
            storage_options=polars_storage_options(minio_config),
        )
        .sort("id")
        .collect()
    )
    assert df["id"].to_list() == [1, 2, 3, 4, 5, 6]
    assert df["region"].to_list() == ["us", "eu", "us", "apac", "eu", "us"]


def test_postgres_catalog_round_trip_via_native_string(
    minio_client: MinIOTestClient,
    minio_config: MinIOConfig,
    postgres_client: PostgresTestClient,
) -> None:
    """Same round-trip via the ``ducklake:postgres:`` native form."""
    pg = postgres_client.config

    with duckdb_writer(minio_config) as con:
        _attach_postgres_catalog(
            con,
            pg=pg,
            data_path=minio_client.s3_uri,
            data_inlining_row_limit=0,
        )
        con.execute("CREATE TABLE t (x INTEGER, label VARCHAR);")
        con.execute("INSERT INTO t VALUES (10,'a'),(20,'b'),(30,'c');")
        con.execute("CHECKPOINT;")

    df = (
        pdl.scan_ducklake(
            pg.ducklake_native_string,
            table="t",
            storage_options=polars_storage_options(minio_config),
        )
        .sort("x")
        .collect()
    )
    assert df["x"].to_list() == [10, 20, 30]
    assert df["label"].to_list() == ["a", "b", "c"]


def test_postgres_catalog_delete_round_trip(
    minio_client: MinIOTestClient,
    minio_config: MinIOConfig,
    postgres_client: PostgresTestClient,
) -> None:
    """Delete-file support works with a Postgres catalog backend.

    This guards against subtle backend-specific issues — different
    timestamp/boolean/null semantics between SQLite and Postgres could
    in theory break the MVCC clauses or the delete-file lookup.
    """
    pg = postgres_client.config

    with duckdb_writer(minio_config) as con:
        _attach_postgres_catalog(
            con,
            pg=pg,
            data_path=minio_client.s3_uri,
            data_inlining_row_limit=0,
        )
        con.execute("CREATE TABLE t (id INTEGER, val VARCHAR);")
        rows = ", ".join(f"({i}, 'v{i}')" for i in range(1, 13))
        con.execute(f"INSERT INTO t VALUES {rows};")
        con.execute("CHECKPOINT;")
        con.execute("DELETE FROM t WHERE id IN (3, 7, 11);")
        con.execute("CHECKPOINT;")

    df = (
        pdl.scan_ducklake(
            pg.sqlalchemy_url,
            table="t",
            storage_options=polars_storage_options(minio_config),
        )
        .sort("id")
        .collect()
    )
    assert df["id"].to_list() == [1, 2, 4, 5, 6, 8, 9, 10, 12]


def test_postgres_catalog_schema_evolution(
    minio_client: MinIOTestClient,
    minio_config: MinIOConfig,
    postgres_client: PostgresTestClient,
) -> None:
    """Schema evolution (add/drop/rename) works with a Postgres catalog."""
    pg = postgres_client.config

    with duckdb_writer(minio_config) as con:
        _attach_postgres_catalog(
            con,
            pg=pg,
            data_path=minio_client.s3_uri,
            data_inlining_row_limit=0,
        )
        con.execute("CREATE TABLE t (id INTEGER, legacy VARCHAR);")
        con.execute("INSERT INTO t VALUES (1,'old1'),(2,'old2');")
        con.execute("CHECKPOINT;")

        con.execute("ALTER TABLE t ADD COLUMN extra DOUBLE;")
        con.execute("INSERT INTO t VALUES (3,'old3',3.14);")
        con.execute("CHECKPOINT;")

        con.execute("ALTER TABLE t DROP COLUMN legacy;")
        con.execute("INSERT INTO t VALUES (4, 4.5);")
        con.execute("CHECKPOINT;")

        con.execute("ALTER TABLE t RENAME COLUMN id TO entity_id;")
        con.execute("INSERT INTO t VALUES (5, 5.5);")
        con.execute("CHECKPOINT;")

    df = (
        pdl.scan_ducklake(
            pg.sqlalchemy_url,
            table="t",
            storage_options=polars_storage_options(minio_config),
        )
        .sort("entity_id")
        .collect()
    )
    assert df.columns == ["entity_id", "extra"]
    assert df["entity_id"].to_list() == [1, 2, 3, 4, 5]
    assert df["extra"].to_list() == [None, None, 3.14, 4.5, 5.5]


def test_postgres_test_db_isolation_guard(
    postgres_client: PostgresTestClient,
) -> None:
    """The guard that prevents touching any other Postgres database."""
    with pytest.raises(RuntimeError, match="restricted"):
        postgres_client._guard_db("postgres")
