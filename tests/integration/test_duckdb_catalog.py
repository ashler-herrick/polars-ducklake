"""DuckDB-as-catalog integration tests.

Drives the DuckDB ducklake-extension writer with a DuckDB catalog file
(no external SQL service needed) and reads via our scan_ducklake using
the ``duckdb-engine`` SQLAlchemy dialect. Data is written to the
project-local MinIO so we still exercise object-storage I/O end-to-end.

This is the only integration test that's self-contained — no
``docker-compose`` services other than MinIO are needed.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

import polars_ducklake as pdl
from tests.integration._minio import (
    MinIOConfig,
    MinIOTestClient,
    duckdb_writer,
    polars_storage_options,
)

pytestmark = pytest.mark.integration


def _attach_duckdb_catalog(
    con: duckdb.DuckDBPyConnection,
    *,
    catalog_path: Path,
    data_path: str,
    data_inlining_row_limit: int | None = None,
) -> None:
    """ATTACH a DuckLake whose catalog is a DuckDB file (not SQLite).

    The DuckDB ducklake extension accepts both ``ducklake:duckdb:<path>``
    and the bare-path form ``ducklake:<path>``; we use the explicit form
    here for symmetry with the other backends.
    """
    if "'" in str(catalog_path) or "'" in data_path:
        raise ValueError("Refusing to ATTACH with single-quote in path")
    con.execute("INSTALL ducklake;")
    con.execute("LOAD ducklake;")
    extras = ""
    if data_inlining_row_limit is not None:
        extras = f", DATA_INLINING_ROW_LIMIT {int(data_inlining_row_limit)}"
    con.execute(
        f"ATTACH 'ducklake:duckdb:{catalog_path}' AS lake (DATA_PATH '{data_path}'{extras})"
    )
    con.execute("USE lake;")


def test_duckdb_catalog_round_trip_via_sqlalchemy_url(
    minio_client: MinIOTestClient, minio_config: MinIOConfig, tmp_path: Path
) -> None:
    """Full DuckDB→MinIO→our reader round-trip with a DuckDB catalog file.

    Reads back via a plain SQLAlchemy URL (``duckdb:///<path>``) — the
    most common form a user would use directly.
    """
    catalog_path = tmp_path / "metadata.duckdb"

    with duckdb_writer(minio_config) as con:
        _attach_duckdb_catalog(
            con,
            catalog_path=catalog_path,
            data_path=minio_client.s3_uri,
            data_inlining_row_limit=0,
        )
        con.execute("CREATE TABLE sales (id INTEGER, amount DOUBLE, region VARCHAR);")
        con.execute(
            "INSERT INTO sales VALUES "
            "(1, 10.0, 'us'), (2, 20.5, 'us'), (3, 30.25, 'eu'), "
            "(4, 40.0, 'eu'), (5, 55.5, 'apac');"
        )
        con.execute("CHECKPOINT;")

    df = (
        pdl.scan_ducklake(
            f"duckdb:///{catalog_path}",
            table="sales",
            storage_options=polars_storage_options(minio_config),
        )
        .sort("id")
        .collect()
    )
    assert df["id"].to_list() == [1, 2, 3, 4, 5]
    assert df["region"].to_list() == ["us", "us", "eu", "eu", "apac"]


def test_duckdb_catalog_round_trip_via_native_string(
    minio_client: MinIOTestClient, minio_config: MinIOConfig, tmp_path: Path
) -> None:
    """Same round-trip via the ``ducklake:duckdb:`` native form.

    Confirms the parser route through ``catalog._translate_ducklake_string``
    yields a working DuckDB engine (vs. the ``duckdb:///`` SQLAlchemy URL
    above, which exercises ``sqlalchemy.create_engine`` directly).
    """
    catalog_path = tmp_path / "metadata.duckdb"

    with duckdb_writer(minio_config) as con:
        _attach_duckdb_catalog(
            con,
            catalog_path=catalog_path,
            data_path=minio_client.s3_uri,
            data_inlining_row_limit=0,
        )
        con.execute("CREATE TABLE t (x INTEGER);")
        con.execute("INSERT INTO t VALUES (10), (20), (30);")
        con.execute("CHECKPOINT;")

    df = (
        pdl.scan_ducklake(
            f"ducklake:duckdb:{catalog_path}",
            table="t",
            storage_options=polars_storage_options(minio_config),
        )
        .sort("x")
        .collect()
    )
    assert df["x"].to_list() == [10, 20, 30]


def test_duckdb_catalog_bare_path_form(
    minio_client: MinIOTestClient, minio_config: MinIOConfig, tmp_path: Path
) -> None:
    """The bare-path form ``ducklake:<path>`` (no backend prefix).

    Equivalent to ``ducklake:duckdb:<path>`` per our parser. Both forms
    are valid in the DuckDB writer too.
    """
    catalog_path = tmp_path / "metadata.duckdb"

    with duckdb_writer(minio_config) as con:
        _attach_duckdb_catalog(
            con,
            catalog_path=catalog_path,
            data_path=minio_client.s3_uri,
            data_inlining_row_limit=0,
        )
        con.execute("CREATE TABLE t (x INTEGER);")
        con.execute("INSERT INTO t VALUES (1);")
        con.execute("CHECKPOINT;")

    df = pdl.scan_ducklake(
        f"ducklake:{catalog_path}",
        table="t",
        storage_options=polars_storage_options(minio_config),
    ).collect()
    assert df["x"].to_list() == [1]
