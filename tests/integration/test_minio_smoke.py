"""Smoke test: DuckDB writes a real DuckLake to MinIO; we read it back.

This is the gating end-to-end test for any future work that touches the
DuckLake spec: if it passes, our reader can consume what the canonical
writer (the DuckDB ducklake extension) actually emits.

The test deliberately uses the spec-compliant writer rather than our
hand-built fixtures so that subtle representational details — type
strings, MVCC encoding, metadata keys — are validated against the
real implementation.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
import polars as pl
import pytest

import polars_ducklake as pdl
from tests.integration._minio import (
    MinIOConfig,
    MinIOTestClient,
    duckdb_writer,
    iter_session_bytes,
    polars_storage_options,
)

pytestmark = pytest.mark.integration


def _attach_ducklake(
    con: duckdb.DuckDBPyConnection,
    *,
    catalog_path: Path,
    data_path: str,
    data_inlining_row_limit: int | None = None,
) -> None:
    """Issue the ducklake ATTACH against a SQLite-backed catalog on MinIO data.

    DuckDB's ATTACH does not support SQL parameters, so the connection
    string is interpolated directly. This is test code with values we
    control end-to-end, so the usual SQL-injection caveats don't apply —
    but we still reject characters that would terminate the literal,
    just to surface an obvious bug fast if the catalog path ever ends up
    user-controlled.

    ``data_inlining_row_limit=0`` disables small-write inlining so every
    insert lands in Parquet — useful for tests that want to exercise the
    Parquet path regardless of insert size.
    """
    if "'" in str(catalog_path) or "'" in data_path:
        raise ValueError("Refusing to ATTACH with single-quote in path")
    con.execute("INSTALL ducklake;")
    con.execute("LOAD ducklake;")
    extras = ""
    if data_inlining_row_limit is not None:
        extras = f", DATA_INLINING_ROW_LIMIT {int(data_inlining_row_limit)}"
    con.execute(
        f"ATTACH 'ducklake:sqlite:{catalog_path}' AS lake (DATA_PATH '{data_path}'{extras})"
    )
    con.execute("USE lake;")


def test_duckdb_writes_we_read(
    minio_client: MinIOTestClient, minio_config: MinIOConfig, tmp_path: Path
) -> None:
    catalog_path = tmp_path / "metadata.db"
    data_path = minio_client.s3_uri  # e.g. s3://polars-ducklake-test/sessions/<uuid>/

    # Phase 1 — DuckDB writes a tiny DuckLake. Disable small-write inlining
    # so this test exercises the Parquet path regardless of row count;
    # the inlining-refusal path is covered separately.
    with duckdb_writer(minio_config) as con:
        _attach_ducklake(
            con,
            catalog_path=catalog_path,
            data_path=data_path,
            data_inlining_row_limit=0,
        )
        con.execute("CREATE TABLE sales (id INTEGER, amount DOUBLE, region VARCHAR);")
        con.execute(
            "INSERT INTO sales VALUES "
            "(1, 10.0, 'us'), (2, 20.5, 'us'), (3, 30.25, 'eu'), "
            "(4, 40.0, 'eu'), (5, 55.5, 'apac');"
        )
        con.execute("CHECKPOINT;")

    # Sanity: the writer actually wrote a Parquet object into our session prefix.
    written = list(iter_session_bytes(minio_client))
    assert written, (
        "DuckLake writer produced no objects under "
        f"{minio_client.s3_uri!r}; check DuckDB s3 settings."
    )
    parquet_objects = [k for k, _ in written if k.endswith(".parquet")]
    assert parquet_objects, f"Expected parquet under {minio_client.s3_uri!r}, got {written!r}"

    # Phase 2 — read it back via our scan_ducklake (no DuckDB!).
    lf = pdl.scan_ducklake(
        f"sqlite:///{catalog_path}",
        table="sales",
        storage_options=polars_storage_options(minio_config),
    )
    df = lf.sort("id").collect()
    assert df.height == 5
    assert df["region"].to_list() == ["us", "us", "eu", "eu", "apac"]
    assert df["amount"].to_list() == [10.0, 20.5, 30.25, 40.0, 55.5]


def test_inlined_data_refusal_against_real_writer(
    minio_client: MinIOTestClient, minio_config: MinIOConfig, tmp_path: Path
) -> None:
    """Confirm the inlined-data refusal fires against actual DuckDB output.

    The unit-test fixture builds the inlined-tables row by hand; this test
    validates that what our reader detects matches what the canonical
    writer actually produces when an insert falls under the inlining
    threshold (default 10 rows, default-on).
    """
    catalog_path = tmp_path / "metadata.db"

    with duckdb_writer(minio_config) as con:
        _attach_ducklake(con, catalog_path=catalog_path, data_path=minio_client.s3_uri)
        con.execute("CREATE TABLE tiny (x INTEGER);")
        con.execute("INSERT INTO tiny VALUES (1), (2), (3);")  # well under 10
        con.execute("CHECKPOINT;")

    with pytest.raises(NotImplementedError, match="inlined"):
        pdl.scan_ducklake(f"sqlite:///{catalog_path}", table="tiny")


def test_delete_round_trip_against_real_writer(
    minio_client: MinIOTestClient, minio_config: MinIOConfig, tmp_path: Path
) -> None:
    """Real DuckDB writes data, deletes some rows, our reader returns the rest.

    This is the primary correctness gate for delete-file support: every
    detail (delete-file Parquet schema, position column name and indexing,
    catalog linkage, MVCC encoding) is validated against the canonical
    writer's actual output rather than our hand-built fixtures.
    """
    catalog_path = tmp_path / "metadata.db"

    with duckdb_writer(minio_config) as con:
        _attach_ducklake(
            con,
            catalog_path=catalog_path,
            data_path=minio_client.s3_uri,
            data_inlining_row_limit=0,
        )
        con.execute("CREATE TABLE t (id INTEGER, val VARCHAR);")
        # 12 rows so the writer takes the Parquet path even without the
        # row-limit override above (defense in depth).
        rows = ", ".join(f"({i}, 'v{i}')" for i in range(1, 13))
        con.execute(f"INSERT INTO t VALUES {rows};")
        con.execute("CHECKPOINT;")
        con.execute("DELETE FROM t WHERE id IN (3, 7, 11);")
        con.execute("CHECKPOINT;")

    df = (
        pdl.scan_ducklake(
            f"sqlite:///{catalog_path}",
            table="t",
            storage_options=polars_storage_options(minio_config),
        )
        .sort("id")
        .collect()
    )
    assert df["id"].to_list() == [1, 2, 4, 5, 6, 8, 9, 10, 12]
    assert df["val"].to_list() == [f"v{i}" for i in [1, 2, 4, 5, 6, 8, 9, 10, 12]]


def test_update_is_delete_plus_insert_round_trip(
    minio_client: MinIOTestClient, minio_config: MinIOConfig, tmp_path: Path
) -> None:
    """DuckLake represents UPDATE as DELETE + INSERT — verify end-to-end.

    Updating a row produces (a) a delete file marking the original row's
    position and (b) a new data file containing the updated row. Our
    reader must skip the original via anti-join *and* include the new
    data file. This test fails loudly if either side breaks.
    """
    catalog_path = tmp_path / "metadata.db"

    with duckdb_writer(minio_config) as con:
        _attach_ducklake(
            con,
            catalog_path=catalog_path,
            data_path=minio_client.s3_uri,
            data_inlining_row_limit=0,
        )
        con.execute("CREATE TABLE products (id INTEGER, price DOUBLE);")
        rows = ", ".join(f"({i}, {i * 1.0})" for i in range(1, 13))
        con.execute(f"INSERT INTO products VALUES {rows};")
        con.execute("CHECKPOINT;")
        con.execute("UPDATE products SET price = 99.99 WHERE id = 5;")
        con.execute("CHECKPOINT;")

    df = (
        pdl.scan_ducklake(
            f"sqlite:///{catalog_path}",
            table="products",
            storage_options=polars_storage_options(minio_config),
        )
        .sort("id")
        .collect()
    )
    assert df["id"].to_list() == list(range(1, 13))
    prices = df["price"].to_list()
    assert prices[4] == 99.99  # id=5 was updated
    # Every other row keeps its original price
    for i, price in enumerate(prices):
        if i + 1 == 5:
            continue
        assert price == float(i + 1)


def test_partitioned_table_round_trip(
    minio_client: MinIOTestClient, minio_config: MinIOConfig, tmp_path: Path
) -> None:
    """Identity-partitioned and non-identity (year) transforms read correctly.

    DuckLake's writer puts the partition column physically into each
    Parquet for identity transforms, so the reader gets it for free. For
    non-identity transforms (year, month, etc.) the source column is
    still written and the transformed value lives only in the catalog —
    correctness still holds for reads (the user gets the source column).
    """
    catalog_path = tmp_path / "metadata.db"
    with duckdb_writer(minio_config) as con:
        _attach_ducklake(
            con,
            catalog_path=catalog_path,
            data_path=minio_client.s3_uri,
            data_inlining_row_limit=0,
        )
        con.execute("CREATE TABLE sales (id INTEGER, region VARCHAR);")
        con.execute("ALTER TABLE sales SET PARTITIONED BY (region);")
        con.execute(
            "INSERT INTO sales VALUES (1,'us'),(2,'eu'),(3,'us'),(4,'apac'),(5,'eu'),(6,'us');"
        )
        con.execute("CHECKPOINT;")

    df = (
        pdl.scan_ducklake(
            f"sqlite:///{catalog_path}",
            table="sales",
            storage_options=polars_storage_options(minio_config),
        )
        .sort("id")
        .collect()
    )
    assert df["id"].to_list() == [1, 2, 3, 4, 5, 6]
    assert df["region"].to_list() == ["us", "eu", "us", "apac", "eu", "us"]


def test_schema_evolution_round_trip(
    minio_client: MinIOTestClient, minio_config: MinIOConfig, tmp_path: Path
) -> None:
    """Real DuckDB ALTER TABLE ADD/DROP/RENAME — older files still read correctly.

    Mixes column-add (extra), column-drop (legacy), and column-rename
    (id → entity_id) across snapshots. Verifies our reader applies the
    catalog's column_id-based translation per file rather than blindly
    projecting by name.
    """
    catalog_path = tmp_path / "metadata.db"
    with duckdb_writer(minio_config) as con:
        _attach_ducklake(
            con,
            catalog_path=catalog_path,
            data_path=minio_client.s3_uri,
            data_inlining_row_limit=0,
        )
        con.execute("CREATE TABLE t (id INTEGER, legacy VARCHAR);")
        con.execute("INSERT INTO t VALUES (1,'old1'),(2,'old2');")
        con.execute("CHECKPOINT;")

        con.execute("ALTER TABLE t ADD COLUMN extra DOUBLE;")
        con.execute("INSERT INTO t VALUES (3,'old3',3.14),(4,'old4',2.71);")
        con.execute("CHECKPOINT;")

        con.execute("ALTER TABLE t DROP COLUMN legacy;")
        con.execute("INSERT INTO t VALUES (5, 5.5);")
        con.execute("CHECKPOINT;")

        con.execute("ALTER TABLE t RENAME COLUMN id TO entity_id;")
        con.execute("INSERT INTO t VALUES (6, 6.0);")
        con.execute("CHECKPOINT;")

    df = (
        pdl.scan_ducklake(
            f"sqlite:///{catalog_path}",
            table="t",
            storage_options=polars_storage_options(minio_config),
        )
        .sort("entity_id")
        .collect()
    )
    assert df.columns == ["entity_id", "extra"]
    assert df["entity_id"].to_list() == [1, 2, 3, 4, 5, 6]
    # Rows 1,2 written before ADD → null-filled for extra.
    assert df["extra"].to_list() == [None, None, 3.14, 2.71, 5.5, 6.0]


def test_compaction_partial_files_round_trip(
    minio_client: MinIOTestClient, minio_config: MinIOConfig, tmp_path: Path
) -> None:
    """Compaction merges small files; merged files carry per-row snapshot ids.

    Reads at the latest snapshot must see all merged rows; reads pinned
    to an earlier snapshot must filter rows whose origin > target.
    """
    catalog_path = tmp_path / "metadata.db"
    with duckdb_writer(minio_config) as con:
        _attach_ducklake(
            con,
            catalog_path=catalog_path,
            data_path=minio_client.s3_uri,
            data_inlining_row_limit=0,
        )
        con.execute("CREATE TABLE t (id INTEGER, val VARCHAR);")
        # Four separate snapshots so there are several files to merge.
        for i in range(1, 5):
            con.execute(f"INSERT INTO t VALUES ({i}, 'v{i}');")
            con.execute("CHECKPOINT;")
        # Force compaction. May be a no-op if DuckDB already auto-merged.
        con.execute("CALL ducklake_merge_adjacent_files('lake');")

    # Latest snapshot sees all 4 rows.
    df = (
        pdl.scan_ducklake(
            f"sqlite:///{catalog_path}",
            table="t",
            storage_options=polars_storage_options(minio_config),
        )
        .sort("id")
        .collect()
    )
    assert df["id"].to_list() == [1, 2, 3, 4]
    assert df["val"].to_list() == ["v1", "v2", "v3", "v4"]

    # Identify the snapshot ids and pin reads to verify time travel works
    # against a partial file. We take the second-earliest data snapshot
    # (the snapshot under which row id=2 was inserted).
    with sqlite3.connect(catalog_path) as sql:
        snapshots = [
            r[0]
            for r in sql.execute("SELECT snapshot_id FROM ducklake_snapshot ORDER BY snapshot_id")
        ]
    # First snapshot is schema-create; second is first INSERT (id=1);
    # third is INSERT id=2; ...
    target = snapshots[2]  # after id=1 and id=2 are visible
    df_old = (
        pdl.scan_ducklake(
            f"sqlite:///{catalog_path}",
            table="t",
            snapshot_id=target,
            storage_options=polars_storage_options(minio_config),
        )
        .sort("id")
        .collect()
    )
    # Assertion is a subset rather than exact equality because the
    # writer's auto-snapshot cadence is implementation-defined; the
    # important property is "no rows beyond the target snapshot" and
    # "at least the rows known to be ≤ target are present".
    ids = df_old["id"].to_list()
    assert 1 in ids
    assert 4 not in ids  # id=4 was inserted at the last snapshot, after target
    assert all(i <= 4 for i in ids)


def test_nested_types_round_trip(
    minio_client: MinIOTestClient, minio_config: MinIOConfig, tmp_path: Path
) -> None:
    """Real DuckDB writer produces nested catalog rows we can read back.

    Validates that LIST / STRUCT / nested-of-nested survive the writer's
    catalog encoding and our reader's recursive type-tree walker. Catches
    representation drift if DuckDB ever changes how it decomposes nested
    types into ``ducklake_column`` rows.

    MAP is omitted intentionally: DuckDB writes the Arrow MAP logical
    type into the Parquet, and Polars' native reader currently panics on
    that form ("MapArray expects DataType::Struct as its inner logical
    type"). Our catalog-side mapping is verified in
    tests/test_backends_matrix.py::test_map_column, where the Parquet is
    written by Polars itself in the equivalent ``List(Struct{key,value})``
    shape that Polars can round-trip.
    """
    catalog_path = tmp_path / "metadata.db"
    with duckdb_writer(minio_config) as con:
        _attach_ducklake(
            con,
            catalog_path=catalog_path,
            data_path=minio_client.s3_uri,
            data_inlining_row_limit=0,
        )
        con.execute(
            """
            CREATE TABLE t (
                id INTEGER,
                tags INTEGER[],
                payload STRUCT(a INTEGER, b VARCHAR),
                events STRUCT(a INTEGER, b VARCHAR)[]
            );
            """
        )
        con.execute(
            """
            INSERT INTO t VALUES
                (1, [10, 20], {a: 1, b: 'x'},
                    [{a: 100, b: 'p'}, {a: 200, b: 'q'}]),
                (2, [30],     {a: 2, b: 'y'},
                    [{a: 300, b: 'r'}]);
            """
        )
        con.execute("CHECKPOINT;")

    df = (
        pdl.scan_ducklake(
            f"sqlite:///{catalog_path}",
            "t",
            storage_options=polars_storage_options(minio_config),
        )
        .sort("id")
        .collect()
    )
    assert df["tags"].to_list() == [[10, 20], [30]]
    assert df["payload"].to_list() == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    # Nested-of-nested: list of structs, the most common real-world shape.
    assert df["events"].to_list() == [
        [{"a": 100, "b": "p"}, {"a": 200, "b": "q"}],
        [{"a": 300, "b": "r"}],
    ]


def test_json_column_is_string(
    minio_client: MinIOTestClient, minio_config: MinIOConfig, tmp_path: Path
) -> None:
    """JSON columns surface as ``pl.String`` so users can call
    ``.str.json_decode(dtype=...)`` on them.

    DuckDB writes JSON to Parquet with the JSON logical type; Polars'
    native reader returns those columns as ``pl.Binary`` by default.
    Our reader should bridge that to ``pl.String`` so the user-visible
    type matches what the DuckLake spec calls a JSON column. Polars 2.0
    is expected to change the default for JSON-tagged Parquet, at which
    point this cast becomes redundant — but we want stable user-facing
    behaviour today.
    """
    catalog_path = tmp_path / "metadata.db"
    with duckdb_writer(minio_config) as con:
        _attach_ducklake(
            con,
            catalog_path=catalog_path,
            data_path=minio_client.s3_uri,
            data_inlining_row_limit=0,
        )
        con.execute("INSTALL json; LOAD json;")
        con.execute("CREATE TABLE t (id INTEGER, payload JSON);")
        con.execute('INSERT INTO t VALUES (1, \'{"a":1,"b":"x"}\'), (2, \'{"a":2,"b":"y"}\');')
        con.execute("CHECKPOINT;")

    df = (
        pdl.scan_ducklake(
            f"sqlite:///{catalog_path}",
            "t",
            storage_options=polars_storage_options(minio_config),
        )
        .sort("id")
        .collect()
    )
    assert df.schema["payload"] == pl.String
    assert df["payload"].to_list() == ['{"a":1,"b":"x"}', '{"a":2,"b":"y"}']
    # Users can now decode to a Struct themselves.
    decoded = df.with_columns(
        pl.col("payload").str.json_decode(dtype=pl.Struct({"a": pl.Int64, "b": pl.String}))
    )
    assert decoded["payload"].to_list() == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_session_prefix_isolation_only(
    minio_client: MinIOTestClient, minio_config: MinIOConfig
) -> None:
    """Belt-and-braces: confirm the cleanup guards refuse stray operations.

    This test exists because the integration suite runs destructive ops on
    a real bucket — if the guards ever break we want a fast, loud signal.
    """
    # Refuse to delete in any other bucket.
    with pytest.raises(RuntimeError, match="restricted"):
        minio_client._guard_bucket("ducklake")
    # Refuse to operate on keys outside the session prefix.
    with pytest.raises(RuntimeError, match="outside the per-session prefix"):
        minio_client._guard_prefix("not-mine/foo.parquet")


def test_catalog_metadata_uses_minio_uri(
    minio_client: MinIOTestClient, minio_config: MinIOConfig, tmp_path: Path
) -> None:
    """Confirm the writer stored a proper s3:// data_path in ducklake_metadata.

    This is the metadata side of the round-trip: it tells us the writer
    persists URIs correctly so our path-resolution code (paths.py) is
    being exercised against real-world inputs.
    """
    catalog_path = tmp_path / "metadata.db"
    data_path = minio_client.s3_uri

    with duckdb_writer(minio_config) as con:
        _attach_ducklake(con, catalog_path=catalog_path, data_path=data_path)
        con.execute("CREATE TABLE t (x INTEGER); INSERT INTO t VALUES (1);")
        con.execute("CHECKPOINT;")

    with sqlite3.connect(catalog_path) as sql:
        row = sql.execute("SELECT value FROM ducklake_metadata WHERE key = 'data_path'").fetchone()
    assert row is not None, "DuckLake writer did not persist data_path"
    stored: str = row[0]
    assert stored.startswith("s3://"), f"Expected s3:// data_path, got {stored!r}"
    assert minio_config.bucket in stored
