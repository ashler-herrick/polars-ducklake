"""Cross-backend matrix tests for ``scan_ducklake``.

The unit-test fixture in ``tests/conftest.py`` originally targeted
SQLite-only via the stdlib ``sqlite3`` driver. To exercise our SQL on
the other catalog backends we support, the same fixture is parametrised
here over every reachable backend in the project-local docker-compose
stack. Tests skip when a backend isn't reachable.

This file is the contract surface across catalogs. If a behavior
verified here passes for SQLite but fails for Postgres, the bug is
in our SQL or our reader; the dialect of the metadata catalog is the
only thing that varies.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest

import polars_ducklake as pdl
from tests.conftest import DuckLakeBuilder
from tests.integration import _mysql, _postgres


# Backends to parametrise over. Each value is a string id (used in test
# names) plus a builder factory callable that returns a fresh
# DuckLakeBuilder pointed at that backend. The factory may call
# pytest.skip if its backend isn't reachable.
def _sqlite_builder(tmp_path: Path) -> DuckLakeBuilder:
    return DuckLakeBuilder(tmp_path)


def _postgres_builder(tmp_path: Path) -> DuckLakeBuilder:
    config = _postgres.load_config()
    if config is None:
        pytest.skip("Postgres config not found in tests/.env")
    client = _postgres.PostgresTestClient(config)
    if not client.reachable():
        pytest.skip(f"Postgres at {config.host}:{config.port} not reachable")
    client.drop_all_ducklake_tables()
    return DuckLakeBuilder(tmp_path, sqlalchemy_url=config.sqlalchemy_url)


def _duckdb_builder(tmp_path: Path) -> DuckLakeBuilder:
    # No external service — DuckDB-as-catalog is just a file. Requires
    # the duckdb-engine SQLAlchemy dialect (installed in the dev group).
    catalog_file = tmp_path / "metadata.duckdb"
    return DuckLakeBuilder(tmp_path, sqlalchemy_url=f"duckdb:///{catalog_file}")


def _mysql_builder(tmp_path: Path) -> DuckLakeBuilder:
    config = _mysql.load_config()
    if config is None:
        pytest.skip("MySQL config not found in tests/.env")
    client = _mysql.MySQLTestClient(config)
    if not client.reachable():
        pytest.skip(f"MySQL at {config.host}:{config.port} not reachable")
    client.drop_all_ducklake_tables()
    return DuckLakeBuilder(tmp_path, sqlalchemy_url=config.sqlalchemy_url)


@pytest.fixture(params=["sqlite", "postgres", "duckdb", "mysql"])
def backend_builder(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[DuckLakeBuilder]:
    """A fresh DuckLake on the parametrised metadata catalog backend.

    Data files are always written to ``tmp_path`` (local FS); only the
    catalog database varies. After the test, Postgres tables are dropped
    so subsequent tests start from a clean catalog.
    """
    backend_name: str = request.param
    factories = {
        "sqlite": _sqlite_builder,
        "postgres": _postgres_builder,
        "duckdb": _duckdb_builder,
        "mysql": _mysql_builder,
    }
    builder = factories[backend_name](tmp_path)
    try:
        yield builder
    finally:
        builder.cleanup()


def test_round_trip(backend_builder: DuckLakeBuilder) -> None:
    """Tracer bullet: build a single-file lake and read it back via scan_ducklake."""
    snap = backend_builder.take_snapshot()
    schema_id = backend_builder.add_schema(name="main", begin_snapshot=snap)
    table_id = backend_builder.add_table(
        schema_id=schema_id,
        name="sales",
        begin_snapshot=snap,
        columns=[("id", "INTEGER"), ("region", "VARCHAR")],
    )
    backend_builder.write_data_file(
        table_id=table_id,
        begin_snapshot=snap,
        df=pl.DataFrame({"id": [1, 2, 3], "region": ["us", "eu", "us"]}),
    )

    df = pdl.scan_ducklake(backend_builder.url, "sales").sort("id").collect()
    assert df["id"].to_list() == [1, 2, 3]
    assert df["region"].to_list() == ["us", "eu", "us"]


def test_multi_snapshot_time_travel(backend_builder: DuckLakeBuilder) -> None:
    """MVCC clauses behave correctly across all backends.

    Build three snapshots that each append a new data file, then read
    pinned to each. The MVCC clause is the most dialect-sensitive SQL
    we issue (boolean / null comparisons, snapshot_time bind types) so
    this catches per-backend regressions.
    """
    from datetime import datetime as _dt

    snap1 = backend_builder.take_snapshot(snapshot_time=_dt(2026, 1, 1))
    schema_id = backend_builder.add_schema(name="main", begin_snapshot=snap1)
    table_id = backend_builder.add_table(
        schema_id=schema_id,
        name="rolling",
        begin_snapshot=snap1,
        columns=[("id", "INTEGER")],
    )
    backend_builder.write_data_file(
        table_id=table_id,
        begin_snapshot=snap1,
        df=pl.DataFrame({"id": [1, 2]}),
    )
    snap2 = backend_builder.take_snapshot(snapshot_time=_dt(2026, 2, 1))
    backend_builder.write_data_file(
        table_id=table_id,
        begin_snapshot=snap2,
        df=pl.DataFrame({"id": [3, 4]}),
    )
    snap3 = backend_builder.take_snapshot(snapshot_time=_dt(2026, 3, 1))
    backend_builder.write_data_file(
        table_id=table_id,
        begin_snapshot=snap3,
        df=pl.DataFrame({"id": [5, 6]}),
    )

    # Default: latest snapshot — sees all 6 rows.
    latest = pdl.scan_ducklake(backend_builder.url, "rolling").collect()
    assert sorted(latest["id"].to_list()) == [1, 2, 3, 4, 5, 6]

    # Pinned by snapshot_id.
    at_s1 = pdl.scan_ducklake(backend_builder.url, "rolling", snapshot_id=snap1).collect()
    assert sorted(at_s1["id"].to_list()) == [1, 2]

    at_s2 = pdl.scan_ducklake(backend_builder.url, "rolling", snapshot_id=snap2).collect()
    assert sorted(at_s2["id"].to_list()) == [1, 2, 3, 4]

    # Pinned by as_of timestamp — exercises the dialect's TIMESTAMP
    # comparison semantics.
    as_of_feb15 = pdl.scan_ducklake(
        backend_builder.url, "rolling", as_of=_dt(2026, 2, 15)
    ).collect()
    assert sorted(as_of_feb15["id"].to_list()) == [1, 2, 3, 4]


def test_deletes(backend_builder: DuckLakeBuilder) -> None:
    """Positional deletes (anti-join on `pos`) work on every backend.

    The catalog query that locates delete files is
    backend-portable, but the join through ``data_file_id`` on
    BIGINT is dialect-sensitive (signed vs unsigned, null handling).
    """
    snap1 = backend_builder.take_snapshot()
    schema_id = backend_builder.add_schema(name="main", begin_snapshot=snap1)
    table_id = backend_builder.add_table(
        schema_id=schema_id,
        name="t",
        begin_snapshot=snap1,
        columns=[("id", "INTEGER"), ("val", "VARCHAR")],
    )
    df_id = backend_builder.write_data_file(
        table_id=table_id,
        begin_snapshot=snap1,
        df=pl.DataFrame({"id": [1, 2, 3, 4, 5], "val": ["a", "b", "c", "d", "e"]}),
    )
    snap2 = backend_builder.take_snapshot()
    backend_builder.add_delete_file(
        table_id=table_id,
        begin_snapshot=snap2,
        data_file_id=df_id,
        positions=[1, 3],  # drop id=2 and id=4
    )
    df = pdl.scan_ducklake(backend_builder.url, "t").sort("id").collect()
    assert df["id"].to_list() == [1, 3, 5]
    assert df["val"].to_list() == ["a", "c", "e"]


def test_schema_evolution(backend_builder: DuckLakeBuilder) -> None:
    """ADD / DROP / RENAME column work on every backend.

    The schema-evolution path queries ``ducklake_column`` at each
    file's begin_snapshot — so any backend-specific MVCC bug surfaces
    here as missing or extra columns in the output.
    """
    snap1 = backend_builder.take_snapshot()
    schema_id = backend_builder.add_schema(name="main", begin_snapshot=snap1)
    table_id = backend_builder.add_table(
        schema_id=schema_id,
        name="t",
        begin_snapshot=snap1,
        columns=[("id", "INTEGER"), ("legacy", "VARCHAR")],
    )
    backend_builder.write_data_file(
        table_id=table_id,
        begin_snapshot=snap1,
        df=pl.DataFrame({"id": [1, 2], "legacy": ["x", "y"]}),
    )
    # snap2: ADD COLUMN extra; DROP legacy.
    snap2 = backend_builder.take_snapshot()
    backend_builder.drop_column(column_id=2, end_snapshot=snap2)  # legacy
    backend_builder.add_column(
        table_id=table_id,
        begin_snapshot=snap2,
        column_name="extra",
        column_type="DOUBLE",
        column_order=1,
    )
    snap3 = backend_builder.take_snapshot()
    backend_builder.write_data_file(
        table_id=table_id,
        begin_snapshot=snap3,
        df=pl.DataFrame({"id": [3, 4], "extra": [3.14, 2.71]}),
    )

    df = pdl.scan_ducklake(backend_builder.url, "t").sort("id").collect()
    assert df.columns == ["id", "extra"]
    assert df["id"].to_list() == [1, 2, 3, 4]
    # First file (pre-ADD) gets null-filled for extra.
    assert df["extra"].to_list() == [None, None, 3.14, 2.71]


def test_partial_files(backend_builder: DuckLakeBuilder) -> None:
    """Compaction-merged files filter rows by per-row snapshot id.

    The reader injects a ``WHERE _ducklake_internal_snapshot_id <= :ts``
    when reading partial files for time-travel — but the *catalog* must
    correctly report ``partial_max`` as a non-null BIGINT.
    """
    snap1 = backend_builder.take_snapshot()
    snap2 = backend_builder.take_snapshot()
    snap3 = backend_builder.take_snapshot()
    snap4 = backend_builder.take_snapshot()
    schema_id = backend_builder.add_schema(name="main", begin_snapshot=snap1)
    table_id = backend_builder.add_table(
        schema_id=schema_id,
        name="t",
        begin_snapshot=snap1,
        columns=[("id", "INTEGER"), ("val", "VARCHAR")],
    )
    # Simulate a compaction-merged file with rows from snaps 1, 2, 3.
    merged = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "val": ["a", "b", "c"],
            "_ducklake_internal_snapshot_id": pl.Series([snap1, snap2, snap3], dtype=pl.Int64()),
        }
    )
    backend_builder.write_data_file(
        table_id=table_id,
        begin_snapshot=snap1,
        df=merged,
        partial_max=snap3,
    )

    # Read at snap4: target >= partial_max → no row filter, all 3 rows.
    latest = pdl.scan_ducklake(backend_builder.url, "t", snapshot_id=snap4).sort("id").collect()
    assert latest["id"].to_list() == [1, 2, 3]
    assert latest.columns == ["id", "val"]  # internal column dropped

    # Time travel below partial_max → filter applies.
    at_s2 = pdl.scan_ducklake(backend_builder.url, "t", snapshot_id=snap2).sort("id").collect()
    assert at_s2["id"].to_list() == [1, 2]


def test_list_column(backend_builder: DuckLakeBuilder) -> None:
    """LIST<INTEGER> column reads back as ``pl.List(pl.Int32)``.

    DuckLake decomposes a list column into two ``ducklake_column`` rows:
    a parent with ``column_type='list'`` and a single child named
    ``element`` that carries the element type. The reader walks the
    parent_column linkage to assemble a recursive ``pl.DataType``.
    """
    snap = backend_builder.take_snapshot()
    schema_id = backend_builder.add_schema(name="main", begin_snapshot=snap)
    # column_id allocator: id=1, tags=2 (parent), element=3 (child of 2).
    table_id = backend_builder.add_table(
        schema_id=schema_id,
        name="t",
        begin_snapshot=snap,
        columns=[("id", "INTEGER"), ("tags", "list")],
        nested_columns=[("element", "int32", 2)],
    )
    # Explicit dtype on the inner list so the Parquet matches what the
    # catalog declares (``int32``); without it Polars defaults to Int64.
    backend_builder.write_data_file(
        table_id=table_id,
        begin_snapshot=snap,
        df=pl.DataFrame(
            {
                "id": [1, 2],
                "tags": pl.Series([[10, 20], [30]], dtype=pl.List(pl.Int32)),
            }
        ),
    )

    lf = pdl.scan_ducklake(backend_builder.url, "t")
    df = lf.sort("id").collect()
    assert df.schema["tags"] == pl.List(pl.Int32)
    assert df["tags"].to_list() == [[10, 20], [30]]


def test_struct_column(backend_builder: DuckLakeBuilder) -> None:
    """STRUCT(a INT32, b VARCHAR) — multi-child case."""
    snap = backend_builder.take_snapshot()
    schema_id = backend_builder.add_schema(name="main", begin_snapshot=snap)
    # column_id allocator: id=1, payload=2 (struct parent),
    # a=3, b=4 (children of 2).
    table_id = backend_builder.add_table(
        schema_id=schema_id,
        name="t",
        begin_snapshot=snap,
        columns=[("id", "INTEGER"), ("payload", "struct")],
        nested_columns=[
            ("a", "int32", 2),
            ("b", "varchar", 2),
        ],
    )
    backend_builder.write_data_file(
        table_id=table_id,
        begin_snapshot=snap,
        df=pl.DataFrame(
            {
                "id": [1, 2],
                "payload": pl.Series(
                    [{"a": 10, "b": "x"}, {"a": 20, "b": "y"}],
                    dtype=pl.Struct({"a": pl.Int32, "b": pl.String}),
                ),
            }
        ),
    )
    df = pdl.scan_ducklake(backend_builder.url, "t").sort("id").collect()
    assert df.schema["payload"] == pl.Struct({"a": pl.Int32, "b": pl.String})
    assert df["payload"].to_list() == [{"a": 10, "b": "x"}, {"a": 20, "b": "y"}]


def test_map_column(backend_builder: DuckLakeBuilder) -> None:
    """MAP<VARCHAR, INT32> — surfaced as ``List(Struct{key, value})``.

    DuckLake's catalog decomposes MAP into a parent with
    ``column_type='map'`` and two children named ``key`` and ``value``.
    Polars represents maps the same way Parquet does on disk:
    ``List(Struct{key, value})``. Our reader should produce that shape
    so users get back exactly what the DuckDB writer wrote.
    """
    snap = backend_builder.take_snapshot()
    schema_id = backend_builder.add_schema(name="main", begin_snapshot=snap)
    # column_id allocator: id=1, attrs=2 (parent map), key=3, value=4.
    table_id = backend_builder.add_table(
        schema_id=schema_id,
        name="t",
        begin_snapshot=snap,
        columns=[("id", "INTEGER"), ("attrs", "map")],
        nested_columns=[
            ("key", "varchar", 2),
            ("value", "int32", 2),
        ],
    )
    map_dtype = pl.List(pl.Struct({"key": pl.String, "value": pl.Int32}))
    backend_builder.write_data_file(
        table_id=table_id,
        begin_snapshot=snap,
        df=pl.DataFrame(
            {
                "id": [1, 2],
                "attrs": pl.Series(
                    [
                        [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
                        [{"key": "c", "value": 3}],
                    ],
                    dtype=map_dtype,
                ),
            }
        ),
    )
    df = pdl.scan_ducklake(backend_builder.url, "t").sort("id").collect()
    assert df.schema["attrs"] == map_dtype
    assert df["attrs"].to_list() == [
        [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
        [{"key": "c", "value": 3}],
    ]


def test_list_of_struct(backend_builder: DuckLakeBuilder) -> None:
    """LIST<STRUCT(a INT32, b VARCHAR)> — recursion correctness.

    Real-world event payloads often look like this — a list of objects
    each with a few scalar fields. Catalog rows are linked
    parent → element → fields, and our recursive type-tree walker has
    to chain through.
    """
    snap = backend_builder.take_snapshot()
    schema_id = backend_builder.add_schema(name="main", begin_snapshot=snap)
    # column_id allocator: id=1, events=2 (list parent),
    # element=3 (struct, child of 2),
    # a=4 (child of 3), b=5 (child of 3).
    table_id = backend_builder.add_table(
        schema_id=schema_id,
        name="t",
        begin_snapshot=snap,
        columns=[("id", "INTEGER"), ("events", "list")],
        nested_columns=[
            ("element", "struct", 2),
            ("a", "int32", 3),
            ("b", "varchar", 3),
        ],
    )
    inner = pl.Struct({"a": pl.Int32, "b": pl.String})
    nested_dtype = pl.List(inner)
    backend_builder.write_data_file(
        table_id=table_id,
        begin_snapshot=snap,
        df=pl.DataFrame(
            {
                "id": [1, 2],
                "events": pl.Series(
                    [
                        [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
                        [{"a": 3, "b": "z"}],
                    ],
                    dtype=nested_dtype,
                ),
            }
        ),
    )
    df = pdl.scan_ducklake(backend_builder.url, "t").sort("id").collect()
    assert df.schema["events"] == nested_dtype
    assert df["events"].to_list() == [
        [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
        [{"a": 3, "b": "z"}],
    ]


def test_empty_table(backend_builder: DuckLakeBuilder) -> None:
    """A table with zero data files returns a zero-row LazyFrame."""
    snap = backend_builder.take_snapshot()
    schema_id = backend_builder.add_schema(name="main", begin_snapshot=snap)
    backend_builder.add_table(
        schema_id=schema_id,
        name="empty_t",
        begin_snapshot=snap,
        columns=[("id", "INTEGER"), ("name", "VARCHAR")],
    )
    df = pdl.scan_ducklake(backend_builder.url, "empty_t").collect()
    assert df.height == 0
    assert df.columns == ["id", "name"]
