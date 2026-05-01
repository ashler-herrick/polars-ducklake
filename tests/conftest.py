"""Test fixtures for polars-ducklake.

Test DuckLakes are constructed *by hand* against the v1.0 spec — we
deliberately do not use the DuckDB ducklake extension to generate fixtures,
since the whole point of this package is to be independent of it. Building
fixtures directly from the spec is the most rigorous validation that our
reader matches the spec.

The :class:`DuckLakeBuilder` helper handles the SQL boilerplate. Each test
fixture below builds a focused scenario (single file, multi-snapshot,
deletes, etc.) and yields the catalog URL plus relevant metadata.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import sqlalchemy
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine


def _stat_string(value: Any) -> str:
    """Stringify a Python value the way DuckLake's stats column expects.

    Mirrors what DuckDB's ducklake writer emits: integers/floats as
    decimal, strings verbatim, booleans as ``"0"``/``"1"``, dates as
    ISO ``YYYY-MM-DD``, naive datetimes as ``YYYY-MM-DD HH:MM:SS[.ffffff]``.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        # Naive only — tests don't currently exercise tz-aware stats.
        return value.isoformat(sep=" ", timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    raise ValueError(f"don't know how to stringify {type(value).__name__}")


class DuckLakeBuilder:
    """Programmatic builder for a DuckLake test fixture on any catalog backend.

    The builder owns a Parquet ``data_path`` (always local, on tmp_path)
    and a metadata catalog (SQLite by default, any SQLAlchemy-supported
    backend via ``sqlalchemy_url=``). Schema/table/column/file ids and
    snapshot ids are integer counters managed by the builder — a stripped
    down replica of what real DuckLake writers do.

    All catalog SQL goes through SQLAlchemy ``text()`` with named bind
    parameters so the same code targets every backend without per-dialect
    branches.
    """

    def __init__(self, root: Path, *, sqlalchemy_url: str | None = None) -> None:
        self.root = root
        self.data_dir = root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Default to a SQLite file inside ``root``; callers can target
        # any other backend by passing a SQLAlchemy URL.
        self.catalog_path = root / "metadata.db"
        self._url = sqlalchemy_url or f"sqlite:///{self.catalog_path}"
        self._engine: Engine = sqlalchemy.create_engine(self._url)
        self._snapshot_counter = 0
        self._schema_counter = 0
        self._table_counter = 0
        self._column_counter = 0
        self._data_file_counter = 0
        self._delete_file_counter = 0
        self._partition_counter = 0
        # Track each data file's resolved on-disk path so add_delete_file
        # can populate the spec-required (file_path, pos) columns.
        self._data_file_paths: dict[int, Path] = {}
        self._init_catalog()

    @property
    def url(self) -> str:
        """SQLAlchemy URL pointing at the catalog."""
        return self._url

    @property
    def data_path(self) -> str:
        """Lake-wide ``data_path`` value (with trailing slash)."""
        return f"{self.data_dir}/"

    @contextmanager
    def _conn(self) -> Iterator[Connection]:
        """Open a transactional SQLAlchemy connection.

        Uses ``engine.begin()`` so DDL/DML auto-commits at block exit
        across every backend (Postgres especially needs the explicit
        commit; the previous sqlite3 path used implicit commits).
        """
        with self._engine.begin() as conn:
            yield conn

    def _quote(self, identifier: str) -> str:
        """Dialect-aware identifier quoting.

        SQLAlchemy's preparer knows that, e.g., ``key`` is a reserved word
        in MySQL (needs backticks) but not in PostgreSQL (no quoting
        needed). Using it keeps the DDL portable across every backend.
        """
        return self._engine.dialect.identifier_preparer.quote(identifier)

    def cleanup(self) -> None:
        """Drop every ``ducklake_*`` table and dispose the engine.

        Idempotent: skips tables that don't exist. Used by parametrised
        tests so a non-SQLite backend doesn't leak state between runs.
        SQLite catalogs are file-scoped so this is a no-op in practice
        for them, but harmless.
        """
        try:
            with self._conn() as conn:
                inspector = sqlalchemy.inspect(conn)
                # Sort descending so dependent tables go before their
                # parents (none of ours have FKs but be safe).
                tables = sorted(
                    (t for t in inspector.get_table_names() if t.startswith("ducklake_")),
                    reverse=True,
                )
                for t in tables:
                    # Quote the identifier per-dialect (MySQL uses backticks,
                    # PG/DuckDB use double quotes); ``ducklake_*`` aren't
                    # reserved but cleanup also needs to handle the optional
                    # ``ducklake_inlined_data_tables`` etc. that some tests
                    # create.
                    conn.execute(text(f"DROP TABLE IF EXISTS {self._quote(t)}"))
        finally:
            self._engine.dispose()

    def _init_catalog(self) -> None:
        ddl = [
            """
            CREATE TABLE ducklake_snapshot (
                snapshot_id BIGINT PRIMARY KEY,
                snapshot_time TIMESTAMP NOT NULL,
                schema_version BIGINT NOT NULL,
                next_catalog_id BIGINT NOT NULL,
                next_file_id BIGINT NOT NULL
            )
            """,
            """
            CREATE TABLE ducklake_schema (
                schema_id BIGINT NOT NULL,
                schema_uuid VARCHAR(64),
                begin_snapshot BIGINT NOT NULL,
                end_snapshot BIGINT,
                schema_name VARCHAR(255) NOT NULL,
                path VARCHAR(1024),
                path_is_relative BOOLEAN NOT NULL DEFAULT TRUE
            )
            """,
            """
            CREATE TABLE ducklake_table (
                table_id BIGINT NOT NULL,
                table_uuid VARCHAR(64),
                begin_snapshot BIGINT NOT NULL,
                end_snapshot BIGINT,
                schema_id BIGINT NOT NULL,
                table_name VARCHAR(255) NOT NULL,
                path VARCHAR(1024),
                path_is_relative BOOLEAN NOT NULL DEFAULT TRUE
            )
            """,
            """
            CREATE TABLE ducklake_column (
                column_id BIGINT NOT NULL,
                begin_snapshot BIGINT NOT NULL,
                end_snapshot BIGINT,
                table_id BIGINT NOT NULL,
                column_order BIGINT NOT NULL,
                column_name VARCHAR(255) NOT NULL,
                column_type VARCHAR(255) NOT NULL,
                initial_default VARCHAR(1024),
                default_value VARCHAR(1024),
                nulls_allowed BOOLEAN NOT NULL DEFAULT TRUE,
                parent_column BIGINT,
                default_value_type VARCHAR(255),
                default_value_dialect VARCHAR(255)
            )
            """,
            """
            CREATE TABLE ducklake_data_file (
                data_file_id BIGINT NOT NULL,
                table_id BIGINT NOT NULL,
                begin_snapshot BIGINT NOT NULL,
                end_snapshot BIGINT,
                file_order BIGINT NOT NULL,
                path VARCHAR(1024) NOT NULL,
                path_is_relative BOOLEAN NOT NULL DEFAULT TRUE,
                file_format VARCHAR(32) NOT NULL DEFAULT 'parquet',
                record_count BIGINT NOT NULL,
                file_size_bytes BIGINT,
                footer_size BIGINT,
                row_id_start BIGINT,
                partition_id BIGINT,
                encryption_key VARCHAR(255),
                mapping_id BIGINT,
                partial_max BIGINT
            )
            """,
            """
            CREATE TABLE ducklake_delete_file (
                delete_file_id BIGINT NOT NULL,
                table_id BIGINT NOT NULL,
                begin_snapshot BIGINT NOT NULL,
                end_snapshot BIGINT,
                data_file_id BIGINT NOT NULL,
                path VARCHAR(1024) NOT NULL,
                path_is_relative BOOLEAN NOT NULL DEFAULT TRUE,
                format VARCHAR(32) NOT NULL DEFAULT 'parquet',
                delete_count BIGINT NOT NULL,
                file_size_bytes BIGINT,
                footer_size BIGINT,
                encryption_key VARCHAR(255),
                partial_max BIGINT
            )
            """,
            """
            CREATE TABLE ducklake_file_column_stats (
                data_file_id BIGINT NOT NULL,
                table_id BIGINT NOT NULL,
                column_id BIGINT NOT NULL,
                column_size_bytes BIGINT,
                value_count BIGINT,
                null_count BIGINT,
                min_value VARCHAR(1024),
                max_value VARCHAR(1024),
                contains_nan BIGINT,
                extra_stats VARCHAR(1024)
            )
            """,
            # Per the DuckLake spec, partition info is split into a
            # per-table partition_info row (one per active spec version)
            # and partition_column rows for each key in the spec. Files
            # carry their per-key values in file_partition_value.
            """
            CREATE TABLE ducklake_partition_info (
                partition_id BIGINT NOT NULL,
                table_id BIGINT NOT NULL,
                begin_snapshot BIGINT NOT NULL,
                end_snapshot BIGINT
            )
            """,
            """
            CREATE TABLE ducklake_partition_column (
                partition_id BIGINT NOT NULL,
                partition_key_index BIGINT NOT NULL,
                column_id BIGINT NOT NULL,
                transform VARCHAR(64) NOT NULL
            )
            """,
            """
            CREATE TABLE ducklake_file_partition_value (
                data_file_id BIGINT NOT NULL,
                table_id BIGINT NOT NULL,
                partition_key_index BIGINT NOT NULL,
                partition_value VARCHAR(1024)
            )
            """,
            # ``key`` is reserved in MySQL — quote it per-dialect.
            f"""
            CREATE TABLE ducklake_metadata (
                {self._quote("key")} VARCHAR(255) NOT NULL,
                value VARCHAR(1024) NOT NULL,
                scope VARCHAR(64),
                scope_id BIGINT
            )
            """,
        ]
        with self._conn() as conn:
            for stmt in ddl:
                conn.execute(text(stmt))
            conn.execute(
                text(
                    f"INSERT INTO ducklake_metadata ({self._quote('key')}, value, scope, scope_id) "
                    "VALUES (:key, :value, NULL, NULL)"
                ),
                {"key": "data_path", "value": self.data_path},
            )

    def take_snapshot(self, *, snapshot_time: datetime | None = None) -> int:
        """Create the next snapshot row and return its id."""
        self._snapshot_counter += 1
        snapshot_id = self._snapshot_counter
        ts = snapshot_time or datetime(2026, 1, 1, tzinfo=timezone.utc).replace(
            day=min(28, snapshot_id), tzinfo=None
        )
        # Pass naive datetime to SQLAlchemy for portability across backends.
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        with self._conn() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO ducklake_snapshot
                        (snapshot_id, snapshot_time, schema_version, next_catalog_id, next_file_id)
                    VALUES (:sid, :ts, :sv, :nci, :nfi)
                    """
                ),
                {
                    "sid": snapshot_id,
                    "ts": ts,
                    "sv": snapshot_id,
                    "nci": snapshot_id * 100,
                    "nfi": snapshot_id * 100,
                },
            )
        return snapshot_id

    def add_schema(self, *, name: str, begin_snapshot: int) -> int:
        self._schema_counter += 1
        schema_id = self._schema_counter
        with self._conn() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO ducklake_schema
                        (schema_id, schema_uuid, begin_snapshot, end_snapshot,
                         schema_name, path, path_is_relative)
                    VALUES (:sid, :uuid, :begin, NULL, :name, :path, :rel)
                    """
                ),
                {
                    "sid": schema_id,
                    "uuid": f"schema-{schema_id}",
                    "begin": begin_snapshot,
                    "name": name,
                    "path": "",
                    "rel": True,
                },
            )
        return schema_id

    def add_table(
        self,
        *,
        schema_id: int,
        name: str,
        begin_snapshot: int,
        columns: list[tuple[str, str]],
        nested_columns: list[tuple[str, str, int]] | None = None,
    ) -> int:
        """Add a table and its columns.

        ``columns`` is a list of ``(column_name, column_type)`` tuples for
        top-level columns. ``nested_columns`` is an optional list of
        ``(column_name, column_type, parent_column_id)`` tuples used by
        the nested-type-refusal test.
        """
        self._table_counter += 1
        table_id = self._table_counter
        with self._conn() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO ducklake_table
                        (table_id, table_uuid, begin_snapshot, end_snapshot,
                         schema_id, table_name, path, path_is_relative)
                    VALUES (:tid, :uuid, :begin, NULL, :sid, :name, :path, :rel)
                    """
                ),
                {
                    "tid": table_id,
                    "uuid": f"table-{table_id}",
                    "begin": begin_snapshot,
                    "sid": schema_id,
                    "name": name,
                    "path": "",
                    "rel": True,
                },
            )
            for order, (col_name, col_type) in enumerate(columns):
                self._column_counter += 1
                conn.execute(
                    text(
                        """
                        INSERT INTO ducklake_column
                            (column_id, begin_snapshot, end_snapshot, table_id,
                             column_order, column_name, column_type, parent_column)
                        VALUES (:cid, :begin, NULL, :tid, :ord, :name, :type, NULL)
                        """
                    ),
                    {
                        "cid": self._column_counter,
                        "begin": begin_snapshot,
                        "tid": table_id,
                        "ord": order,
                        "name": col_name,
                        "type": col_type,
                    },
                )
            for col_name, col_type, parent_id in nested_columns or []:
                self._column_counter += 1
                conn.execute(
                    text(
                        """
                        INSERT INTO ducklake_column
                            (column_id, begin_snapshot, end_snapshot, table_id,
                             column_order, column_name, column_type, parent_column)
                        VALUES (:cid, :begin, NULL, :tid, :ord, :name, :type, :parent)
                        """
                    ),
                    {
                        "cid": self._column_counter,
                        "begin": begin_snapshot,
                        "tid": table_id,
                        "ord": 999,
                        "name": col_name,
                        "type": col_type,
                        "parent": parent_id,
                    },
                )
        return table_id

    def add_partition(
        self,
        *,
        table_id: int,
        begin_snapshot: int,
        columns: list[tuple[int, str]],
    ) -> int:
        """Register an identity (or transformed) partition spec for a table.

        ``columns`` is a list of ``(column_id, transform)`` tuples in
        partition_key_index order. ``transform`` is a string like
        ``"identity"``, ``"year"``, etc.; the reader/builder only push
        identity-transform clauses today.
        """
        self._partition_counter += 1
        partition_id = self._partition_counter
        with self._conn() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO ducklake_partition_info
                        (partition_id, table_id, begin_snapshot, end_snapshot)
                    VALUES (:pid, :tid, :begin, NULL)
                    """
                ),
                {"pid": partition_id, "tid": table_id, "begin": begin_snapshot},
            )
            for idx, (column_id, transform) in enumerate(columns):
                conn.execute(
                    text(
                        """
                        INSERT INTO ducklake_partition_column
                            (partition_id, partition_key_index, column_id, transform)
                        VALUES (:pid, :idx, :cid, :tr)
                        """
                    ),
                    {
                        "pid": partition_id,
                        "idx": idx,
                        "cid": column_id,
                        "tr": transform,
                    },
                )
        return partition_id

    def write_data_file(
        self,
        *,
        table_id: int,
        begin_snapshot: int,
        df: pl.DataFrame,
        filename: str | None = None,
        path_is_relative: bool = True,
        partial_max: int | None = None,
        compute_stats: bool = False,
        partition_values: dict[int, Any] | None = None,
    ) -> int:
        """Write a Parquet data file and register it in ducklake_data_file.

        ``partial_max`` simulates a compaction-merged file. When set, the
        DataFrame must already include a ``_ducklake_internal_snapshot_id``
        column carrying the per-row origin snapshot id.

        ``compute_stats=True`` populates ``ducklake_file_column_stats``
        with min/max/null counts derived from ``df``. Used for predicate-
        pruning tests; left off by default so existing fixtures stay
        focused on whatever they're testing.

        ``partition_values`` maps ``partition_key_index → Python value``
        (stringified per the writer's stat-string rules). Required for
        files in a partitioned table; ignored otherwise.
        """
        self._data_file_counter += 1
        data_file_id = self._data_file_counter
        rel_path = filename or f"part-{data_file_id:04d}.parquet"
        full_path = self.data_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(full_path)
        self._data_file_paths[data_file_id] = full_path
        with self._conn() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO ducklake_data_file
                        (data_file_id, table_id, begin_snapshot, end_snapshot,
                         file_order, path, path_is_relative, file_format,
                         record_count, file_size_bytes, partial_max)
                    VALUES (:dfid, :tid, :begin, NULL, :ord, :path, :rel,
                            'parquet', :rc, :size, :pmax)
                    """
                ),
                {
                    "dfid": data_file_id,
                    "tid": table_id,
                    "begin": begin_snapshot,
                    "ord": data_file_id,
                    "path": rel_path if path_is_relative else str(full_path),
                    "rel": path_is_relative,
                    "rc": df.height,
                    "size": full_path.stat().st_size,
                    "pmax": partial_max,
                },
            )
        if compute_stats:
            self._populate_file_column_stats(
                data_file_id=data_file_id, table_id=table_id, df=df
            )
        if partition_values:
            with self._conn() as conn:
                for key_idx, value in partition_values.items():
                    conn.execute(
                        text(
                            """
                            INSERT INTO ducklake_file_partition_value
                                (data_file_id, table_id,
                                 partition_key_index, partition_value)
                            VALUES (:dfid, :tid, :idx, :val)
                            """
                        ),
                        {
                            "dfid": data_file_id,
                            "tid": table_id,
                            "idx": key_idx,
                            "val": _stat_string(value) if value is not None else None,
                        },
                    )
        return data_file_id

    def _populate_file_column_stats(
        self, *, data_file_id: int, table_id: int, df: pl.DataFrame
    ) -> None:
        """Insert per-column min/max into ducklake_file_column_stats.

        Looks up the catalog column_id for each DataFrame column; columns
        the catalog doesn't know about (e.g. _ducklake_internal_snapshot_id
        on partial files) are skipped silently.
        """
        with self._conn() as conn:
            cols = conn.execute(
                text(
                    "SELECT column_id, column_name FROM ducklake_column "
                    "WHERE table_id = :tid AND parent_column IS NULL"
                ),
                {"tid": table_id},
            ).all()
        column_id_by_name = {row[1]: row[0] for row in cols}
        for name in df.columns:
            column_id = column_id_by_name.get(name)
            if column_id is None:
                continue
            col = df[name]
            non_null = col.drop_nulls()
            null_count = int(col.null_count())
            value_count = int(col.len())
            if non_null.len() == 0:
                min_v: str | None = None
                max_v: str | None = None
            else:
                min_v = _stat_string(non_null.min())
                max_v = _stat_string(non_null.max())
            with self._conn() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO ducklake_file_column_stats
                            (data_file_id, table_id, column_id,
                             value_count, null_count, min_value, max_value,
                             contains_nan)
                        VALUES (:dfid, :tid, :cid,
                                :vc, :nc, :mn, :mx, NULL)
                        """
                    ),
                    {
                        "dfid": data_file_id,
                        "tid": table_id,
                        "cid": column_id,
                        "vc": value_count,
                        "nc": null_count,
                        "mn": min_v,
                        "mx": max_v,
                    },
                )

    def expire_data_file(self, *, data_file_id: int, end_snapshot: int) -> None:
        """Mark a data file as no longer visible at and after ``end_snapshot``."""
        with self._conn() as conn:
            conn.execute(
                text(
                    "UPDATE ducklake_data_file SET end_snapshot = :end WHERE data_file_id = :dfid"
                ),
                {"end": end_snapshot, "dfid": data_file_id},
            )

    def add_column(
        self,
        *,
        table_id: int,
        begin_snapshot: int,
        column_name: str,
        column_type: str,
        column_order: int,
    ) -> int:
        """ALTER TABLE ADD COLUMN: register a new column starting at this snapshot."""
        self._column_counter += 1
        column_id = self._column_counter
        with self._conn() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO ducklake_column
                        (column_id, begin_snapshot, end_snapshot, table_id, column_order,
                         column_name, column_type, parent_column)
                    VALUES (:cid, :begin, NULL, :tid, :ord, :name, :type, NULL)
                    """
                ),
                {
                    "cid": column_id,
                    "begin": begin_snapshot,
                    "tid": table_id,
                    "ord": column_order,
                    "name": column_name,
                    "type": column_type,
                },
            )
        return column_id

    def drop_column(self, *, column_id: int, end_snapshot: int) -> None:
        """ALTER TABLE DROP COLUMN: close the active row at this snapshot."""
        with self._conn() as conn:
            conn.execute(
                text(
                    "UPDATE ducklake_column SET end_snapshot = :end "
                    "WHERE column_id = :cid AND end_snapshot IS NULL"
                ),
                {"end": end_snapshot, "cid": column_id},
            )

    def rename_column(
        self, *, column_id: int, table_id: int, at_snapshot: int, new_name: str
    ) -> None:
        """ALTER TABLE RENAME COLUMN: close the prior row, open a new row with same column_id.

        DuckLake encodes a rename as two ``ducklake_column`` rows that
        share the column_id but have different (column_name, begin_snapshot)
        — the older row gets ``end_snapshot = at_snapshot``, the new row
        starts at ``begin_snapshot = at_snapshot`` with the new name.
        """
        with self._conn() as conn:
            old = conn.execute(
                text(
                    """
                    SELECT column_type, column_order
                    FROM ducklake_column
                    WHERE column_id = :cid AND end_snapshot IS NULL
                    """
                ),
                {"cid": column_id},
            ).fetchone()
            if old is None:
                raise ValueError(f"No active column row for column_id={column_id}")
            column_type, column_order = old
            conn.execute(
                text(
                    "UPDATE ducklake_column SET end_snapshot = :end "
                    "WHERE column_id = :cid AND end_snapshot IS NULL"
                ),
                {"end": at_snapshot, "cid": column_id},
            )
            conn.execute(
                text(
                    """
                    INSERT INTO ducklake_column
                        (column_id, begin_snapshot, end_snapshot, table_id, column_order,
                         column_name, column_type, parent_column)
                    VALUES (:cid, :begin, NULL, :tid, :ord, :name, :type, NULL)
                    """
                ),
                {
                    "cid": column_id,
                    "begin": at_snapshot,
                    "tid": table_id,
                    "ord": column_order,
                    "name": new_name,
                    "type": column_type,
                },
            )

    def add_delete_file(
        self,
        *,
        table_id: int,
        begin_snapshot: int,
        data_file_id: int,
        positions: list[int],
        rel_path: str | None = None,
        write_parquet: bool = True,
    ) -> int:
        """Write a positional delete file and register it in the catalog.

        Mirrors the format the DuckDB writer emits: a Parquet with schema
        ``(file_path: String, pos: Int64)`` where each row marks a row
        position to drop from the referenced data file.

        ``write_parquet=False`` skips the on-disk write — only useful for
        tests of catalog-level behavior (e.g., MVCC visibility) that don't
        actually scan.
        """
        self._delete_file_counter += 1
        delete_file_id = self._delete_file_counter
        path = rel_path or f"deletes/del-{delete_file_id:04d}.parquet"
        full_path = self.data_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if write_parquet:
            referenced = self._data_file_paths.get(data_file_id)
            if referenced is None:
                raise ValueError(
                    f"data_file_id={data_file_id} has no recorded on-disk path; "
                    "call write_data_file before add_delete_file."
                )
            delete_df = pl.DataFrame(
                {
                    "file_path": [str(referenced)] * len(positions),
                    "pos": pl.Series(positions, dtype=pl.Int64()),
                }
            )
            delete_df.write_parquet(full_path)

        with self._conn() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO ducklake_delete_file
                        (delete_file_id, table_id, begin_snapshot, end_snapshot,
                         data_file_id, path, path_is_relative, format, delete_count)
                    VALUES (:dfid, :tid, :begin, NULL, :data_id, :path, :rel,
                            'parquet', :cnt)
                    """
                ),
                {
                    "dfid": delete_file_id,
                    "tid": table_id,
                    "begin": begin_snapshot,
                    "data_id": data_file_id,
                    "path": path,
                    "rel": True,
                    "cnt": len(positions),
                },
            )
        return delete_file_id

    def add_inlined_data_table(
        self, *, table_id: int, schema_version: int, populated: bool = True
    ) -> str:
        """Register a tracker in ``ducklake_inlined_data_tables`` and create
        the per-(table_id, schema_version) tracker table itself.

        When ``populated`` is True, the tracker is given a placeholder row
        — modelling the pre-flush state. When False, the tracker is created
        but left empty, modelling the state DuckDB leaves behind after
        ``ducklake_flush_inlined_data``: registry row intact, tracker empty.
        """
        tracker_name = f"ducklake_inlined_data_{table_id}_{schema_version}"
        with self._conn() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS ducklake_inlined_data_tables (
                        table_id BIGINT NOT NULL,
                        table_name VARCHAR(255) NOT NULL,
                        schema_version BIGINT NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(
                    "INSERT INTO ducklake_inlined_data_tables "
                    "(table_id, table_name, schema_version) "
                    "VALUES (:tid, :name, :sv)"
                ),
                {"tid": table_id, "name": tracker_name, "sv": schema_version},
            )
            conn.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS {self._quote(tracker_name)} "
                    "(row_id BIGINT)"
                )
            )
            if populated:
                conn.execute(
                    text(f"INSERT INTO {self._quote(tracker_name)} (row_id) VALUES (1)")
                )
        return tracker_name


@pytest.fixture()
def builder(tmp_path: Path) -> DuckLakeBuilder:
    """A fresh DuckLake on disk for the test."""
    return DuckLakeBuilder(tmp_path)


@pytest.fixture()
def empty_table_lake(builder: DuckLakeBuilder) -> dict[str, Any]:
    """A lake with one snapshot and a table that has zero data files."""
    snap = builder.take_snapshot()
    schema_id = builder.add_schema(name="main", begin_snapshot=snap)
    table_id = builder.add_table(
        schema_id=schema_id,
        name="empty_t",
        begin_snapshot=snap,
        columns=[("id", "INTEGER"), ("name", "VARCHAR")],
    )
    return {"builder": builder, "schema_id": schema_id, "table_id": table_id}


@pytest.fixture()
def single_file_lake(builder: DuckLakeBuilder) -> dict[str, Any]:
    """A lake with one snapshot, one table, one Parquet data file."""
    snap = builder.take_snapshot()
    schema_id = builder.add_schema(name="main", begin_snapshot=snap)
    table_id = builder.add_table(
        schema_id=schema_id,
        name="sales",
        begin_snapshot=snap,
        columns=[("id", "INTEGER"), ("amount", "DOUBLE"), ("region", "VARCHAR")],
    )
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "amount": [10.0, 20.5, 30.25, 40.0, 55.5],
            "region": ["us", "us", "eu", "eu", "apac"],
        }
    )
    builder.write_data_file(table_id=table_id, begin_snapshot=snap, df=df)
    return {"builder": builder, "table_id": table_id, "row_count": df.height}


@pytest.fixture()
def multi_file_lake(builder: DuckLakeBuilder) -> dict[str, Any]:
    """A lake with one snapshot and several Parquet files of varying size."""
    snap = builder.take_snapshot()
    schema_id = builder.add_schema(name="main", begin_snapshot=snap)
    table_id = builder.add_table(
        schema_id=schema_id,
        name="events",
        begin_snapshot=snap,
        columns=[("id", "BIGINT"), ("kind", "VARCHAR")],
    )
    parts = [
        pl.DataFrame({"id": [1, 2, 3], "kind": ["a", "b", "a"]}),
        pl.DataFrame({"id": [4, 5], "kind": ["c", "a"]}),
        pl.DataFrame({"id": [6, 7, 8, 9], "kind": ["b", "b", "c", "a"]}),
    ]
    for p in parts:
        builder.write_data_file(table_id=table_id, begin_snapshot=snap, df=p)
    return {
        "builder": builder,
        "table_id": table_id,
        "row_count": sum(p.height for p in parts),
    }


@pytest.fixture()
def multi_snapshot_lake(builder: DuckLakeBuilder) -> dict[str, Any]:
    """A lake with three snapshots, each appending a new data file.

    Snapshot 1: rows 1-2.
    Snapshot 2: rows 1-2 + 3-4 (file from snap 1 still visible).
    Snapshot 3: rows 1-2 + 3-4 + 5-6.
    """
    snap1 = builder.take_snapshot(snapshot_time=datetime(2026, 1, 1))
    schema_id = builder.add_schema(name="main", begin_snapshot=snap1)
    table_id = builder.add_table(
        schema_id=schema_id,
        name="rolling",
        begin_snapshot=snap1,
        columns=[("id", "INTEGER")],
    )
    builder.write_data_file(
        table_id=table_id,
        begin_snapshot=snap1,
        df=pl.DataFrame({"id": [1, 2]}),
    )

    snap2 = builder.take_snapshot(snapshot_time=datetime(2026, 2, 1))
    builder.write_data_file(
        table_id=table_id,
        begin_snapshot=snap2,
        df=pl.DataFrame({"id": [3, 4]}),
    )

    snap3 = builder.take_snapshot(snapshot_time=datetime(2026, 3, 1))
    builder.write_data_file(
        table_id=table_id,
        begin_snapshot=snap3,
        df=pl.DataFrame({"id": [5, 6]}),
    )

    return {
        "builder": builder,
        "table_id": table_id,
        "snap1": snap1,
        "snap2": snap2,
        "snap3": snap3,
    }
