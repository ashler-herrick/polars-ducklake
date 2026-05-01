"""Tests for issue #5: one connection per scan phase, not three.

Verifies:

* :func:`_engine_from_metadata_catalog` memoises by URL across scans.
* :func:`scan_ducklake().collect()` opens exactly two SQLAlchemy
  ``Connection``\\ s — one for the eager identity resolution and one
  shared across the deferred plan/prune/file-list phase.

The ``engine_connect`` SQLAlchemy event fires on every ``engine.connect()``
call, so it's the right hook for counting checkouts (not ``connect``,
which fires only on new DBAPI connections).
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest
import sqlalchemy
from sqlalchemy import event

import polars_ducklake as pdl
from polars_ducklake._catalog import (
    _ENGINE_CACHE,
    _engine_from_metadata_catalog,
)


@pytest.fixture(autouse=True)
def _reset_engine_cache() -> Any:
    """Each test starts with a clean engine cache.

    Engines are kept for process lifetime in production; in tests we drop
    them between cases so independent tests don't share pools (and so
    the cache-hit count assertions are meaningful).
    """
    _ENGINE_CACHE.clear()
    yield
    for engine in list(_ENGINE_CACHE.values()):
        engine.dispose()
    _ENGINE_CACHE.clear()


def _count_connects(engine: sqlalchemy.engine.Engine) -> dict[str, int]:
    counts = {"n": 0}

    @event.listens_for(engine, "engine_connect")
    def _on_connect(_conn: Any) -> None:
        counts["n"] += 1

    return counts


class TestEngineCache:
    def test_same_url_returns_same_engine(self, builder: Any) -> None:
        e1 = _engine_from_metadata_catalog(builder.url)
        e2 = _engine_from_metadata_catalog(builder.url)
        assert e1 is e2

    def test_ducklake_prefix_normalises_to_same_entry(
        self, tmp_path: Any
    ) -> None:
        path = tmp_path / "lake.db"
        sa_url = f"sqlite:///{path}"
        ducklake_url = f"ducklake:sqlite:{path}"
        e1 = _engine_from_metadata_catalog(ducklake_url)
        e2 = _engine_from_metadata_catalog(sa_url)
        assert e1 is e2

    def test_prebuilt_engine_bypasses_cache(self) -> None:
        eng = sqlalchemy.create_engine("sqlite:///:memory:")
        assert _engine_from_metadata_catalog(eng) is eng
        # Pre-built engines are not interned: passing the same engine
        # twice still returns the caller's instance, and nothing was added
        # to the cache.
        assert _engine_from_metadata_catalog(eng) is eng
        assert eng not in _ENGINE_CACHE.values()

    def test_create_engine_called_once_per_url(
        self,
        empty_table_lake: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polars_ducklake import _catalog as catalog_mod

        builder = empty_table_lake["builder"]
        calls = {"n": 0}
        real = sqlalchemy.create_engine

        def counting(*args: Any, **kw: Any) -> Any:
            calls["n"] += 1
            return real(*args, **kw)

        monkeypatch.setattr(catalog_mod.sqlalchemy, "create_engine", counting)
        # Cache was cleared by the autouse fixture, so the first scan
        # builds an engine and the second hits the cache.
        pdl.scan_ducklake(builder.url, table="empty_t").collect()
        pdl.scan_ducklake(builder.url, table="empty_t").collect()
        assert calls["n"] == 1


class TestConnectionCount:
    """Issue #5: scans should open one connection per phase, not three."""

    def _scan_table(self, builder: Any) -> tuple[Any, dict[str, int]]:
        """Build an engine, attach a counter, and return both."""
        engine = _engine_from_metadata_catalog(builder.url)
        counts = _count_connects(engine)
        return engine, counts

    def test_collect_opens_two_connections(
        self, single_file_lake: dict[str, Any]
    ) -> None:
        builder = single_file_lake["builder"]
        engine, counts = self._scan_table(builder)
        df = pdl.scan_ducklake(engine, table="sales").collect()
        assert df.height == single_file_lake["row_count"]
        # One eager connection (identity resolution) + one deferred
        # connection (plan + file list).
        assert counts["n"] == 2

    def test_collect_with_predicate_still_two_connections(
        self, builder: Any
    ) -> None:
        # Build a predicate-prunable lake inline so the prune path runs.
        snap = builder.take_snapshot()
        schema_id = builder.add_schema(name="main", begin_snapshot=snap)
        table_id = builder.add_table(
            schema_id=schema_id,
            name="t",
            begin_snapshot=snap,
            columns=[("id", "INTEGER"), ("v", "VARCHAR")],
        )
        builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap,
            df=pl.DataFrame({"id": [1, 2, 3], "v": ["a", "b", "c"]}),
            compute_stats=True,
        )

        engine, counts = self._scan_table(builder)
        df = (
            pdl.scan_ducklake(engine, table="t")
            .filter(pl.col("id") == 2)
            .collect()
        )
        assert df["v"].to_list() == ["b"]
        # Prune now runs on the deferred reader instead of opening its
        # own — still 2 total.
        assert counts["n"] == 2

    def test_schema_only_opens_one_connection(
        self, single_file_lake: dict[str, Any]
    ) -> None:
        # Reading just the schema (no collect) only drives the eager
        # phase — column list is cached on _PlanContext, so the schema
        # callable answers without touching the catalog.
        builder = single_file_lake["builder"]
        engine, counts = self._scan_table(builder)
        lf = pdl.scan_ducklake(engine, table="sales")
        _ = lf.collect_schema()
        assert counts["n"] == 1

    def test_time_travel_two_connections(
        self, multi_snapshot_lake: dict[str, Any]
    ) -> None:
        builder = multi_snapshot_lake["builder"]
        engine, counts = self._scan_table(builder)
        df = (
            pdl.scan_ducklake(
                engine, table="rolling", snapshot_id=multi_snapshot_lake["snap2"]
            )
            .sort("id")
            .collect()
        )
        assert df["id"].to_list() == [1, 2, 3, 4]
        assert counts["n"] == 2


class TestDeferredQueryCount:
    """Issue #4: file enumeration is one CTE query, not three.

    Previously the deferred phase issued ``fetch_data_files`` +
    ``fetch_delete_files`` + (with predicate) ``fetch_file_stats``. The
    new path collapses these into one ``fetch_candidate_files`` CTE
    plus a separate ``fetch_partition_spec`` query.
    """

    @staticmethod
    def _file_query_counts(engine: sqlalchemy.engine.Engine) -> dict[str, int]:
        """Tally how often each ducklake_* table is read in this engine.

        Filters to ``ducklake_data_file`` / ``ducklake_delete_file`` /
        ``ducklake_file_column_stats`` / ``ducklake_partition_*`` so the
        per-file column lookups (``ducklake_column``) and the snapshot
        resolution don't muddle the counts.
        """
        counts: dict[str, int] = {
            "ducklake_data_file": 0,
            "ducklake_delete_file": 0,
            "ducklake_file_column_stats": 0,
            "ducklake_partition_info": 0,
            "ducklake_file_partition_value": 0,
        }

        @event.listens_for(engine, "before_execute")
        def _before(
            conn: Any,
            clauseelement: Any,
            multiparams: Any,
            params: Any,
            execution_options: Any,
        ) -> None:
            sql = str(clauseelement)
            for table in counts:
                if table in sql:
                    counts[table] += 1

        return counts

    def test_collect_with_predicate_runs_one_file_query(
        self, builder: Any
    ) -> None:
        # Build a stats-prunable lake.
        snap = builder.take_snapshot()
        schema_id = builder.add_schema(name="main", begin_snapshot=snap)
        table_id = builder.add_table(
            schema_id=schema_id,
            name="t",
            begin_snapshot=snap,
            columns=[("id", "INTEGER"), ("v", "VARCHAR")],
        )
        builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap,
            df=pl.DataFrame({"id": [1, 2, 3], "v": ["a", "b", "c"]}),
            compute_stats=True,
        )

        engine = _engine_from_metadata_catalog(builder.url)
        counts = self._file_query_counts(engine)
        df = (
            pdl.scan_ducklake(engine, table="t")
            .filter(pl.col("id") == 2)
            .collect()
        )
        assert df["v"].to_list() == ["b"]
        # The single CTE enumeration query references all three of these
        # tables (visible_files, visible_deletes, per-column stats CTE),
        # so each is mentioned once. Crucially: not three separate queries.
        assert counts["ducklake_data_file"] == 1
        assert counts["ducklake_delete_file"] == 1
        assert counts["ducklake_file_column_stats"] == 1

    def test_collect_without_predicate_does_not_query_stats(
        self, single_file_lake: dict[str, Any]
    ) -> None:
        # No predicate → no per-column stats CTE → file_column_stats
        # never appears in the file query.
        builder = single_file_lake["builder"]
        engine = _engine_from_metadata_catalog(builder.url)
        counts = self._file_query_counts(engine)
        df = pdl.scan_ducklake(engine, table="sales").collect()
        assert df.height == single_file_lake["row_count"]
        assert counts["ducklake_file_column_stats"] == 0
        assert counts["ducklake_data_file"] == 1
        assert counts["ducklake_delete_file"] == 1

    def test_partitioned_table_loads_spec_and_filters(
        self, builder: Any
    ) -> None:
        # The partition spec is a separate query (one round trip, cached
        # under issue #6 follow-up).
        snap = builder.take_snapshot()
        schema_id = builder.add_schema(name="main", begin_snapshot=snap)
        table_id = builder.add_table(
            schema_id=schema_id,
            name="t",
            begin_snapshot=snap,
            columns=[("id", "INTEGER"), ("region", "VARCHAR")],
        )
        builder.add_partition(
            table_id=table_id, begin_snapshot=snap, columns=[(2, "identity")]
        )
        builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap,
            df=pl.DataFrame({"id": [1, 2], "region": ["us", "us"]}),
            partition_values={0: "us"},
        )
        builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap,
            df=pl.DataFrame({"id": [3, 4], "region": ["eu", "eu"]}),
            partition_values={0: "eu"},
        )
        engine = _engine_from_metadata_catalog(builder.url)
        counts = self._file_query_counts(engine)
        df = (
            pdl.scan_ducklake(engine, table="t")
            .filter(pl.col("region") == "eu")
            .sort("id")
            .collect()
        )
        assert df["id"].to_list() == [3, 4]
        # Partition spec query is one round trip; the file query CTE
        # references partition_value once (the part_X_filter CTE).
        assert counts["ducklake_partition_info"] == 1
        assert counts["ducklake_file_partition_value"] == 1
