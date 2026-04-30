"""Tests for :class:`polars_ducklake._catalog.CatalogReader`.

The reader is the package's only catalog-access surface (private —
not re-exported from ``polars_ducklake``). It runs against any SQLAlchemy
URL; these tests use SQLite (no docker needed). The cross-backend
matrix in ``tests/test_backends_matrix.py`` exercises the same query
surface against Postgres, MySQL, and DuckDB-as-catalog.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl
import pytest

from polars_ducklake._catalog import CatalogReader


def _reader(builder: Any) -> CatalogReader:
    return CatalogReader.from_metadata_catalog(builder.url)


class TestSnapshotResolution:
    def test_latest_snapshot(self, multi_snapshot_lake: dict[str, Any]) -> None:
        with _reader(multi_snapshot_lake["builder"]) as r:
            assert r.resolve_snapshot_id(snapshot_id=None, as_of=None) == 3

    def test_explicit_snapshot_id(self, multi_snapshot_lake: dict[str, Any]) -> None:
        with _reader(multi_snapshot_lake["builder"]) as r:
            assert r.resolve_snapshot_id(snapshot_id=2, as_of=None) == 2

    def test_invalid_snapshot_id(self, multi_snapshot_lake: dict[str, Any]) -> None:
        with (
            _reader(multi_snapshot_lake["builder"]) as r,
            pytest.raises(LookupError, match="snapshot_id=99"),
        ):
            r.resolve_snapshot_id(snapshot_id=99, as_of=None)

    def test_as_of_resolves_to_latest_at_or_before(
        self, multi_snapshot_lake: dict[str, Any]
    ) -> None:
        with _reader(multi_snapshot_lake["builder"]) as r:
            # snap1 = 2026-01-01, snap2 = 2026-02-01, snap3 = 2026-03-01
            assert r.resolve_snapshot_id(snapshot_id=None, as_of=datetime(2026, 2, 15)) == 2
            assert r.resolve_snapshot_id(snapshot_id=None, as_of=datetime(2026, 1, 1)) == 1

    def test_as_of_before_first_snapshot(self, multi_snapshot_lake: dict[str, Any]) -> None:
        with (
            _reader(multi_snapshot_lake["builder"]) as r,
            pytest.raises(LookupError, match="as_of"),
        ):
            r.resolve_snapshot_id(snapshot_id=None, as_of=datetime(2025, 1, 1))

    def test_conflicting_args(self, multi_snapshot_lake: dict[str, Any]) -> None:
        with (
            _reader(multi_snapshot_lake["builder"]) as r,
            pytest.raises(ValueError, match="mutually exclusive"),
        ):
            r.resolve_snapshot_id(snapshot_id=1, as_of=datetime(2026, 1, 1))


class TestSchemaAndTableResolution:
    def test_schema_lookup(self, single_file_lake: dict[str, Any]) -> None:
        with _reader(single_file_lake["builder"]) as r:
            info = r.resolve_schema(schema_name="main", snapshot_id=1)
        assert info.schema_id == 1
        assert info.path == ""
        assert info.path_is_relative is True

    def test_unknown_schema(self, single_file_lake: dict[str, Any]) -> None:
        with (
            _reader(single_file_lake["builder"]) as r,
            pytest.raises(LookupError, match="missing"),
        ):
            r.resolve_schema(schema_name="missing", snapshot_id=1)

    def test_table_lookup(self, single_file_lake: dict[str, Any]) -> None:
        with _reader(single_file_lake["builder"]) as r:
            info = r.resolve_table(schema_id=1, table_name="sales", snapshot_id=1)
        assert info.table_id == 1
        assert info.path == ""

    def test_unknown_table(self, single_file_lake: dict[str, Any]) -> None:
        with (
            _reader(single_file_lake["builder"]) as r,
            pytest.raises(LookupError, match="absent"),
        ):
            r.resolve_table(schema_id=1, table_name="absent", snapshot_id=1)


class TestColumnSchemaAndDataFiles:
    def test_columns_returned_in_order(self, single_file_lake: dict[str, Any]) -> None:
        with _reader(single_file_lake["builder"]) as r:
            cols = r.fetch_columns(table_id=1, snapshot_id=1)
        assert [c.column_name for c in cols] == ["id", "amount", "region"]
        assert [c.column_type for c in cols] == ["INTEGER", "DOUBLE", "VARCHAR"]

    def test_data_files_listed_in_file_order(self, multi_file_lake: dict[str, Any]) -> None:
        builder = multi_file_lake["builder"]
        with _reader(builder) as r:
            files = r.fetch_data_files(table_id=1, snapshot_id=1)
        assert len(files) == 3
        assert all(f.path_is_relative for f in files)
        assert sum(f.record_count for f in files) == multi_file_lake["row_count"]

    def test_no_delete_files(self, single_file_lake: dict[str, Any]) -> None:
        with _reader(single_file_lake["builder"]) as r:
            assert r.fetch_delete_files(table_id=1, snapshot_id=1) == []

    def test_data_path_read(self, single_file_lake: dict[str, Any]) -> None:
        builder = single_file_lake["builder"]
        with _reader(builder) as r:
            assert r.fetch_data_path() == builder.data_path

    def test_inlined_data_table_absent(self, single_file_lake: dict[str, Any]) -> None:
        with _reader(single_file_lake["builder"]) as r:
            assert r.has_inlined_data(table_id=1) is False


class TestMVCCVisibility:
    """The MVCC clause is applied correctly across snapshots."""

    def test_files_visible_at_each_snapshot(self, multi_snapshot_lake: dict[str, Any]) -> None:
        with _reader(multi_snapshot_lake["builder"]) as r:
            assert len(r.fetch_data_files(table_id=1, snapshot_id=1)) == 1
            assert len(r.fetch_data_files(table_id=1, snapshot_id=2)) == 2
            assert len(r.fetch_data_files(table_id=1, snapshot_id=3)) == 3

    def test_expired_file_invisible_after_end_snapshot(self, builder: Any) -> None:
        snap1 = builder.take_snapshot()
        schema_id = builder.add_schema(name="main", begin_snapshot=snap1)
        table_id = builder.add_table(
            schema_id=schema_id,
            name="t",
            begin_snapshot=snap1,
            columns=[("id", "INTEGER")],
        )
        df_id = builder.write_data_file(
            table_id=table_id, begin_snapshot=snap1, df=pl.DataFrame({"id": [1]})
        )
        snap2 = builder.take_snapshot()
        builder.expire_data_file(data_file_id=df_id, end_snapshot=snap2)

        with _reader(builder) as r:
            assert len(r.fetch_data_files(table_id=table_id, snapshot_id=snap1)) == 1
            assert len(r.fetch_data_files(table_id=table_id, snapshot_id=snap2)) == 0


class TestInlinedDataDetection:
    def test_inlined_table_present(self, builder: Any) -> None:
        snap = builder.take_snapshot()
        schema_id = builder.add_schema(name="main", begin_snapshot=snap)
        table_id = builder.add_table(
            schema_id=schema_id,
            name="t",
            begin_snapshot=snap,
            columns=[("id", "INTEGER")],
        )
        builder.add_inlined_data_table(table_id=table_id, schema_version=1)
        with _reader(builder) as r:
            assert r.has_inlined_data(table_id=table_id) is True


class TestReaderLifecycle:
    """Context-manager invariants: methods only valid inside `with`,
    reader can be entered multiple times."""

    def test_methods_outside_with_raise(self, single_file_lake: dict[str, Any]) -> None:
        reader = _reader(single_file_lake["builder"])
        with pytest.raises(RuntimeError, match="outside a `with` block"):
            reader.fetch_data_path()

    def test_reader_can_be_reopened(self, single_file_lake: dict[str, Any]) -> None:
        reader = _reader(single_file_lake["builder"])
        with reader as r1:
            assert r1.fetch_data_path() is not None
        with reader as r2:
            assert r2.fetch_data_path() is not None

    def test_dialect_property(self, single_file_lake: dict[str, Any]) -> None:
        reader = _reader(single_file_lake["builder"])
        assert reader.dialect == "sqlite"
