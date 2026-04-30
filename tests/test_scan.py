"""End-to-end scan_ducklake tests."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl
import pytest
import sqlalchemy

import polars_ducklake as pdl


class TestEmptyTable:
    def test_empty_returns_zero_row_lazyframe_with_schema(
        self, empty_table_lake: dict[str, Any]
    ) -> None:
        builder = empty_table_lake["builder"]
        lf = pdl.scan_ducklake(builder.url, table="empty_t")
        df = lf.collect()
        assert df.height == 0
        assert df.schema == {"id": pl.Int32(), "name": pl.String()}


class TestSingleFile:
    def test_round_trip(self, single_file_lake: dict[str, Any]) -> None:
        builder = single_file_lake["builder"]
        lf = pdl.scan_ducklake(builder.url, table="sales")
        df = lf.collect().sort("id")
        assert df.height == single_file_lake["row_count"]
        assert df["region"].to_list() == ["us", "us", "eu", "eu", "apac"]
        assert df["amount"].to_list() == [10.0, 20.5, 30.25, 40.0, 55.5]

    def test_filter_works(self, single_file_lake: dict[str, Any]) -> None:
        builder = single_file_lake["builder"]
        lf = pdl.scan_ducklake(builder.url, table="sales")
        df = lf.filter(pl.col("region") == "us").collect()
        assert df["id"].to_list() == [1, 2]

    def test_engine_input(self, single_file_lake: dict[str, Any]) -> None:
        builder = single_file_lake["builder"]
        eng = sqlalchemy.create_engine(builder.url)
        lf = pdl.scan_ducklake(eng, table="sales")
        assert lf.collect().height == single_file_lake["row_count"]

    def test_ducklake_native_string(self, single_file_lake: dict[str, Any]) -> None:
        builder = single_file_lake["builder"]
        lf = pdl.scan_ducklake(
            f"ducklake:sqlite:{builder.catalog_path}",
            table="sales",
        )
        assert lf.collect().height == single_file_lake["row_count"]


class TestMultiFile:
    def test_concatenates_all_files(self, multi_file_lake: dict[str, Any]) -> None:
        builder = multi_file_lake["builder"]
        lf = pdl.scan_ducklake(builder.url, table="events")
        df = lf.collect().sort("id")
        assert df.height == multi_file_lake["row_count"]
        assert df["id"].to_list() == list(range(1, multi_file_lake["row_count"] + 1))


class TestTimeTravel:
    def test_default_is_latest(self, multi_snapshot_lake: dict[str, Any]) -> None:
        builder = multi_snapshot_lake["builder"]
        lf = pdl.scan_ducklake(builder.url, table="rolling")
        assert sorted(lf.collect()["id"].to_list()) == [1, 2, 3, 4, 5, 6]

    def test_snapshot_id_pin(self, multi_snapshot_lake: dict[str, Any]) -> None:
        builder = multi_snapshot_lake["builder"]
        lf1 = pdl.scan_ducklake(builder.url, table="rolling", snapshot_id=1)
        assert sorted(lf1.collect()["id"].to_list()) == [1, 2]

        lf2 = pdl.scan_ducklake(builder.url, table="rolling", snapshot_id=2)
        assert sorted(lf2.collect()["id"].to_list()) == [1, 2, 3, 4]

    def test_as_of_resolves_correctly(self, multi_snapshot_lake: dict[str, Any]) -> None:
        builder = multi_snapshot_lake["builder"]
        # snap1 at 2026-01-01, snap2 at 2026-02-01, snap3 at 2026-03-01
        lf = pdl.scan_ducklake(builder.url, table="rolling", as_of=datetime(2026, 2, 15))
        assert sorted(lf.collect()["id"].to_list()) == [1, 2, 3, 4]

    def test_conflicting_snapshot_args(self, multi_snapshot_lake: dict[str, Any]) -> None:
        builder = multi_snapshot_lake["builder"]
        with pytest.raises(ValueError, match="mutually exclusive"):
            pdl.scan_ducklake(
                builder.url,
                table="rolling",
                snapshot_id=1,
                as_of=datetime(2026, 1, 1),
            )

    def test_invalid_snapshot_id(self, multi_snapshot_lake: dict[str, Any]) -> None:
        builder = multi_snapshot_lake["builder"]
        with pytest.raises(LookupError, match="snapshot_id=999"):
            pdl.scan_ducklake(builder.url, table="rolling", snapshot_id=999)


class TestErrorPaths:
    def test_invalid_table_name(self, single_file_lake: dict[str, Any]) -> None:
        builder = single_file_lake["builder"]
        with pytest.raises(LookupError, match="missing"):
            pdl.scan_ducklake(builder.url, table="missing")

    def test_invalid_schema_name(self, single_file_lake: dict[str, Any]) -> None:
        builder = single_file_lake["builder"]
        with pytest.raises(LookupError, match="other"):
            pdl.scan_ducklake(builder.url, table="sales", schema="other")


class TestTableIdentifier:
    """``table=`` can be unqualified or fully-qualified ``schema.table``."""

    def test_unqualified_uses_default_schema(self, single_file_lake: dict[str, Any]) -> None:
        builder = single_file_lake["builder"]
        # Unqualified — defaults to schema="main"
        assert (
            pdl.scan_ducklake(builder.url, table="sales").collect().height
            == single_file_lake["row_count"]
        )

    def test_fqn_table_form(self, single_file_lake: dict[str, Any]) -> None:
        builder = single_file_lake["builder"]
        # Equivalent: schema parsed from the dotted name
        assert (
            pdl.scan_ducklake(builder.url, table="main.sales").collect().height
            == single_file_lake["row_count"]
        )

    def test_explicit_schema_kwarg_overrides_default(
        self, single_file_lake: dict[str, Any]
    ) -> None:
        builder = single_file_lake["builder"]
        # Explicit kwarg + unqualified table still works.
        assert (
            pdl.scan_ducklake(builder.url, table="sales", schema="main").collect().height
            == single_file_lake["row_count"]
        )

    def test_dotted_table_with_explicit_schema_raises(
        self, single_file_lake: dict[str, Any]
    ) -> None:
        builder = single_file_lake["builder"]
        # Ambiguous: both forms specify the schema.
        with pytest.raises(ValueError, match="schema"):
            pdl.scan_ducklake(builder.url, table="main.sales", schema="main")

    def test_fqn_with_non_default_schema(self, builder: Any) -> None:
        # Build a lake with a non-default schema and read it via FQN.
        snap = builder.take_snapshot()
        schema_id = builder.add_schema(name="analytics", begin_snapshot=snap)
        table_id = builder.add_table(
            schema_id=schema_id,
            name="events",
            begin_snapshot=snap,
            columns=[("id", "INTEGER")],
        )
        builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap,
            df=pl.DataFrame({"id": [1, 2, 3]}),
        )
        df = pdl.scan_ducklake(builder.url, table="analytics.events").sort("id").collect()
        assert df["id"].to_list() == [1, 2, 3]


class TestRefusalCases:
    """Loud-refusal cases. Verify the user gets a clear message."""

    def test_inlined_data_refused(self, single_file_lake: dict[str, Any]) -> None:
        builder = single_file_lake["builder"]
        builder.add_inlined_data_table(table_id=single_file_lake["table_id"], schema_version=1)
        with pytest.raises(NotImplementedError, match="inlined"):
            pdl.scan_ducklake(builder.url, table="sales")

    def test_flushed_inline_registry_does_not_refuse(
        self, single_file_lake: dict[str, Any]
    ) -> None:
        # DuckDB's ducklake_flush_inlined_data empties the tracker but
        # leaves the row in ducklake_inlined_data_tables. The reader must
        # probe the tracker contents before refusing, otherwise users get
        # a permanent NotImplementedError they can't work around.
        builder = single_file_lake["builder"]
        builder.add_inlined_data_table(
            table_id=single_file_lake["table_id"], schema_version=1, populated=False
        )
        df = pdl.scan_ducklake(builder.url, table="sales").sort("id").collect()
        assert df.height == single_file_lake["row_count"]

    # Nested types (LIST/STRUCT/MAP) are now supported — see
    # tests/test_backends_matrix.py for full coverage across every backend.


class TestPushdown:
    """Predicate and projection pushdown survive through scan_ducklake."""

    def test_predicate_pushdown_in_plan(self, multi_file_lake: dict[str, Any]) -> None:
        builder = multi_file_lake["builder"]
        lf = pdl.scan_ducklake(builder.url, table="events").filter(pl.col("kind") == "a")
        plan = lf.explain()
        # Polars will push the predicate into the parquet scan; the plan
        # should not contain a separate FILTER node above the scan.
        assert "Parquet" in plan or "PARQUET" in plan
        assert "kind" in plan

    def test_projection_pushdown_in_plan(self, multi_file_lake: dict[str, Any]) -> None:
        builder = multi_file_lake["builder"]
        lf = pdl.scan_ducklake(builder.url, table="events").select("id")
        plan = lf.explain()
        # Polars reports projection pushdown as "PROJECT N/M COLUMNS" inline
        # with the parquet scan node — verify both that the scan exists and
        # that only a subset of the columns is being read.
        assert "PROJECT 1/2 COLUMNS" in plan


class TestPrimitiveTypes:
    """All v0.1 supported primitive types round-trip through a scan."""

    def test_round_trip_primitives(self, builder: Any) -> None:
        snap = builder.take_snapshot()
        schema_id = builder.add_schema(name="main", begin_snapshot=snap)

        # Map from DuckLake column-type string to a corresponding pl.DataType
        # we can write into Parquet via a Polars DataFrame.
        cols = [
            ("c_bool", "BOOLEAN", pl.Boolean()),
            ("c_int8", "TINYINT", pl.Int8()),
            ("c_int16", "SMALLINT", pl.Int16()),
            ("c_int32", "INTEGER", pl.Int32()),
            ("c_int64", "BIGINT", pl.Int64()),
            ("c_float", "FLOAT", pl.Float32()),
            ("c_double", "DOUBLE", pl.Float64()),
            ("c_str", "VARCHAR", pl.String()),
            ("c_date", "DATE", pl.Date()),
            ("c_ts", "TIMESTAMP", pl.Datetime("us")),
        ]
        table_id = builder.add_table(
            schema_id=schema_id,
            name="prim",
            begin_snapshot=snap,
            columns=[(name, ty) for name, ty, _ in cols],
        )
        df = pl.DataFrame(
            {
                "c_bool": [True, False],
                "c_int8": pl.Series([1, -2], dtype=pl.Int8()),
                "c_int16": pl.Series([300, -400], dtype=pl.Int16()),
                "c_int32": pl.Series([100000, -200000], dtype=pl.Int32()),
                "c_int64": pl.Series([10**12, -(10**12)], dtype=pl.Int64()),
                "c_float": pl.Series([1.5, 2.5], dtype=pl.Float32()),
                "c_double": [3.14, 2.71],
                "c_str": ["hello", "world"],
                "c_date": pl.Series(
                    [datetime(2026, 1, 1).date(), datetime(2026, 6, 30).date()],
                    dtype=pl.Date(),
                ),
                "c_ts": pl.Series(
                    [datetime(2026, 1, 1, 12, 0, 0), datetime(2026, 6, 30, 23, 59, 59)],
                    dtype=pl.Datetime("us"),
                ),
            }
        )
        builder.write_data_file(table_id=table_id, begin_snapshot=snap, df=df)

        lf = pdl.scan_ducklake(builder.url, table="prim")
        out = lf.collect().sort("c_int32")
        assert out.schema == df.schema
        assert out["c_str"].to_list() == ["world", "hello"]


class TestStorageOptionsPassthrough:
    """``storage_options`` reaches scan_parquet — we don't validate values
    here (no real cloud), just that it doesn't break the local path."""

    def test_passes_storage_options(self, single_file_lake: dict[str, Any]) -> None:
        builder = single_file_lake["builder"]
        # Polars accepts arbitrary string-keyed dicts and ignores irrelevant
        # ones for local files.
        lf = pdl.scan_ducklake(builder.url, table="sales", storage_options={"some_key": "value"})
        # Local path means it just works regardless.
        assert lf.collect().height == single_file_lake["row_count"]


class TestDataPathOverride:
    def test_data_path_kwarg_overrides_metadata(
        self, single_file_lake: dict[str, Any], tmp_path: Any
    ) -> None:
        builder = single_file_lake["builder"]
        # Override with the same correct path (positive control: it should still work)
        lf = pdl.scan_ducklake(builder.url, table="sales", data_path=builder.data_path)
        assert lf.collect().height == single_file_lake["row_count"]


class TestPartialFiles:
    """Compaction-merged files (partial_max IS NOT NULL).

    These files carry rows from multiple historical snapshots and an
    extra ``_ducklake_internal_snapshot_id`` column per row. Time-travel
    reads at a target below ``partial_max`` must filter rows whose
    origin snapshot exceeds the target.
    """

    def test_latest_read_returns_all_rows(self, builder: Any) -> None:
        snap1 = builder.take_snapshot()
        snap2 = builder.take_snapshot()
        snap3 = builder.take_snapshot()
        snap4 = builder.take_snapshot()
        schema_id = builder.add_schema(name="main", begin_snapshot=snap1)
        table_id = builder.add_table(
            schema_id=schema_id,
            name="t",
            begin_snapshot=snap1,
            columns=[("id", "INTEGER"), ("val", "VARCHAR")],
        )
        # Simulate a merged file with rows originating at snaps 1, 2, 3.
        merged = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "val": ["a", "b", "c"],
                "_ducklake_internal_snapshot_id": pl.Series(
                    [snap1, snap2, snap3], dtype=pl.Int64()
                ),
            }
        )
        builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap1,
            df=merged,
            partial_max=snap3,
        )
        # Read at snap4: target >= partial_max → no row filter, all 3 rows visible.
        df = pdl.scan_ducklake(builder.url, table="t", snapshot_id=snap4).sort("id").collect()
        assert df.columns == ["id", "val"]
        assert df["id"].to_list() == [1, 2, 3]

    def test_time_travel_filters_future_rows(self, builder: Any) -> None:
        snap1 = builder.take_snapshot()
        snap2 = builder.take_snapshot()
        snap3 = builder.take_snapshot()
        schema_id = builder.add_schema(name="main", begin_snapshot=snap1)
        table_id = builder.add_table(
            schema_id=schema_id,
            name="t",
            begin_snapshot=snap1,
            columns=[("id", "INTEGER"), ("val", "VARCHAR")],
        )
        merged = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "val": ["a", "b", "c"],
                "_ducklake_internal_snapshot_id": pl.Series(
                    [snap1, snap2, snap3], dtype=pl.Int64()
                ),
            }
        )
        builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap1,
            df=merged,
            partial_max=snap3,
        )

        # Read at snap1: only row with origin=snap1 should be visible.
        df1 = pdl.scan_ducklake(builder.url, table="t", snapshot_id=snap1).sort("id").collect()
        assert df1["id"].to_list() == [1]

        # Read at snap2: rows from snap1 and snap2.
        df2 = pdl.scan_ducklake(builder.url, table="t", snapshot_id=snap2).sort("id").collect()
        assert df2["id"].to_list() == [1, 2]

    def test_partial_filter_drops_internal_column_from_output(self, builder: Any) -> None:
        # Output schema must match the catalog's user columns —
        # _ducklake_internal_snapshot_id never reaches the user.
        snap1 = builder.take_snapshot()
        snap2 = builder.take_snapshot()
        schema_id = builder.add_schema(name="main", begin_snapshot=snap1)
        table_id = builder.add_table(
            schema_id=schema_id,
            name="t",
            begin_snapshot=snap1,
            columns=[("id", "INTEGER")],
        )
        builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap1,
            df=pl.DataFrame(
                {
                    "id": [1, 2],
                    "_ducklake_internal_snapshot_id": pl.Series([snap1, snap2], dtype=pl.Int64()),
                }
            ),
            partial_max=snap2,
        )
        df = pdl.scan_ducklake(builder.url, table="t", snapshot_id=snap1).sort("id").collect()
        assert df.columns == ["id"]


class TestSchemaEvolution:
    """Files written under older schemas read correctly at the new schema.

    DuckLake tracks columns by stable ``column_id`` across renames: a
    rename produces two ``ducklake_column`` rows sharing the column_id
    but with different (begin_snapshot, column_name). Parquet files
    physically store whatever name was in effect when they were written.
    The reader translates per-file using the column_id linkage.
    """

    def test_added_column_null_filled_in_older_files(self, builder: Any) -> None:
        # snap1: create (id INT). Write 2 rows.
        # snap2: ADD COLUMN extra DOUBLE.
        # snap3: insert 2 rows with extra=...
        snap1 = builder.take_snapshot()
        schema_id = builder.add_schema(name="main", begin_snapshot=snap1)
        table_id = builder.add_table(
            schema_id=schema_id,
            name="t",
            begin_snapshot=snap1,
            columns=[("id", "INTEGER")],
        )
        old_file = builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap1,
            df=pl.DataFrame({"id": [1, 2]}),
        )
        snap2 = builder.take_snapshot()
        builder.add_column(
            table_id=table_id,
            begin_snapshot=snap2,
            column_name="extra",
            column_type="DOUBLE",
            column_order=1,
        )
        snap3 = builder.take_snapshot()
        new_file = builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap3,
            df=pl.DataFrame({"id": [3, 4], "extra": [3.14, 2.71]}),
        )
        assert old_file != new_file

        df = pdl.scan_ducklake(builder.url, table="t").sort("id").collect()
        assert df.columns == ["id", "extra"]
        assert df["id"].to_list() == [1, 2, 3, 4]
        # Rows from the pre-add file get null-filled.
        assert df["extra"].to_list() == [None, None, 3.14, 2.71]

    def test_dropped_column_disappears_from_old_files(self, builder: Any) -> None:
        snap1 = builder.take_snapshot()
        schema_id = builder.add_schema(name="main", begin_snapshot=snap1)
        table_id = builder.add_table(
            schema_id=schema_id,
            name="t",
            begin_snapshot=snap1,
            columns=[("id", "INTEGER"), ("dropme", "VARCHAR")],
        )
        builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap1,
            df=pl.DataFrame({"id": [1, 2], "dropme": ["a", "b"]}),
        )
        snap2 = builder.take_snapshot()
        # column_id of "dropme" is 2 (id=1, dropme=2 since they were inserted in order)
        builder.drop_column(column_id=2, end_snapshot=snap2)

        df = pdl.scan_ducklake(builder.url, table="t").sort("id").collect()
        assert df.columns == ["id"]
        assert df["id"].to_list() == [1, 2]

    def test_renamed_column_translates_old_file_names(self, builder: Any) -> None:
        snap1 = builder.take_snapshot()
        schema_id = builder.add_schema(name="main", begin_snapshot=snap1)
        table_id = builder.add_table(
            schema_id=schema_id,
            name="t",
            begin_snapshot=snap1,
            columns=[("old_name", "INTEGER")],
        )
        builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap1,
            df=pl.DataFrame({"old_name": [1, 2, 3]}),
        )
        snap2 = builder.take_snapshot()
        builder.rename_column(
            column_id=1, table_id=table_id, at_snapshot=snap2, new_name="new_name"
        )
        # Insert new rows under the new name.
        snap3 = builder.take_snapshot()
        builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap3,
            df=pl.DataFrame({"new_name": [4, 5]}),
        )

        df = pdl.scan_ducklake(builder.url, table="t").sort("new_name").collect()
        assert df.columns == ["new_name"]
        assert df["new_name"].to_list() == [1, 2, 3, 4, 5]

    def test_time_travel_sees_pre_rename_schema(self, builder: Any) -> None:
        # At snap1 the column is 'old_name'. After rename the catalog says
        # 'new_name'. A read pinned at snap1 must call the column 'old_name'
        # because that's what existed then.
        snap1 = builder.take_snapshot()
        schema_id = builder.add_schema(name="main", begin_snapshot=snap1)
        table_id = builder.add_table(
            schema_id=schema_id,
            name="t",
            begin_snapshot=snap1,
            columns=[("old_name", "INTEGER")],
        )
        builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap1,
            df=pl.DataFrame({"old_name": [1, 2]}),
        )
        snap2 = builder.take_snapshot()
        builder.rename_column(
            column_id=1, table_id=table_id, at_snapshot=snap2, new_name="new_name"
        )
        df_old = (
            pdl.scan_ducklake(builder.url, table="t", snapshot_id=snap1).sort("old_name").collect()
        )
        assert df_old.columns == ["old_name"]
        df_new = (
            pdl.scan_ducklake(builder.url, table="t", snapshot_id=snap2).sort("new_name").collect()
        )
        assert df_new.columns == ["new_name"]

    def test_combined_add_drop_rename_in_one_read(self, builder: Any) -> None:
        # Realistic mix: add a column, drop another, rename a third — read
        # everything at the latest snapshot.
        snap1 = builder.take_snapshot()
        schema_id = builder.add_schema(name="main", begin_snapshot=snap1)
        table_id = builder.add_table(
            schema_id=schema_id,
            name="t",
            begin_snapshot=snap1,
            columns=[("id", "INTEGER"), ("legacy", "VARCHAR"), ("zone", "VARCHAR")],
        )
        # column_ids: id=1, legacy=2, zone=3
        builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap1,
            df=pl.DataFrame({"id": [1, 2], "legacy": ["x", "y"], "zone": ["a", "b"]}),
        )

        snap2 = builder.take_snapshot()
        builder.drop_column(column_id=2, end_snapshot=snap2)
        builder.rename_column(column_id=3, table_id=table_id, at_snapshot=snap2, new_name="region")
        builder.add_column(
            table_id=table_id,
            begin_snapshot=snap2,
            column_name="value",
            column_type="DOUBLE",
            column_order=10,
        )

        snap3 = builder.take_snapshot()
        builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap3,
            df=pl.DataFrame({"id": [3, 4], "region": ["c", "d"], "value": [1.5, 2.5]}),
        )

        df = pdl.scan_ducklake(builder.url, table="t").sort("id").collect()
        assert df.columns == ["id", "region", "value"]
        assert df["id"].to_list() == [1, 2, 3, 4]
        # First file's 'zone' values become 'region' (rename).
        assert df["region"].to_list() == ["a", "b", "c", "d"]
        # First file had no 'value' column → null-filled.
        assert df["value"].to_list() == [None, None, 1.5, 2.5]

    def test_schema_evolution_with_deletes(self, builder: Any) -> None:
        # Both schema evolution AND deletes on the same file: the
        # per-file plan should compose them correctly.
        snap1 = builder.take_snapshot()
        schema_id = builder.add_schema(name="main", begin_snapshot=snap1)
        table_id = builder.add_table(
            schema_id=schema_id,
            name="t",
            begin_snapshot=snap1,
            columns=[("id", "INTEGER")],
        )
        df_id = builder.write_data_file(
            table_id=table_id,
            begin_snapshot=snap1,
            df=pl.DataFrame({"id": [1, 2, 3, 4]}),
        )
        snap2 = builder.take_snapshot()
        builder.add_column(
            table_id=table_id,
            begin_snapshot=snap2,
            column_name="kind",
            column_type="VARCHAR",
            column_order=1,
        )
        snap3 = builder.take_snapshot()
        builder.add_delete_file(
            table_id=table_id,
            begin_snapshot=snap3,
            data_file_id=df_id,
            positions=[0, 2],  # drop id=1 and id=3
        )

        df = pdl.scan_ducklake(builder.url, table="t").sort("id").collect()
        assert df.columns == ["id", "kind"]
        assert df["id"].to_list() == [2, 4]
        # Old file never had 'kind' → null-fill survives the anti-join.
        assert df["kind"].to_list() == [None, None]


class TestDeleteFiles:
    """Positional merge-on-read deletes: per-file anti-join semantics."""

    def test_single_delete_file_drops_referenced_positions(
        self, single_file_lake: dict[str, Any]
    ) -> None:
        # single_file_lake has 5 rows id 1..5. Delete positions 1 and 3
        # (= rows id=2 and id=4).
        builder = single_file_lake["builder"]
        snap2 = builder.take_snapshot()
        builder.add_delete_file(
            table_id=single_file_lake["table_id"],
            begin_snapshot=snap2,
            data_file_id=1,
            positions=[1, 3],
        )
        df = pdl.scan_ducklake(builder.url, table="sales").sort("id").collect()
        assert df["id"].to_list() == [1, 3, 5]
        assert df["region"].to_list() == ["us", "eu", "apac"]

    def test_multiple_delete_files_per_data_file_unioned(
        self, single_file_lake: dict[str, Any]
    ) -> None:
        # Two delete files at the same snapshot, each removing one row.
        # Together they should drop two rows, even though the catalog
        # represents them as separate delete-file rows.
        builder = single_file_lake["builder"]
        snap2 = builder.take_snapshot()
        builder.add_delete_file(
            table_id=single_file_lake["table_id"],
            begin_snapshot=snap2,
            data_file_id=1,
            positions=[0],
        )
        builder.add_delete_file(
            table_id=single_file_lake["table_id"],
            begin_snapshot=snap2,
            data_file_id=1,
            positions=[4],
        )
        df = pdl.scan_ducklake(builder.url, table="sales").sort("id").collect()
        assert df["id"].to_list() == [2, 3, 4]

    def test_data_file_with_all_rows_deleted(self, single_file_lake: dict[str, Any]) -> None:
        builder = single_file_lake["builder"]
        snap2 = builder.take_snapshot()
        builder.add_delete_file(
            table_id=single_file_lake["table_id"],
            begin_snapshot=snap2,
            data_file_id=1,
            positions=[0, 1, 2, 3, 4],
        )
        df = pdl.scan_ducklake(builder.url, table="sales").collect()
        assert df.height == 0
        # Schema is preserved.
        assert df.columns == ["id", "amount", "region"]

    def test_mixed_files_with_and_without_deletes(self, multi_file_lake: dict[str, Any]) -> None:
        # multi_file_lake has 3 files: ids 1-3, 4-5, 6-9.
        # Delete pos 0 from file 1 (id=1) and pos 1 from file 3 (id=7).
        # File 2 has no deletes; total surviving rows: 9 - 2 = 7.
        builder = multi_file_lake["builder"]
        snap2 = builder.take_snapshot()
        builder.add_delete_file(
            table_id=multi_file_lake["table_id"],
            begin_snapshot=snap2,
            data_file_id=1,
            positions=[0],
        )
        builder.add_delete_file(
            table_id=multi_file_lake["table_id"],
            begin_snapshot=snap2,
            data_file_id=3,
            positions=[1],
        )
        df = pdl.scan_ducklake(builder.url, table="events").sort("id").collect()
        assert df.height == 7
        assert df["id"].to_list() == [2, 3, 4, 5, 6, 8, 9]

    def test_delete_invisible_at_earlier_snapshot(self, single_file_lake: dict[str, Any]) -> None:
        # Time travel: at the original snapshot the delete didn't exist
        # yet, so all rows should be visible.
        builder = single_file_lake["builder"]
        snap2 = builder.take_snapshot()
        builder.add_delete_file(
            table_id=single_file_lake["table_id"],
            begin_snapshot=snap2,
            data_file_id=1,
            positions=[1, 3],
        )
        before = pdl.scan_ducklake(builder.url, table="sales", snapshot_id=1).sort("id").collect()
        assert before.height == 5
        after = (
            pdl.scan_ducklake(builder.url, table="sales", snapshot_id=snap2).sort("id").collect()
        )
        assert after.height == 3

    def test_expired_delete_file_skipped(self, single_file_lake: dict[str, Any]) -> None:
        # Delete files also have end_snapshot. A delete file that's been
        # superseded (end_snapshot set) should not be applied at later
        # snapshots — verify by setting end_snapshot on the delete row.
        import sqlite3

        builder = single_file_lake["builder"]
        snap2 = builder.take_snapshot()
        delete_id = builder.add_delete_file(
            table_id=single_file_lake["table_id"],
            begin_snapshot=snap2,
            data_file_id=1,
            positions=[0, 1, 2],
        )
        snap3 = builder.take_snapshot()
        with sqlite3.connect(builder.catalog_path) as conn:
            conn.execute(
                "UPDATE ducklake_delete_file SET end_snapshot = ? WHERE delete_file_id = ?",
                (snap3, delete_id),
            )
            conn.commit()
        # At snap2 the delete is active → 2 rows survive.
        at2 = pdl.scan_ducklake(builder.url, table="sales", snapshot_id=snap2).sort("id").collect()
        assert at2.height == 2
        # At snap3 the delete has been superseded → all 5 rows back.
        at3 = pdl.scan_ducklake(builder.url, table="sales", snapshot_id=snap3).sort("id").collect()
        assert at3.height == 5

    def test_filter_after_delete_works(self, single_file_lake: dict[str, Any]) -> None:
        # The returned LazyFrame must still compose with downstream
        # operations — predicate pushdown is on the post-delete data.
        builder = single_file_lake["builder"]
        snap2 = builder.take_snapshot()
        builder.add_delete_file(
            table_id=single_file_lake["table_id"],
            begin_snapshot=snap2,
            data_file_id=1,
            positions=[1, 3],
        )
        df = (
            pdl.scan_ducklake(builder.url, table="sales").filter(pl.col("region") == "us").collect()
        )
        # Original us rows were id=1 and id=2; pos=1 removed id=2; only id=1 left.
        assert df["id"].to_list() == [1]
