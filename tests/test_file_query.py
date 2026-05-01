"""Unit tests for the catalog-side file query builder (issue #4).

Tests focus on emitted SQL shape and bind params — the actual SQL
execution against each backend is exercised by integration tests.
"""

from __future__ import annotations

from datetime import date, datetime

from polars_ducklake._file_query import (
    _BIND_UNSUPPORTED,
    PartitionField,
    _bind_value_for,
    _column_family,
    _translate_partition_clause,
    _typed_stat_expr,
    build_files_query,
    translate_clause,
)
from polars_ducklake._predicate import _AtomicClause


class TestColumnFamily:
    def test_int(self) -> None:
        assert _column_family("INTEGER") == "int"
        assert _column_family("bigint") == "int"
        assert _column_family("uint64") == "int"

    def test_float(self) -> None:
        assert _column_family("DOUBLE") == "float"
        assert _column_family("real") == "float"

    def test_string(self) -> None:
        assert _column_family("VARCHAR") == ""
        assert _column_family("text") == ""

    def test_boolean(self) -> None:
        assert _column_family("BOOLEAN") == ""

    def test_date(self) -> None:
        assert _column_family("date") == "date"

    def test_timestamp(self) -> None:
        assert _column_family("timestamp") == "timestamp"
        assert _column_family("timestamp with time zone") == "timestamp"

    def test_unknown(self) -> None:
        assert _column_family("blob") is None
        assert _column_family("json") is None


class TestTypedStatExpr:
    def test_int_postgres(self) -> None:
        assert _typed_stat_expr("min_value", "bigint", "postgresql") == (
            "CAST(min_value AS BIGINT)"
        )

    def test_int_sqlite(self) -> None:
        assert _typed_stat_expr("min_value", "int", "sqlite") == (
            "CAST(min_value AS INTEGER)"
        )

    def test_int_mysql(self) -> None:
        # MySQL CAST uses SIGNED for int conversion, not BIGINT.
        assert _typed_stat_expr("max_value", "int", "mysql") == (
            "CAST(max_value AS SIGNED)"
        )

    def test_float_dialects(self) -> None:
        assert _typed_stat_expr("min_value", "double", "postgresql") == (
            "CAST(min_value AS DOUBLE PRECISION)"
        )
        assert _typed_stat_expr("min_value", "float", "duckdb") == (
            "CAST(min_value AS DOUBLE)"
        )
        assert _typed_stat_expr("min_value", "real", "sqlite") == (
            "CAST(min_value AS REAL)"
        )

    def test_string_no_cast(self) -> None:
        # Varchar / text never need a cast — string compare is correct.
        assert _typed_stat_expr("min_value", "varchar", "postgresql") == "min_value"
        assert _typed_stat_expr("min_value", "text", "sqlite") == "min_value"

    def test_boolean_no_cast(self) -> None:
        # Booleans are written as "0"/"1" strings; literal is bound the
        # same way, so direct string compare is correct.
        assert _typed_stat_expr("min_value", "boolean", "postgresql") == "min_value"

    def test_date_sqlite_no_cast(self) -> None:
        # SQLite stores dates as TEXT and compares ISO strings lex-
        # equivalently to chronological order — no CAST needed.
        assert _typed_stat_expr("min_value", "date", "sqlite") == "min_value"

    def test_date_postgres_casts(self) -> None:
        assert _typed_stat_expr("min_value", "date", "postgresql") == (
            "CAST(min_value AS DATE)"
        )

    def test_timestamp_mysql_casts_to_datetime(self) -> None:
        # MySQL's CAST target for timestamps is DATETIME, not TIMESTAMP.
        assert _typed_stat_expr("min_value", "timestamp", "mysql") == (
            "CAST(min_value AS DATETIME)"
        )

    def test_unknown_type_returns_none(self) -> None:
        assert _typed_stat_expr("min_value", "blob", "postgresql") is None

    def test_unknown_dialect_falls_back(self) -> None:
        # Unrecognized dialect falls back to postgres-style CAST.
        assert _typed_stat_expr("min_value", "bigint", "weirdsql") == (
            "CAST(min_value AS BIGINT)"
        )


class TestBindValue:
    def test_int_passthrough(self) -> None:
        assert _bind_value_for(5, "int") == 5

    def test_float_passthrough(self) -> None:
        assert _bind_value_for(3.14, "double") == 3.14

    def test_string_passthrough(self) -> None:
        assert _bind_value_for("hello", "varchar") == "hello"

    def test_bool_on_boolean_becomes_string(self) -> None:
        assert _bind_value_for(True, "boolean") == "1"
        assert _bind_value_for(False, "boolean") == "0"
        # Case-insensitive on the catalog type.
        assert _bind_value_for(True, "BOOLEAN") == "1"

    def test_bool_on_non_boolean_rejected(self) -> None:
        # bool is a subclass of int — must not silently bind as 1/0
        # against a numeric column.
        assert _bind_value_for(True, "bigint") is _BIND_UNSUPPORTED

    def test_date_passthrough(self) -> None:
        d = date(2024, 3, 15)
        assert _bind_value_for(d, "date") == d

    def test_datetime_passthrough(self) -> None:
        dt = datetime(2024, 3, 15, 12, 30)
        assert _bind_value_for(dt, "timestamp") == dt


class TestTranslateClauseNullOps:
    def test_is_null_no_params(self) -> None:
        c = translate_clause(
            _AtomicClause("a", "is_null", None),
            column_type="int",
            dialect="postgresql",
            param_prefix="p",
        )
        assert c is not None
        assert c.params == {}
        # Drop only when null_count is known to be 0.
        assert "null_count" in c.sql
        assert "null_count > 0" in c.sql

    def test_is_not_null_no_params(self) -> None:
        c = translate_clause(
            _AtomicClause("a", "is_not_null", None),
            column_type="int",
            dialect="postgresql",
            param_prefix="p",
        )
        assert c is not None
        assert c.params == {}
        assert "null_count < value_count" in c.sql


class TestTranslateClauseComparisons:
    def test_eq_int_postgres(self) -> None:
        c = translate_clause(
            _AtomicClause("a", "eq", 5),
            column_type="bigint",
            dialect="postgresql",
            param_prefix="c1",
        )
        assert c is not None
        assert c.params == {"c1_v": 5}
        # Both bounds reference the typed cast.
        assert "CAST(min_value AS BIGINT) <= :c1_v" in c.sql
        assert "CAST(max_value AS BIGINT) >= :c1_v" in c.sql
        # NaN + null guards present.
        assert "contains_nan" in c.sql
        assert "min_value IS NULL" in c.sql

    def test_eq_string_no_cast(self) -> None:
        c = translate_clause(
            _AtomicClause("region", "eq", "us"),
            column_type="varchar",
            dialect="postgresql",
            param_prefix="c0",
        )
        assert c is not None
        assert c.params == {"c0_v": "us"}
        assert "CAST(" not in c.sql

    def test_eq_bool(self) -> None:
        c = translate_clause(
            _AtomicClause("flag", "eq", True),
            column_type="boolean",
            dialect="postgresql",
            param_prefix="cf",
        )
        assert c is not None
        # True translated to "1" string.
        assert c.params == {"cf_v": "1"}

    def test_lt_uses_min_only(self) -> None:
        c = translate_clause(
            _AtomicClause("a", "lt", 100),
            column_type="int",
            dialect="postgresql",
            param_prefix="c2",
        )
        assert c is not None
        assert "min_value AS BIGINT) < :c2_v" in c.sql

    def test_gt_uses_max_only(self) -> None:
        c = translate_clause(
            _AtomicClause("a", "gt", 100),
            column_type="int",
            dialect="postgresql",
            param_prefix="c3",
        )
        assert c is not None
        assert "max_value AS BIGINT) > :c3_v" in c.sql

    def test_ne_includes_null_count_carve_out(self) -> None:
        c = translate_clause(
            _AtomicClause("a", "ne", 5),
            column_type="int",
            dialect="postgresql",
            param_prefix="c4",
        )
        assert c is not None
        # Drop condition is "min == max == lit AND value_count > null_count".
        # Negation: <> on either bound, OR all-null.
        assert "<> :c4_v" in c.sql
        assert "value_count <= null_count" in c.sql

    def test_unsupported_column_type(self) -> None:
        c = translate_clause(
            _AtomicClause("a", "eq", 5),
            column_type="blob",
            dialect="postgresql",
            param_prefix="c5",
        )
        assert c is None

    def test_date_clause_postgres(self) -> None:
        c = translate_clause(
            _AtomicClause("d", "ge", date(2024, 1, 1)),
            column_type="date",
            dialect="postgresql",
            param_prefix="cd",
        )
        assert c is not None
        assert c.params == {"cd_v": date(2024, 1, 1)}
        assert "CAST(max_value AS DATE) >= :cd_v" in c.sql

    def test_date_clause_sqlite_no_cast(self) -> None:
        c = translate_clause(
            _AtomicClause("d", "ge", date(2024, 1, 1)),
            column_type="date",
            dialect="sqlite",
            param_prefix="cd",
        )
        assert c is not None
        # SQLite compares ISO date strings; no CAST in the bound.
        assert "max_value >= :cd_v" in c.sql
        assert "CAST(max_value AS DATE)" not in c.sql

    def test_param_prefix_namespacing(self) -> None:
        # Two clauses with different prefixes must not collide on bind names.
        c1 = translate_clause(
            _AtomicClause("a", "eq", 1),
            column_type="int",
            dialect="postgresql",
            param_prefix="c0",
        )
        c2 = translate_clause(
            _AtomicClause("a", "eq", 2),
            column_type="int",
            dialect="postgresql",
            param_prefix="c1",
        )
        assert c1 is not None and c2 is not None
        assert set(c1.params).isdisjoint(c2.params)


class TestTranslatePartitionClause:
    def test_eq_int(self) -> None:
        c = _translate_partition_clause(
            _AtomicClause("yr", "eq", 2024),
            column_type="int",
            dialect="postgresql",
            param_prefix="p0",
        )
        assert c is not None
        assert c.params == {"p0_v": 2024}
        assert "CAST(partition_value AS BIGINT) = :p0_v" in c.sql
        # NULL partition_value kept conservatively.
        assert "partition_value IS NULL OR" in c.sql

    def test_is_null_no_params(self) -> None:
        c = _translate_partition_clause(
            _AtomicClause("region", "is_null", None),
            column_type="varchar",
            dialect="postgresql",
            param_prefix="p1",
        )
        assert c is not None
        assert c.params == {}
        assert c.sql == "(partition_value IS NULL)"

    def test_is_not_null_no_params(self) -> None:
        c = _translate_partition_clause(
            _AtomicClause("region", "is_not_null", None),
            column_type="varchar",
            dialect="postgresql",
            param_prefix="p1",
        )
        assert c is not None
        assert c.sql == "(partition_value IS NOT NULL)"

    def test_unsupported_type(self) -> None:
        c = _translate_partition_clause(
            _AtomicClause("x", "eq", 1),
            column_type="blob",
            dialect="postgresql",
            param_prefix="p2",
        )
        assert c is None


class TestBuildFilesQuery:
    """The full CTE builder. Asserts emitted SQL shape, not execution."""

    def test_no_clauses_emits_just_files_and_deletes(self) -> None:
        sql, params = build_files_query(
            table_id=7,
            snapshot_id=42,
            column_meta={},
            partition_spec=[],
            clauses=[],
            dialect="postgresql",
        )
        assert params == {"table_id": 7, "snapshot_id": 42}
        # Both visibility CTEs are present.
        assert "visible_files AS (" in sql
        assert "visible_deletes AS (" in sql
        # No predicate CTEs and no WHERE filter.
        assert "col_" not in sql
        assert "part_" not in sql
        assert "WHERE vf." not in sql
        # MVCC fragment present.
        assert ":snapshot_id >= begin_snapshot" in sql
        # Final SELECT shape.
        assert "FROM visible_files vf" in sql
        assert "LEFT JOIN visible_deletes vd USING (data_file_id)" in sql
        assert "ORDER BY vf.file_order, vd.delete_path" in sql

    def test_one_int_clause_emits_stats_cte(self) -> None:
        sql, params = build_files_query(
            table_id=7,
            snapshot_id=42,
            column_meta={"id": (101, "int")},
            partition_spec=[],
            clauses=[_AtomicClause("id", "eq", 5)],
            dialect="postgresql",
        )
        # Stats CTE present and filtered into main SELECT.
        assert "col_101_stats AS (" in sql
        assert "vf.data_file_id IN (SELECT data_file_id FROM col_101_stats)" in sql
        # No-stats UNION fallback present.
        assert "UNION ALL" in sql
        assert "data_file_id NOT IN (" in sql
        # Bind parameters: column_id + clause literal.
        assert params["col_101_id"] == 101
        assert params["col_101_0_v"] == 5
        assert params["table_id"] == 7
        assert params["snapshot_id"] == 42

    def test_clause_on_unknown_column_dropped(self) -> None:
        # column_meta has no entry for 'zzz' → clause dropped, no CTE.
        sql, params = build_files_query(
            table_id=1,
            snapshot_id=1,
            column_meta={"id": (101, "int")},
            partition_spec=[],
            clauses=[_AtomicClause("zzz", "eq", 5)],
            dialect="postgresql",
        )
        assert "col_" not in sql
        assert "WHERE vf." not in sql
        # Bind params don't include the dropped clause.
        assert all(not k.startswith("col_") for k in params)

    def test_unsupported_column_type_dropped(self) -> None:
        sql, params = build_files_query(
            table_id=1,
            snapshot_id=1,
            column_meta={"x": (200, "blob")},
            partition_spec=[],
            clauses=[_AtomicClause("x", "eq", 5)],
            dialect="postgresql",
        )
        # Stats CTE not emitted because no clauses translated.
        assert "col_200_stats AS" not in sql
        assert "WHERE vf." not in sql

    def test_multiple_clauses_same_column_anded(self) -> None:
        sql, params = build_files_query(
            table_id=1,
            snapshot_id=1,
            column_meta={"id": (101, "int")},
            partition_spec=[],
            clauses=[
                _AtomicClause("id", "ge", 10),
                _AtomicClause("id", "le", 100),
            ],
            dialect="postgresql",
        )
        # One CTE, two clauses AND'd.
        assert sql.count("col_101_stats AS (") == 1
        assert "col_101_0_v" in params
        assert "col_101_1_v" in params
        # Only one IN filter for this column.
        assert sql.count(
            "vf.data_file_id IN (SELECT data_file_id FROM col_101_stats)"
        ) == 1

    def test_multiple_columns_each_get_their_own_cte(self) -> None:
        sql, _ = build_files_query(
            table_id=1,
            snapshot_id=1,
            column_meta={"id": (101, "int"), "region": (102, "varchar")},
            partition_spec=[],
            clauses=[
                _AtomicClause("id", "gt", 0),
                _AtomicClause("region", "eq", "us"),
            ],
            dialect="postgresql",
        )
        assert "col_101_stats AS (" in sql
        assert "col_102_stats AS (" in sql
        assert sql.count("vf.data_file_id IN (SELECT data_file_id FROM col_") == 2

    def test_identity_partition_clause_emits_partition_cte(self) -> None:
        sql, params = build_files_query(
            table_id=1,
            snapshot_id=1,
            column_meta={"yr": (300, "int")},
            partition_spec=[
                PartitionField(
                    column_id=300,
                    column_name="yr",
                    column_type="int",
                    partition_key_index=0,
                    transform="identity",
                ),
            ],
            clauses=[_AtomicClause("yr", "eq", 2024)],
            dialect="postgresql",
        )
        # Both CTEs emitted (stats AND partition); they intersect.
        assert "col_300_stats AS (" in sql
        assert "part_0_filter AS (" in sql
        assert "vf.data_file_id IN (SELECT data_file_id FROM col_300_stats)" in sql
        assert "vf.data_file_id IN (SELECT data_file_id FROM part_0_filter)" in sql
        # Partition param uses its own namespace.
        assert params["pki_0_0_v"] == 2024

    def test_non_identity_partition_skipped(self) -> None:
        # Year transform isn't translatable yet — no partition CTE emitted.
        sql, _ = build_files_query(
            table_id=1,
            snapshot_id=1,
            column_meta={"ts": (400, "timestamp")},
            partition_spec=[
                PartitionField(
                    column_id=400,
                    column_name="ts",
                    column_type="timestamp",
                    partition_key_index=0,
                    transform="year",
                ),
            ],
            clauses=[_AtomicClause("ts", "ge", datetime(2024, 1, 1))],
            dialect="postgresql",
        )
        # Stats CTE still emitted (column_meta has the column); no
        # partition CTE for unsupported transform.
        assert "col_400_stats AS (" in sql
        assert "part_0_filter AS (" not in sql

