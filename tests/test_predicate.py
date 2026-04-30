"""Unit tests for the predicate AST walker and stats matcher."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from polars_ducklake._catalog import FileColumnStat
from polars_ducklake._predicate import (
    _AtomicClause,
    _coerce_stat,
    extract_clauses,
    file_can_match,
)


class TestExtractClauses:
    """``extract_clauses`` recovers a flat AND of leaf comparisons."""

    def test_simple_eq(self) -> None:
        c = extract_clauses(pl.col("a") == 5)
        assert c == [_AtomicClause("a", "eq", 5)]

    def test_all_comparison_ops(self) -> None:
        cases = [
            (pl.col("a") == 1, "eq"),
            (pl.col("a") != 1, "ne"),
            (pl.col("a") < 1, "lt"),
            (pl.col("a") <= 1, "le"),
            (pl.col("a") > 1, "gt"),
            (pl.col("a") >= 1, "ge"),
        ]
        for expr, op in cases:
            assert extract_clauses(expr) == [_AtomicClause("a", op, 1)]

    def test_and_splits_into_clauses(self) -> None:
        c = extract_clauses((pl.col("a") > 0) & (pl.col("b") == "x"))
        assert c == [
            _AtomicClause("a", "gt", 0),
            _AtomicClause("b", "eq", "x"),
        ]

    def test_nested_and(self) -> None:
        c = extract_clauses(
            (pl.col("a") > 0) & (pl.col("b") == "x") & (pl.col("c") <= 10)
        )
        assert c == [
            _AtomicClause("a", "gt", 0),
            _AtomicClause("b", "eq", "x"),
            _AtomicClause("c", "le", 10),
        ]

    def test_is_null_and_is_not_null(self) -> None:
        assert extract_clauses(pl.col("a").is_null()) == [
            _AtomicClause("a", "is_null", None)
        ]
        assert extract_clauses(pl.col("a").is_not_null()) == [
            _AtomicClause("a", "is_not_null", None)
        ]

    def test_string_literal(self) -> None:
        assert extract_clauses(pl.col("a") == "hello") == [
            _AtomicClause("a", "eq", "hello")
        ]

    def test_float_literal(self) -> None:
        c = extract_clauses(pl.col("a") < 3.14)
        assert c == [_AtomicClause("a", "lt", 3.14)]

    def test_bool_literal(self) -> None:
        c = extract_clauses(pl.col("a") == True)  # noqa: E712
        assert c == [_AtomicClause("a", "eq", True)]

    def test_date_literal(self) -> None:
        c = extract_clauses(pl.col("a") >= date(2024, 3, 15))
        assert c == [_AtomicClause("a", "ge", date(2024, 3, 15))]

    def test_naive_datetime_literal(self) -> None:
        c = extract_clauses(pl.col("a") < datetime(2024, 3, 15, 12, 30))
        assert c == [_AtomicClause("a", "lt", datetime(2024, 3, 15, 12, 30))]

    # -- bail-out cases (None) --------------------------------------------

    def test_or_returns_none(self) -> None:
        assert extract_clauses((pl.col("a") > 0) | (pl.col("b") == 1)) is None

    def test_arithmetic_on_column_returns_none(self) -> None:
        assert extract_clauses(pl.col("a") + 1 > 5) is None

    def test_function_other_than_null_check_returns_none(self) -> None:
        assert extract_clauses(pl.col("a").str.contains("x")) is None

    def test_is_in_returns_none(self) -> None:
        # Supporting is_in needs decoding the embedded series payload —
        # not done in v0.2; bail and let Polars handle the predicate.
        assert extract_clauses(pl.col("a").is_in([1, 2, 3])) is None

    def test_partial_unsupported_branch_taints_whole(self) -> None:
        # If half of an AND is unsupported we must return None for the
        # whole thing — partial pruning could drop files that the
        # unsupported half would have kept.
        assert (
            extract_clauses((pl.col("a") > 0) & (pl.col("b").is_in([1, 2]))) is None
        )

    def test_typed_int_literal_after_optimizer_cast(self) -> None:
        # When Polars' optimizer pushes a predicate through a typed
        # column it casts the literal to the column's dtype, switching
        # the serialized form from {"Dyn": {"Int": 5}} to
        # {"Scalar": {"Int32": 5}} (or Int8/Int64/UInt32/etc). The
        # extractor must accept all of these — missing this class of
        # literal silently disabled pruning end-to-end.
        for dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
            expr = pl.col("a") == pl.lit(5, dtype=dtype)
            assert extract_clauses(expr) == [_AtomicClause("a", "eq", 5)]

    def test_typed_float_literal(self) -> None:
        for dtype in (pl.Float32, pl.Float64):
            expr = pl.col("a") < pl.lit(2.5, dtype=dtype)
            assert extract_clauses(expr) == [_AtomicClause("a", "lt", 2.5)]


class TestCoerceStat:
    def test_int(self) -> None:
        assert _coerce_stat("42", "int32") == 42
        assert _coerce_stat("42", "bigint") == 42

    def test_float(self) -> None:
        assert _coerce_stat("3.14", "double") == 3.14

    def test_string(self) -> None:
        assert _coerce_stat("hello", "varchar") == "hello"

    def test_bool(self) -> None:
        # DuckLake writes booleans as "0"/"1" in stats.
        assert _coerce_stat("0", "boolean") is False
        assert _coerce_stat("1", "boolean") is True

    def test_date(self) -> None:
        assert _coerce_stat("2024-03-15", "date") == date(2024, 3, 15)

    def test_timestamp(self) -> None:
        assert _coerce_stat(
            "2024-03-15 12:30:45", "timestamp"
        ) == datetime(2024, 3, 15, 12, 30, 45)

    def test_none_passes_through(self) -> None:
        assert _coerce_stat(None, "int32") is None

    def test_unrecognized_type_returns_none(self) -> None:
        assert _coerce_stat("anything", "blob") is None

    def test_malformed_returns_none(self) -> None:
        assert _coerce_stat("not-a-number", "int32") is None


class TestFileCanMatch:
    """`file_can_match` is the per-file decision: keep or drop."""

    @staticmethod
    def _stat(
        column_id: int,
        *,
        min_value: str | None = None,
        max_value: str | None = None,
        null_count: int | None = 0,
        value_count: int | None = 100,
        contains_nan: bool | None = None,
    ) -> FileColumnStat:
        return FileColumnStat(
            data_file_id=1,
            column_id=column_id,
            min_value=min_value,
            max_value=max_value,
            null_count=null_count,
            value_count=value_count,
            contains_nan=contains_nan,
        )

    def test_eq_outside_range_prunes(self) -> None:
        stats = {1: self._stat(1, min_value="10", max_value="20")}
        assert not file_can_match(
            stats_by_column=stats,
            clauses=[_AtomicClause("a", "eq", 5)],
            column_meta={"a": (1, "int32")},
        )

    def test_eq_inside_range_keeps(self) -> None:
        stats = {1: self._stat(1, min_value="10", max_value="20")}
        assert file_can_match(
            stats_by_column=stats,
            clauses=[_AtomicClause("a", "eq", 15)],
            column_meta={"a": (1, "int32")},
        )

    def test_lt_below_min_prunes(self) -> None:
        stats = {1: self._stat(1, min_value="10", max_value="20")}
        # col < 10: anything below 10 — but min is 10, so impossible.
        assert not file_can_match(
            stats_by_column=stats,
            clauses=[_AtomicClause("a", "lt", 10)],
            column_meta={"a": (1, "int32")},
        )

    def test_le_at_min_keeps(self) -> None:
        stats = {1: self._stat(1, min_value="10", max_value="20")}
        # col <= 10: 10 itself qualifies.
        assert file_can_match(
            stats_by_column=stats,
            clauses=[_AtomicClause("a", "le", 10)],
            column_meta={"a": (1, "int32")},
        )

    def test_gt_above_max_prunes(self) -> None:
        stats = {1: self._stat(1, min_value="10", max_value="20")}
        assert not file_can_match(
            stats_by_column=stats,
            clauses=[_AtomicClause("a", "gt", 20)],
            column_meta={"a": (1, "int32")},
        )

    def test_string_range_prunes(self) -> None:
        stats = {1: self._stat(1, min_value="m", max_value="z")}
        assert not file_can_match(
            stats_by_column=stats,
            clauses=[_AtomicClause("a", "eq", "a")],
            column_meta={"a": (1, "varchar")},
        )

    def test_date_range_prunes(self) -> None:
        stats = {1: self._stat(1, min_value="2024-06-01", max_value="2024-06-30")}
        assert not file_can_match(
            stats_by_column=stats,
            clauses=[_AtomicClause("a", "eq", date(2024, 1, 1))],
            column_meta={"a": (1, "date")},
        )

    def test_is_null_with_no_nulls_prunes(self) -> None:
        stats = {1: self._stat(1, min_value="1", max_value="9", null_count=0)}
        assert not file_can_match(
            stats_by_column=stats,
            clauses=[_AtomicClause("a", "is_null", None)],
            column_meta={"a": (1, "int32")},
        )

    def test_is_not_null_when_all_null_prunes(self) -> None:
        stats = {1: self._stat(1, null_count=100, value_count=100)}
        assert not file_can_match(
            stats_by_column=stats,
            clauses=[_AtomicClause("a", "is_not_null", None)],
            column_meta={"a": (1, "int32")},
        )

    def test_no_stats_keeps_conservatively(self) -> None:
        # No stats row for the column → no information → keep the file.
        assert file_can_match(
            stats_by_column={},
            clauses=[_AtomicClause("a", "eq", 5)],
            column_meta={"a": (1, "int32")},
        )

    def test_unknown_column_keeps_conservatively(self) -> None:
        stats = {1: self._stat(1, min_value="10", max_value="20")}
        assert file_can_match(
            stats_by_column=stats,
            clauses=[_AtomicClause("zzz", "eq", 5)],
            column_meta={"a": (1, "int32")},
        )

    def test_contains_nan_keeps_conservatively(self) -> None:
        # NaN poisons ordering, so we can't trust min/max.
        stats = {
            1: self._stat(
                1, min_value="1.0", max_value="2.0", contains_nan=True
            )
        }
        assert file_can_match(
            stats_by_column=stats,
            clauses=[_AtomicClause("a", "gt", 100.0)],
            column_meta={"a": (1, "float64")},
        )

    def test_multiple_clauses_all_must_pass(self) -> None:
        stats = {
            1: self._stat(1, min_value="10", max_value="20"),
            2: self._stat(2, min_value="m", max_value="z"),
        }
        # First clause keeps; second prunes → file is dropped.
        assert not file_can_match(
            stats_by_column=stats,
            clauses=[
                _AtomicClause("a", "eq", 15),
                _AtomicClause("b", "eq", "a"),
            ],
            column_meta={"a": (1, "int32"), "b": (2, "varchar")},
        )

    def test_ne_singleton_value_prunes(self) -> None:
        # All values in this file are exactly 5; "col != 5" can be pruned.
        stats = {1: self._stat(1, min_value="5", max_value="5", null_count=0)}
        assert not file_can_match(
            stats_by_column=stats,
            clauses=[_AtomicClause("a", "ne", 5)],
            column_meta={"a": (1, "int32")},
        )

    def test_ne_range_keeps(self) -> None:
        # Range, so some value != 5 exists → can't prune.
        stats = {1: self._stat(1, min_value="1", max_value="10")}
        assert file_can_match(
            stats_by_column=stats,
            clauses=[_AtomicClause("a", "ne", 5)],
            column_meta={"a": (1, "int32")},
        )


@pytest.mark.parametrize(
    "expr,expected",
    [
        # quick sanity check that the API accepts a serializable expr
        (pl.col("a") == 1, [_AtomicClause("a", "eq", 1)]),
    ],
)
def test_extract_clauses_roundtrip(
    expr: pl.Expr, expected: list[_AtomicClause]
) -> None:
    assert extract_clauses(expr) == expected
