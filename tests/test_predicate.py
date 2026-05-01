"""Unit tests for the predicate AST walker."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from polars_ducklake._predicate import _AtomicClause, extract_clauses


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

    # -- lossy drop cases -------------------------------------------------

    def test_or_drops_both_sides(self) -> None:
        # Pushing only one side of a disjunction would lose files that
        # match the other side, so we drop the whole OR.
        assert extract_clauses((pl.col("a") > 0) | (pl.col("b") == 1)) == []

    def test_arithmetic_on_column_drops_clause(self) -> None:
        assert extract_clauses(pl.col("a") + 1 > 5) == []

    def test_function_other_than_null_check_drops_clause(self) -> None:
        assert extract_clauses(pl.col("a").str.contains("x")) == []

    def test_is_in_drops_clause(self) -> None:
        # Supporting is_in needs decoding the embedded series payload —
        # not done in v0.2; drop and let Polars handle the predicate.
        assert extract_clauses(pl.col("a").is_in([1, 2, 3])) == []

    def test_partial_unsupported_branch_keeps_supported_half(self) -> None:
        # AND with one supported and one unsupported leaf: keep the
        # supported leaf. Polars still applies the full predicate to the
        # rows we yield, so widening the candidate set is safe.
        assert extract_clauses(
            (pl.col("a") > 0) & (pl.col("b").is_in([1, 2]))
        ) == [_AtomicClause("a", "gt", 0)]

    def test_or_inside_and_is_dropped_other_side_kept(self) -> None:
        # (A OR B) AND C: drop the OR subtree, keep C.
        assert extract_clauses(
            ((pl.col("a") > 0) | (pl.col("b") == 1)) & (pl.col("c") == 9)
        ) == [_AtomicClause("c", "eq", 9)]

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
