"""Tests for polars_ducklake.types."""

from __future__ import annotations

import polars as pl
import pytest

from polars_ducklake.types import is_nested_type, map_type


class TestPrimitiveMapping:
    """Spec lowercase names + DuckDB uppercase aliases both resolve."""

    @pytest.mark.parametrize(
        ("ducklake_type", "expected"),
        [
            ("boolean", pl.Boolean()),
            ("BOOLEAN", pl.Boolean()),
            ("int8", pl.Int8()),
            ("TINYINT", pl.Int8()),
            ("int16", pl.Int16()),
            ("SMALLINT", pl.Int16()),
            ("int32", pl.Int32()),
            ("INTEGER", pl.Int32()),
            ("int64", pl.Int64()),
            ("BIGINT", pl.Int64()),
            ("uint8", pl.UInt8()),
            ("uint64", pl.UInt64()),
            ("float32", pl.Float32()),
            ("FLOAT", pl.Float32()),
            ("float64", pl.Float64()),
            ("DOUBLE", pl.Float64()),
            ("varchar", pl.String()),
            ("VARCHAR", pl.String()),
            ("blob", pl.Binary()),
            ("BLOB", pl.Binary()),
            ("date", pl.Date()),
            ("timestamp", pl.Datetime(time_unit="us")),
            ("timestamp_ms", pl.Datetime(time_unit="ms")),
            ("timestamp_ns", pl.Datetime(time_unit="ns")),
            ("uuid", pl.String()),
            ("json", pl.String()),
        ],
    )
    def test_scalar_mapping(self, ducklake_type: str, expected: pl.DataType) -> None:
        assert map_type(ducklake_type) == expected


class TestDecimal:
    """``DECIMAL(p, s)`` parses precision and scale."""

    def test_basic_decimal(self) -> None:
        assert map_type("DECIMAL(18, 3)") == pl.Decimal(precision=18, scale=3)

    def test_lowercase_decimal(self) -> None:
        assert map_type("decimal(10,0)") == pl.Decimal(precision=10, scale=0)

    def test_numeric_alias(self) -> None:
        # SQL NUMERIC is the spec-equivalent alias for DECIMAL in some writers.
        assert map_type("NUMERIC(20, 5)") == pl.Decimal(precision=20, scale=5)


class TestNestedTypesNotResolvableInIsolation:
    """``map_type`` covers scalar types only.

    Nested types (LIST / STRUCT / MAP) must be assembled from the
    catalog's full row tree via ``build_nested_type``; calling
    ``map_type`` on a nested type alone produces an incomplete
    answer, so the function raises with a pointer to the right API.
    Real read-path coverage of nested types lives in the cross-backend
    matrix tests.
    """

    @pytest.mark.parametrize("ty", ["LIST", "list<int32>", "STRUCT", "MAP<INT, VARCHAR>"])
    def test_nested_raises(self, ty: str) -> None:
        with pytest.raises(ValueError, match="build_nested_type"):
            map_type(ty)

    def test_nested_uses_column_name_in_message(self) -> None:
        with pytest.raises(ValueError, match="my_col"):
            map_type("LIST<INT>", column_name="my_col")

    @pytest.mark.parametrize("ty", ["LIST<INT>", "struct(a int)", "map(int,int)"])
    def test_is_nested_type(self, ty: str) -> None:
        assert is_nested_type(ty) is True

    @pytest.mark.parametrize("ty", ["INTEGER", "VARCHAR", "DECIMAL(10,2)"])
    def test_scalar_is_not_nested(self, ty: str) -> None:
        assert is_nested_type(ty) is False


class TestUnknownType:
    """An unrecognised type string surfaces a ValueError with context."""

    def test_unknown_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unrecognised"):
            map_type("WIDGET")

    def test_unknown_includes_column_name(self) -> None:
        with pytest.raises(ValueError, match="my_col"):
            map_type("WIDGET", column_name="my_col")
