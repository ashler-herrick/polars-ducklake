"""Tests for polars_ducklake.paths."""

from __future__ import annotations

import pytest

from polars_ducklake.paths import PathSegment, resolve_path


class TestRelativePathJoining:
    """Joining a relative path with a data_path prefix."""

    def test_local_path_with_trailing_slash(self) -> None:
        assert (
            resolve_path(path="part-0001.parquet", path_is_relative=True, data_path="/tmp/lake/")
            == "/tmp/lake/part-0001.parquet"
        )

    def test_local_path_without_trailing_slash(self) -> None:
        assert (
            resolve_path(path="part-0001.parquet", path_is_relative=True, data_path="/tmp/lake")
            == "/tmp/lake/part-0001.parquet"
        )

    def test_relative_with_leading_slash_is_normalised(self) -> None:
        # Even if the writer accidentally stored a leading slash on the
        # relative path, we must produce exactly one separator.
        assert (
            resolve_path(path="/sub/part.parquet", path_is_relative=True, data_path="/tmp/lake/")
            == "/tmp/lake/sub/part.parquet"
        )

    def test_s3_uri(self) -> None:
        assert (
            resolve_path(
                path="t1/part.parquet",
                path_is_relative=True,
                data_path="s3://bucket/prefix/",
            )
            == "s3://bucket/prefix/t1/part.parquet"
        )

    def test_s3_uri_no_trailing_slash(self) -> None:
        assert (
            resolve_path(
                path="t1/part.parquet",
                path_is_relative=True,
                data_path="s3://bucket/prefix",
            )
            == "s3://bucket/prefix/t1/part.parquet"
        )

    def test_relative_without_data_path_raises(self) -> None:
        with pytest.raises(ValueError, match="data_path"):
            resolve_path(path="x.parquet", path_is_relative=True, data_path=None)


class TestAbsolutePath:
    """Absolute paths are passed through verbatim."""

    def test_s3_absolute(self) -> None:
        absolute = "s3://other-bucket/x/y.parquet"
        assert resolve_path(path=absolute, path_is_relative=False, data_path=None) == absolute

    def test_absolute_ignores_data_path(self) -> None:
        absolute = "/var/lake/x.parquet"
        assert (
            resolve_path(path=absolute, path_is_relative=False, data_path="/something/else/")
            == absolute
        )


class TestSchemaAndTableChain:
    """The full data_path -> schema.path -> table.path -> file.path chain."""

    def test_all_relative(self) -> None:
        # This is the common DuckDB-writer case: schema "main/", table "sales/",
        # file "ducklake-...parquet", everything relative to data_path.
        out = resolve_path(
            path="ducklake-001.parquet",
            path_is_relative=True,
            data_path="s3://bucket/lake/",
            schema_segment=PathSegment(path="main/", is_relative=True),
            table_segment=PathSegment(path="sales/", is_relative=True),
        )
        assert out == "s3://bucket/lake/main/sales/ducklake-001.parquet"

    def test_table_absolute_overrides(self) -> None:
        # A table whose storage lives in a different bucket: the catalog
        # marks the table-level path absolute, so it replaces the prefix.
        out = resolve_path(
            path="part-1.parquet",
            path_is_relative=True,
            data_path="s3://lake/main/",
            schema_segment=PathSegment(path="main/", is_relative=True),
            table_segment=PathSegment(path="s3://elsewhere/external_table/", is_relative=False),
        )
        assert out == "s3://elsewhere/external_table/part-1.parquet"

    def test_schema_absolute_overrides(self) -> None:
        out = resolve_path(
            path="part.parquet",
            path_is_relative=True,
            data_path="s3://lake/",
            schema_segment=PathSegment(path="s3://other/sch/", is_relative=False),
            table_segment=PathSegment(path="t/", is_relative=True),
        )
        assert out == "s3://other/sch/t/part.parquet"

    def test_empty_segments_are_skipped(self) -> None:
        # Hand-built fixtures (and lakes whose schemas/tables have no
        # path component) supply empty strings — those should be no-ops.
        out = resolve_path(
            path="x.parquet",
            path_is_relative=True,
            data_path="/lake/",
            schema_segment=PathSegment(path="", is_relative=True),
            table_segment=PathSegment(path="", is_relative=True),
        )
        assert out == "/lake/x.parquet"

    def test_file_absolute_short_circuits(self) -> None:
        # Even with non-empty schema/table segments, an absolute file
        # path is returned as-is.
        out = resolve_path(
            path="s3://override/foo.parquet",
            path_is_relative=False,
            data_path="/lake/",
            schema_segment=PathSegment(path="main/", is_relative=True),
            table_segment=PathSegment(path="t/", is_relative=True),
        )
        assert out == "s3://override/foo.parquet"
