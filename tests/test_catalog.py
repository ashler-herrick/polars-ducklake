"""Tests for connection-string parsing and CatalogReader factory.

The string parsers (`_translate_ducklake_string` and helpers) live in
``polars_ducklake._catalog``; they're private but the parsing rules are
worth pinning so we don't accidentally break the user-facing
``ducklake:...`` form.
"""

from __future__ import annotations

import pytest
import sqlalchemy

from polars_ducklake._catalog import CatalogReader, _translate_ducklake_string


class TestTranslateDuckLakeString:
    """Pure string translation — no engine creation."""

    def test_sqlite(self) -> None:
        assert _translate_ducklake_string("ducklake:sqlite:metadata.db") == "sqlite:///metadata.db"

    def test_sqlite_nested_path(self) -> None:
        assert (
            _translate_ducklake_string("ducklake:sqlite:dir/sub/metadata.db")
            == "sqlite:///dir/sub/metadata.db"
        )

    def test_sqlite_empty_path_raises(self) -> None:
        with pytest.raises(ValueError, match="path"):
            _translate_ducklake_string("ducklake:sqlite:")

    def test_postgres_basic(self) -> None:
        url = _translate_ducklake_string("ducklake:postgres:dbname=foo host=bar user=baz")
        assert url == "postgresql+psycopg://baz@bar/foo"

    def test_postgres_with_port(self) -> None:
        url = _translate_ducklake_string("ducklake:postgres:dbname=foo host=bar user=baz port=5433")
        assert url == "postgresql+psycopg://baz@bar:5433/foo"

    def test_postgres_with_password(self) -> None:
        url = _translate_ducklake_string(
            "ducklake:postgres:dbname=foo host=bar user=baz password=secret"
        )
        assert url == "postgresql+psycopg://baz:secret@bar/foo"

    def test_postgres_database_alias(self) -> None:
        url = _translate_ducklake_string("ducklake:postgres:database=foo host=bar user=baz")
        assert url == "postgresql+psycopg://baz@bar/foo"

    def test_postgres_missing_dbname(self) -> None:
        with pytest.raises(ValueError, match="dbname"):
            _translate_ducklake_string("ducklake:postgres:host=bar user=baz")

    def test_postgres_malformed_pair(self) -> None:
        with pytest.raises(ValueError, match="key=value"):
            _translate_ducklake_string("ducklake:postgres:dbname=foo not_a_pair")

    def test_postgres_empty_body(self) -> None:
        with pytest.raises(ValueError, match="key=value"):
            _translate_ducklake_string("ducklake:postgres:")

    def test_duckdb_default(self) -> None:
        # No backend prefix → DuckDB-as-catalog.
        assert (
            _translate_ducklake_string("ducklake:metadata.ducklake")
            == "duckdb:///metadata.ducklake"
        )

    def test_duckdb_explicit_prefix(self) -> None:
        # The DuckDB ducklake extension also accepts an explicit duckdb: prefix;
        # we mirror the same form for symmetry with sqlite/postgres.
        assert (
            _translate_ducklake_string("ducklake:duckdb:metadata.duckdb")
            == "duckdb:///metadata.duckdb"
        )

    def test_duckdb_explicit_prefix_empty_path_raises(self) -> None:
        with pytest.raises(ValueError, match="path"):
            _translate_ducklake_string("ducklake:duckdb:")

    def test_mysql_basic(self) -> None:
        url = _translate_ducklake_string("ducklake:mysql:database=foo host=bar user=root")
        assert url == "mysql+pymysql://root@bar/foo"

    def test_mysql_with_port_and_password(self) -> None:
        url = _translate_ducklake_string(
            "ducklake:mysql:database=foo host=bar port=3307 user=root password=secret"
        )
        assert url == "mysql+pymysql://root:secret@bar:3307/foo"

    def test_mysql_db_alias(self) -> None:
        # DuckLake's MySQL backend accepts both `database=` and `db=`.
        url = _translate_ducklake_string("ducklake:mysql:db=foo host=bar user=root")
        assert url == "mysql+pymysql://root@bar/foo"

    def test_mysql_missing_database(self) -> None:
        with pytest.raises(ValueError, match="database"):
            _translate_ducklake_string("ducklake:mysql:host=bar user=root")

    def test_mysql_empty_body(self) -> None:
        with pytest.raises(ValueError, match="key=value"):
            _translate_ducklake_string("ducklake:mysql:")

    def test_unsupported_backend(self) -> None:
        # postgresql: prefix is rejected (use postgres: instead).
        with pytest.raises(ValueError, match="postgresql"):
            _translate_ducklake_string("ducklake:postgresql:dbname=foo")

    def test_empty_body(self) -> None:
        with pytest.raises(ValueError, match="path"):
            _translate_ducklake_string("ducklake:")


class TestCatalogReaderFactory:
    """End-to-end: ``CatalogReader.from_metadata_catalog`` accepts every
    documented form and produces a reader that can open a connection."""

    def test_engine_passthrough(self) -> None:
        eng = sqlalchemy.create_engine("sqlite:///:memory:")
        reader = CatalogReader.from_metadata_catalog(eng)
        assert reader.engine is eng

    def test_sqlalchemy_url_string(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        url = f"sqlite:///{tmp_path / 'x.db'}"
        with CatalogReader.from_metadata_catalog(url) as r:
            # Smoke-test that the connection works at all.
            assert r.dialect == "sqlite"

    def test_ducklake_sqlite_string(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "lake.db"
        with CatalogReader.from_metadata_catalog(f"ducklake:sqlite:{path}") as r:
            assert r.dialect == "sqlite"

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="metadata_catalog must be"):
            CatalogReader.from_metadata_catalog(42)  # type: ignore[arg-type]

    def test_duckdb_without_engine_extra(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # Simulate the duckdb-engine package being absent and confirm the
        # error message points users at the right install extra. (The dev
        # group installs duckdb-engine so the integration suite can use
        # it; this test patches the import to exercise the error path
        # without requiring the package be uninstalled.)
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "duckdb_engine":
                raise ImportError("No module named 'duckdb_engine'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="duckdb-catalog"):
            CatalogReader.from_metadata_catalog("ducklake:metadata.ducklake")
