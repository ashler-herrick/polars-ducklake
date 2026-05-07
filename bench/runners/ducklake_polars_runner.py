"""``ducklake-polars`` reader: a third-party rival package.

The PyPI package ``ducklake-polars`` (different author) implements the
same idea — read a DuckLake table directly into Polars without going
through DuckDB — with a slightly different API:

* takes a raw catalog *file path* or a ``postgresql://`` connection
  string, not a SQLAlchemy URL
* uses ``snapshot_version=`` (we use ``snapshot_id=``)
* does not accept ``storage_options``; relies on Polars' default
  credential chain. That means the full tier on MinIO won't work
  without environment-variable shims; the CI tier (local FS) is fine.

The ``duckdb`` catalog backend (DuckDB-format catalog file) is not
supported by ducklake-polars — its docs say SQLite/Postgres only.
"""

from __future__ import annotations

import ducklake_polars as dlp

from bench.fixtures import BenchLake
from bench.runners import RunResult
from bench.workloads import Workload


_SQLITE_PREFIX = "sqlite:///"
_PG_PREFIXES = ("postgresql://", "postgresql+psycopg://", "postgres://")


def catalog_path_from_url(sqlalchemy_url: str) -> str:
    """Translate the SQLAlchemy URL the harness already has into the
    catalog reference ducklake-polars expects.

    SQLite catalogs collapse to the raw file path; Postgres catalogs
    drop the SQLAlchemy driver fragment so the connection string is
    the bare libpq form.
    """
    if sqlalchemy_url.startswith(_SQLITE_PREFIX):
        return sqlalchemy_url[len(_SQLITE_PREFIX):]
    for pre in _PG_PREFIXES:
        if sqlalchemy_url.startswith(pre):
            tail = sqlalchemy_url[len(pre):]
            return f"postgresql://{tail}"
    raise ValueError(
        f"ducklake-polars cannot read catalog URL {sqlalchemy_url!r}; "
        "supported: sqlite:/// and postgresql:// (no duckdb-format catalogs)"
    )


def run(lake: BenchLake, workload: Workload) -> RunResult:
    lf = workload.ducklake_polars(lake)
    df = lf.collect()
    return RunResult(rows_returned=len(df))


__all__ = ["run", "catalog_path_from_url", "dlp"]
