"""polars-ducklake: a native DuckLake reader for Polars.

Public API is a single function — :func:`scan_ducklake` — which returns
a :class:`polars.LazyFrame` backed by the underlying Parquet files of a
DuckLake table. The metadata catalog (SQL database hosting DuckLake's
bookkeeping tables) can be SQLite, PostgreSQL, MySQL, or DuckDB; the
right backend is selected automatically from the connection string.

See the project README and the DuckLake specification at
https://ducklake.select/docs/stable/specification/introduction for
details.
"""

from __future__ import annotations

from polars_ducklake.scan import scan_ducklake

__all__ = ["scan_ducklake"]
__version__ = "0.2.0"
