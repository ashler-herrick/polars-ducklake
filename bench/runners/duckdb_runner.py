"""DuckDB ducklake-extension reader, in two modes.

* ``arrow_only`` materializes a pyarrow Table — measures DuckDB-side
  work plus the Arrow handoff, but no Polars conversion.
* ``arrow_to_polars`` runs the same query and then ``pl.from_arrow`` on
  the result — apples-to-apples with the polars-ducklake runner.

DuckDB is configured once per benchmark run (single connection,
ducklake extension loaded, lake attached as ``L``). Each rep just
issues the SQL string from the workload definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import duckdb
import polars as pl

from polars_ducklake._catalog import _ENGINE_CACHE, _ENGINE_CACHE_LOCK

from bench.fixtures import BenchLake
from bench.runners import RunResult
from bench.workloads import Workload


Mode = Literal["arrow_only", "arrow_to_polars"]


@dataclass
class DuckDBContext:
    con: duckdb.DuckDBPyConnection

    def close(self) -> None:
        self.con.close()


_BACKEND_EXTENSION = {"sqlite": "sqlite", "postgres": "postgres", "duckdb": None}


def setup(lake: BenchLake) -> DuckDBContext:
    # When the catalog is itself a DuckDB file, polars-ducklake's cached
    # SQLAlchemy engine holds the file open in this process. DuckDB
    # enforces a unique-file-handle rule, so our ATTACH below would fail
    # with "already attached by database 'metadata'". Drop the cached
    # engine first.
    if lake.backend == "duckdb":
        with _ENGINE_CACHE_LOCK:
            cached = _ENGINE_CACHE.pop(lake.sqlalchemy_url, None)
        if cached is not None:
            cached.dispose()

    con = duckdb.connect(":memory:")
    for stmt in lake.duckdb_setup_sql:
        con.execute(stmt)
    con.execute("INSTALL ducklake;")
    con.execute("LOAD ducklake;")
    ext = _BACKEND_EXTENSION.get(lake.backend)
    if ext is not None:
        con.execute(f"INSTALL {ext};")
        con.execute(f"LOAD {ext};")
    con.execute(
        f"ATTACH '{lake.ducklake_native}' AS L (DATA_PATH '{lake.data_path}')"
    )
    return DuckDBContext(con=con)


def run(
    ctx: DuckDBContext, lake: BenchLake, workload: Workload, *, mode: Mode
) -> RunResult:
    sql = workload.duckdb_sql(lake)
    table = ctx.con.execute(sql).fetch_arrow_table()
    rows = table.num_rows
    if mode == "arrow_to_polars":
        df = pl.from_arrow(table)
        # pl.from_arrow can return DataFrame or Series; we always pass a Table here.
        assert isinstance(df, pl.DataFrame)
        rows = len(df)
    return RunResult(rows_returned=rows)
