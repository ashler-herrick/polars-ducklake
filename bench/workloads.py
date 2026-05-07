"""Declarative workload triples for the bench harness.

Each workload knows how to produce three equivalent reads against a
:class:`bench.fixtures.BenchLake`:

* a polars-ducklake :class:`polars.LazyFrame` (this project)
* a ducklake-polars :class:`polars.LazyFrame` (rival pure-Python reader,
  separate PyPI package)
* a DuckDB SQL string (consumed via the ducklake extension)

They are intentionally written so the result *should* match. The bench
harness records ``rows_returned`` per run so an obvious mismatch is
visible without making the harness do a full equality check (we are
measuring wall-clock, not correctness).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import ducklake_polars as dlp
import polars as pl

import polars_ducklake as pdl

from bench.fixtures import BenchLake


@dataclass(frozen=True)
class Workload:
    name: str
    polars_ducklake: Callable[[BenchLake], pl.LazyFrame]
    duckdb_sql: Callable[[BenchLake], str]
    ducklake_polars: Callable[[BenchLake], pl.LazyFrame]


def _scan(lake: BenchLake, table: str, **kwargs: object) -> pl.LazyFrame:
    return pdl.scan_ducklake(
        lake.sqlalchemy_url,
        table,
        storage_options=lake.storage_options,
        **kwargs,  # type: ignore[arg-type]
    )


def _dlp_path(lake: BenchLake) -> str:
    # Local import keeps this module importable even when the rival
    # package is missing — only the runner module hard-requires it.
    from bench.runners.ducklake_polars_runner import catalog_path_from_url

    return catalog_path_from_url(lake.sqlalchemy_url)


def _dlp_scan(lake: BenchLake, table: str, **kwargs: object) -> pl.LazyFrame:
    return dlp.scan_ducklake(_dlp_path(lake), table, **kwargs)  # type: ignore[arg-type]


def _wl_tiny_limit(lake: BenchLake) -> pl.LazyFrame:
    return _scan(lake, lake.table_main).limit(100)


def _dlp_tiny_limit(lake: BenchLake) -> pl.LazyFrame:
    return _dlp_scan(lake, lake.table_main).limit(100)


def _sql_tiny_limit(lake: BenchLake) -> str:
    return f"SELECT * FROM L.{lake.table_main} LIMIT 100"


def _wl_full_scan(lake: BenchLake) -> pl.LazyFrame:
    return _scan(lake, lake.table_main)


def _dlp_full_scan(lake: BenchLake) -> pl.LazyFrame:
    return _dlp_scan(lake, lake.table_main)


def _sql_full_scan(lake: BenchLake) -> str:
    return f"SELECT * FROM L.{lake.table_main}"


def _wl_predicate_partition(lake: BenchLake) -> pl.LazyFrame:
    return _scan(lake, lake.table_main).filter(pl.col("region") == "us")


def _dlp_predicate_partition(lake: BenchLake) -> pl.LazyFrame:
    return _dlp_scan(lake, lake.table_main).filter(pl.col("region") == "us")


def _sql_predicate_partition(lake: BenchLake) -> str:
    return f"SELECT * FROM L.{lake.table_main} WHERE region = 'us'"


def _wl_predicate_nonpartition(lake: BenchLake) -> pl.LazyFrame:
    return _scan(lake, lake.table_main).filter(pl.col("symbol") == "SYM007")


def _dlp_predicate_nonpartition(lake: BenchLake) -> pl.LazyFrame:
    return _dlp_scan(lake, lake.table_main).filter(pl.col("symbol") == "SYM007")


def _sql_predicate_nonpartition(lake: BenchLake) -> str:
    return f"SELECT * FROM L.{lake.table_main} WHERE symbol = 'SYM007'"


def _wl_time_travel(lake: BenchLake) -> pl.LazyFrame:
    return _scan(lake, lake.table_main, snapshot_id=lake.mid_snapshot_id)


def _dlp_time_travel(lake: BenchLake) -> pl.LazyFrame:
    # ducklake-polars uses ``snapshot_version`` for the same concept
    # polars-ducklake calls ``snapshot_id``.
    return _dlp_scan(
        lake, lake.table_main, snapshot_version=lake.mid_snapshot_id
    )


def _sql_time_travel(lake: BenchLake) -> str:
    return (
        f"SELECT * FROM L.{lake.table_main} AT (VERSION => {lake.mid_snapshot_id})"
    )


def _wl_schema_evolved(lake: BenchLake) -> pl.LazyFrame:
    return _scan(lake, lake.table_evolved)


def _dlp_schema_evolved(lake: BenchLake) -> pl.LazyFrame:
    return _dlp_scan(lake, lake.table_evolved)


def _sql_schema_evolved(lake: BenchLake) -> str:
    return f"SELECT * FROM L.{lake.table_evolved}"


WORKLOADS: tuple[Workload, ...] = (
    Workload(
        "tiny_limit",
        _wl_tiny_limit,
        _sql_tiny_limit,
        _dlp_tiny_limit,
    ),
    Workload(
        "full_scan",
        _wl_full_scan,
        _sql_full_scan,
        _dlp_full_scan,
    ),
    Workload(
        "predicate_partition",
        _wl_predicate_partition,
        _sql_predicate_partition,
        _dlp_predicate_partition,
    ),
    Workload(
        "predicate_nonpartition",
        _wl_predicate_nonpartition,
        _sql_predicate_nonpartition,
        _dlp_predicate_nonpartition,
    ),
    Workload(
        "time_travel",
        _wl_time_travel,
        _sql_time_travel,
        _dlp_time_travel,
    ),
    Workload(
        "schema_evolved",
        _wl_schema_evolved,
        _sql_schema_evolved,
        _dlp_schema_evolved,
    ),
)


def workloads_by_name(names: list[str] | None) -> list[Workload]:
    if names is None:
        return list(WORKLOADS)
    by_name = {w.name: w for w in WORKLOADS}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"unknown workload(s): {missing}; available: {sorted(by_name)}")
    return [by_name[n] for n in names]
