"""``polars-ducklake`` reader: ``scan_ducklake(...).collect()`` to Polars."""

from __future__ import annotations

from bench.fixtures import BenchLake
from bench.runners import RunResult
from bench.workloads import Workload


def run(lake: BenchLake, workload: Workload) -> RunResult:
    lf = workload.polars_ducklake(lake)
    df = lf.collect()
    return RunResult(rows_returned=len(df))
