"""Run the bench matrix and emit JSONL results.

For each (workload, reader) pair: 1 untimed warmup + 5 timed reps, drop
min/max, report median + p95 over the remaining 3. The harness is
deliberately not a regression gate — it's a "way off base?" signal.

Reader handling:

* ``polars_ducklake`` re-uses the cached SQLAlchemy engine the library
  builds for the catalog URL. The instrumentation context attaches
  listeners only for the timed call so we don't count warmup activity.
* ``duckdb_arrow_only`` and ``duckdb_arrow_to_polars`` share one
  ``DuckDBContext`` (one connection, one ATTACH) across all workloads
  for that reader; the per-rep work is just the query.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from polars_ducklake._catalog import _engine_from_metadata_catalog

from bench.fixtures import BenchLake
from bench.instrumentation import count_catalog_activity
from bench.runners import RunResult
from bench.runners import (
    ducklake_polars_runner,
    duckdb_runner,
    polars_ducklake_runner,
)
from bench.workloads import Workload


READERS = (
    "polars_ducklake",
    "ducklake_polars",
    "duckdb_arrow_only",
    "duckdb_arrow_to_polars",
)
WARMUP_REPS = 1
TIMED_REPS = 5


@dataclass
class MeasurementRow:
    git_sha: str
    host_fingerprint: dict[str, object]
    timestamp: str
    tier: str
    backend: str
    table_shape: str
    workload: str
    reader: str
    wallclock_ms_median: float
    wallclock_ms_p95: float
    conn_count: int | None
    metadata_query_count: int | None
    rows_returned: int


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _host_fingerprint() -> dict[str, object]:
    uname = platform.uname()
    fp: dict[str, object] = {
        "system": uname.system,
        "release": uname.release,
        "machine": uname.machine,
        "processor": uname.processor,
        "python": platform.python_version(),
    }
    try:
        import os
        fp["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        # /proc/meminfo on Linux; cheap, no psutil dep.
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    fp["ram_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except OSError:
        pass
    return fp


def _summarize(samples_ms: list[float]) -> tuple[float, float]:
    """Drop min and max, return (median, p95) of what remains.

    With ``TIMED_REPS=5`` that's 3 samples — the median is the middle
    one and p95 collapses to the max-of-three. Good enough for a "way
    off base" signal.
    """
    trimmed = sorted(samples_ms)[1:-1]
    median = statistics.median(trimmed)
    p95 = max(trimmed)
    return median, p95


def _time_polars_ducklake(
    lake: BenchLake, workload: Workload
) -> tuple[list[float], RunResult]:
    engine = _engine_from_metadata_catalog(lake.sqlalchemy_url)

    # Warmup (untimed, no counters).
    for _ in range(WARMUP_REPS):
        polars_ducklake_runner.run(lake, workload)

    samples_ms: list[float] = []
    last: RunResult | None = None
    for _ in range(TIMED_REPS):
        with count_catalog_activity(engine) as counters:
            t0 = time.perf_counter()
            res = polars_ducklake_runner.run(lake, workload)
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
        last = RunResult(
            rows_returned=res.rows_returned,
            conn_count=counters.conn_count,
            metadata_query_count=counters.metadata_query_count,
        )
    assert last is not None
    return samples_ms, last


def _time_ducklake_polars(
    lake: BenchLake, workload: Workload
) -> tuple[list[float], RunResult]:
    """Time the rival ducklake-polars reader.

    Unlike :func:`_time_polars_ducklake`, this reader does not go
    through our SQLAlchemy engine, so there is nothing to instrument:
    ``conn_count`` and ``metadata_query_count`` come back as ``None``,
    same as the DuckDB readers.
    """
    for _ in range(WARMUP_REPS):
        ducklake_polars_runner.run(lake, workload)

    samples_ms: list[float] = []
    last: RunResult | None = None
    for _ in range(TIMED_REPS):
        t0 = time.perf_counter()
        res = ducklake_polars_runner.run(lake, workload)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
        last = res
    assert last is not None
    return samples_ms, last


def _time_duckdb(
    lake: BenchLake,
    workload: Workload,
    *,
    mode: duckdb_runner.Mode,
) -> tuple[list[float], RunResult]:
    ctx = duckdb_runner.setup(lake)
    try:
        for _ in range(WARMUP_REPS):
            duckdb_runner.run(ctx, lake, workload, mode=mode)

        samples_ms: list[float] = []
        last: RunResult | None = None
        for _ in range(TIMED_REPS):
            t0 = time.perf_counter()
            res = duckdb_runner.run(ctx, lake, workload, mode=mode)
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
            last = res
        assert last is not None
        return samples_ms, last
    finally:
        ctx.close()


def run_bench(
    lake: BenchLake,
    *,
    tier: str,
    backend: str,
    table_shape: str,
    workloads: list[Workload],
    readers: list[str],
) -> list[MeasurementRow]:
    git_sha = _git_sha()
    host = _host_fingerprint()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows: list[MeasurementRow] = []
    skipped_ducklake_polars_reason: str | None = None
    for workload in workloads:
        for reader in readers:
            if reader == "polars_ducklake":
                samples, result = _time_polars_ducklake(lake, workload)
            elif reader == "ducklake_polars":
                if skipped_ducklake_polars_reason is not None:
                    continue
                try:
                    samples, result = _time_ducklake_polars(lake, workload)
                except ValueError as exc:
                    # The rival package raises ValueError when it does
                    # not understand the catalog version (e.g. v1.0 vs
                    # the 0.3 it supports). Skip every remaining
                    # ducklake_polars cell and surface the reason once.
                    skipped_ducklake_polars_reason = str(exc)
                    print(
                        f"  ducklake_polars: skipped — {exc}",
                        file=sys.stderr,
                    )
                    continue
            elif reader == "duckdb_arrow_only":
                samples, result = _time_duckdb(lake, workload, mode="arrow_only")
            elif reader == "duckdb_arrow_to_polars":
                samples, result = _time_duckdb(lake, workload, mode="arrow_to_polars")
            else:
                raise ValueError(f"unknown reader: {reader}")

            median, p95 = _summarize(samples)
            rows.append(
                MeasurementRow(
                    git_sha=git_sha,
                    host_fingerprint=host,
                    timestamp=timestamp,
                    tier=tier,
                    backend=backend,
                    table_shape=table_shape,
                    workload=workload.name,
                    reader=reader,
                    wallclock_ms_median=round(median, 2),
                    wallclock_ms_p95=round(p95, 2),
                    conn_count=result.conn_count,
                    metadata_query_count=result.metadata_query_count,
                    rows_returned=result.rows_returned,
                )
            )
    return rows


def write_jsonl(rows: list[MeasurementRow], dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    stamp = rows[0].timestamp.replace(":", "").replace("-", "")
    sha = rows[0].git_sha
    out = dest_dir / f"{stamp}-{sha}.jsonl"
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")
    return out
