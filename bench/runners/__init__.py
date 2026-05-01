"""Per-reader runners. Each returns a :class:`RunResult`.

A runner is a callable ``(lake, workload) -> RunResult`` that performs
exactly one read end-to-end. The harness wraps the call in timing and
counter context. The runner itself is responsible for any
reader-specific setup that should be amortized across reps (handle to a
DuckDB connection, etc.) but the actual read happens inside the call.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunResult:
    rows_returned: int
    # Filled by the harness's instrumentation context for the
    # polars-ducklake reader. ``None`` for the DuckDB readers (their
    # catalog access goes through a DuckDB extension we don't sit
    # inside, so we cannot count it the same way).
    conn_count: int | None = None
    metadata_query_count: int | None = None
