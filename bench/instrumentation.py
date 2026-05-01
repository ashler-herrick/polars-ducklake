"""SQLAlchemy event-based counters for the polars-ducklake reader.

Used by the bench harness to record how many catalog connections the
reader checks out and how many SQL statements it issues, alongside
wall-clock. The counters live in ``bench/`` rather than the library so
production code never carries the listener overhead.

The hooks mirror what ``tests/test_connection_count.py`` already
exercises against the live engine: ``engine_connect`` fires on every
checkout, ``before_cursor_execute`` fires on every cursor execute. We
attach both, hand a snapshot back to the harness, then detach.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Engine


@dataclass
class CatalogCounters:
    conn_count: int = 0
    metadata_query_count: int = 0


@contextmanager
def count_catalog_activity(engine: Engine) -> Iterator[CatalogCounters]:
    """Count engine connect/execute events on ``engine`` for the duration of the block.

    Listeners are removed in ``finally`` so a counter context can wrap
    one timed iteration without leaking listeners across reps.
    """
    counters = CatalogCounters()

    def _on_connect(_conn: Any) -> None:
        counters.conn_count += 1

    def _on_execute(
        _conn: Any,
        _cursor: Any,
        _stmt: Any,
        _params: Any,
        _context: Any,
        _executemany: Any,
    ) -> None:
        counters.metadata_query_count += 1

    event.listen(engine, "engine_connect", _on_connect)
    event.listen(engine, "before_cursor_execute", _on_execute)
    try:
        yield counters
    finally:
        event.remove(engine, "engine_connect", _on_connect)
        event.remove(engine, "before_cursor_execute", _on_execute)
