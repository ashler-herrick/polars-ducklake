"""Resolve a :class:`BenchLake` for a given ``(tier, backend)``.

For the CI tier this builds the lake from scratch into a caller-owned
temporary directory — fast (~half a second), self-contained, no infra
required. For the full tier we look up a previously seeded lake from
``bench/state.json``; the user is expected to have run
``python -m bench.seed --tier full --backend X`` first.

``BenchLake`` is the read-side projection of :class:`bench.state.LakeEntry`:
just the fields the runners and workloads actually need.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from bench.seed import materialize_lake
from bench.shapes import BENCH_CI
from bench.state import LakeEntry, get_entry
from bench.targets import ci_sqlite_local


@dataclass(frozen=True)
class BenchLake:
    """The read-side handle a workload + runner needs.

    ``storage_options`` is forwarded to :func:`scan_ducklake` (None for
    local-FS data paths). ``duckdb_setup_sql`` is run on a fresh DuckDB
    connection before ATTACH (e.g. ``CREATE SECRET`` for MinIO).
    """

    backend: str
    sqlalchemy_url: str
    ducklake_native: str
    data_path: str
    table_main: str
    table_evolved: str
    mid_snapshot_id: int
    storage_options: dict[str, str] | None = None
    duckdb_setup_sql: tuple[str, ...] = ()


def _from_entry(entry: LakeEntry) -> BenchLake:
    if entry.table_evolved is None:
        raise RuntimeError(
            f"lake {entry.shape_name} has no evolved table; "
            "the schema_evolved workload requires has_evolved_table=True"
        )
    return BenchLake(
        backend=entry.backend,
        sqlalchemy_url=entry.target.sqlalchemy_url,
        ducklake_native=entry.target.ducklake_native,
        data_path=entry.target.data_path,
        table_main=entry.table_main,
        table_evolved=entry.table_evolved,
        mid_snapshot_id=entry.mid_snapshot_id,
        storage_options=entry.target.storage_options,
        duckdb_setup_sql=tuple(entry.target.duckdb_setup_sql),
    )


def build_bench_ci_lake(root: Path) -> BenchLake:
    """Materialize the CI tier lake under ``root`` and return its handle.

    The CI tier is rebuilt-per-run because the build is fast and we
    want every CI invocation to start from a known-clean state. The
    full tier uses :func:`get_full_lake` to load a pre-seeded lake.
    """
    target = ci_sqlite_local(root)
    staging = root / "_staging"
    entry = materialize_lake(BENCH_CI, target, staging_dir=staging, progress=False)
    return _from_entry(entry)


def get_full_lake(backend: str) -> BenchLake:
    """Load the persisted full-tier lake for ``backend`` from state.json.

    Raises if no such lake has been seeded; the user is told to run
    :mod:`bench.seed` first.
    """
    entry = get_entry("full", backend)
    if entry is None:
        raise RuntimeError(
            f"no seeded lake for tier=full backend={backend!r}; "
            f"run: python -m bench.seed --tier full --backend {backend}"
        )
    return _from_entry(entry)


def make_temp_ci_lake() -> tuple[BenchLake, tempfile.TemporaryDirectory[str]]:
    """Convenience for the harness: build a CI lake in a temp dir.

    Returns the ``TemporaryDirectory`` so the caller can keep it alive
    for the duration of the bench run and clean it up explicitly.
    """
    tmp = tempfile.TemporaryDirectory(prefix="bench-ci-")
    lake = build_bench_ci_lake(Path(tmp.name))
    return lake, tmp
