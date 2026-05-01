"""Materialize a :class:`bench.shapes.Lakeshape` into a target catalog/data path.

This is the canonical writer for bench lakes. It drives the DuckDB
ducklake extension — the same writer the integration suite uses — so
the resulting catalog is exactly what a real DuckLake user would
produce.

CLI: ``python -m bench.seed --tier {ci,full} --backend {sqlite,postgres,duckdb}``

After a successful build the resolved handle (catalog URLs, snapshot
ids, table names) is written to :data:`bench.state.STATE_PATH` so
``python -m bench --tier full --backend X`` can find it later.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb

from bench.shapes import (
    BENCH_CI,
    BENCH_FULL,
    Lakeshape,
    evolved_chunk,
    generate_chunk,
    shape_for_tier,
)
from bench.state import LakeEntry, upsert_entry
from bench.targets import (
    CatalogTarget,
    SUPPORTED_BACKENDS_FULL,
    ci_sqlite_local,
    full_duckdb_minio,
    full_postgres_minio,
    full_sqlite_minio,
)


_DEFAULT_BENCH_ROOT = Path.home() / ".cache" / "polars-ducklake-bench"


def _attach(con: duckdb.DuckDBPyConnection, target: CatalogTarget) -> None:
    for stmt in target.duckdb_setup_sql:
        con.execute(stmt)
    con.execute("INSTALL ducklake;")
    con.execute("LOAD ducklake;")
    extension_for_backend = {
        "sqlite": "sqlite",
        "postgres": "postgres",
        "duckdb": None,  # built in
    }
    ext = extension_for_backend.get(target.backend)
    if ext is not None:
        con.execute(f"INSTALL {ext};")
        con.execute(f"LOAD {ext};")
    con.execute(
        f"ATTACH '{target.ducklake_native}' AS lake "
        f"(DATA_PATH '{target.data_path}', DATA_INLINING_ROW_LIMIT 0)"
    )
    con.execute("USE lake;")


def materialize_lake(
    shape: Lakeshape,
    target: CatalogTarget,
    *,
    staging_dir: Path,
    progress: bool = True,
) -> LakeEntry:
    """Build ``shape`` into ``target`` and return the resolved entry.

    Chunks are streamed: generate → write to a staging Parquet → INSERT
    via ``read_parquet(...)`` → free. This keeps peak memory at one
    chunk's worth even for the 30M-row full tier.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    table_main = "trades"
    table_evolved = "trades_evolved" if shape.has_evolved_table else None

    con = duckdb.connect(":memory:")
    try:
        _attach(con, target)

        sizes = shape.chunk_sizes()
        mid_snapshot_id: int | None = None
        for i, n_rows in enumerate(sizes):
            if progress:
                print(
                    f"  chunk {i + 1}/{len(sizes)}: {n_rows:,} rows -> {table_main}",
                    file=sys.stderr,
                    flush=True,
                )
            df = generate_chunk(shape, i)
            chunk_path = staging_dir / f"chunk_{i:03d}.parquet"
            df.write_parquet(chunk_path)
            del df  # free before duckdb reads it back
            if i == 0:
                con.execute(
                    f"CREATE TABLE {table_main} AS "
                    f"SELECT * FROM read_parquet('{chunk_path}');"
                )
                con.execute(
                    f"ALTER TABLE {table_main} SET PARTITIONED BY ({shape.partition_col});"
                )
            else:
                con.execute(
                    f"INSERT INTO {table_main} "
                    f"SELECT * FROM read_parquet('{chunk_path}');"
                )

            if i + 1 == shape.mid_chunk:
                row = con.execute("SELECT max(snapshot_id) FROM lake.snapshots()").fetchone()
                if row is None or row[0] is None:
                    raise RuntimeError("could not capture mid snapshot id")
                mid_snapshot_id = int(row[0])

            chunk_path.unlink()  # we won't reuse staged Parquet between chunks

        if mid_snapshot_id is None:
            raise RuntimeError(
                f"shape {shape.name}: mid_chunk={shape.mid_chunk} did not produce a snapshot"
            )

        if shape.has_evolved_table:
            if progress:
                print("  building evolved sibling table...", file=sys.stderr, flush=True)
            df0 = generate_chunk(shape, 0)
            p0 = staging_dir / "evolved_0.parquet"
            df0.write_parquet(p0)
            del df0
            con.execute(
                f"CREATE TABLE {table_evolved} AS "
                f"SELECT * FROM read_parquet('{p0}');"
            )
            con.execute(f"ALTER TABLE {table_evolved} ADD COLUMN venue VARCHAR;")
            p0.unlink()

            df1 = evolved_chunk(shape, 1)
            p1 = staging_dir / "evolved_1.parquet"
            df1.write_parquet(p1)
            del df1
            con.execute(
                f"INSERT INTO {table_evolved} "
                f"SELECT * FROM read_parquet('{p1}');"
            )
            p1.unlink()

        con.execute("CHECKPOINT;")
    finally:
        con.close()

    return LakeEntry(
        tier="ci" if shape is BENCH_CI else "full" if shape is BENCH_FULL else "custom",
        backend=target.backend,
        shape_name=shape.name,
        shape_hash=shape.hash(),
        target=target,
        table_main=table_main,
        table_evolved=table_evolved,
        mid_snapshot_id=mid_snapshot_id,
        built_sha=_git_sha(),
    )


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# --------------------------------- CLI ---------------------------------


def _build_target(tier: str, backend: str, root: Path) -> CatalogTarget:
    if tier == "ci":
        if backend != "sqlite":
            raise SystemExit(
                f"tier=ci only supports backend=sqlite (got {backend!r}); "
                "use --tier full for postgres/duckdb"
            )
        return ci_sqlite_local(root / "ci-sqlite-local")

    # full tier
    from tests.integration._minio import (  # local import: needs tests/.env loaded
        MinIOTestClient,
        load_config as load_minio_config,
    )

    minio_cfg = load_minio_config()
    if minio_cfg is None:
        raise SystemExit(
            "MinIO config missing — set MINIO_* in tests/.env or "
            "tests/.env.local before seeding the full tier"
        )
    minio_client = MinIOTestClient(minio_cfg, prefix="")  # bench buckets, not session-prefixed

    if backend == "sqlite":
        return full_sqlite_minio(minio_client, root / "full-sqlite-catalog")
    if backend == "duckdb":
        return full_duckdb_minio(minio_client, root / "full-duckdb-catalog")
    if backend == "postgres":
        from tests.integration._postgres import (
            ensure_database_exists,
            load_bench_config,
        )

        pg_cfg = load_bench_config()
        if pg_cfg is None:
            raise SystemExit(
                "Postgres bench config missing — set POSTGRES_BENCH_DATABASE "
                "(and the rest of POSTGRES_*) in tests/.env or tests/.env.local "
                "before seeding the postgres backend"
            )
        # Bench uses a dedicated DB so the integration suite's
        # drop-all-ducklake-tables doesn't wipe a seeded lake.
        ensure_database_exists(pg_cfg)
        return full_postgres_minio(minio_client, pg_cfg)

    raise SystemExit(f"unsupported backend: {backend!r}; use one of {SUPPORTED_BACKENDS_FULL}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.seed")
    parser.add_argument("--tier", choices=("ci", "full"), required=True)
    parser.add_argument(
        "--backend",
        choices=("sqlite", "postgres", "duckdb"),
        default="sqlite",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_DEFAULT_BENCH_ROOT,
        help="local cache directory for catalog files (default: ~/.cache/polars-ducklake-bench)",
    )
    args = parser.parse_args(argv)

    shape = shape_for_tier(args.tier)
    args.root.mkdir(parents=True, exist_ok=True)

    print(
        f"seeding tier={args.tier} backend={args.backend} shape={shape.name} "
        f"({shape.rows:,} rows, {shape.n_chunks} chunks)",
        file=sys.stderr,
    )
    target = _build_target(args.tier, args.backend, args.root)

    with tempfile.TemporaryDirectory(prefix="bench-staging-") as staging:
        entry = materialize_lake(shape, target, staging_dir=Path(staging))

    upsert_entry(entry)
    print(
        f"\nseeded {entry.tier}|{entry.backend} (shape_hash={entry.shape_hash}, "
        f"mid_snapshot_id={entry.mid_snapshot_id})",
        file=sys.stderr,
    )
    print(f"persisted to bench/state.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
