# polars-ducklake

A native [DuckLake](https://ducklake.select/) reader for [Polars](https://pola.rs/)
— with no DuckDB runtime dependency.

DuckLake stores all metadata in a SQL catalog database and all data in Parquet
files. `polars-ducklake` exploits that design directly: `scan_ducklake` is
registered as a Polars IO plugin (`polars.io.plugins.register_io_source`) and
returns a single `IO_SOURCE` LazyFrame node. Polars hands the deserialized
predicate, projected columns, and any `n_rows` cap back to the reader, which
prunes whole files using the catalog's min/max stats and then forwards what
remains into per-file `pl.scan_parquet` calls — so projection pushdown,
predicate pushdown, and streaming all still reach the Parquet engine.

Requires `polars >= 1.20`.

## Install

```bash
pip install polars-ducklake
```

Optional extras for non-SQLite catalogs:

```bash
pip install "polars-ducklake[postgres]"        # Postgres catalog
pip install "polars-ducklake[mysql]"           # MySQL catalog (read-only; see below)
pip install "polars-ducklake[duckdb-catalog]"  # DuckDB-as-catalog
```

## Catalog backends

Each backend is verified by an end-to-end integration test: the canonical
DuckDB ducklake-extension writer creates a real lake against the backend,
then this package reads it back and checks the results.

| Backend | Status | Notes |
|---|---|---|
| **SQLite** | Verified | No extra dependency; stdlib only. |
| **PostgreSQL** | Verified | Requires `[postgres]`; tested against PG 17. |
| **DuckDB-as-catalog** | Verified | Requires `[duckdb-catalog]` (`duckdb-engine`). |
| **MySQL** | Reader verified | `[mysql]` installs `pymysql`. Every read behavior (round-trip, time travel, deletes, schema evolution, partial files) is verified against MySQL 8.0 in the cross-backend test matrix. End-to-end writer round-trip via the DuckDB ducklake+mysql extension is currently unstable upstream — until that's resolved, populate MySQL-backed catalogs with another writer. |

## Quickstart

```python
import polars as pl
import polars_ducklake as pdl

# `scan_ducklake(metadata_catalog, table, ...)` — first arg identifies
# the metadata catalog (the SQL database hosting DuckLake's bookkeeping
# tables), second arg names the table.
lf = pdl.scan_ducklake(
    "sqlite:///metadata.db",
    "sales",
)

print(lf.filter(pl.col("region") == "us").select("amount").collect())
```

Tables can be addressed by an unqualified name (the default schema
``main`` is assumed) or a fully-qualified ``schema.table`` form:

```python
lf = pdl.scan_ducklake("sqlite:///metadata.db", "analytics.events")
# equivalent to:
lf = pdl.scan_ducklake("sqlite:///metadata.db", "events", schema="analytics")
```

The DuckLake-native connection string form is also supported:

```python
lf = pdl.scan_ducklake("ducklake:sqlite:metadata.db", "sales")
```

You can also pass a pre-built SQLAlchemy `Engine` — useful when you
need connection-pool options, SSL settings, or a custom driver:

```python
import sqlalchemy
engine = sqlalchemy.create_engine(
    "postgresql+psycopg://user@host/db?application_name=my-app",
    pool_size=10,
)
lf = pdl.scan_ducklake(engine, "analytics.sales")
```

### Time travel

```python
from datetime import datetime, timedelta

# By snapshot id
lf = pdl.scan_ducklake(engine, table="sales", snapshot_id=42)

# By timestamp (resolves to latest snapshot at or before the given time)
lf = pdl.scan_ducklake(
    engine,
    table="sales",
    as_of=datetime.now() - timedelta(days=7),
)
```

### Object storage

`storage_options` is passed through to `pl.scan_parquet`:

```python
lf = pdl.scan_ducklake(
    "postgresql+psycopg://user@host/db",
    table="sales",
    storage_options={"aws_region": "us-east-1"},
)
```

## What works

- Append-only reads, with multi-snapshot time travel by `snapshot_id`
  or `as_of` timestamp.
- Tables that have seen `DELETE` or `UPDATE` (DuckLake represents
  `UPDATE` as delete + insert).
- **Schema evolution**: `ADD COLUMN` (older files null-fill at the
  target's dtype), `DROP COLUMN` (column disappears), and
  `RENAME COLUMN` (older files' physical names are translated through
  the catalog's stable `column_id`).
- **Partitioned tables** — identity transforms (the partition column is
  written into each Parquet) and non-identity transforms (`year`,
  `month`, `bucket(N)`, etc., where the source column is in the
  Parquet) both read correctly.
- **Compacted lakes** — files merged across snapshots (`partial_max IS
  NOT NULL`) get a per-row snapshot filter applied for time-travel
  reads, so pinning to an older snapshot returns only rows whose origin
  was at-or-before that snapshot.
- **Catalog-stats file pruning** — supported predicate shapes (a flat
  AND of `eq`/`ne`/`lt`/`le`/`gt`/`ge` against a literal, plus
  `is_null`/`is_not_null`) are matched against
  `ducklake_file_column_stats` so files whose per-column min/max
  provably can't satisfy the filter are dropped before any Parquet I/O.
  Pruning is conservative: unrecognized predicate shapes (`OR`,
  `is_in`, arithmetic on a column, NaN-tainted floats, missing stats)
  keep the file and Polars handles the filter post-yield.

The returned `LazyFrame` is a single `IO_SOURCE` node — `.explain()`
shows `PYTHON SCAN` rather than `Parquet SCAN`. `lf.filter(...)` and
`lf.select(...)` work exactly as they do on any other Polars source;
the predicate and projection are deserialized and pushed back into the
reader, which forwards them to per-file `pl.scan_parquet` calls.

## Current limitations

The following are **not** yet supported and will raise
`NotImplementedError` rather than silently produce wrong results:

- **No writes.** Read-only.
- **No inlined data.** Tables with rows stored directly in the catalog
  DB (small writes under DuckLake's inlining threshold — default 10
  rows, on by default in the DuckDB writer) are detected and refused.
  `scan_ducklake` is designed for the large-dataset path; for tiny
  interactive lakes either set `DATA_INLINING_ROW_LIMIT=0` on the
  catalog or run the writer's inline-flush command.

**Known upstream issue:** Polars' native Parquet reader currently
panics on the Arrow MAP logical type produced by DuckDB's writer
("MapArray expects DataType::Struct as its inner logical type"). Our
catalog mapping for `MAP` columns is correct (assembled as
`List(Struct{key, value})`), and reads work when the on-disk Parquet
uses that shape directly. A workaround for DuckDB-MAP Parquet is to
read those columns via PyArrow until the upstream Polars fix lands.

## How it works

`scan_ducklake` is split into eager identity resolution and a deferred
plan that Polars drives through the IO-plugin contract.

**Eagerly, at call time:**

1. Normalize the catalog argument to a SQLAlchemy `Engine`.
2. Resolve the target snapshot (latest, by `snapshot_id`, or by `as_of`).
3. Look up the `schema_id` / `table_id` from `ducklake_schema` /
   `ducklake_table` using DuckLake's MVCC visibility filter, and
   refuse inlined-data tables. These raise `LookupError` /
   `NotImplementedError` directly from the call rather than being
   wrapped in `ComputeError` at `.collect()` time.
4. Register an IO source via
   `polars.io.plugins.register_io_source(..., is_pure=True)` and
   return its `LazyFrame`. Heavier catalog work is deferred to a
   memoized `_PlanContext`.

**Lazily, when Polars drives the read:**

5. Read the active column set at the target snapshot (renames, adds,
   drops applied), the data files, the positional-delete files, and
   the lake-wide `data_path`.
6. Take the pushed predicate from Polars and walk its serialized
   form to extract a flat AND of supported leaves; match those
   against `ducklake_file_column_stats` and drop files whose per-column
   min/max can't satisfy them. Anything unrecognized falls through and
   keeps the file (false negatives, never false positives).
7. Plan each surviving data file individually:
   - **Schema evolution**: query `ducklake_column` at the file's
     `begin_snapshot` and translate physical names through stable
     `column_id`s (`pl.col(old_name).alias(new_name)`); null-fill
     columns added after the file was written.
   - **Deletes**: per-file anti-join on the delete file's `pos`.
   - **Compacted files**: when the target is older than `partial_max`,
     filter rows by the writer-emitted `_ducklake_internal_snapshot_id`.
8. Yield each file's `pl.scan_parquet` LazyFrame to Polars with the
   pushed `with_columns` / `predicate` / `n_rows` applied.

All catalog queries use parameterized SQL (no f-string interpolation) and
the spec's MVCC clause:

```
WHERE :snapshot_id >= begin_snapshot
  AND (end_snapshot IS NULL OR :snapshot_id < end_snapshot)
```

## References

- DuckLake specification: <https://ducklake.select/docs/stable/specification/introduction>
- DuckLake queries (canonical SQL pattern): <https://ducklake.select/docs/stable/specification/queries>
- DuckLake table specs: <https://ducklake.select/docs/stable/specification/tables/overview>
- Polars `scan_parquet`: <https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html>

## Development

```bash
# Set up the venv with all dev / catalog dependencies pinned by uv.lock
uv sync --group dev

# Bring up the project-local docker stack (MinIO + Postgres + MySQL).
# Required only for the integration suite — unit tests run without it.
docker compose up -d

# Run everything.
uv run pytest

# Run only fast unit tests (no docker needed).
uv run pytest --ignore=tests/integration

# Run only integration tests.
uv run pytest tests/integration -m integration

# Tear the docker stack down (and wipe its volumes).
docker compose down -v
```

The docker-compose stack uses non-default ports (MinIO 19000/19001,
Postgres 15432, MySQL 13306) and a project-prefixed volume namespace,
so it is fully isolated from anything else you have running locally.

## License

MIT.
