# bench/

On-demand performance harness comparing `polars-ducklake` against the
DuckDB ducklake extension reading the same lake. Not run in CI; not
shipped in the published package.

## Tiers

| Tier  | Lake size | Catalog backends     | Data path                      | Infra needed              |
|-------|-----------|----------------------|--------------------------------|---------------------------|
| `ci`  | ~100k rows| `sqlite` only        | local FS (caller's tmp dir)    | nothing — fully self-contained |
| `full`| ~30M rows | `sqlite`, `postgres`, `duckdb` | MinIO buckets, one per backend | `docker compose up -d`    |

`mysql` is intentionally not supported — the DuckDB ducklake-mysql
extension is upstream-unstable per the project README.

The full-tier buckets are split per catalog backend
(`polars-ducklake-bench-{sqlite,postgres,duckdb}`) so any bleed across
ducklake's catalog-migration semantics cannot contaminate another
backend's seeded data. Created automatically by `minio-init` in
`docker-compose.yml`.

## Run — CI tier

```bash
# Self-contained, ~3-5 seconds:
uv run python -m bench --tier ci
```

The CI tier rebuilds the lake every invocation into a temp dir; nothing
to clean up.

## Run — full tier

The full tier separates **seeding** (build the 30M-row lake once,
persist its handle) from **measuring** (read the persisted lake).

```bash
docker compose up -d                                 # MinIO + Postgres up

uv run python -m bench.seed --tier full --backend sqlite     # ~30s
uv run python -m bench.seed --tier full --backend postgres
uv run python -m bench.seed --tier full --backend duckdb

uv run python -m bench --tier full --backend sqlite          # measure
uv run python -m bench --tier full --backend postgres
uv run python -m bench --tier full --backend duckdb
```

`bench.seed` writes a handle to `bench/state.json` (gitignored) keyed
by `(tier, backend)`. `python -m bench --tier full --backend X` reads
that handle and refuses to run if you haven't seeded yet.

`state.json` records the shape's content hash, so if you change a
shape definition in `bench/shapes.py` you should re-seed (the harness
doesn't auto-detect drift today; that's a future addition if it
becomes a footgun).

## Workloads

| Name                    | What it measures                                          |
|-------------------------|-----------------------------------------------------------|
| `tiny_limit`            | `LIMIT 100` — predicate-pushed read, dominated by metadata |
| `full_scan`             | `SELECT *` — bulk decode + transport                      |
| `predicate_partition`   | filter on partition column — should prune most files      |
| `predicate_nonpartition`| filter on non-partition column — runtime filter only      |
| `time_travel`           | read at an older snapshot id                              |
| `schema_evolved`        | read a table after add-column events                      |

## Readers

- `polars_ducklake` — this project: `scan_ducklake(...).collect()` straight to Polars.
- `ducklake_polars` — the rival pure-Python reader on PyPI (different
  author, similar idea). Wired in via the same `scan_ducklake(...).collect()`
  pattern, with API translation (raw catalog path instead of SQLAlchemy
  URL, `snapshot_version=` instead of `snapshot_id=`, no
  `storage_options`). See "Comparing against `ducklake-polars`" below
  for current limitations.
- `duckdb_arrow_only` — DuckDB ducklake extension materializing a pyarrow Table (no Polars hop).
- `duckdb_arrow_to_polars` — DuckDB → arrow → `pl.from_arrow`. The natural apples-to-apples comparison.

## Output schema

One JSONL row per `(workload, reader)`:

```json
{
  "git_sha": "abc1234",
  "host_fingerprint": {"system": "...", "machine": "...", "cpu_count": 8, "ram_gb": 16},
  "timestamp": "2026-05-01T12:34:56Z",
  "tier": "ci",
  "backend": "sqlite",
  "table_shape": "bench_ci",
  "workload": "full_scan",
  "reader": "polars_ducklake",
  "wallclock_ms_median": 142.3,
  "wallclock_ms_p95": 158.7,
  "conn_count": 2,
  "metadata_query_count": 7,
  "rows_returned": 100000
}
```

`conn_count` and `metadata_query_count` are populated only for
`polars_ducklake` (instrumented via SQLAlchemy event listeners on the
catalog engine). For DuckDB readers they are reported as `null`.

## Methodology

- 1 untimed warmup + 5 timed reps per `(workload, reader)`.
- Min and max are dropped; median + p95 are reported over the remaining 3.
- Numbers are only meaningful relative to the host they were captured
  on. Each run records `platform.uname()` + CPU count + total RAM
  under `host_fingerprint`. Don't compare medians across hosts.

This is a "way off base?" signal, not a regression gate.

## Comparing against `ducklake-polars`

There is a separate, similarly-named package on PyPI —
[`ducklake-polars`](https://pypi.org/project/ducklake-polars/) (by a
different author) — that pursues the same goal: read DuckLake tables
into Polars without DuckDB at runtime. The bench harness wires it in
as a fourth reader so we can measure both side-by-side.

**Today, a head-to-head wallclock comparison on the bench lake is not
possible.** The reason is a non-overlapping spec-version matrix:

| Component                       | DuckLake spec it speaks |
|---------------------------------|--------------------------|
| `polars-ducklake` (this project)| v1.0 only (uses the `partial_max` column added in v1.0) |
| `ducklake-polars` 0.1.1 (rival) | v0.3 only (rejects v1.0 with `Unsupported DuckLake catalog version '1.0'`) |
| DuckDB 1.3.x ducklake extension | writes v0.2 catalogs |
| DuckDB 1.4.x ducklake extension | writes v0.3 catalogs (and has a `CHECKPOINT` binder bug on the v0.3 metadata) |
| DuckDB 1.5.2+ ducklake extension| writes v1.0 catalogs |

There is no DuckDB version that produces a catalog both readers
accept, so the bench harness emits a single skip message and proceeds
with the other three readers when the lake is v1.0. The skip is
preserved in the runner code so a future ducklake-polars release that
adopts v1.0 will light up the comparison automatically.

If you want to drive the rival writer directly (rather than the
DuckDB-extension-managed seed), add a paradigm-3 fixture using
`ducklake_polars.write_ducklake` — but that is testing the rival
end-to-end, not the read path against canonical DuckDB-extension
output, which is the question the harness is built around.

## Two test paradigms in this repo

The integration suite (`tests/integration/`) and bench harness
exercise *two distinct* paradigms for building DuckLake fixtures, and
both are kept around on purpose:

1. **Hand-built (`tests/conftest.py: DuckLakeBuilder`).** Builds the
   metadata tables directly from the v1.0 spec without DuckDB. Used
   by spec-conformance unit tests (everything in `tests/test_*.py`)
   to prove the reader matches the spec independently of any one
   writer.

2. **DuckDB-extension-managed (`bench/seed.py: materialize_lake`).**
   Drives the canonical DuckDB ducklake-extension writer to produce
   real-world metadata. Used by the bench harness and any future
   integration tests that want representative writer output. This is
   what your end users will actually have written.

The bench harness uses paradigm (2) exclusively because the question
it answers — "are we competitive with DuckDB on lakes that real users
write?" — only makes sense against the canonical writer.
