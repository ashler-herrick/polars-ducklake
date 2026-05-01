"""On-demand benchmark harness for polars-ducklake vs. the DuckDB ducklake extension.

Not part of the published package. Run via ``python -m bench``; results
are written as JSONL under ``bench/results/`` (gitignored).

The CI tier (``--tier ci``) is self-contained: it builds a small
ducklake against a SQLite catalog and a local-FS data path, no
docker-compose stack required. The full tier is a placeholder until the
persistent multi-bucket fixture system in #9 lands.
"""
