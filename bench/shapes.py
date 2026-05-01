"""Declarative lake shapes used by the bench harness.

A :class:`Lakeshape` is a recipe — number of rows, partition column,
how many INSERT chunks (≈ how many files), how schema evolution looks.
:func:`materialize_lake` (in :mod:`bench.seed`) turns a shape into a
real ducklake against any catalog target.

Two named shapes ship with the harness:

* ``bench_ci`` — ~100k rows, 3 chunks, fits anywhere in seconds.
* ``bench_full`` — ~30M rows, 10 chunks, mirrors the issue #7 baseline.

The chunk generator is deterministic per ``(shape.name, shape.seed,
chunk_index)`` so re-seeding produces byte-identical Parquet.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import polars as pl


REGIONS = ("us", "eu", "apac")
N_SYMBOLS = 50
SYMBOLS = tuple(f"SYM{i:03d}" for i in range(N_SYMBOLS))


@dataclass(frozen=True)
class Lakeshape:
    """Describes the lake we want to materialize.

    ``rows`` is split as evenly as possible across ``n_chunks`` INSERTs.
    ``mid_chunk`` is 1-indexed: the post-``mid_chunk``-th-INSERT snapshot
    is captured for time-travel reads. ``mid_chunk`` must be < ``n_chunks``
    so there is at least one chunk after it.

    The schema-evolved sibling table is built when ``has_evolved_table``
    is true: same first chunk, ``ALTER TABLE ... ADD COLUMN venue``,
    one more INSERT.
    """

    name: str
    rows: int
    n_chunks: int
    mid_chunk: int
    partition_col: str = "region"
    has_evolved_table: bool = True
    seed: int = 1

    def chunk_sizes(self) -> list[int]:
        base, rem = divmod(self.rows, self.n_chunks)
        return [base + (1 if i < rem else 0) for i in range(self.n_chunks)]

    def hash(self) -> str:
        """Stable identifier for the shape definition.

        Used in ``bench/state.json`` so we can warn when a persisted lake
        was built against a different shape than the current code defines.
        """
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:12]


BENCH_CI = Lakeshape(
    name="bench_ci",
    rows=100_000,
    n_chunks=3,
    mid_chunk=2,
)

BENCH_FULL = Lakeshape(
    name="bench_full",
    rows=30_000_000,
    n_chunks=10,
    mid_chunk=7,
)

SHAPES: dict[str, Lakeshape] = {s.name: s for s in (BENCH_CI, BENCH_FULL)}


def shape_for_tier(tier: str) -> Lakeshape:
    if tier == "ci":
        return BENCH_CI
    if tier == "full":
        return BENCH_FULL
    raise ValueError(f"unknown tier: {tier!r}")


def generate_chunk(shape: Lakeshape, chunk_index: int) -> pl.DataFrame:
    """Deterministic per ``(shape.name, shape.seed, chunk_index)``.

    Columns: id, ts, region, symbol, amount.
    """
    sizes = shape.chunk_sizes()
    if not 0 <= chunk_index < len(sizes):
        raise ValueError(f"chunk_index {chunk_index} out of range for shape {shape.name}")
    n = sizes[chunk_index]
    start_id = sum(sizes[:chunk_index])

    # Per-chunk seed so chunks are independent yet reproducible.
    rng = random.Random(hash((shape.name, shape.seed, chunk_index)) & 0xFFFFFFFF)
    base = datetime(2024, 1, 1)
    one_year_seconds = 60 * 60 * 24 * 365

    return pl.DataFrame(
        {
            "id": list(range(start_id, start_id + n)),
            "ts": [base + timedelta(seconds=rng.randint(0, one_year_seconds)) for _ in range(n)],
            "region": [rng.choice(REGIONS) for _ in range(n)],
            "symbol": [rng.choice(SYMBOLS) for _ in range(n)],
            "amount": [round(rng.uniform(0, 1000), 2) for _ in range(n)],
        }
    )


def iter_chunks(shape: Lakeshape) -> Iterator[tuple[int, pl.DataFrame]]:
    """Yield ``(chunk_index, chunk_df)`` so callers can stream a large shape."""
    for i in range(shape.n_chunks):
        yield i, generate_chunk(shape, i)


def evolved_chunk(shape: Lakeshape, chunk_index: int, venue: str = "NYSE") -> pl.DataFrame:
    """A chunk for the schema-evolved sibling table — same data, plus ``venue``."""
    df = generate_chunk(shape, chunk_index)
    return df.with_columns(pl.Series("venue", [venue] * len(df)))
