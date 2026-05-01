"""``python -m bench`` CLI entry point."""

from __future__ import annotations

import argparse
import contextlib
import sys
import tempfile
from pathlib import Path

from bench.fixtures import build_bench_ci_lake, get_full_lake
from bench.harness import READERS, run_bench, write_jsonl
from bench.shapes import shape_for_tier
from bench.workloads import WORKLOADS, workloads_by_name


def _parse_csv(s: str | None) -> list[str] | None:
    if s is None:
        return None
    return [p.strip() for p in s.split(",") if p.strip()]


@contextlib.contextmanager
def _resolve_lake(tier: str, backend: str):  # type: ignore[no-untyped-def]
    """Yield a :class:`BenchLake` for ``(tier, backend)``.

    CI tier: built fresh into a temp dir, cleaned up at exit. Full
    tier: looked up in ``bench/state.json`` (the user is expected to
    have run ``python -m bench.seed`` first).
    """
    if tier == "ci":
        if backend != "sqlite":
            raise SystemExit(
                f"tier=ci only supports backend=sqlite (got {backend!r}); "
                "use --tier full for postgres/duckdb"
            )
        tmp = tempfile.TemporaryDirectory(prefix="bench-ci-")
        try:
            print(f"building bench_ci lake under {tmp.name}...", file=sys.stderr)
            yield build_bench_ci_lake(Path(tmp.name))
        finally:
            tmp.cleanup()
    elif tier == "full":
        lake = get_full_lake(backend)
        print(
            f"using seeded full lake: backend={backend} catalog={lake.sqlalchemy_url}",
            file=sys.stderr,
        )
        yield lake
    else:
        raise SystemExit(f"unknown tier: {tier!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench")
    parser.add_argument("--tier", choices=("ci", "full"), default="ci")
    parser.add_argument(
        "--backend",
        choices=("sqlite", "postgres", "duckdb"),
        default="sqlite",
        help="catalog backend (CI tier supports sqlite only; full tier supports all three)",
    )
    parser.add_argument(
        "--workloads",
        type=str,
        default=None,
        help=f"comma-separated subset of {[w.name for w in WORKLOADS]}",
    )
    parser.add_argument(
        "--readers",
        type=str,
        default=None,
        help=f"comma-separated subset of {list(READERS)}",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    args = parser.parse_args(argv)

    workloads = workloads_by_name(_parse_csv(args.workloads))
    readers = _parse_csv(args.readers) or list(READERS)
    for r in readers:
        if r not in READERS:
            print(f"unknown reader: {r!r}; available: {list(READERS)}", file=sys.stderr)
            return 2

    shape = shape_for_tier(args.tier)

    with _resolve_lake(args.tier, args.backend) as lake:
        print(
            f"running {len(workloads)} workloads x {len(readers)} readers "
            f"(tier={args.tier}, backend={args.backend})...",
            file=sys.stderr,
        )
        rows = run_bench(
            lake,
            tier=args.tier,
            backend=args.backend,
            table_shape=shape.name,
            workloads=workloads,
            readers=readers,
        )

    out = write_jsonl(rows, args.results_dir)
    print(f"\nwrote {len(rows)} rows to {out}\n", file=sys.stderr)

    name_w = max(len(r.workload) for r in rows)
    reader_w = max(len(r.reader) for r in rows)
    print(
        f"{'workload':<{name_w}}  {'reader':<{reader_w}}  "
        f"{'median_ms':>10}  {'p95_ms':>10}  {'rows':>8}  {'conns':>6}  {'queries':>8}"
    )
    for r in rows:
        cc = "-" if r.conn_count is None else str(r.conn_count)
        mq = "-" if r.metadata_query_count is None else str(r.metadata_query_count)
        print(
            f"{r.workload:<{name_w}}  {r.reader:<{reader_w}}  "
            f"{r.wallclock_ms_median:>10.2f}  {r.wallclock_ms_p95:>10.2f}  "
            f"{r.rows_returned:>8}  {cc:>6}  {mq:>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
