"""Compare bench JSONL results across runs to spot regressions.

Loads every ``*.jsonl`` under ``bench/results/`` (or a passed-in directory),
groups by git SHA, and prints a per-(workload, reader) delta table against a
chosen baseline SHA. Cross-host comparisons are noisy, so by default we
filter to a single host fingerprint — the one matching the most recent run,
unless ``--host`` is given.

CLI::

    python -m bench.report --baseline caf2cfd
    python -m bench.report --baseline caf2cfd --target 478bbd6
    python -m bench.report --baseline caf2cfd --threshold 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl


_DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"
_DEFAULT_THRESHOLD_PCT = 5.0
_HOST_KEYS = ("system", "machine", "processor", "cpu_count", "ram_gb")


def _host_id(fp: dict[str, object]) -> str:
    return "|".join(f"{k}={fp.get(k)}" for k in _HOST_KEYS)


def load_results(results_dir: Path) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["host_id"] = _host_id(rec.get("host_fingerprint") or {})
                rec.pop("host_fingerprint", None)
                rows.append(rec)
    if not rows:
        raise SystemExit(f"no JSONL records under {results_dir}")
    return pl.DataFrame(rows)


def _latest_sha(df: pl.DataFrame) -> str:
    last = df.sort("timestamp", descending=True).head(1)
    return str(last["git_sha"][0])


def _aggregate(df: pl.DataFrame) -> pl.DataFrame:
    """Median wallclock per (sha, tier, backend, workload, reader) — collapses repeat runs."""
    return (
        df.group_by(["git_sha", "tier", "backend", "workload", "reader"])
        .agg(
            pl.col("wallclock_ms_median").median().alias("ms"),
            pl.len().alias("runs"),
        )
        .sort(["tier", "backend", "workload", "reader"])
    )


def compare(
    df: pl.DataFrame, baseline: str, target: str, threshold_pct: float
) -> pl.DataFrame:
    agg = _aggregate(df)
    base = agg.filter(pl.col("git_sha") == baseline).rename(
        {"ms": "baseline_ms", "runs": "baseline_runs"}
    ).drop("git_sha")
    tgt = agg.filter(pl.col("git_sha") == target).rename(
        {"ms": "target_ms", "runs": "target_runs"}
    ).drop("git_sha")
    if base.is_empty():
        raise SystemExit(f"no rows for baseline SHA {baseline!r}")
    if tgt.is_empty():
        raise SystemExit(f"no rows for target SHA {target!r}")

    joined = base.join(
        tgt, on=["tier", "backend", "workload", "reader"], how="full", coalesce=True
    )
    joined = joined.with_columns(
        delta_pct=(
            (pl.col("target_ms") - pl.col("baseline_ms")) / pl.col("baseline_ms") * 100.0
        ),
    ).with_columns(
        status=pl.when(pl.col("delta_pct").is_null())
        .then(pl.lit("missing"))
        .when(pl.col("delta_pct") > threshold_pct)
        .then(pl.lit("regressed"))
        .when(pl.col("delta_pct") < -threshold_pct)
        .then(pl.lit("improved"))
        .otherwise(pl.lit("flat")),
    )
    return joined.sort(["tier", "backend", "workload", "reader"])


def _print_sha_index(df: pl.DataFrame) -> None:
    """Show every SHA with its run timestamps, host, and tier/backend coverage."""
    idx = (
        df.group_by(["git_sha", "host_id"])
        .agg(
            pl.col("timestamp").min().alias("first_run"),
            pl.col("timestamp").max().alias("last_run"),
            pl.col("timestamp").n_unique().alias("runs"),
            pl.concat_str(
                [pl.col("tier"), pl.lit("|"), pl.col("backend")]
            ).unique().sort().str.join(",").alias("tier_backends"),
        )
        .sort("last_run", descending=True)
    )
    print(
        f"{'sha':<10} {'runs':>4}  {'first_run':<25} {'last_run':<25}  "
        f"{'host':<22}  tier|backend"
    )
    for r in idx.iter_rows(named=True):
        host_short = r["host_id"].split("|")[0].replace("system=", "")
        host_short += " " + (
            r["host_id"].split("cpu_count=")[-1].split("|")[0] + "c"
            if "cpu_count=" in r["host_id"] else ""
        )
        print(
            f"{r['git_sha']:<10} {r['runs']:>4}  "
            f"{r['first_run']:<25} {r['last_run']:<25}  "
            f"{host_short:<22}  {r['tier_backends']}"
        )


def _print_table(rows: pl.DataFrame, baseline: str, target: str) -> None:
    print(
        f"\n{'workload':<24} {'reader':<24} {'tier':<6} {'backend':<10} "
        f"{baseline[:8]:>10} {target[:8]:>10} {'delta_%':>9}  status"
    )
    for r in rows.iter_rows(named=True):
        b = r["baseline_ms"]
        t = r["target_ms"]
        d = r["delta_pct"]
        b_s = f"{b:>10.2f}" if b is not None else f"{'-':>10}"
        t_s = f"{t:>10.2f}" if t is not None else f"{'-':>10}"
        d_s = f"{d:>+8.1f}%" if d is not None else f"{'-':>9}"
        print(
            f"{r['workload']:<24} {r['reader']:<24} {r['tier']:<6} "
            f"{r['backend']:<10} {b_s} {t_s} {d_s}  {r['status']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.report")
    parser.add_argument(
        "--results", type=Path, default=_DEFAULT_RESULTS_DIR,
        help="directory of bench JSONL files (default: bench/results)",
    )
    parser.add_argument("--baseline", default=None, help="git SHA to compare against")
    parser.add_argument(
        "--list", action="store_true",
        help="list SHAs with run timestamps and host then exit",
    )
    parser.add_argument(
        "--target", default=None,
        help="git SHA to compare (default: most recent run on the chosen host)",
    )
    parser.add_argument(
        "--host", default=None,
        help="host_id substring filter (default: host of the most recent run)",
    )
    parser.add_argument(
        "--threshold", type=float, default=_DEFAULT_THRESHOLD_PCT,
        help="percent change to flag as regressed/improved (default: 5)",
    )
    parser.add_argument(
        "--show-missing", action="store_true",
        help="include rows where one side has no data (default: hide)",
    )
    args = parser.parse_args(argv)

    df = load_results(args.results)

    if args.list:
        _print_sha_index(df)
        return 0

    if args.baseline is None:
        parser.error("--baseline is required (use --list to see available SHAs)")

    if args.host is not None:
        df = df.filter(pl.col("host_id").str.contains(args.host, literal=True))
        if df.is_empty():
            raise SystemExit(f"no rows match --host {args.host!r}")
    else:
        latest_host = (
            df.sort("timestamp", descending=True).head(1)["host_id"][0]
        )
        df = df.filter(pl.col("host_id") == latest_host)

    target = args.target or _latest_sha(df)
    if target == args.baseline:
        print(
            f"warning: target SHA == baseline SHA ({target}); pick --target explicitly",
            file=sys.stderr,
        )

    result = compare(df, args.baseline, target, args.threshold)
    if not args.show_missing:
        result = result.filter(pl.col("status") != "missing")
    _print_table(result, args.baseline, target)

    regressed = result.filter(pl.col("status") == "regressed")
    if not regressed.is_empty():
        print(
            f"\n{len(regressed)} workload(s) regressed > {args.threshold:.0f}%",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
