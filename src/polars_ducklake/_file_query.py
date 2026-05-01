"""Catalog-side file enumeration query (issue #4).

A single CTE-shaped query that replaces the prior three-query pattern
(fetch_data_files + fetch_delete_files + fetch_file_stats). Predicate
clauses translate to per-column stats CTEs; identity-partition clauses
translate to per-partition-key CTEs. Clauses we can't translate are
silently dropped — Polars still applies the full predicate at read
time, so a dropped clause widens the candidate file set without
affecting correctness.

This module is pure SQL string construction; it issues no queries. The
actual execution lives on :class:`polars_ducklake._catalog.CatalogReader`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping

from polars_ducklake._predicate import _AtomicClause


# ---------------------------------------------------------------------------
# Catalog column-type families
# ---------------------------------------------------------------------------

_INT_COLUMN_TYPES = frozenset({
    "tinyint", "smallint", "integer", "int", "bigint", "hugeint",
    "utinyint", "usmallint", "uinteger", "ubigint",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
})
_FLOAT_COLUMN_TYPES = frozenset({
    "float", "double", "real", "float4", "float8", "float32", "float64",
})
_STRING_COLUMN_TYPES = frozenset({"varchar", "string", "text"})


# Per-dialect CAST type names for translating stat min/max VARCHARs to a
# typed comparable. Empty string means "no cast" — stats compare as
# string for varchar/boolean (boolean stats are written as "0"/"1", and
# both the literal and the stat are bound/stored as those same strings).
_CAST_BY_DIALECT: dict[str, dict[str, str]] = {
    "sqlite":     {"int": "INTEGER", "float": "REAL",             "date": "",     "timestamp": ""},
    "postgresql": {"int": "BIGINT",  "float": "DOUBLE PRECISION", "date": "DATE", "timestamp": "TIMESTAMP"},
    "mysql":      {"int": "SIGNED",  "float": "DOUBLE",           "date": "DATE", "timestamp": "DATETIME"},
    "duckdb":     {"int": "BIGINT",  "float": "DOUBLE",           "date": "DATE", "timestamp": "TIMESTAMP"},
}

# Fallback dialect for unknowns. ANSI-ish; works on most engines.
_DEFAULT_DIALECT = "postgresql"


def _column_family(column_type: str) -> str | None:
    """Map a catalog ``column_type`` string to a CAST family.

    Returns ``None`` for column types we don't know how to compare
    (caller drops the clause). ``""`` family means "no cast needed".
    """
    ct = column_type.strip().lower()
    if ct in _STRING_COLUMN_TYPES or ct == "boolean":
        return ""
    if ct in _INT_COLUMN_TYPES:
        return "int"
    if ct in _FLOAT_COLUMN_TYPES:
        return "float"
    if ct == "date":
        return "date"
    if ct.startswith("timestamp"):
        return "timestamp"
    return None


def _typed_stat_expr(stat_col: str, column_type: str, dialect: str) -> str | None:
    """SQL expression that yields ``stat_col`` cast to a comparable type.

    Returns ``None`` for column types we don't know how to translate.
    Returns ``stat_col`` unchanged for varchar/boolean (string compare).
    Otherwise wraps in a dialect-appropriate ``CAST(... AS T)``.
    """
    family = _column_family(column_type)
    if family is None:
        return None
    if family == "":
        return stat_col
    cast_map = _CAST_BY_DIALECT.get(dialect, _CAST_BY_DIALECT[_DEFAULT_DIALECT])
    cast_type = cast_map.get(family, "")
    if not cast_type:
        return stat_col
    return f"CAST({stat_col} AS {cast_type})"


# ---------------------------------------------------------------------------
# Per-clause translation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ClauseSql:
    """SQL keep-condition for one atomic clause + its bind params.

    ``sql`` is meant to be AND'd inside a per-column stats CTE. It
    already wraps the bound check in conservative-keep guards (NaN-
    tainted, NULL min/max), so a row is kept whenever stats are
    incomplete or unreliable.
    """

    sql: str
    params: dict[str, Any]


def translate_clause(
    clause: _AtomicClause,
    *,
    column_type: str,
    dialect: str,
    param_prefix: str,
) -> _ClauseSql | None:
    """Translate one :class:`_AtomicClause` to a stats keep-condition.

    Returns ``None`` when ``column_type`` isn't one we know how to
    translate (e.g. unsupported type, struct/list). ``param_prefix``
    namespaces the bind parameter names so multiple clauses on the same
    column can coexist in one CTE.
    """
    op = clause.op
    if op == "is_null":
        return _ClauseSql(
            sql="(null_count IS NULL OR null_count > 0)",
            params={},
        )
    if op == "is_not_null":
        return _ClauseSql(
            sql=(
                "(null_count IS NULL OR value_count IS NULL "
                "OR null_count < value_count)"
            ),
            params={},
        )

    typed_min = _typed_stat_expr("min_value", column_type, dialect)
    typed_max = _typed_stat_expr("max_value", column_type, dialect)
    if typed_min is None or typed_max is None:
        return None

    bind_value = _bind_value_for(clause.literal, column_type)
    if bind_value is _BIND_UNSUPPORTED:
        return None

    p_name = f"{param_prefix}_v"
    p_ref = f":{p_name}"

    if op == "eq":
        bound = f"({typed_min} <= {p_ref} AND {typed_max} >= {p_ref})"
    elif op == "ne":
        # Drop only when every non-null value is provably ``lit``:
        # min == max == lit AND value_count > null_count. Keep otherwise.
        bound = (
            f"({typed_min} <> {p_ref} OR {typed_max} <> {p_ref} "
            f"OR null_count IS NULL OR value_count IS NULL "
            f"OR value_count <= null_count)"
        )
    elif op == "lt":
        bound = f"({typed_min} < {p_ref})"
    elif op == "le":
        bound = f"({typed_min} <= {p_ref})"
    elif op == "gt":
        bound = f"({typed_max} > {p_ref})"
    elif op == "ge":
        bound = f"({typed_max} >= {p_ref})"
    else:
        return None

    # Conservative-keep guards: NaN poisons numeric ordering, missing
    # min/max means "no info". Either case keeps the file regardless of
    # the bound check.
    nan_guard = "(contains_nan IS NOT NULL AND contains_nan = 1)"
    null_guard = "(min_value IS NULL OR max_value IS NULL)"
    full = f"({nan_guard} OR {null_guard} OR {bound})"
    return _ClauseSql(sql=full, params={p_name: bind_value})


# Sentinel for "literal cannot be bound" — distinguishes from the
# legitimate value ``None``.
_BIND_UNSUPPORTED: Any = object()


def _bind_value_for(literal: Any, column_type: str) -> Any:
    """Convert an extractor literal to a bind value for the typed CAST.

    Booleans on a ``boolean`` catalog column become the same ``"0"``/``"1"``
    strings the writer puts in ``min_value``/``max_value``. Everything
    else passes through and SQLAlchemy binds by Python type.
    """
    if isinstance(literal, bool):
        # bool is a subclass of int; intercept before the int branch.
        if column_type.strip().lower() == "boolean":
            return "1" if literal else "0"
        return _BIND_UNSUPPORTED
    if isinstance(literal, (int, float, str, date, datetime)):
        return literal
    return _BIND_UNSUPPORTED


# ---------------------------------------------------------------------------
# Partition spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartitionField:
    """One partition column descriptor at a given snapshot.

    ``transform`` is the DuckLake partition transform name (``identity``,
    ``year``, ``month``, ``day``, ``bucket``, etc.). Only ``identity`` is
    pushable in this iteration; the rest are kept on the field for
    future work but produce no CTE today.
    """

    column_id: int
    column_name: str
    column_type: str
    partition_key_index: int
    transform: str


# ---------------------------------------------------------------------------
# Full query builder
# ---------------------------------------------------------------------------


# MVCC visibility fragment used inside both visible_files and
# visible_deletes CTEs. References the bind ``:snapshot_id``.
_MVCC = (
    ":snapshot_id >= begin_snapshot AND "
    "(end_snapshot IS NULL OR :snapshot_id < end_snapshot)"
)


def build_files_query(
    *,
    table_id: int,
    snapshot_id: int,
    column_meta: Mapping[str, tuple[int, str]],
    partition_spec: list[PartitionField],
    clauses: list[_AtomicClause],
    dialect: str,
) -> tuple[str, dict[str, Any]]:
    """Compose the single CTE-shaped file enumeration query.

    Returns ``(sql_text, bind_params)`` ready for SQLAlchemy ``text()``.
    The result is files visible at ``snapshot_id`` for ``table_id``,
    LEFT-joined to their delete files, with predicate pushdown into
    per-column stats CTEs and identity-partition CTEs.

    ``column_meta`` maps user-visible column name → (column_id,
    column_type). Clauses on columns not in ``column_meta`` are dropped.
    ``clauses`` is the lossy output of :func:`extract_clauses`; this
    builder only tightens the file set, never widens it.

    The returned rows are ordered by ``ducklake_data_file.file_order``
    so callers see files in writer-order; multiple delete files for the
    same data file appear as multiple rows (Python-side aggregation).
    """
    params: dict[str, Any] = {
        "table_id": table_id,
        "snapshot_id": snapshot_id,
    }

    clauses_by_col: dict[str, list[_AtomicClause]] = {}
    for clause in clauses:
        clauses_by_col.setdefault(clause.column, []).append(clause)

    stats_ctes: list[str] = []
    partition_ctes: list[str] = []
    where_filters: list[str] = []

    partition_by_column: dict[str, PartitionField] = {
        f.column_name: f
        for f in partition_spec
        if f.transform == "identity"
    }

    for col_name, col_clauses in clauses_by_col.items():
        col_ref = column_meta.get(col_name)
        if col_ref is not None:
            cte_sql, cte_params, alias = _build_stats_cte(
                column_id=col_ref[0],
                column_type=col_ref[1],
                clauses=col_clauses,
                dialect=dialect,
            )
            if cte_sql is not None:
                stats_ctes.append(cte_sql)
                params.update(cte_params)
                where_filters.append(
                    f"vf.data_file_id IN (SELECT data_file_id FROM {alias})"
                )

        pf = partition_by_column.get(col_name)
        if pf is not None:
            cte_sql, cte_params, alias = _build_partition_cte(
                field=pf,
                clauses=col_clauses,
                dialect=dialect,
            )
            if cte_sql is not None:
                partition_ctes.append(cte_sql)
                params.update(cte_params)
                where_filters.append(
                    f"vf.data_file_id IN (SELECT data_file_id FROM {alias})"
                )

    cte_pieces = [
        _visible_files_cte(),
        _visible_deletes_cte(),
        *stats_ctes,
        *partition_ctes,
    ]
    with_block = "WITH " + ",\n".join(cte_pieces)

    where_clause = ""
    if where_filters:
        where_clause = "WHERE " + "\n  AND ".join(where_filters) + "\n"

    sql = (
        f"{with_block}\n"
        "SELECT vf.data_file_id, vf.path, vf.path_is_relative,\n"
        "       vf.record_count, vf.begin_snapshot, vf.partial_max,\n"
        "       vd.delete_path, vd.delete_path_is_relative\n"
        "FROM visible_files vf\n"
        "LEFT JOIN visible_deletes vd USING (data_file_id)\n"
        f"{where_clause}"
        "ORDER BY vf.file_order, vd.delete_path"
    )
    return sql, params


def _visible_files_cte() -> str:
    return (
        "visible_files AS (\n"
        "  SELECT data_file_id, path, path_is_relative, record_count,\n"
        "         begin_snapshot, partial_max, file_order\n"
        "  FROM ducklake_data_file\n"
        f"  WHERE table_id = :table_id AND ({_MVCC})\n"
        ")"
    )


def _visible_deletes_cte() -> str:
    return (
        "visible_deletes AS (\n"
        "  SELECT data_file_id,\n"
        "         path AS delete_path,\n"
        "         path_is_relative AS delete_path_is_relative\n"
        "  FROM ducklake_delete_file\n"
        f"  WHERE table_id = :table_id AND ({_MVCC})\n"
        ")"
    )


def _build_stats_cte(
    *,
    column_id: int,
    column_type: str,
    clauses: list[_AtomicClause],
    dialect: str,
) -> tuple[str | None, dict[str, Any], str]:
    """Build a per-column stats CTE: file_ids that may match all clauses,
    UNION'd with file_ids that have no stats row for the column.

    Returns ``(cte_sql, params, alias)``; ``cte_sql`` is ``None`` when
    no clauses on this column translated.
    """
    alias = f"col_{column_id}_stats"
    parts: list[str] = []
    params: dict[str, Any] = {f"col_{column_id}_id": column_id}
    for i, c in enumerate(clauses):
        translated = translate_clause(
            c,
            column_type=column_type,
            dialect=dialect,
            param_prefix=f"col_{column_id}_{i}",
        )
        if translated is None:
            continue
        parts.append(translated.sql)
        params.update(translated.params)
    if not parts:
        return None, {}, alias
    bound_block = "\n    AND ".join(parts)
    cte_sql = (
        f"{alias} AS (\n"
        "  SELECT data_file_id FROM ducklake_file_column_stats\n"
        f"  WHERE table_id = :table_id AND column_id = :col_{column_id}_id\n"
        f"    AND {bound_block}\n"
        "  UNION ALL\n"
        "  SELECT data_file_id FROM visible_files\n"
        "  WHERE data_file_id NOT IN (\n"
        "    SELECT data_file_id FROM ducklake_file_column_stats\n"
        f"    WHERE column_id = :col_{column_id}_id\n"
        "  )\n"
        ")"
    )
    return cte_sql, params, alias


def _build_partition_cte(
    *,
    field: PartitionField,
    clauses: list[_AtomicClause],
    dialect: str,
) -> tuple[str | None, dict[str, Any], str]:
    """Build an identity-partition CTE: file_ids whose partition_value
    matches the clause, UNION'd with files that have no partition_value
    row for this key (kept conservatively).
    """
    pki = field.partition_key_index
    alias = f"part_{pki}_filter"
    parts: list[str] = []
    params: dict[str, Any] = {f"pki_{pki}": pki}
    for i, c in enumerate(clauses):
        translated = _translate_partition_clause(
            c,
            column_type=field.column_type,
            dialect=dialect,
            param_prefix=f"pki_{pki}_{i}",
        )
        if translated is None:
            continue
        parts.append(translated.sql)
        params.update(translated.params)
    if not parts:
        return None, {}, alias
    bound_block = "\n    AND ".join(parts)
    cte_sql = (
        f"{alias} AS (\n"
        "  SELECT data_file_id FROM ducklake_file_partition_value\n"
        f"  WHERE table_id = :table_id AND partition_key_index = :pki_{pki}\n"
        f"    AND {bound_block}\n"
        "  UNION ALL\n"
        "  SELECT vf.data_file_id FROM visible_files vf\n"
        "  WHERE NOT EXISTS (\n"
        "    SELECT 1 FROM ducklake_file_partition_value fpv\n"
        "    WHERE fpv.data_file_id = vf.data_file_id\n"
        f"      AND fpv.partition_key_index = :pki_{pki}\n"
        "  )\n"
        ")"
    )
    return cte_sql, params, alias


def _translate_partition_clause(
    clause: _AtomicClause,
    *,
    column_type: str,
    dialect: str,
    param_prefix: str,
) -> _ClauseSql | None:
    """Translate a clause to a single-value match against ``partition_value``.

    Identity-transform only. Files with NULL ``partition_value`` are
    kept conservatively (the writer may have left it unset).
    """
    op = clause.op
    if op == "is_null":
        return _ClauseSql("(partition_value IS NULL)", {})
    if op == "is_not_null":
        return _ClauseSql("(partition_value IS NOT NULL)", {})

    typed = _typed_stat_expr("partition_value", column_type, dialect)
    if typed is None:
        return None
    bind = _bind_value_for(clause.literal, column_type)
    if bind is _BIND_UNSUPPORTED:
        return None

    op_sql = {
        "eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
    }.get(op)
    if op_sql is None:
        return None

    p_name = f"{param_prefix}_v"
    p_ref = f":{p_name}"
    return _ClauseSql(
        sql=f"(partition_value IS NULL OR {typed} {op_sql} {p_ref})",
        params={p_name: bind},
    )
