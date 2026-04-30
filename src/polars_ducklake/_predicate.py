"""Catalog-stats predicate pruning.

The Polars IO plugin hands us the user's predicate as a deserialized
:class:`polars.Expr`. We walk its JSON form and try to extract a flat
list of "atomic" clauses — each one a ``column <op> literal`` test or
an ``is_null`` / ``is_not_null`` check, joined only by AND. On anything
we don't recognize (OR, arithmetic on a column, function calls,
``is_in``, struct/list ops, etc) we return ``None`` and the caller
falls back to "don't prune" — Polars will still apply the predicate
itself on the rows we yield.

For each kept clause we evaluate it against a file's per-column
``ducklake_file_column_stats`` row (min, max, null counts) and drop the
file when the clause is provably false everywhere in it. The min/max
values are stored as VARCHAR by DuckLake's writer; we coerce them to
typed Python values per the catalog's ``column_type`` so comparisons
behave correctly (e.g. integer order, not string order).

This is a best-effort layer above Polars' Parquet-level row-group
pruning: we get to skip files we never open at all, while Polars still
gets to skip row groups within the files we do open.

This module is private; behavior may change without notice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal

import polars as pl

from polars_ducklake._catalog import FileColumnStat

_Op = Literal["eq", "ne", "lt", "le", "gt", "ge", "is_null", "is_not_null"]

# Polars BinaryExpr op names → our internal op tags. Anything not in this
# map is unsupported (And is handled separately as a node-splitter; Or is
# deliberately absent — we'd need to fold disjunctions across files,
# which we don't.)
_BINOP_TO_ATOMIC: dict[str, _Op] = {
    "Eq": "eq",
    "NotEq": "ne",
    "Lt": "lt",
    "LtEq": "le",
    "Gt": "gt",
    "GtEq": "ge",
}

# DuckLake catalog column-type strings (lowercased) we know how to coerce.
_INT_TYPES = {
    "tinyint", "smallint", "integer", "int", "bigint", "hugeint",
    "utinyint", "usmallint", "uinteger", "ubigint",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
}
_FLOAT_TYPES = {
    "float", "double", "real", "float4", "float8", "float32", "float64",
}
_STRING_TYPES = {"varchar", "string", "text"}

_EPOCH = date(1970, 1, 1)


@dataclass(frozen=True)
class _AtomicClause:
    """One leaf comparison: ``col(name) <op> literal``.

    ``literal`` is ``None`` for the unary ``is_null`` / ``is_not_null``
    operators and a plain Python value otherwise (int / float / str /
    bool / date / datetime).
    """

    column: str
    op: _Op
    literal: Any


def extract_clauses(predicate: pl.Expr) -> list[_AtomicClause] | None:
    """Walk the predicate AST; return its atomic clauses or ``None``.

    Returns ``None`` (caller falls back to no pruning) on any of:

    * top-level OR
    * arithmetic on a column reference (``col + 1 > 5``)
    * function calls other than ``is_null`` / ``is_not_null``
    * ``is_in``, ``between``, struct/list field access, etc
    * literals we don't know how to coerce (e.g. timezone-aware datetimes)

    Returns ``[]`` (truthy = "no clauses to apply") only if the predicate
    is structurally empty, which shouldn't happen — Polars wouldn't
    serialize that — but is harmless if it does.
    """
    try:
        node = json.loads(predicate.meta.serialize(format="json"))
    except Exception:
        return None
    out: list[_AtomicClause] = []
    if not _walk(node, out):
        return None
    return out


def _walk(node: Any, out: list[_AtomicClause]) -> bool:
    """Recursively split ANDs and append atomic clauses. Returns False on
    the first unsupported subtree we encounter."""
    if not isinstance(node, dict):
        return False
    if "BinaryExpr" in node:
        be = node["BinaryExpr"]
        op = be.get("op")
        if op == "And":
            # Both sides must be supported; partial extraction would let
            # us drop files that the *unsupported* side would have kept.
            return _walk(be["left"], out) and _walk(be["right"], out)
        atomic = _BINOP_TO_ATOMIC.get(op)
        if atomic is None:
            return False
        col = _as_column(be.get("left"))
        if col is None:
            return False
        lit = _as_literal(be.get("right"))
        if lit is _UNSUPPORTED:
            return False
        out.append(_AtomicClause(column=col, op=atomic, literal=lit))
        return True
    if "Function" in node:
        fn = node["Function"]
        boolean_fn = fn.get("function", {}).get("Boolean")
        if boolean_fn not in ("IsNull", "IsNotNull"):
            return False
        inputs = fn.get("input") or []
        if len(inputs) != 1:
            return False
        col = _as_column(inputs[0])
        if col is None:
            return False
        out.append(
            _AtomicClause(
                column=col,
                op="is_null" if boolean_fn == "IsNull" else "is_not_null",
                literal=None,
            )
        )
        return True
    return False


def _as_column(node: Any) -> str | None:
    if isinstance(node, dict):
        c = node.get("Column")
        if isinstance(c, str):
            return c
    return None


# Sentinel: distinguishes "literal we don't recognize" (caller bails on
# the whole predicate) from "literal value is the Python None" (which
# we don't currently produce — null literals fail the `is_in` / `Null`
# branch of _as_literal and return _UNSUPPORTED).
_UNSUPPORTED: Any = object()


# Typed integer scalar tags Polars emits when its optimizer casts a
# literal to match the column's dtype. We treat all of them as plain
# Python ints — width matters to the engine, not to us.
_INT_KINDS = frozenset({
    "Int", "Int8", "Int16", "Int32", "Int64", "Int128",
    "UInt8", "UInt16", "UInt32", "UInt64",
})
_FLOAT_KINDS = frozenset({"Float", "Float32", "Float64"})


def _as_literal(node: Any) -> Any:
    """Extract a Python value from a Literal node, or _UNSUPPORTED."""
    if not isinstance(node, dict):
        return _UNSUPPORTED
    inner = node.get("Literal")
    if not isinstance(inner, dict):
        return _UNSUPPORTED
    # Polars wraps literals in either {"Dyn": {kind: value}} or
    # {"Scalar": {kind: value}} — Dyn for "untyped" literals (e.g. an
    # int written as 5 in Python), Scalar for typed ones (anything Polars
    # has cast to a specific dtype, plus dates/datetimes/etc).
    for wrapper in ("Dyn", "Scalar"):
        payload = inner.get(wrapper)
        if not isinstance(payload, dict) or len(payload) != 1:
            continue
        ((kind, value),) = payload.items()
        if kind in _INT_KINDS and isinstance(value, int):
            return value
        if kind in _FLOAT_KINDS and isinstance(value, (int, float)):
            return float(value)
        if kind == "String" and isinstance(value, str):
            return value
        if kind == "Boolean" and isinstance(value, bool):
            return value
        if kind == "Date" and isinstance(value, int):
            # Polars encodes dates as days since the Unix epoch.
            return _EPOCH + timedelta(days=value)
        if kind == "Datetime" and isinstance(value, list) and len(value) == 3:
            scalar, unit, tz = value
            if tz is not None:
                # Tz-aware comparisons against tz-naive stat strings would
                # need conversion; bail conservatively for now.
                return _UNSUPPORTED
            if not isinstance(scalar, int) or unit not in (
                "Microseconds", "Milliseconds", "Nanoseconds",
            ):
                return _UNSUPPORTED
            us = scalar
            if unit == "Milliseconds":
                us = scalar * 1000
            elif unit == "Nanoseconds":
                us = scalar // 1000
            return datetime(1970, 1, 1) + timedelta(microseconds=us)
    return _UNSUPPORTED


def _coerce_stat(raw: str | None, column_type: str) -> Any:
    """Catalog-VARCHAR → Python value matching the column's logical type.

    Returns ``None`` if ``raw`` is ``None`` (the catalog stored a NULL,
    e.g. all-null column) or if coercion fails. Callers should treat
    ``None`` as "no information" and conservatively keep the file.
    """
    if raw is None:
        return None
    ct = column_type.strip().lower()
    try:
        if ct in _INT_TYPES:
            return int(raw)
        if ct in _FLOAT_TYPES:
            return float(raw)
        if ct in _STRING_TYPES:
            return raw
        if ct == "boolean":
            # DuckLake writes booleans as "0"/"1" in stats.
            return raw == "1"
        if ct == "date":
            return date.fromisoformat(raw)
        if ct.startswith("timestamp"):
            # ISO with a space separator; fromisoformat tolerates the
            # space in 3.11+ but not all minor versions, so normalize.
            return datetime.fromisoformat(raw.replace(" ", "T"))
    except (ValueError, TypeError):
        return None
    return None


def file_can_match(
    *,
    stats_by_column: dict[int, FileColumnStat],
    clauses: list[_AtomicClause],
    column_meta: dict[str, tuple[int, str]],
) -> bool:
    """Decide whether a single data file could possibly match every clause.

    ``stats_by_column`` maps catalog column_id → its stats row for this
    file (missing keys mean no stats are available for that column on
    this file). ``column_meta`` maps user-visible column name → (column_id,
    column_type) so we can look up the right stats row and coerce its
    min/max correctly.

    Returns ``False`` only when at least one clause is provably false.
    Anything ambiguous (no stats, NaN-poisoned numerics, unrecognized
    column types) keeps the file — pruning errs on the side of False
    negatives, never False positives.
    """
    for clause in clauses:
        meta = column_meta.get(clause.column)
        if meta is None:
            # Predicate references a column we don't know about — keep.
            continue
        column_id, column_type = meta
        stat = stats_by_column.get(column_id)
        if stat is None:
            continue  # No stats for this column on this file → keep.

        if clause.op == "is_null":
            if stat.null_count is not None and stat.null_count == 0:
                return False
            continue
        if clause.op == "is_not_null":
            if (
                stat.null_count is not None
                and stat.value_count is not None
                and stat.null_count == stat.value_count
            ):
                return False
            continue

        # Comparison clauses need typed min/max + a typed literal.
        # NaN-tainted floats invalidate ordering: keep conservatively.
        if stat.contains_nan:
            continue

        cmin = _coerce_stat(stat.min_value, column_type)
        cmax = _coerce_stat(stat.max_value, column_type)
        if cmin is None or cmax is None:
            continue
        lit = clause.literal

        # If types don't compare cleanly (e.g. int vs str — shouldn't
        # happen if the predicate is well-formed, but defensive), keep.
        try:
            if clause.op == "eq" and (lit < cmin or lit > cmax):
                return False
            elif clause.op == "ne":
                # Only prune if every value equals lit, i.e. min == max == lit.
                if cmin == cmax == lit:
                    # Still need to make sure there are non-null values; if
                    # the column is all nulls, "ne lit" is vacuously true and
                    # we shouldn't prune.
                    has_non_null = (
                        stat.null_count is None
                        or stat.value_count is None
                        or stat.value_count > stat.null_count
                    )
                    if has_non_null:
                        return False
            elif (
                (clause.op == "lt" and lit <= cmin)
                or (clause.op == "le" and lit < cmin)
                or (clause.op == "gt" and lit >= cmax)
                or (clause.op == "ge" and lit > cmax)
            ):
                return False
        except TypeError:
            continue

    return True
