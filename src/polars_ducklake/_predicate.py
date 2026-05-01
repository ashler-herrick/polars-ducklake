"""Predicate AST extraction for catalog-side pushdown.

The Polars IO plugin hands us the user's predicate as a deserialized
:class:`polars.Expr`. We walk its JSON form and extract a list of
"atomic" clauses — each one a ``column <op> literal`` test or an
``is_null`` / ``is_not_null`` check. The walk is lossy by design: we
keep what we can express in SQL and silently drop anything we don't
(OR, arithmetic on a column, ``is_in``, struct/list ops, tz-aware
datetimes, etc). Polars still applies the full predicate to rows we
yield, so dropping a clause widens the candidate file set without
changing the result.

The actual translation from these clauses into the catalog-side CTE
query lives in :mod:`polars_ducklake._file_query`; this module's job
is purely to parse the Polars expression tree.

This module is private; behavior may change without notice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal

import polars as pl

_Op = Literal["eq", "ne", "lt", "le", "gt", "ge", "is_null", "is_not_null"]

# Polars BinaryExpr op names → our internal op tags. Anything not in this
# map is unsupported (And is handled separately as a node-splitter; Or
# drops both branches because pushing one side of a disjunction would
# lose files matching only the other side.)
_BINOP_TO_ATOMIC: dict[str, _Op] = {
    "Eq": "eq",
    "NotEq": "ne",
    "Lt": "lt",
    "LtEq": "le",
    "Gt": "gt",
    "GtEq": "ge",
}

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


def extract_clauses(predicate: pl.Expr) -> list[_AtomicClause]:
    """Walk the predicate AST; return atomic clauses we can push down.

    Lossy: returns whatever supported leaves we can extract from an outer
    AND, and silently drops the rest. Anything we don't translate — OR,
    arithmetic on a column, function calls beyond ``is_null`` /
    ``is_not_null``, ``is_in``, ``between``, struct/list ops, tz-aware
    datetimes — is dropped. Polars still applies the full predicate on
    the rows we yield, so a dropped clause widens the candidate file set
    without changing the result.

    Returns ``[]`` when nothing is pushable (including a top-level OR).
    """
    try:
        node = json.loads(predicate.meta.serialize(format="json"))
    except Exception:
        return []
    out: list[_AtomicClause] = []
    _walk(node, out)
    return out


def _walk(node: Any, out: list[_AtomicClause]) -> None:
    """Recursively split top-level ANDs and append atomic clauses.

    Drops unsupported subtrees silently. For OR we drop *both* branches:
    pushing only one side of a disjunction would lose files that match
    only the other side. AND is the one composing operator we honor —
    partial extraction across an AND is safe because Polars re-applies
    the full predicate at read time.
    """
    if not isinstance(node, dict):
        return
    if "BinaryExpr" in node:
        be = node["BinaryExpr"]
        op = be.get("op")
        if op == "And":
            _walk(be.get("left"), out)
            _walk(be.get("right"), out)
            return
        if op == "Or":
            return
        atomic = _BINOP_TO_ATOMIC.get(op)
        if atomic is None:
            return
        col = _as_column(be.get("left"))
        if col is None:
            return
        lit = _as_literal(be.get("right"))
        if lit is _UNSUPPORTED:
            return
        out.append(_AtomicClause(column=col, op=atomic, literal=lit))
        return
    if "Function" in node:
        fn = node["Function"]
        boolean_fn = fn.get("function", {}).get("Boolean")
        if boolean_fn not in ("IsNull", "IsNotNull"):
            return
        inputs = fn.get("input") or []
        if len(inputs) != 1:
            return
        col = _as_column(inputs[0])
        if col is None:
            return
        out.append(
            _AtomicClause(
                column=col,
                op="is_null" if boolean_fn == "IsNull" else "is_not_null",
                literal=None,
            )
        )


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


