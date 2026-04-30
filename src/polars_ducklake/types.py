"""Map DuckLake column type strings to Polars data types.

Type strings live in ``ducklake_column.column_type``. The DuckLake v1.0 spec
(https://ducklake.select/docs/stable/specification/data_types) defines them in
lowercase form (``int32``, ``varchar``, ``decimal(P, S)``), but DuckDB — the
canonical writer — emits the uppercase SQL aliases (``INTEGER``, ``BIGINT``,
``VARCHAR``, ``DECIMAL(18,3)``). We accept both forms and map them
case-insensitively.

Scalar types and ``DECIMAL(p, s)`` map directly via :func:`map_type`.
Nested types (``LIST``, ``STRUCT``, ``MAP``) are decomposed by the catalog
across multiple ``ducklake_column`` rows linked by ``parent_column``,
and are assembled via :func:`build_nested_type` once each child has
been resolved (recursively, for nested-of-nested cases).
"""

from __future__ import annotations

import re

import polars as pl

# Spec lowercase names and DuckDB-emitted uppercase aliases all collapse to the
# same Polars type. Keys are stored lowercase; lookups normalise case.
_SCALAR_MAP: dict[str, pl.DataType] = {
    "boolean": pl.Boolean(),
    "bool": pl.Boolean(),
    "int8": pl.Int8(),
    "tinyint": pl.Int8(),
    "int16": pl.Int16(),
    "smallint": pl.Int16(),
    "int32": pl.Int32(),
    "integer": pl.Int32(),
    "int": pl.Int32(),
    "int64": pl.Int64(),
    "bigint": pl.Int64(),
    "uint8": pl.UInt8(),
    "utinyint": pl.UInt8(),
    "uint16": pl.UInt16(),
    "usmallint": pl.UInt16(),
    "uint32": pl.UInt32(),
    "uinteger": pl.UInt32(),
    "uint64": pl.UInt64(),
    "ubigint": pl.UInt64(),
    "float32": pl.Float32(),
    "float": pl.Float32(),
    "real": pl.Float32(),
    "float64": pl.Float64(),
    "double": pl.Float64(),
    "varchar": pl.String(),
    "string": pl.String(),
    "text": pl.String(),
    "blob": pl.Binary(),
    "binary": pl.Binary(),
    "bytea": pl.Binary(),
    "date": pl.Date(),
    "time": pl.Time(),
    # timetz has no direct Polars equivalent; v0.1 maps it to Time and lets the
    # user handle timezone semantics on their side. Surfaced explicitly so a
    # future revision can revisit.
    "timetz": pl.Time(),
    "timestamp": pl.Datetime(time_unit="us"),
    "timestamp_s": pl.Datetime(time_unit="ms"),  # closest available unit
    "timestamp_ms": pl.Datetime(time_unit="ms"),
    "timestamp_ns": pl.Datetime(time_unit="ns"),
    "timestamptz": pl.Datetime(time_unit="us", time_zone="UTC"),
    "uuid": pl.String(),  # Polars has no UUID type; surface as String
    "json": pl.String(),  # JSON stored as text; users can parse downstream
}

_DECIMAL_RE = re.compile(r"^\s*decimal\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*$", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"^\s*numeric\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*$", re.IGNORECASE)
_NESTED_RE = re.compile(r"^\s*(list|struct|map)\b", re.IGNORECASE)


def map_type(column_type: str, *, column_name: str | None = None) -> pl.DataType:
    """Translate a *scalar* DuckLake column-type string to a Polars dtype.

    For nested types (LIST / STRUCT / MAP) the catalog stores additional
    rows linked by ``parent_column``, and the full type can only be
    assembled from the tree — see :func:`build_nested_type`. This
    function raises :class:`ValueError` if asked to map a nested type
    in isolation, since the result would be incomplete.

    Parameters
    ----------
    column_type:
        The string stored in ``ducklake_column.column_type``.
    column_name:
        Optional column name used only to produce more useful error messages.

    Raises
    ------
    ValueError
        If the type string is not recognised, or if it names a nested
        type that should be assembled via :func:`build_nested_type`.
    """
    raw = column_type.strip()
    lowered = raw.lower()

    if lowered in _SCALAR_MAP:
        return _SCALAR_MAP[lowered]

    decimal_match = _DECIMAL_RE.match(raw) or _NUMERIC_RE.match(raw)
    if decimal_match:
        precision = int(decimal_match.group(1))
        scale = int(decimal_match.group(2))
        return pl.Decimal(precision=precision, scale=scale)

    if _NESTED_RE.match(raw):
        prefix = f"{column_name}: " if column_name else ""
        raise ValueError(
            f"{prefix}nested DuckLake types must be assembled via "
            f"build_nested_type(), not map_type(); got {column_type!r}."
        )

    prefix = f"{column_name}: " if column_name else ""
    raise ValueError(
        f"{prefix}unrecognised DuckLake column type {column_type!r}. "
        "If this is a valid v1.0 type that polars-ducklake should support, please "
        "file an issue."
    )


def is_nested_type(column_type: str) -> bool:
    """Return True if the type string names a DuckLake nested type."""
    return bool(_NESTED_RE.match(column_type.strip()))


# Spec field-name conventions for nested children, used to drive the
# Polars-side type construction. Confirmed against the DuckLake spec at
# https://ducklake.select/docs/stable/specification/data_types.
_LIST_ELEMENT_NAME = "element"
_MAP_KEY_NAME = "key"
_MAP_VALUE_NAME = "value"


def build_nested_type(
    column_type: str,
    children: list[tuple[str, pl.DataType]],
    *,
    column_name: str | None = None,
) -> pl.DataType:
    """Construct a Polars dtype for a nested column from its already-resolved children.

    ``children`` is a list of ``(child_name, child_dtype)`` pairs in the
    catalog's stored order. Caller is responsible for resolving each
    child recursively — for STRUCT children that are themselves nested,
    the dtype passed in must already be the assembled nested type.

    Maps are encoded as ``pl.List(pl.Struct({key, value}))`` to mirror
    Parquet's on-disk representation; users get back exactly what
    Polars writes for a Map column.
    """
    raw = column_type.strip().lower()
    if raw == "list":
        if len(children) != 1:
            raise ValueError(
                f"LIST column expected exactly 1 child (the element), got {len(children)}: "
                f"{[name for name, _ in children]!r}"
            )
        return pl.List(children[0][1])
    if raw == "struct":
        if not children:
            raise ValueError("STRUCT column must have at least one field child")
        return pl.Struct({name: dtype for name, dtype in children})
    if raw == "map":
        # Polars represents maps as List(Struct{key, value}) — same as
        # the Parquet on-disk form, which the DuckDB writer produces.
        if len(children) != 2:
            raise ValueError(
                f"MAP column expected key + value children, got {len(children)}: "
                f"{[name for name, _ in children]!r}"
            )
        # The catalog stores them in a fixed order, but we sort by name
        # for safety so we always emit (key, value) regardless of how
        # the writer ordered them.
        by_name = dict(children)
        if _MAP_KEY_NAME not in by_name or _MAP_VALUE_NAME not in by_name:
            raise ValueError(
                f"MAP column children must be named {_MAP_KEY_NAME!r} and "
                f"{_MAP_VALUE_NAME!r}; got {sorted(by_name)!r}"
            )
        return pl.List(
            pl.Struct(
                {_MAP_KEY_NAME: by_name[_MAP_KEY_NAME], _MAP_VALUE_NAME: by_name[_MAP_VALUE_NAME]}
            )
        )
    prefix = f"{column_name}: " if column_name else ""
    raise ValueError(f"{prefix}unrecognised nested DuckLake type {column_type!r}")
