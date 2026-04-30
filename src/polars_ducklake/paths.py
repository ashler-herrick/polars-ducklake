"""Resolve DuckLake data-file paths against the lake's path chain.

DuckLake locates a data file by composing four path segments, each
carrying its own ``path_is_relative`` flag:

1. ``ducklake_metadata.data_path`` — the lake-wide root.
2. ``ducklake_schema.path`` — per-schema sub-prefix.
3. ``ducklake_table.path`` — per-table sub-prefix.
4. ``ducklake_data_file.path`` — the file itself.

When a level's flag is *true*, that segment is appended to the
already-resolved prefix from the parent levels. When *false*, the
segment is treated as a fully-qualified location and replaces the
prefix entirely (used when a schema or table lives in a different
bucket from the rest of the lake).

The data path can be a local filesystem path or any object-storage URI
(``s3://``, ``gs://``, ``az://``, ``file://``); we deliberately do not
parse or normalise it beyond ensuring exactly one slash separates each
joined pair. Polars' ``scan_parquet`` (and the underlying object-store
crate) handles the actual URI resolution.

See https://ducklake.select/docs/stable/specification/tables/ducklake_metadata
and the ``path`` / ``path_is_relative`` columns on ``ducklake_schema``
and ``ducklake_table``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PathSegment:
    """A single (path, path_is_relative) tuple from the catalog."""

    path: str
    is_relative: bool


def resolve_path(
    *,
    path: str,
    path_is_relative: bool,
    data_path: str | None,
    schema_segment: PathSegment | None = None,
    table_segment: PathSegment | None = None,
) -> str:
    """Resolve a DuckLake data-file path through the full schema+table chain.

    Parameters
    ----------
    path:
        The ``path`` column from ``ducklake_data_file`` /
        ``ducklake_delete_file``.
    path_is_relative:
        The corresponding ``path_is_relative`` flag.
    data_path:
        The lake-wide ``data_path`` from ``ducklake_metadata``. Required
        when *every* segment from this point downward is relative.
    schema_segment:
        ``ducklake_schema``'s ``(path, path_is_relative)`` for this table's
        schema. Optional only for back-compat with simple test fixtures
        whose schemas have no path component.
    table_segment:
        ``ducklake_table``'s ``(path, path_is_relative)``. Same caveat.
    """
    base = _resolve_base(
        data_path=data_path,
        schema_segment=schema_segment,
        table_segment=table_segment,
    )

    if not path_is_relative:
        return path

    if base is None:
        raise ValueError(
            f"Data file path {path!r} is marked relative but no resolvable "
            "base prefix is available. The catalog has no 'data_path' entry "
            "in ducklake_metadata, the schema/table paths are also relative, "
            "and no data_path= override was passed to scan_ducklake."
        )

    return _join(base, path)


def _resolve_base(
    *,
    data_path: str | None,
    schema_segment: PathSegment | None,
    table_segment: PathSegment | None,
) -> str | None:
    """Compose the lake/schema/table prefixes into a single base for the file.

    Returns None only if every segment supplied is relative *and* there
    is no ``data_path`` to anchor them.
    """
    base = data_path

    if schema_segment is not None and schema_segment.path:
        base = _attach(base, schema_segment)

    if table_segment is not None and table_segment.path:
        base = _attach(base, table_segment)

    return base


def _attach(base: str | None, segment: PathSegment) -> str:
    """Apply one PathSegment on top of an existing base."""
    if not segment.is_relative:
        # Absolute segment overrides the parent prefix entirely.
        return segment.path
    if base is None:
        # Relative segment with no parent — nothing to anchor against;
        # treat the segment itself as the base. Resolution will only
        # succeed if a later level supplies an absolute path.
        return segment.path
    return _join(base, segment.path)


def _join(prefix: str, tail: str) -> str:
    """Join a prefix and a relative tail with exactly one separator slash."""
    return f"{prefix.rstrip('/')}/{tail.lstrip('/')}"
