"""The public ``scan_ducklake`` entry point.

This is the only function exported from the package. It wires together
the catalog-resolution, metadata-query, and path-resolution layers,
then hands the resolved Parquet path list to :func:`polars.scan_parquet`
so Polars' native query engine takes over (predicate pushdown, projection
pushdown, streaming, etc).

The architecture is deliberately minimal: there is no custom IO source,
no Arrow handoff, and no DuckDB process. Catalog access lives entirely
in :class:`polars_ducklake._catalog.CatalogReader`; this module focuses
on translating its results into a Polars LazyFrame.

See https://ducklake.select/docs/stable/specification/queries for the
canonical catalog SQL pattern.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from polars_ducklake._catalog import CatalogReader, ColumnInfo, DataFileInfo
from polars_ducklake.paths import PathSegment, resolve_path
from polars_ducklake.types import build_nested_type, is_nested_type, map_type

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


# Internal column name used to assign positions to data-file rows so we can
# anti-join against the delete-file's `pos` column. Underscore-prefixed and
# double-underscored to make accidental user collisions vanishingly unlikely.
_ROW_INDEX_COL = "__pdl_row_idx"

# Per-row snapshot column written by DuckLake's compactor into merged
# (partial) data files. Required to filter rows during time-travel reads
# at a target below ``partial_max`` — see _FilePlan.build.
_PARTIAL_SNAPSHOT_COL = "_ducklake_internal_snapshot_id"


@dataclass(frozen=True)
class _TargetColumn:
    """One column at the read target's snapshot."""

    column_id: int
    name: str
    dtype: pl.DataType
    # Catalog ``column_type`` string (lowercased). We retain this so the
    # projection layer can apply per-type physical-to-logical bridges —
    # currently just JSON columns, which Polars surfaces as ``pl.Binary``
    # from a DuckDB-written Parquet but our catalog mapping declares as
    # ``pl.String`` (matching how Polars treats raw JSON elsewhere).
    column_type: str = ""

    @property
    def needs_string_cast(self) -> bool:
        """True if reads of this column must be cast to ``pl.String``.

        Today only JSON columns trigger this. Polars 2.0 is expected
        to register the JSON Arrow extension by default, at which
        point we can drop the cast.
        """
        return self.column_type == "json" and self.dtype == pl.String()


def scan_ducklake(
    metadata_catalog: str | Engine,
    table: str,
    *,
    schema: str | None = None,
    snapshot_id: int | None = None,
    as_of: datetime | None = None,
    storage_options: dict[str, Any] | None = None,
    data_path: str | None = None,
) -> pl.LazyFrame:
    """Lazily read from a DuckLake table.

    The returned :class:`polars.LazyFrame` is the concatenation of
    :func:`polars.scan_parquet` calls over the table's data files (with
    positional deletes and schema-evolution projections applied
    per-file). All predicate / projection pushdown is handled by the
    Polars query engine on the resulting LazyFrame — there is no
    custom predicate handling in this package.

    Parameters
    ----------
    metadata_catalog:
        Identifies the SQL database hosting DuckLake's metadata tables —
        the *metadata* catalog, not the per-table catalog level of a
        ``catalog.schema.table`` fully qualified name. Accepts:

        * a SQLAlchemy URL string (e.g. ``sqlite:///metadata.db``,
          ``postgresql+psycopg://user@host/db``),
        * a DuckLake-native connection string (e.g.
          ``ducklake:sqlite:metadata.db``,
          ``ducklake:postgres:dbname=foo host=bar user=baz``,
          ``ducklake:duckdb:metadata.duckdb``,
          ``ducklake:mysql:database=foo host=bar user=root``),
        * a pre-built :class:`sqlalchemy.engine.Engine`.

        The input is parsed and a SQLAlchemy engine is constructed
        internally; users do not need to interact with the catalog
        layer directly.
    table:
        The DuckLake table to read. Either an unqualified name
        (``"sales"`` — uses the default schema ``"main"``) or a
        fully-qualified ``"schema.table"`` form (``"analytics.sales"``).
        For tables whose name itself contains a dot, pass the unqualified
        name here and the schema explicitly via ``schema=``.
    schema:
        DuckLake schema name. When ``None`` (the default), parsed from
        ``table`` if it is dotted, otherwise falls back to ``"main"``.
        Passing both an explicit ``schema=`` *and* a dotted ``table``
        raises :class:`ValueError`.
    snapshot_id:
        Pin the read to a specific snapshot id (time travel).
        Mutually exclusive with ``as_of``.
    as_of:
        Read the latest snapshot whose ``snapshot_time`` is at or before
        this timestamp. Mutually exclusive with ``snapshot_id``.
    storage_options:
        Forwarded verbatim to :func:`polars.scan_parquet`. Useful for cloud
        credentials and region settings (``aws_region``, ``aws_access_key_id``,
        etc).
    data_path:
        Optional override for the lake's ``data_path`` prefix. Normally
        read from ``ducklake_metadata``; useful when the lake has been
        relocated or for testing.

    Returns
    -------
    polars.LazyFrame
        Backed by ``scan_parquet`` over the resolved data-file paths,
        with positional deletes applied per-file via anti-join and
        per-file column projections that translate older Parquet schemas
        into the target snapshot's column layout.

    Raises
    ------
    ValueError
        If ``snapshot_id`` and ``as_of`` are both supplied, or if both
        ``schema=`` is given and ``table`` is dotted.
    LookupError
        If the requested snapshot, schema, or table is not visible in
        the metadata catalog.
    NotImplementedError
        If the table has any nested-type columns or any inlined data.
        These are refused loudly rather than silently producing wrong
        results.
    """
    if snapshot_id is not None and as_of is not None:
        raise ValueError("snapshot_id and as_of are mutually exclusive")

    schema, table = _resolve_table_identifier(schema=schema, table=table)

    with CatalogReader.from_metadata_catalog(metadata_catalog) as reader:
        target_snapshot = reader.resolve_snapshot_id(snapshot_id=snapshot_id, as_of=as_of)
        schema_info = reader.resolve_schema(schema_name=schema, snapshot_id=target_snapshot)
        table_info = reader.resolve_table(
            schema_id=schema_info.schema_id,
            table_name=table,
            snapshot_id=target_snapshot,
        )
        table_id = table_info.table_id

        if reader.has_inlined_data(table_id=table_id):
            raise NotImplementedError(
                f"Table {table!r} has inlined data stored directly in the catalog "
                "(see ducklake_inlined_data_tables). polars-ducklake's scan_ducklake "
                "is designed for the large-dataset path; inlined-data support is on "
                "the future-work list. To unblock today, set "
                "DATA_INLINING_ROW_LIMIT=0 on the catalog or run the writer's "
                "inline-flush command to materialize inlined rows into Parquet."
            )

        target_columns = _build_target_columns(
            reader.fetch_columns_with_nested(table_id=table_id, snapshot_id=target_snapshot)
        )
        polars_schema = {tc.name: tc.dtype for tc in target_columns}

        data_files = reader.fetch_data_files(table_id=table_id, snapshot_id=target_snapshot)
        delete_files = reader.fetch_delete_files(table_id=table_id, snapshot_id=target_snapshot)
        effective_data_path = data_path if data_path is not None else reader.fetch_data_path()

        if not data_files:
            # Empty table: build a zero-row LazyFrame with the catalog schema.
            # Returning an empty pl.scan_parquet([]) would raise; this preserves
            # the public contract (LazyFrame with the right schema, .collect()
            # yields zero rows) without that.
            return pl.LazyFrame(schema=polars_schema)

        schema_segment = PathSegment(
            path=schema_info.path, is_relative=schema_info.path_is_relative
        )
        table_segment = PathSegment(path=table_info.path, is_relative=table_info.path_is_relative)

        def _resolve(path: str, path_is_relative: bool) -> str:
            return resolve_path(
                path=path,
                path_is_relative=path_is_relative,
                data_path=effective_data_path,
                schema_segment=schema_segment,
                table_segment=table_segment,
            )

        # Group delete files by the data file they apply to. The catalog
        # link via data_file_id is authoritative: even if a delete-file
        # Parquet were to carry rows for multiple data files (its on-disk
        # `file_path` column would distinguish them), the catalog tells us
        # which data file each catalog row applies to.
        deletes_by_file: dict[int, list[str]] = defaultdict(list)
        for df in delete_files:
            deletes_by_file[df.data_file_id].append(_resolve(df.path, df.path_is_relative))

        # Decide each file's projection vs target. When *every* file's schema
        # at its begin_snapshot matches target (no rename, no add/drop), and
        # no file has deletes, we take the fast path: a single scan_parquet
        # over all paths so Polars can optimize multi-file reads. Otherwise
        # we go file-by-file, which is robust to schema evolution at the
        # cost of losing cross-file optimization.
        per_file_plans = [
            _plan_file(
                reader,
                table_id=table_id,
                file=f,
                target_snapshot=target_snapshot,
                target_columns=target_columns,
                full_path=_resolve(f.path, f.path_is_relative),
                delete_paths=deletes_by_file.get(f.data_file_id, []),
            )
            for f in data_files
        ]

    fast_path = not delete_files and all(plan.is_pure_passthrough for plan in per_file_plans)
    if fast_path:
        clean_paths = [plan.full_path for plan in per_file_plans]
        return pl.scan_parquet(clean_paths, storage_options=storage_options).select(
            list(polars_schema.keys())
        )

    frames = [plan.build(storage_options=storage_options) for plan in per_file_plans]
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="vertical")


@dataclass(frozen=True)
class _FilePlan:
    """One data file's read plan: scan, optional anti-join, optional row
    filter for partial files, then column projection.

    The build order matters: the partial-file row filter must read
    ``_ducklake_internal_snapshot_id`` *before* the projection drops it.
    """

    full_path: str
    delete_paths: tuple[str, ...]
    projection: tuple[pl.Expr, ...]
    is_pure_passthrough: bool
    partial_filter_at: int | None  # target snapshot id, or None if no filter needed

    def build(self, *, storage_options: dict[str, Any] | None) -> pl.LazyFrame:
        if self.delete_paths:
            frame = _scan_with_deletes(
                data_path=self.full_path,
                delete_paths=list(self.delete_paths),
                storage_options=storage_options,
            )
        else:
            frame = pl.scan_parquet(self.full_path, storage_options=storage_options)
        if self.partial_filter_at is not None:
            frame = frame.filter(pl.col(_PARTIAL_SNAPSHOT_COL) <= self.partial_filter_at)
        return frame.select(list(self.projection))


def _plan_file(
    reader: CatalogReader,
    *,
    table_id: int,
    file: DataFileInfo,
    target_snapshot: int,
    target_columns: list[_TargetColumn],
    full_path: str,
    delete_paths: list[str],
) -> _FilePlan:
    """Build a per-file projection that maps the file's physical schema to target.

    For each target column, look up what name it had at the file's
    ``begin_snapshot``: that's the name physically stored in the Parquet.
    If the column existed at that snapshot, project it (renaming if the
    name has changed since); otherwise null-fill at the target dtype.

    Partial files (``partial_max`` IS NOT NULL) need an extra row filter
    when the read target is older than ``partial_max`` — see _FilePlan.
    """
    file_columns_by_id = {
        c.column_id: c
        for c in reader.fetch_columns(table_id=table_id, snapshot_id=file.begin_snapshot)
    }

    needs_partial_filter = file.partial_max is not None and target_snapshot < file.partial_max

    exprs: list[pl.Expr] = []
    is_pure = (
        not delete_paths
        and not needs_partial_filter
        and len(file_columns_by_id) == len(target_columns)
    )
    for tc in target_columns:
        historical = file_columns_by_id.get(tc.column_id)
        if historical is None:
            # Column was added after this file was written → null-fill.
            exprs.append(pl.lit(None, dtype=tc.dtype).alias(tc.name))
            is_pure = False
        elif historical.column_name != tc.name:
            # Column was renamed → read by old name, alias to new.
            source = pl.col(historical.column_name)
            if tc.needs_string_cast:
                source = source.cast(pl.String)
                is_pure = False
            exprs.append(source.alias(tc.name))
            is_pure = False
        elif tc.needs_string_cast:
            # JSON column: Polars surfaces it as Binary; bridge to String
            # so the user-visible dtype matches our declared schema.
            exprs.append(pl.col(tc.name).cast(pl.String).alias(tc.name))
            is_pure = False
        else:
            exprs.append(pl.col(tc.name))
    return _FilePlan(
        full_path=full_path,
        delete_paths=tuple(delete_paths),
        projection=tuple(exprs),
        is_pure_passthrough=is_pure,
        partial_filter_at=target_snapshot if needs_partial_filter else None,
    )


def _resolve_table_identifier(*, schema: str | None, table: str) -> tuple[str, str]:
    """Parse the user-provided schema/table into a ``(schema, table)`` pair.

    * Explicit ``schema=`` always wins; passing it together with a
      dotted ``table`` is ambiguous and raises :class:`ValueError`.
    * Dotted ``table`` (no explicit ``schema=``) is split on the first
      dot — ``"analytics.sales"`` → ``("analytics", "sales")``.
    * Otherwise ``schema`` defaults to ``"main"`` (DuckLake's default).
    """
    if schema is not None:
        if "." in table:
            raise ValueError(
                f"Cannot pass both an explicit schema={schema!r} and a "
                f"dotted table name {table!r}; choose one form."
            )
        return schema, table
    if "." in table:
        parsed_schema, _, parsed_table = table.partition(".")
        return parsed_schema, parsed_table
    return "main", table


def _build_target_columns(rows: list[ColumnInfo]) -> list[_TargetColumn]:
    """Assemble user-visible columns from the catalog's flat row set.

    ``rows`` includes every ``ducklake_column`` row for the table, top-
    level rows (``parent_column_id is None``) and nested-decomposition
    rows together. Top-level rows become :class:`_TargetColumn` entries;
    nested rows are folded into their parent's ``pl.DataType`` via
    :func:`build_nested_type`.

    Type-mapping happens here (eagerly) so unsupported types raise at
    scan-build time, not when the user calls ``.collect()`` minutes later.
    """
    children_by_parent: dict[int, list[ColumnInfo]] = {}
    for c in rows:
        if c.parent_column_id is not None:
            children_by_parent.setdefault(c.parent_column_id, []).append(c)

    def _resolve(c: ColumnInfo) -> pl.DataType:
        if not is_nested_type(c.column_type):
            return map_type(c.column_type, column_name=c.column_name)
        # Nested: recursively resolve each child, then assemble.
        kids = children_by_parent.get(c.column_id, [])
        # Sort by column_order to keep struct field order deterministic.
        kids = sorted(kids, key=lambda k: k.column_order)
        resolved_kids = [(k.column_name, _resolve(k)) for k in kids]
        return build_nested_type(c.column_type, resolved_kids, column_name=c.column_name)

    return [
        _TargetColumn(
            column_id=c.column_id,
            name=c.column_name,
            dtype=_resolve(c),
            column_type=c.column_type.strip().lower(),
        )
        for c in rows
        if c.parent_column_id is None
    ]


def _scan_with_deletes(
    *,
    data_path: str,
    delete_paths: list[str],
    storage_options: dict[str, Any] | None,
) -> pl.LazyFrame:
    """Read one data file with positional deletes applied via anti-join.

    DuckLake delete files are Parquet with schema ``(file_path: String,
    pos: Int64)`` where ``pos`` is the 0-indexed row position in the
    referenced data file. We assign matching row indices to the data-file
    scan and anti-join against the union of all ``pos`` values from every
    delete file pointing at this data file.
    """
    data_lf = pl.scan_parquet(data_path, storage_options=storage_options).with_row_index(
        _ROW_INDEX_COL
    )
    delete_lf = pl.scan_parquet(delete_paths, storage_options=storage_options).select(
        pl.col("pos").cast(pl.UInt32)
    )
    return data_lf.join(delete_lf, left_on=_ROW_INDEX_COL, right_on="pos", how="anti").drop(
        _ROW_INDEX_COL
    )
