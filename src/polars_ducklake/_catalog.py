"""Internal: connect to a DuckLake metadata catalog and read its tables.

This module owns every metadata-catalog operation
:func:`polars_ducklake.scan_ducklake` performs:

* parsing the user-supplied connection string (SQLAlchemy URL,
  DuckLake-native ``ducklake:...`` form, or a pre-built ``Engine``),
* opening / closing the SQL connection,
* issuing each catalog query with the spec's MVCC visibility filter,
* returning the result as a small set of dataclasses the rest of the
  package consumes.

The whole reader is a single class — :class:`CatalogReader` — used as
a context manager. Earlier iterations split this into a Backend / Session
pair behind ABCs with per-dialect subclasses; that abstraction had a
single real implementation and was deleted as boilerplate. If a future
non-SQLAlchemy catalog source ever appears (a REST catalog, a direct
DuckDB driver fast-path), the seam is one small refactor away — but
adding it now to satisfy a hypothetical second adapter is the wrong
trade.

This module is private; the only public surface in the package is
:func:`polars_ducklake.scan_ducklake`. Names here may change without
notice.

See https://ducklake.select/docs/stable/specification/queries for the
canonical catalog SQL pattern this code follows.
"""

from __future__ import annotations

import shlex
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import sqlalchemy
from sqlalchemy import text
from sqlalchemy.engine import Engine

if TYPE_CHECKING:
    from types import TracebackType

    from sqlalchemy.engine import Connection


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaInfo:
    """A DuckLake schema row, including its path contribution."""

    schema_id: int
    path: str
    path_is_relative: bool


@dataclass(frozen=True)
class TableInfo:
    """A DuckLake table row, including its path contribution."""

    table_id: int
    path: str
    path_is_relative: bool


@dataclass(frozen=True)
class ColumnInfo:
    """A column of a DuckLake table at a given snapshot.

    Top-level columns have ``parent_column_id is None``. For nested
    types (LIST / STRUCT / MAP) the catalog stores additional rows
    whose ``parent_column_id`` points back at the column they belong
    to — see :meth:`CatalogReader.fetch_columns_with_nested`.
    """

    column_id: int
    column_name: str
    column_type: str
    column_order: int
    parent_column_id: int | None = None


@dataclass(frozen=True)
class DataFileInfo:
    """A single Parquet data file participating in a snapshot scan.

    ``begin_snapshot`` is needed by the schema-evolution path: a file
    written at snapshot N has the table schema as it existed at N, which
    may differ from the read-target snapshot's schema (column adds/drops/
    renames). Querying ``ducklake_column`` at ``begin_snapshot`` recovers
    the file's physical column names and types.

    ``partial_max`` is non-NULL only for files produced by compaction
    that span multiple snapshots. When set, the file carries an extra
    ``_ducklake_internal_snapshot_id`` column and time-travel reads at
    a target below ``partial_max`` must filter rows whose origin
    snapshot exceeds the target.
    """

    data_file_id: int
    path: str
    path_is_relative: bool
    record_count: int
    begin_snapshot: int
    partial_max: int | None


@dataclass(frozen=True)
class DeleteFileInfo:
    """A single positional delete file referenced from a snapshot."""

    data_file_id: int
    path: str
    path_is_relative: bool


@dataclass(frozen=True)
class FileColumnStat:
    """Per-file, per-column statistics from ``ducklake_file_column_stats``.

    ``min_value`` and ``max_value`` are the *raw* VARCHAR forms written by
    the writer. Coercing them to typed Python values is the predicate
    pruner's job (it knows the column's logical type).
    """

    data_file_id: int
    column_id: int
    min_value: str | None
    max_value: str | None
    null_count: int | None
    value_count: int | None
    contains_nan: bool | None


# ---------------------------------------------------------------------------
# Connection-string parsing
# ---------------------------------------------------------------------------


_DUCKLAKE_PREFIX = "ducklake:"


def _translate_ducklake_string(metadata_catalog: str) -> str:
    """Translate a ``ducklake:...`` connection string to a SQLAlchemy URL.

    Supported forms:

    ``ducklake:sqlite:path/to/file.db``
        -> ``sqlite:///path/to/file.db``
    ``ducklake:postgres:dbname=foo host=bar user=baz [password=...] [port=...]``
        -> ``postgresql+psycopg://baz[:pw]@bar[:port]/foo``
    ``ducklake:duckdb:path/to/file.duckdb``
        -> ``duckdb:///path/to/file.duckdb`` (requires ``duckdb-engine``).
    ``ducklake:mysql:database=foo host=bar user=root [password=...] [port=...]``
        -> ``mysql+pymysql://root[:pw]@bar[:port]/foo``.
    ``ducklake:metadata.ducklake`` (no backend prefix, treated as DuckDB file)
        -> ``duckdb:///metadata.ducklake`` (requires ``duckdb-engine``).
    """
    assert metadata_catalog.startswith(_DUCKLAKE_PREFIX)
    body = metadata_catalog[len(_DUCKLAKE_PREFIX) :]

    if body.startswith("sqlite:"):
        path = body[len("sqlite:") :]
        if not path:
            raise ValueError("ducklake:sqlite: requires a database path")
        return f"sqlite:///{path}"

    if body.startswith("postgres:"):
        return _translate_postgres(body[len("postgres:") :])

    if body.startswith("duckdb:"):
        path = body[len("duckdb:") :]
        if not path:
            raise ValueError("ducklake:duckdb: requires a database path")
        return f"duckdb:///{path}"

    if body.startswith("mysql:"):
        return _translate_mysql(body[len("mysql:") :])

    if body.startswith("postgresql:"):
        raise ValueError(
            "Unsupported DuckLake backend prefix 'postgresql:'; "
            "use ducklake:postgres:..., ducklake:sqlite:..., ducklake:duckdb:..., "
            "ducklake:mysql:..., or pass a SQLAlchemy URL directly"
        )

    if not body:
        raise ValueError("ducklake: requires a path or backend specifier after the prefix")

    return f"duckdb:///{body}"


def _translate_postgres(body: str) -> str:
    """Translate libpq-style key=value parameters to a Postgres SQLAlchemy URL.

    Example: ``dbname=foo host=bar user=baz`` ->
    ``postgresql+psycopg://baz@bar/foo``.
    """
    if not body.strip():
        raise ValueError("ducklake:postgres: requires libpq-style key=value parameters")

    pairs = _parse_kv(body, kind="postgres")
    dbname = pairs.get("dbname") or pairs.get("database")
    if not dbname:
        raise ValueError("ducklake:postgres: requires a 'dbname' parameter")
    return _build_url("postgresql+psycopg", pairs, dbname)


def _translate_mysql(body: str) -> str:
    """Translate libpq-style key=value parameters to a MySQL SQLAlchemy URL.

    DuckLake's MySQL writer accepts the same key=value form as Postgres
    (``host``, ``port``, ``user``, ``password``, ``database`` / ``db``).
    We mirror it for symmetry.
    """
    if not body.strip():
        raise ValueError("ducklake:mysql: requires libpq-style key=value parameters")

    pairs = _parse_kv(body, kind="mysql")
    dbname = pairs.get("database") or pairs.get("db") or pairs.get("dbname")
    if not dbname:
        raise ValueError("ducklake:mysql: requires a 'database' (or 'db') parameter")
    return _build_url("mysql+pymysql", pairs, dbname)


def _parse_kv(body: str, *, kind: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for token in shlex.split(body):
        if "=" not in token:
            raise ValueError(f"Malformed {kind} parameter {token!r}; expected key=value")
        key, value = token.split("=", 1)
        pairs[key.strip().lower()] = value
    return pairs


def _build_url(scheme: str, pairs: dict[str, str], dbname: str) -> str:
    host = pairs.get("host", "localhost")
    user = pairs.get("user", "")
    password = pairs.get("password", "")
    port = pairs.get("port")

    userinfo = ""
    if user:
        userinfo = user
        if password:
            userinfo = f"{user}:{password}"
        userinfo = f"{userinfo}@"

    netloc = host
    if port:
        netloc = f"{host}:{port}"

    return f"{scheme}://{userinfo}{netloc}/{dbname}"


def _require_duckdb_engine() -> None:
    """Raise a clear ImportError if duckdb-engine isn't installed."""
    try:
        import duckdb_engine  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Using DuckDB as the DuckLake catalog backend requires the optional "
            "'duckdb-engine' dependency. Install it with: "
            "pip install 'polars-ducklake[duckdb-catalog]'"
        ) from exc


# Process-wide cache of engines built from string inputs. Keyed by the
# post-translation SQLAlchemy URL so ``ducklake:sqlite:foo.db`` and
# ``sqlite:///foo.db`` collapse to the same entry. Pre-built ``Engine``
# arguments bypass the cache entirely — the caller owns disposal.
#
# Engines are kept for process lifetime; they hold a connection pool
# (typically idle) but no checked-out resources between scans, so the
# memory cost is small compared to the per-scan handshake we save.
_ENGINE_CACHE: dict[str, Engine] = {}
_ENGINE_CACHE_LOCK = threading.Lock()


def _engine_from_metadata_catalog(metadata_catalog: str | Engine) -> Engine:
    """Normalise any catalog input into a SQLAlchemy ``Engine``.

    Internal helper for :meth:`CatalogReader.from_metadata_catalog`; not
    exported. Accepts:

    * an already-built :class:`sqlalchemy.engine.Engine`,
    * a SQLAlchemy URL string,
    * a DuckLake-native ``ducklake:...`` connection string.

    String inputs are memoised by their post-translation URL so successive
    scans against the same catalog reuse one SQLAlchemy pool.
    """
    if isinstance(metadata_catalog, Engine):
        return metadata_catalog

    if not isinstance(metadata_catalog, str):
        raise TypeError(
            "metadata_catalog must be a SQLAlchemy Engine, a SQLAlchemy URL string, "
            f"or a DuckLake-native connection string; got {type(metadata_catalog).__name__}"
        )

    url = (
        _translate_ducklake_string(metadata_catalog)
        if metadata_catalog.startswith(_DUCKLAKE_PREFIX)
        else metadata_catalog
    )

    with _ENGINE_CACHE_LOCK:
        cached = _ENGINE_CACHE.get(url)
    if cached is not None:
        return cached

    if url.startswith("duckdb:"):
        _require_duckdb_engine()

    try:
        engine = sqlalchemy.create_engine(url)
    except Exception as exc:
        raise ValueError(
            f"Could not build SQLAlchemy engine from {metadata_catalog!r}: {exc}"
        ) from exc

    # Two threads racing on the same URL each build their own engine; the
    # loser's engine is disposed so its pool is released cleanly.
    with _ENGINE_CACHE_LOCK:
        existing = _ENGINE_CACHE.setdefault(url, engine)
    if existing is not engine:
        engine.dispose()
    return existing


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


# Reusable SQL fragment for MVCC visibility. Bind parameter is :snapshot_id.
_MVCC_CLAUSE = (
    "(:snapshot_id >= begin_snapshot AND (end_snapshot IS NULL OR :snapshot_id < end_snapshot))"
)


class CatalogReader(AbstractContextManager["CatalogReader"]):
    """Read access to a DuckLake metadata catalog.

    Holds a SQLAlchemy ``Engine`` and, while inside a ``with`` block, an
    open ``Connection``. Every catalog query method runs against that
    connection. Outside the ``with`` block the reader is dormant — the
    engine is set up but no connection is open.

    Construction is cheap (no I/O); use the
    :meth:`from_metadata_catalog` factory to build one from a string or
    a pre-built ``Engine``::

        with CatalogReader.from_metadata_catalog("sqlite:///lake.db") as reader:
            target = reader.resolve_snapshot_id(snapshot_id=None, as_of=None)
            ...

    Every query method applies the spec's MVCC visibility clause where
    appropriate; forgetting it on a query produces silent phantom rows.
    The clause lives in :data:`_MVCC_CLAUSE` and is reused.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._conn: Connection | None = None

    @classmethod
    def from_metadata_catalog(cls, metadata_catalog: str | Engine) -> CatalogReader:
        """Build a reader from any supported catalog argument.

        See the module docstring for accepted forms.
        """
        return cls(_engine_from_metadata_catalog(metadata_catalog))

    @classmethod
    def from_engine(cls, engine: Engine) -> CatalogReader:
        """Build a reader directly from an already-resolved engine.

        Used inside the scan after :func:`_engine_from_metadata_catalog`
        has resolved the user's argument once — successive readers in
        the same scan skip the parse/cache lookup that
        :meth:`from_metadata_catalog` performs.
        """
        return cls(engine)

    @property
    def engine(self) -> Engine:
        """Underlying SQLAlchemy engine (for advanced / debug use)."""
        return self._engine

    @property
    def dialect(self) -> str:
        return self._engine.dialect.name

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> CatalogReader:
        self._conn = self._engine.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def _active_conn(self) -> Connection:
        if self._conn is None:
            raise RuntimeError(
                "CatalogReader used outside a `with` block; methods can only be "
                "called while a connection is open."
            )
        return self._conn

    # -- snapshot resolution -------------------------------------------------

    def resolve_snapshot_id(self, *, snapshot_id: int | None, as_of: datetime | None) -> int:
        """Resolve the target snapshot to read at.

        Exactly one of ``snapshot_id`` and ``as_of`` may be set, or both
        ``None`` (latest). Raises ``LookupError`` if no matching snapshot
        exists, ``ValueError`` if both arguments are supplied.
        """
        if snapshot_id is not None and as_of is not None:
            raise ValueError("snapshot_id and as_of are mutually exclusive")
        conn = self._active_conn

        if snapshot_id is not None:
            row = conn.execute(
                text("SELECT snapshot_id FROM ducklake_snapshot WHERE snapshot_id = :snapshot_id"),
                {"snapshot_id": snapshot_id},
            ).first()
            if row is None:
                raise LookupError(f"No snapshot with snapshot_id={snapshot_id} in catalog")
            return int(row[0])

        if as_of is not None:
            row = conn.execute(
                text(
                    "SELECT max(snapshot_id) FROM ducklake_snapshot WHERE snapshot_time <= :as_of"
                ),
                {"as_of": as_of},
            ).first()
            if row is None or row[0] is None:
                raise LookupError(f"No snapshot at or before as_of={as_of!r} in catalog")
            return int(row[0])

        row = conn.execute(text("SELECT max(snapshot_id) FROM ducklake_snapshot")).first()
        if row is None or row[0] is None:
            raise LookupError(
                "Catalog has no snapshots — is this a freshly initialised lake "
                "with no data committed yet?"
            )
        return int(row[0])

    # -- schema / table resolution ------------------------------------------

    def resolve_schema(self, *, schema_name: str, snapshot_id: int) -> SchemaInfo:
        row = self._active_conn.execute(
            text(
                f"""
                SELECT schema_id, path, path_is_relative
                FROM ducklake_schema
                WHERE schema_name = :schema_name
                  AND {_MVCC_CLAUSE}
                """
            ),
            {"schema_name": schema_name, "snapshot_id": snapshot_id},
        ).first()
        if row is None:
            raise LookupError(
                f"No DuckLake schema named {schema_name!r} visible at snapshot {snapshot_id}"
            )
        return SchemaInfo(
            schema_id=int(row[0]),
            path=str(row[1]) if row[1] is not None else "",
            path_is_relative=bool(row[2]),
        )

    def resolve_table(self, *, schema_id: int, table_name: str, snapshot_id: int) -> TableInfo:
        row = self._active_conn.execute(
            text(
                f"""
                SELECT table_id, path, path_is_relative
                FROM ducklake_table
                WHERE schema_id = :schema_id
                  AND table_name = :table_name
                  AND {_MVCC_CLAUSE}
                """
            ),
            {
                "schema_id": schema_id,
                "table_name": table_name,
                "snapshot_id": snapshot_id,
            },
        ).first()
        if row is None:
            raise LookupError(
                f"No table named {table_name!r} in schema_id={schema_id} "
                f"visible at snapshot {snapshot_id}"
            )
        return TableInfo(
            table_id=int(row[0]),
            path=str(row[1]) if row[1] is not None else "",
            path_is_relative=bool(row[2]),
        )

    # -- columns -------------------------------------------------------------

    def fetch_columns(self, *, table_id: int, snapshot_id: int) -> list[ColumnInfo]:
        """Top-level columns only, ordered by ``column_order``."""
        rows = self._active_conn.execute(
            text(
                f"""
                SELECT column_id, column_name, column_type, column_order
                FROM ducklake_column
                WHERE table_id = :table_id
                  AND parent_column IS NULL
                  AND {_MVCC_CLAUSE}
                ORDER BY column_order
                """
            ),
            {"table_id": table_id, "snapshot_id": snapshot_id},
        ).all()
        return [
            ColumnInfo(
                column_id=int(r[0]),
                column_name=str(r[1]),
                column_type=str(r[2]),
                column_order=int(r[3]),
            )
            for r in rows
        ]

    def fetch_columns_with_nested(self, *, table_id: int, snapshot_id: int) -> list[ColumnInfo]:
        """Every column row for the table, including nested children.

        Top-level rows have ``parent_column IS NULL``; nested children
        link via ``parent_column``. Sorted by ``column_order`` so callers
        can preserve the spec's defined ordering when assembling the
        type tree.
        """
        rows = self._active_conn.execute(
            text(
                f"""
                SELECT column_id, column_name, column_type, column_order, parent_column
                FROM ducklake_column
                WHERE table_id = :table_id
                  AND {_MVCC_CLAUSE}
                ORDER BY column_order
                """
            ),
            {"table_id": table_id, "snapshot_id": snapshot_id},
        ).all()
        return [
            ColumnInfo(
                column_id=int(r[0]),
                column_name=str(r[1]),
                column_type=str(r[2]),
                column_order=int(r[3]),
                parent_column_id=int(r[4]) if r[4] is not None else None,
            )
            for r in rows
        ]

    # -- data + delete files -------------------------------------------------

    def fetch_data_files(self, *, table_id: int, snapshot_id: int) -> list[DataFileInfo]:
        rows = self._active_conn.execute(
            text(
                f"""
                SELECT data_file_id, path, path_is_relative, record_count,
                       begin_snapshot, partial_max
                FROM ducklake_data_file
                WHERE table_id = :table_id
                  AND {_MVCC_CLAUSE}
                ORDER BY file_order
                """
            ),
            {"table_id": table_id, "snapshot_id": snapshot_id},
        ).all()
        return [
            DataFileInfo(
                data_file_id=int(r[0]),
                path=str(r[1]),
                path_is_relative=bool(r[2]),
                record_count=int(r[3]) if r[3] is not None else 0,
                begin_snapshot=int(r[4]),
                partial_max=int(r[5]) if r[5] is not None else None,
            )
            for r in rows
        ]

    def fetch_delete_files(self, *, table_id: int, snapshot_id: int) -> list[DeleteFileInfo]:
        rows = self._active_conn.execute(
            text(
                f"""
                SELECT data_file_id, path, path_is_relative
                FROM ducklake_delete_file
                WHERE table_id = :table_id
                  AND {_MVCC_CLAUSE}
                """
            ),
            {"table_id": table_id, "snapshot_id": snapshot_id},
        ).all()
        return [
            DeleteFileInfo(
                data_file_id=int(r[0]),
                path=str(r[1]),
                path_is_relative=bool(r[2]),
            )
            for r in rows
        ]

    def fetch_file_stats(
        self, *, data_file_ids: list[int], column_ids: list[int]
    ) -> list[FileColumnStat]:
        """Fetch min/max/null counts for the given (file, column) pairs.

        Visibility is established by the caller (``data_file_ids`` should
        already be filtered to files visible at the read snapshot), so
        this method does not apply MVCC filters of its own — it's a pure
        lookup against ``ducklake_file_column_stats``.

        Returns an empty list if either input is empty (so callers don't
        have to special-case "no predicate columns" before calling).
        """
        if not data_file_ids or not column_ids:
            return []
        stmt = text(
            """
            SELECT data_file_id, column_id, min_value, max_value,
                   null_count, value_count, contains_nan
            FROM ducklake_file_column_stats
            WHERE data_file_id IN :file_ids
              AND column_id IN :column_ids
            """
        ).bindparams(
            sqlalchemy.bindparam("file_ids", expanding=True),
            sqlalchemy.bindparam("column_ids", expanding=True),
        )
        rows = self._active_conn.execute(
            stmt, {"file_ids": data_file_ids, "column_ids": column_ids}
        ).all()
        return [
            FileColumnStat(
                data_file_id=int(r[0]),
                column_id=int(r[1]),
                min_value=str(r[2]) if r[2] is not None else None,
                max_value=str(r[3]) if r[3] is not None else None,
                null_count=int(r[4]) if r[4] is not None else None,
                value_count=int(r[5]) if r[5] is not None else None,
                contains_nan=bool(r[6]) if r[6] is not None else None,
            )
            for r in rows
        ]

    # -- inlined data + global metadata --------------------------------------

    def has_inlined_data(self, *, table_id: int) -> bool:
        """Detect rows stored directly in the catalog (small writes).

        ``ducklake_inlined_data_tables`` is a *registry* of tracker tables
        — one row per ``(table_id, schema_version)`` pair — not the inlined
        rows themselves. DuckDB's ``ducklake_flush_inlined_data`` empties
        the trackers but leaves the registry rows in place, so the registry
        alone is not a reliable signal. Probe each tracker for at least one
        row before refusing the read.
        """
        if not self._table_exists("ducklake_inlined_data_tables"):
            return False
        rows = self._active_conn.execute(
            text(
                "SELECT table_name FROM ducklake_inlined_data_tables "
                "WHERE table_id = :table_id"
            ),
            {"table_id": table_id},
        ).all()
        if not rows:
            return False
        preparer = self._active_conn.engine.dialect.identifier_preparer
        for (tracker_name,) in rows:
            tracker_name = str(tracker_name)
            # Defensive: the registry can outlive the tracker table itself
            # in some lifecycle paths; treat a missing tracker as empty.
            if not self._table_exists(tracker_name):
                continue
            probe = self._active_conn.execute(
                text(f"SELECT 1 FROM {preparer.quote(tracker_name)} LIMIT 1")
            ).first()
            if probe is not None:
                return True
        return False

    def fetch_data_path(self) -> str | None:
        """Read the lake-wide ``data_path`` value from ``ducklake_metadata``.

        Returns ``None`` if no entry is present.
        """
        # ``key`` is reserved in MySQL — quote it via the dialect preparer
        # so this query is portable across every backend.
        key_col = self._active_conn.engine.dialect.identifier_preparer.quote("key")
        row = self._active_conn.execute(
            text(
                f"""
                SELECT value
                FROM ducklake_metadata
                WHERE {key_col} = 'data_path'
                  AND (scope IS NULL OR scope = 'global')
                LIMIT 1
                """
            )
        ).first()
        if row is None:
            return None
        return str(row[0])

    # -- internals -----------------------------------------------------------

    def _table_exists(self, table_name: str) -> bool:
        """Backend-agnostic check for a catalog table's existence."""
        return sqlalchemy.inspect(self._active_conn).has_table(table_name)
