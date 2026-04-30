"""MySQL-as-catalog integration tests.

Status (as of authoring): the DuckDB ducklake+mysql extension exhibits
intermittent and persistent failures when used as a catalog backend
against MySQL 8.0 (mysql_native_password and caching_sha2_password
both tested) — symptoms include "Got packets out of order", "Server
has gone away" mid-CHECKPOINT, and segfaults on consecutive
connections within a single Python process. The same flow works
reliably for SQLite, DuckDB-as-catalog, and PostgreSQL backends.

These tests are therefore marked ``xfail(strict=False)`` so they
neither block CI nor falsely advertise support. Our parser does
accept the ``ducklake:mysql:...`` connection form correctly (see
``tests/test_catalog.py::TestTranslateDuckLakeString::test_mysql_*``)
and the ``[mysql]`` extra installs ``pymysql`` for SQLAlchemy reads —
so the *reader* path works, but a writer-roundtrip integration test
isn't currently reliable.

When the upstream issue is resolved, drop the ``xfail`` marks and
require pass.
"""

from __future__ import annotations

import pytest

from tests.integration._mysql import MySQLTestClient

pytestmark = pytest.mark.integration


def test_mysql_round_trip_unstable_in_duckdb_extension(
    mysql_client: MySQLTestClient,
) -> None:
    """Document the upstream-blocked round-trip so it's visible in CI output.

    The DuckDB ducklake+mysql extension is currently unstable as a
    catalog backend (segfaults on consecutive connections, "Got packets
    out of order" on protocol-level operations, "Server has gone away"
    mid-CHECKPOINT). We can't reliably exercise an end-to-end round-trip
    until that's fixed upstream.

    What *does* work end-to-end and is exercised elsewhere:

    * The reader path against a MySQL-backed catalog —
      ``tests/test_backends_matrix.py`` runs every behavior
      (round-trip, time travel, deletes, schema evolution, partial
      files, empty table) against MySQL via the SQLAlchemy reader.
    * Connection-string parsing — see ``test_catalog.py``'s MySQL cases.
    * Database-isolation guard rails (validated below).

    Skipping rather than xfail-ing because xfail can't catch a segfault
    that takes down the pytest process.
    """
    pytest.skip(
        "DuckDB ducklake+mysql extension is unstable; writer round-trip "
        "blocked on upstream fix. Reader path is exercised via "
        "tests/test_backends_matrix.py."
    )


def test_mysql_test_db_isolation_guard(mysql_client: MySQLTestClient) -> None:
    """The DB-name guard works regardless of writer-extension stability."""
    with pytest.raises(RuntimeError, match="restricted"):
        mysql_client._guard_db("mysql")
