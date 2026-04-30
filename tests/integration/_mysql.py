"""MySQL helper for catalog-backend integration tests.

Mirrors :mod:`tests.integration._postgres` but for MySQL. Runs against
the dedicated database configured via ``MYSQL_TEST_DATABASE`` in
``tests/.env`` (matching the project-local ``docker-compose.yml`` stack).
Every destructive operation is gated on the test-database name.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import sqlalchemy
from sqlalchemy.engine import Engine

from tests.integration._env import env


@dataclass(frozen=True)
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )

    @property
    def ducklake_native_string(self) -> str:
        return (
            f"ducklake:mysql:database={self.database} host={self.host} "
            f"port={self.port} user={self.user} password={self.password}"
        )

    @property
    def duckdb_attach_string(self) -> str:
        """The ATTACH form DuckDB's ducklake extension uses for MySQL."""
        return (
            f"ducklake:mysql:database={self.database} host={self.host} "
            f"port={self.port} user={self.user} password={self.password}"
        )


def load_config() -> MySQLConfig | None:
    values = env()
    required = (
        "MYSQL_HOST",
        "MYSQL_PORT",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
        "MYSQL_TEST_DATABASE",
    )
    if not all(values.get(k) for k in required):
        return None
    return MySQLConfig(
        host=values["MYSQL_HOST"],
        port=int(values["MYSQL_PORT"]),
        user=values["MYSQL_USER"],
        password=values["MYSQL_PASSWORD"],
        database=values["MYSQL_TEST_DATABASE"],
    )


class MySQLTestClient:
    """SQLAlchemy-backed wrapper restricted to the test database."""

    def __init__(self, config: MySQLConfig) -> None:
        self.config = config
        self._engine = sqlalchemy.create_engine(config.sqlalchemy_url)

    @property
    def engine(self) -> Engine:
        return self._engine

    def reachable(self) -> bool:
        try:
            with self._engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            return True
        except Exception:
            return False

    def _guard_db(self, db: str) -> None:
        if db != self.config.database:
            raise RuntimeError(
                f"Refusing destructive op on database {db!r}: integration "
                f"tests are restricted to {self.config.database!r}."
            )

    def drop_all_ducklake_tables(self) -> int:
        """Drop every ``ducklake_*`` table in the test database.

        Disposes of the SQLAlchemy connection pool when done — DuckDB's
        mysql extension opens its own MySQL connections and we've seen
        "server has gone away" errors when SQLAlchemy idle connections
        are still in the pool while DuckDB does DDL on the same DB. The
        helper is short-lived per-test, so disposing is cheap.
        """
        self._guard_db(self.config.database)
        dropped = 0
        with self._engine.begin() as conn:
            inspector = sqlalchemy.inspect(conn)
            for table_name in inspector.get_table_names():
                if table_name.startswith("ducklake_"):
                    conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS `{table_name}`"))
                    dropped += 1
        self._engine.dispose()
        return dropped


@contextmanager
def isolated_mysql_catalog(client: MySQLTestClient) -> Iterator[MySQLConfig]:
    client.drop_all_ducklake_tables()
    try:
        yield client.config
    finally:
        client.drop_all_ducklake_tables()
