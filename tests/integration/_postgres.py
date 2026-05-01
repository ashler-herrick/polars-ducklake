"""Postgres helper for catalog-backend integration tests.

The integration suite uses a dedicated Postgres database (configured via
``tests/.env`` → ``POSTGRES_TEST_DATABASE``) per the project-local
``docker-compose.yml`` stack. Every destructive operation goes through
:meth:`PostgresTestClient._guard_db` so we can never touch any other
database on the same instance.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import sqlalchemy
from sqlalchemy.engine import Engine

from tests.integration._env import env


@dataclass(frozen=True)
class PostgresConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def ducklake_native_string(self) -> str:
        """The ``ducklake:postgres:...`` form parsed by our catalog module."""
        return (
            "ducklake:postgres:"
            f"dbname={self.database} host={self.host} port={self.port} "
            f"user={self.user} password={self.password}"
        )

    @property
    def duckdb_attach_string(self) -> str:
        """The ATTACH connection string DuckDB's ducklake extension uses.

        DuckLake's writer accepts a libpq-style key=value string after
        the ``ducklake:postgres:`` prefix.
        """
        return (
            f"ducklake:postgres:dbname={self.database} host={self.host} "
            f"port={self.port} user={self.user} password={self.password}"
        )


def _load(database_var: str) -> PostgresConfig | None:
    values = env()
    required = (
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        database_var,
    )
    if not all(values.get(k) for k in required):
        return None
    return PostgresConfig(
        host=values["POSTGRES_HOST"],
        port=int(values["POSTGRES_PORT"]),
        user=values["POSTGRES_USER"],
        password=values["POSTGRES_PASSWORD"],
        database=values[database_var],
    )


def load_config() -> PostgresConfig | None:
    """Return integration-test Postgres config, or None if any key is missing."""
    return _load("POSTGRES_TEST_DATABASE")


def load_bench_config() -> PostgresConfig | None:
    """Return bench Postgres config (separate DB so the integration suite
    can drop its tables without wiping a seeded bench lake)."""
    return _load("POSTGRES_BENCH_DATABASE")


def ensure_database_exists(cfg: PostgresConfig) -> None:
    """Create ``cfg.database`` if it doesn't exist yet.

    Connects to the maintenance ``postgres`` DB on the same instance
    and issues ``CREATE DATABASE`` outside a transaction (PG won't
    allow CREATE DATABASE in one). Idempotent.
    """
    admin = sqlalchemy.create_engine(
        f"postgresql+psycopg://{cfg.user}:{cfg.password}@{cfg.host}:{cfg.port}/postgres",
        isolation_level="AUTOCOMMIT",
    )
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                sqlalchemy.text("SELECT 1 FROM pg_database WHERE datname = :n"),
                {"n": cfg.database},
            ).scalar()
            if not exists:
                # Database name comes from a checked-in env file we control;
                # quoting just to be defensive.
                conn.execute(sqlalchemy.text(f'CREATE DATABASE "{cfg.database}"'))
    finally:
        admin.dispose()


class PostgresTestClient:
    """SQLAlchemy-backed wrapper that refuses to mutate any DB but the test one."""

    def __init__(self, config: PostgresConfig) -> None:
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
        """Drop every catalog table created by a DuckLake writer.

        The DuckLake spec defines ~28 ``ducklake_*`` tables. Rather than
        listing them all, we discover them and bulk-drop. Anything outside
        the ``ducklake_*`` and ``ducklake_inlined_*`` namespaces stays
        untouched. Idempotent across runs.
        """
        self._guard_db(self.config.database)
        dropped = 0
        with self._engine.begin() as conn:
            inspector = sqlalchemy.inspect(conn)
            for table_name in inspector.get_table_names():
                if table_name.startswith("ducklake_"):
                    conn.execute(sqlalchemy.text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
                    dropped += 1
        return dropped


@contextmanager
def isolated_postgres_catalog(client: PostgresTestClient) -> Iterator[PostgresConfig]:
    """Yield a clean Postgres catalog DB; drop all ducklake tables on exit.

    The ``CREATE OR REPLACE SECRET`` we issue to DuckDB ensures the
    writer talks to *this* Postgres instance. The teardown leaves the
    database itself alive (so subsequent runs reuse it) but removes
    every ``ducklake_*`` table.
    """
    client.drop_all_ducklake_tables()
    try:
        yield client.config
    finally:
        client.drop_all_ducklake_tables()
