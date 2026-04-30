"""Fixtures for the integration test suite.

These tests rely on the project-local ``docker-compose.yml`` stack. Run
``docker compose up -d`` once before invoking pytest; tests that need a
service whose endpoint isn't reachable will skip rather than fail.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.integration import _minio, _mysql, _postgres


@pytest.fixture(scope="session")
def minio_config() -> _minio.MinIOConfig:
    config = _minio.load_config()
    if config is None:
        pytest.skip("MinIO config not found; verify tests/.env exists or set MINIO_* env vars.")
    return config


@pytest.fixture()
def minio_client(minio_config: _minio.MinIOConfig) -> Iterator[_minio.MinIOTestClient]:
    """Per-test MinIO client with an isolated prefix; cleans up on teardown."""
    client = _minio.MinIOTestClient(minio_config)
    if not client.reachable():
        pytest.skip(
            f"MinIO at {minio_config.endpoint} is not reachable — "
            "run `docker compose up -d` to start the project-local stack."
        )
    try:
        yield client
    finally:
        # Cleanup is bounded to the session prefix by construction
        # (see MinIOTestClient._guard_prefix).
        client.cleanup()


@pytest.fixture(scope="session")
def postgres_config() -> _postgres.PostgresConfig:
    config = _postgres.load_config()
    if config is None:
        pytest.skip("Postgres config not found in tests/.env or env vars.")
    return config


@pytest.fixture()
def postgres_client(
    postgres_config: _postgres.PostgresConfig,
) -> Iterator[_postgres.PostgresTestClient]:
    """Per-test Postgres client with auto-cleanup of ducklake_* tables.

    Drops on *entry* and teardown so a previous crashed test (which
    skipped its teardown) doesn't leave stale catalog rows that conflict
    with the next test's DATA_PATH.
    """
    client = _postgres.PostgresTestClient(postgres_config)
    if not client.reachable():
        pytest.skip(
            f"Postgres at {postgres_config.host}:{postgres_config.port} not reachable — "
            "run `docker compose up -d` to start the project-local stack."
        )
    client.drop_all_ducklake_tables()
    try:
        yield client
    finally:
        client.drop_all_ducklake_tables()


@pytest.fixture(scope="session")
def mysql_config() -> _mysql.MySQLConfig:
    config = _mysql.load_config()
    if config is None:
        pytest.skip("MySQL config not found in tests/.env or env vars.")
    return config


@pytest.fixture()
def mysql_client(mysql_config: _mysql.MySQLConfig) -> Iterator[_mysql.MySQLTestClient]:
    """Per-test MySQL client with auto-cleanup of ducklake_* tables.

    Drops on entry and teardown so a previous crashed test doesn't leave
    stale catalog rows.
    """
    client = _mysql.MySQLTestClient(mysql_config)
    if not client.reachable():
        pytest.skip(
            f"MySQL at {mysql_config.host}:{mysql_config.port} not reachable — "
            "run `docker compose up -d` to start the project-local stack."
        )
    client.drop_all_ducklake_tables()
    try:
        yield client
    finally:
        client.drop_all_ducklake_tables()
