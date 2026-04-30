"""Shared env-file loader for integration test config.

Loads in priority order (later overrides earlier):

1. ``tests/.env`` — checked-in defaults that match the project-local
   ``docker-compose.yml`` stack.
2. ``tests/.env.local`` — gitignored per-developer overrides.
3. Real ``os.environ`` — for CI or one-off shell exports.

A single load is shared across all integration helpers (MinIO, Postgres,
MySQL) so configuration stays in one place.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_ENV = _TESTS_DIR / ".env"
_LOCAL_ENV = _TESTS_DIR / ".env.local"


@lru_cache(maxsize=1)
def env() -> dict[str, str]:
    """Return the merged config for the current process.

    Cached because (a) re-reading is wasteful and (b) it makes the
    config a stable object across all fixtures in a session.
    """
    out: dict[str, str] = {}
    for path in (_DEFAULT_ENV, _LOCAL_ENV):
        if path.exists():
            out.update(_read(path))
    out.update(os.environ)
    return out


def _read(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        parsed[key.strip()] = value.strip().strip('"').strip("'")
    return parsed
