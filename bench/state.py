"""Persistent registry of materialized bench lakes.

``bench/state.json`` records a mapping from ``(tier, backend)`` to the
exact :class:`bench.targets.CatalogTarget` plus the resolved snapshot
ids and table names a runner needs. The file is gitignored — it's
machine-local infra, not source.

Each entry stores the shape's content hash so the harness can warn
when the persisted lake was built against an older shape definition
than the current code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bench.targets import CatalogTarget


STATE_PATH = Path(__file__).parent / "state.json"
STATE_VERSION = 1


@dataclass
class LakeEntry:
    tier: str
    backend: str
    shape_name: str
    shape_hash: str
    target: CatalogTarget
    table_main: str
    table_evolved: str | None
    mid_snapshot_id: int
    built_sha: str
    built_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


def _key(tier: str, backend: str) -> str:
    return f"{tier}|{backend}"


def load_state(path: Path = STATE_PATH) -> dict[str, LakeEntry]:
    if not path.exists():
        return {}
    blob = json.loads(path.read_text())
    if blob.get("version") != STATE_VERSION:
        raise RuntimeError(
            f"unsupported bench state version: {blob.get('version')!r}; "
            f"delete {path} and re-seed"
        )
    out: dict[str, LakeEntry] = {}
    for k, v in blob.get("lakes", {}).items():
        target = CatalogTarget(**v["target"])
        entry = LakeEntry(
            tier=v["tier"],
            backend=v["backend"],
            shape_name=v["shape_name"],
            shape_hash=v["shape_hash"],
            target=target,
            table_main=v["table_main"],
            table_evolved=v.get("table_evolved"),
            mid_snapshot_id=v["mid_snapshot_id"],
            built_sha=v["built_sha"],
            built_at=v["built_at"],
        )
        out[k] = entry
    return out


def save_state(entries: dict[str, LakeEntry], path: Path = STATE_PATH) -> None:
    blob = {
        "version": STATE_VERSION,
        "lakes": {k: asdict(v) for k, v in entries.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2))


def upsert_entry(entry: LakeEntry, path: Path = STATE_PATH) -> None:
    entries = load_state(path)
    entries[_key(entry.tier, entry.backend)] = entry
    save_state(entries, path)


def get_entry(tier: str, backend: str, path: Path = STATE_PATH) -> LakeEntry | None:
    return load_state(path).get(_key(tier, backend))
