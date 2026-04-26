"""Append-only ``<vault_root>/log.md`` — the ingest journal."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

__all__ = ["Action", "append_log"]

Action = Literal["created", "updated", "skipped"]


def _utcnow_iso_micros() -> str:
    # Microsecond resolution so adjacent calls in tests don't collide.
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def append_log(vault_root: Path, source_id: str, action: Action) -> None:
    """Append one timestamped ``<ts> <action> <source_id>`` line."""
    vault_root.mkdir(parents=True, exist_ok=True)
    log_path = vault_root / "log.md"
    line = f"{_utcnow_iso_micros()} {action} {source_id}\n"
    with log_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line)
