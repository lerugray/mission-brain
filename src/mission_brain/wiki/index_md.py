"""Maintain ``<vault_root>/index.md`` — the flat wiki catalog.

One line per wiki page, alphabetically by title. Upsert keyed by slug.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from mission_brain.schema import WikiPage
from mission_brain.wiki.store import slugify

__all__ = ["update_index"]


_HEADER = "# mission-brain wiki index"

# One catalog line looks like:
#   - [Title](wiki/slug.md) — sources: a, b — updated: 2026-04-18T12:34:56Z
_ENTRY_RE = re.compile(
    r"^- \[(?P<title>[^\]]+)\]\(wiki/(?P<slug>[a-z0-9-]+)\.md\) "
    r"— sources: (?P<sources>[^—]*?) "
    r"— updated: (?P<updated>\S+)$"
)


def _utcnow_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_entry(title: str, slug: str, source_ids: list[str], updated: str) -> str:
    sources = ", ".join(source_ids) if source_ids else "(none)"
    return f"- [{title}](wiki/{slug}.md) — sources: {sources} — updated: {updated}"


def _parse_existing(raw: str) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for line in raw.splitlines():
        m = _ENTRY_RE.match(line)
        if not m:
            continue
        entries[m.group("slug")] = {
            "title": m.group("title"),
            "slug": m.group("slug"),
            "sources": [
                s.strip() for s in m.group("sources").split(",") if s.strip()
            ]
            if m.group("sources") != "(none)"
            else [],
            "updated": m.group("updated"),
        }
    return entries


def update_index(vault_root: Path, page: WikiPage) -> None:
    """Upsert *page*'s catalog entry in ``<vault_root>/index.md``."""
    vault_root.mkdir(parents=True, exist_ok=True)
    index_path = vault_root / "index.md"
    raw = index_path.read_text(encoding="utf-8") if index_path.exists() else ""

    entries = _parse_existing(raw)
    slug = slugify(page.title)
    entries[slug] = {
        "title": page.title,
        "slug": slug,
        "sources": [str(sid) for sid in page.source_ids],
        "updated": _utcnow_iso(),
    }

    lines = [_HEADER, ""]
    for entry in sorted(entries.values(), key=lambda e: e["title"].lower()):
        lines.append(
            _format_entry(
                entry["title"], entry["slug"], entry["sources"], entry["updated"]
            )
        )
    content = "\n".join(lines) + "\n"
    index_path.write_text(content, encoding="utf-8", newline="\n")
