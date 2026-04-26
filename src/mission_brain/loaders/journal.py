"""Mission-bullet journal loader (rb-036 + partial rb-037).

Walks ``corpus/journal/`` and emits :class:`SourceDocument`
instances tagged with provenance ``"journal-private"`` by
default. The privacy floor is enforced downstream via
``--exclude-provenance journal-private`` on ``mission-brain query``
(rb-032) — bot consumers default-exclude journal data even when
ingest is enabled (rb-035 default ``journal_enabled=True``).

Layers handled:

- **Raw** (rb-036): ``entries/YYYY/MM/DD.md`` — Ray's
  rapid-log entries, untouched. The voice-bearing layer.
- **Monthly** (rb-037): ``entries/YYYY/MM/monthly.md`` — monthly
  aggregate / log file when present.

Skipped (with reasons):

- ``DD.refined.md`` — **AI-polished output**, not Ray's voice.
  Inspected 2026-04-26: mission-bullet's refined files carry an
  explicit header ``"AI-REFINED OUTPUT — do not treat this as
  your own writing"`` with metadata showing the polish is done
  by ``openrouter:anthropic/claude-sonnet-4-6``. The rb-037 task
  description assumed refined = Ray's edit-toward-voice; reality
  is refined = Claude's polish of Ray's notes. Ingesting refined
  pollutes voice retrieval (two layers of LLM removal from Ray's
  actual voice) and adds redundant content (refined covers the
  same topics as raw, just paraphrased). If a future use case
  needs refined-derived content, it should be added back with a
  ``"derived"`` provenance tag (or a ``journal-private-derived``
  variant) and an explicit opt-in flag, NOT bundled with raw.
- ``DD.claude.md`` — chat transcripts, deferred per rb-039.
- ``DD.sketch.excalidraw`` — binary, never ingest.
- Any other ``.md`` at the day-folder depth not matching the
  raw / monthly patterns.

Mirrors :class:`mission_brain.loaders.plain_markdown.PlainMarkdownLoader`
in shape (stable hash, line-range locator, optional frontmatter).
The provenance default is the load-bearing difference: tagging
``"journal-private"`` at the loader is the cleanest seam for
preserving the privacy invariant; consumers that DO want journal
content (the ``consult_voice`` MCP server with explicit opt-in)
override the exclude_provenance default.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

import frontmatter

from mission_brain.loaders.base import SourceDocument, SourceLoader
from mission_brain.schema.wiki_page import Provenance

__all__ = ["MissionBulletLoader"]

# Raw entry: DD.md (two-digit day stem).
# Refined entry: DD.refined.md (rb-037).
# Monthly entry: monthly.md (rb-037).
# Excludes:
#   - DD.claude.md (rb-039 deferred)
#   - any other .md at this depth (e.g., README.md, NOTES.md)
_RAW_ENTRY_STEM = re.compile(r"^\d{2}$")
_REFINED_NAME = re.compile(r"^(?P<day>\d{2})\.refined\.md$")
_MONTHLY_NAME = "monthly.md"
_CHAT_SUFFIX = ".claude.md"


class MissionBulletLoader(SourceLoader):
    source_type = "mission_bullet_journal"
    supported_extensions = [".md"]
    version = "1.0"

    def __init__(self, provenance: Provenance = "journal-private") -> None:
        """``provenance`` defaults to ``"journal-private"`` per rb-035 —
        downstream consumers exclude this provenance by default. Override
        when constructing the loader if a different tag is needed (e.g.,
        a future "journal-public" subset for entries Ray explicitly
        marks shareable).
        """
        self._provenance: Provenance = provenance

    def discover(self, corpus_root: Path) -> Iterable[Path]:
        """Walk corpus_root/journal/entries/YYYY/MM/ for journal layers.

        Yields raw (DD.md) and monthly (monthly.md) entries in
        deterministic sorted order. Refined entries (DD.refined.md)
        are **explicitly skipped** — they're AI-polished, not Ray's
        voice (see module docstring for the 2026-04-26 finding).
        Also skips chat transcripts (DD.claude.md, rb-039 deferred),
        Excalidraw sketches, and any other .md at this depth.
        """
        entries = corpus_root / "journal" / "entries"
        if not entries.is_dir():
            return []
        out: list[Path] = []
        for p in entries.rglob("*.md"):
            if not p.is_file():
                continue
            name = p.name
            if name.endswith(_CHAT_SUFFIX):
                continue
            if _REFINED_NAME.match(name):
                # AI-polished, redundant with raw; see module docstring.
                continue
            if name == _MONTHLY_NAME:
                out.append(p)
                continue
            if _RAW_ENTRY_STEM.match(p.stem):
                out.append(p)
                continue
            # Anything else (README.md, notes.md, etc.) — skip.
        return sorted(out)

    def load(self, path: Path) -> SourceDocument:
        raw = path.read_bytes()
        post = frontmatter.loads(raw.decode("utf-8"))
        body = post.content
        meta = dict(post.metadata)

        title = self._infer_title(meta, path)
        line_count = len(body.splitlines())

        return SourceDocument(
            title=title,
            body=body,
            frontmatter=meta,
            line_count=line_count,
            content_hash=self.content_hash(raw),
            source_path=path,
            source_type=self.source_type,
            loader_version=self.version,
            provenance=self._provenance,
        )

    def content_hash(self, raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def locator_for(self, span: tuple[int, int]) -> str:
        start, end = span
        if start < 1 or end < start:
            raise ValueError(f"invalid span {span!r}: require 1 <= start <= end")
        return f"lines={start}-{end}"

    @staticmethod
    def _infer_title(meta: dict, path: Path) -> str:
        """Derive a date-stamped title for raw / refined / monthly layers.

        Preference order:
          1. ``meta["title"]`` if present and non-empty
          2. Layer-aware fallback derived from path:
             - ``entries/YYYY/MM/DD.md``           -> ``Journal YYYY-MM-DD``
             - ``entries/YYYY/MM/DD.refined.md``   -> ``Journal YYYY-MM-DD (refined)``
             - ``entries/YYYY/MM/monthly.md``      -> ``Journal YYYY-MM (monthly)``
          3. ``path.stem`` fallback (defense if path layout is
             unexpected)
        """
        fm_title = meta.get("title")
        if isinstance(fm_title, str) and fm_title.strip():
            return fm_title.strip()
        try:
            name = path.name
            month = path.parent.name
            year = path.parent.parent.name
            year_ok = len(year) == 4 and year.isdigit()
            month_ok = len(month) == 2 and month.isdigit()
            if year_ok and month_ok:
                if name == _MONTHLY_NAME:
                    return f"Journal {year}-{month} (monthly)"
                refined_match = _REFINED_NAME.match(name)
                if refined_match:
                    day = refined_match.group("day")
                    return f"Journal {year}-{month}-{day} (refined)"
                day = path.stem
                if len(day) == 2 and day.isdigit():
                    return f"Journal {year}-{month}-{day}"
        except (AttributeError, IndexError):
            pass
        return path.stem
