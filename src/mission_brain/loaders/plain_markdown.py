"""Plain-markdown loader — Phase 1 proof-shape (skeleton.md §3, CLAUDE.md §4).

Globs ``*.md`` under a corpus root, parses YAML frontmatter via
``python-frontmatter``, hashes raw bytes for stable citation anchors,
and infers a title + optional date from frontmatter/filename.

Per CLAUDE.md §3.1, this is the first loader because plain markdown
is the easiest source shape to prove the full ingest → wiki → query
pipeline against before fragile formats (PDF, mbox, FB export) land.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

import frontmatter

from mission_brain.loaders.base import SourceDocument, SourceLoader
from mission_brain.schema.wiki_page import Provenance

__all__ = ["PlainMarkdownLoader"]

_FILENAME_YYYY_MM = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})(?!\d)")
_H1 = re.compile(r"^\s{0,3}#\s+(?P<title>.+?)\s*#*\s*$", re.MULTILINE)


class PlainMarkdownLoader(SourceLoader):
    source_type = "plain_markdown"
    supported_extensions = [".md"]
    version = "1.0"

    def __init__(self, provenance: Provenance = "primary") -> None:
        """``provenance`` tags every emitted SourceDocument so downstream
        wiki pages can distinguish primary material (user-authored) from
        derived material (content synthesized by another tool before
        reaching mission-brain — e.g. catalogdna's LLM-analyzed Theory/ pages).
        """
        self._provenance: Provenance = provenance

    def discover(self, corpus_root: Path) -> Iterable[Path]:
        return sorted(p for p in corpus_root.rglob("*.md") if p.is_file())

    def load(self, path: Path) -> SourceDocument:
        raw = path.read_bytes()
        post = frontmatter.loads(raw.decode("utf-8"))
        body = post.content
        meta = dict(post.metadata)

        title = self._infer_title(meta, body, path)
        inferred_date = self._infer_date(meta, path)
        if inferred_date is not None and "date" not in meta:
            meta["date"] = inferred_date

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
    def _infer_title(meta: dict, body: str, path: Path) -> str:
        fm_title = meta.get("title")
        if isinstance(fm_title, str) and fm_title.strip():
            return fm_title.strip()
        m = _H1.search(body)
        if m:
            return m.group("title").strip()
        return path.stem

    @staticmethod
    def _infer_date(meta: dict, path: Path) -> str | None:
        if "date" in meta:
            return None
        m = _FILENAME_YYYY_MM.search(path.stem)
        if m:
            return f"{m.group('year')}-{m.group('month')}"
        return None
