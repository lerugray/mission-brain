"""Abstract loader interface (skeleton.md §3).

Every source type implements :class:`SourceLoader` so the ingest
pipeline is uniform across raw-source shapes. Concrete loaders
(plain markdown, PDF, mbox, ...) land per phase.

:class:`SourceDocument` is the canonical parsed shape returned by
``load()``. It is a plain ``@dataclass`` — not a pydantic model —
so loaders can construct it freely without importing pydantic;
the schema-layer ``WikiPage`` model downstream is the validated
surface. The ``provenance`` field (primary vs derived) carries
through to the wiki page.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mission_brain.schema.wiki_page import Provenance

__all__ = ["SourceDocument", "SourceLoader", "canonical_source_id"]

_SID_SCRUB = re.compile(r"[^a-z0-9_-]+")
_SID_COLLAPSE = re.compile(r"-{2,}")


def canonical_source_id(path: Path, corpus_root: Path) -> str:
    """Derive a CLAUDE.md §1-compliant source_id from *path* under *corpus_root*.

    Grammar: ``[a-z0-9_-]+``. Path separators become ``-``; extensions are
    stripped; non-conforming chars collapse to ``-``; runs of ``-`` collapse
    to a single ``-``; leading/trailing ``-`` stripped.
    """
    rel = path.resolve().relative_to(corpus_root.resolve()).with_suffix("")
    raw = "-".join(rel.parts).lower()
    scrubbed = _SID_SCRUB.sub("-", raw)
    return _SID_COLLAPSE.sub("-", scrubbed).strip("-")


@dataclass
class SourceDocument:
    title: str
    body: str
    frontmatter: dict[str, Any]
    line_count: int
    content_hash: str
    source_path: Path
    source_type: str = ""
    loader_version: str = ""
    source_id: str = ""
    provenance: Provenance = "primary"
    extras: dict[str, Any] = field(default_factory=dict)


class SourceLoader(ABC):
    source_type: str
    supported_extensions: list[str]
    version: str

    @abstractmethod
    def discover(self, corpus_root: Path) -> Iterable[Path]:
        """Walk *corpus_root* and yield candidate files for this source type."""

    @abstractmethod
    def load(self, path: Path) -> SourceDocument:
        """Parse *path* into a canonical :class:`SourceDocument`."""

    @abstractmethod
    def content_hash(self, raw: bytes) -> str:
        """Stable hash of *raw* bytes — anchors citation IDs across regens."""

    @abstractmethod
    def locator_for(self, span: tuple[int, int]) -> str:
        """Render a span into the ``locator`` half of a ``[ref:<source>:<locator>]``."""
